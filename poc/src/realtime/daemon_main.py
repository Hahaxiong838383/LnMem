"""Realtime daemon entrypoint. / 启动采集线程、存储线程和本地 HTTP 状态端点。"""

from __future__ import annotations

from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import queue
import signal
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from .capture_worker import CaptureWorker, CapturedFrame
from .daemon_config import HTTP_HOST, HTTP_PORT, load_config, preflight_check
from .signal_pipeline import SignalPipelineWorker
from .storage_manager import StorageManager


class DaemonApp:
    def __init__(self, device: str | None = None, camera: int | None = None, config_path: str | None = None) -> None:
        self._config = load_config(config_path, device_override=device, camera_override=camera)
        self._preflight = preflight_check(self._config)
        self._stop_event = threading.Event()
        self._started_at = time.monotonic()
        self._session_id = str(uuid4())
        self._capture_out: "queue.Queue[CapturedFrame]" = queue.Queue(maxsize=2)
        self._pipeline_in: "queue.Queue[CapturedFrame]" = queue.Queue(maxsize=2)
        self._pipeline_out: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=64)
        self._storage = StorageManager(self._config.storage, self._stop_event)
        self._capture = CaptureWorker(self._config.capture, self._capture_out, self._stop_event, self._session_id)
        self._pipeline = SignalPipelineWorker(
            self._config.pipeline,
            self._pipeline_in,
            self._pipeline_out,
            self._stop_event,
            self._config.capture.device_name,
        )
        self._router_threads: list[threading.Thread] = []
        self._http_server: ThreadingHTTPServer | None = None

    def start(self) -> None:
        logging.info("starting realtime daemon: %s", json.dumps(self._preflight, ensure_ascii=False))
        self._storage.start()
        self._capture.start()
        self._pipeline.start()
        self._router_threads = [
            threading.Thread(target=self._frame_router_loop, name="lnmem-frame-router", daemon=True),
            threading.Thread(target=self._signal_router_loop, name="lnmem-signal-router", daemon=True),
        ]
        for thread in self._router_threads:
            thread.start()
        self._start_http_server()

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        logging.info("stop requested for session %s", self._session_id)
        self._stop_event.set()
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
        self._capture.join(timeout=5.0)
        self._pipeline.join(timeout=10.0)
        for thread in self._router_threads:
            thread.join(timeout=5.0)
        self._storage.stop()

    def serve_forever(self) -> None:
        self.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(0.2)
        finally:
            self.stop()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "uptime_s": round(time.monotonic() - self._started_at, 3),
            "frames_processed": self._pipeline._frames_processed,
            "last_frame_ts": self._pipeline._last_frame_ts or self._capture.last_frame_ts,
        }

    def recent_signals(self, seconds: int) -> list[dict[str, Any]]:
        return self._storage.get_recent(seconds)

    def _frame_router_loop(self) -> None:
        while not self._stop_event.is_set() or not self._capture_out.empty():
            try:
                item = self._capture_out.get(timeout=0.2)
            except queue.Empty:
                continue
            self._storage.enqueue_frame(item)
            _offer(self._pipeline_in, item)

    def _signal_router_loop(self) -> None:
        while not self._stop_event.is_set() or not self._pipeline_out.empty():
            try:
                item = self._pipeline_out.get(timeout=0.2)
            except queue.Empty:
                continue
            self._storage.enqueue_signal(item)

    def _start_http_server(self) -> None:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    self._write_json(200, app.health())
                    return
                if parsed.path == "/signals/recent":
                    seconds = int(parse_qs(parsed.query).get("seconds", ["300"])[0])
                    self._write_json(200, app.recent_signals(seconds))
                    return
                self._write_json(404, {"error": "not_found"})

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/control/stop":
                    self._write_json(404, {"error": "not_found"})
                    return
                threading.Thread(target=app.stop, name="lnmem-stop", daemon=True).start()
                self._write_json(202, {"status": "stopping"})

            def log_message(self, _fmt: str, *_args: Any) -> None:
                logging.info("http %s", self.path)

            def _write_json(self, code: int, payload: Any) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._http_server = ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), Handler)
        threading.Thread(target=self._http_server.serve_forever, name="lnmem-http", daemon=True).start()


def _request_camera_auth_macos(camera_index: int) -> bool:
    """在主线程触发 macOS TCC 摄像头授权对话框，返回是否授权成功。"""
    import os
    import sys
    if sys.platform != "darwin":
        return True
    # 先不跳过 auth，让系统弹出 TCC 对话框
    os.environ.pop("OPENCV_AVFOUNDATION_SKIP_AUTH", None)
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(camera_index)
        authorized = cap.isOpened()
        cap.release()
        if not authorized:
            logging.warning(
                "camera TCC not authorized. "
                "Grant permission in System Settings → Privacy & Security → Camera, "
                "then restart daemon."
            )
        return authorized
    except Exception as exc:
        logging.warning("camera auth check failed: %s", exc)
        return False


def _load_dotenv() -> None:
    """自动加载项目根目录 poc/.env（含 VLM API key）。不报错，没文件就跳过。"""
    import pathlib
    env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_path, override=False)
            logging.info("daemon: loaded %s", env_path)
        except ImportError:
            # dotenv 未安装时手动解析
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        import os
                        os.environ.setdefault(k, v)
            logging.info("daemon: loaded %s (manual parse)", env_path)


def main() -> None:
    parser = ArgumentParser(description="LnMem realtime capture daemon")
    parser.add_argument("--device", default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-camera-auth", action="store_true",
                        help="跳过主线程 TCC 授权预检（已确认权限时用）")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_dotenv()  # 加载 poc/.env → GEMINI_PROXY_KEY / MINIMAX_API_KEY
    if not args.skip_camera_auth:
        cam_idx = args.camera if args.camera is not None else 0
        _request_camera_auth_macos(cam_idx)
    app = DaemonApp(device=args.device, camera=args.camera, config_path=args.config)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _signum, _frame: app.stop())
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        app.stop()


def _offer(target: queue.Queue[Any], item: Any) -> None:
    try:
        target.put_nowait(item)
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        target.put_nowait(item)


__all__ = ["DaemonApp", "main"]


if __name__ == "__main__":
    main()
