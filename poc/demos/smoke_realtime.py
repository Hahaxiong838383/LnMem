"""Day 2 真实采集 smoke test — cv2 + MediaPipe + sounddevice + osascript

阶段：
1. MediaPipe 静态合成图（不需权限）
2. osascript 活动 app（不需权限）
3. sounddevice 1s 录音（首次会弹麦克风权限框）
4. cv2 摄像头 30fps × 60 帧（首次会弹摄像头权限框）

所有结果落 /tmp/lnmem_smoke_results.jsonl
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("/tmp/lnmem_smoke_results.jsonl")
OUT.unlink(missing_ok=True)


def log(name: str, **fields) -> None:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "test": name, **fields}
    OUT.open("a").write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[{name}] {fields}", flush=True)


# === 1. MediaPipe 静态合成图（不需权限） ===
def smoke_mediapipe_static() -> None:
    import mediapipe as mp

    print("\n=== 1. MediaPipe FaceMesh on synthetic image ===", flush=True)
    # 合成 480x640 灰图（无脸 → 应返回 0 landmarks）
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
    ) as fm:
        result = fm.process(blank)
    has_faces = bool(result.multi_face_landmarks)
    log("mediapipe_static", has_faces=has_faces, mp_version=mp.__version__)


# === 2. osascript 活动 app（不需权限） ===
def smoke_active_app() -> None:
    import subprocess

    print("\n=== 2. osascript active app ===", flush=True)
    out = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to get name of first application process whose frontmost is true'],
        capture_output=True,
        text=True,
        timeout=5,
    )
    app = out.stdout.strip()
    log("osascript_active_app", active_app=app, returncode=out.returncode)


# === 3. sounddevice 1s 录音（首次弹麦克风权限） ===
def smoke_audio() -> None:
    import sounddevice as sd

    print("\n=== 3. sounddevice 1s @ 16kHz ===", flush=True)
    sample_rate = 16000
    duration_s = 1.0
    print(f"录 {duration_s}s @ {sample_rate}Hz... 首次可能弹麦克风权限框", flush=True)
    try:
        audio = sd.rec(
            int(duration_s * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        rms = float(np.sqrt(np.mean(audio**2)))
        log(
            "sounddevice_record",
            samples=int(audio.shape[0]),
            actual_sample_rate=sample_rate,
            rms=rms,
            non_silent=rms > 0.0001,
        )
    except Exception as e:
        log("sounddevice_record", error=str(e), error_type=type(e).__name__)


# === 4. cv2 + MediaPipe 摄像头 30fps × 60 帧 ===
def smoke_camera() -> None:
    import cv2
    import mediapipe as mp

    print("\n=== 4. cv2 + MediaPipe 摄像头 60 帧 ===", flush=True)
    print("首次可能弹摄像头权限框（请点'允许'）", flush=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log("cv2_camera", error="VideoCapture(0) failed to open")
        return
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    target_frames = 60
    landmarks_per_frame = []
    t_start = time.time()
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
    ) as fm:
        for i in range(target_frames):
            ok, frame = cap.read()
            if not ok:
                landmarks_per_frame.append(-1)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
            n = (
                len(res.multi_face_landmarks[0].landmark)
                if res.multi_face_landmarks
                else 0
            )
            landmarks_per_frame.append(n)
    elapsed = time.time() - t_start
    cap.release()

    fps = target_frames / elapsed if elapsed > 0 else 0
    successful = sum(1 for n in landmarks_per_frame if n > 0)
    log(
        "cv2_camera_30fps",
        target_frames=target_frames,
        elapsed_s=round(elapsed, 2),
        actual_fps=round(fps, 1),
        face_detected_frames=successful,
        landmarks_max=max(landmarks_per_frame) if landmarks_per_frame else 0,
    )


def main() -> int:
    args = sys.argv[1:]
    if "all" in args or not args:
        smoke_mediapipe_static()
        smoke_active_app()
        smoke_audio()
        smoke_camera()
    else:
        for a in args:
            globals()[f"smoke_{a}"]()
    print(f"\n=== Done. Results in {OUT} ===", flush=True)
    print(OUT.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
