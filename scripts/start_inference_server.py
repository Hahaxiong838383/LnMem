"""Start realtime inference websocket server."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.realtime.inference_server import InferenceServer, RealtimeLNNRuntime


def parse_args() -> argparse.Namespace:
    """Parse CLI args. 英文一句话 + 中文说明：定义启动参数。"""
    p = argparse.ArgumentParser(description="Start LnMem realtime inference websocket server")
    p.add_argument("--ckpt", type=Path, default=None, help="Optional dt-aware checkpoint path")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8765, help="Bind port")
    p.add_argument("--persona-idx", type=int, default=0, help="Persona index for tau bias")
    p.add_argument("--target-fps", type=float, default=30.0, help="Expected incoming frame rate")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    runtime = RealtimeLNNRuntime(
        ckpt_path=args.ckpt,
        target_fps=float(args.target_fps),
        persona_idx=int(args.persona_idx),
    )
    server = InferenceServer(runtime=runtime, host=str(args.host), port=int(args.port))
    await server.serve_forever()


def main() -> None:
    """CLI entrypoint. 英文一句话 + 中文说明：启动并常驻服务。"""
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
