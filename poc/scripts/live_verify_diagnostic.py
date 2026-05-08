#!/usr/bin/env python3
"""live_verify 诊断版 — 查根因：真人输入分布 vs 训练数据分布是否匹配。

打印完整 18 维特征 + 8 state 全概率分布 + 实时统计。
跑 30 秒，输出最大/最小/均值。

用法：
  PYTHONPATH=. .venv-poc/bin/python -u poc/scripts/live_verify_diagnostic.py
"""
from __future__ import annotations

import os
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np
import torch

from poc.src.capture.face_signals import FaceSignalConfig, FaceSignalExtractor
from poc.src.fusion import MultimodalFusion, FusionConfig
from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.dataset import _STATE_LABELS
from poc.src.lnn.model import DefaultLNNFactory

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False


DIM_NAMES = [
    "eye", "pitch", "yaw", "gaze",
    "fatigue", "attn", "pose", "blink",
    "smile", "brow", "cheek", "lip-d", "jaw", "mouth",
    "voice", "utt", "ctxK", "focus",
]


class SimpleVAD:
    def __init__(self, threshold_rms: float = 0.02) -> None:
        self._threshold = threshold_rms
        self._is_speaking = False
        self._rms = 0.0
        self._stream = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        if not _SD_AVAILABLE:
            return False
        try:
            self._stream = sd.InputStream(channels=1, samplerate=16000, blocksize=1600,
                                           callback=self._cb)
            self._stream.start()
            return True
        except Exception:
            return False

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass

    def _cb(self, indata, *_) -> None:
        rms = float(np.sqrt(np.mean(indata ** 2)))
        with self._lock:
            self._rms = rms
            self._is_speaking = rms > self._threshold

    def get(self) -> dict:
        with self._lock:
            sp = self._is_speaking
            rms = self._rms
        return {
            "transcript": "speaking" if sp else "",
            "lang": "zh",
            "duration_s": (rms * 100) if sp else 0.0,
        }


class AppCtx:
    def __init__(self) -> None:
        self._app = ""
        self._last = 0.0
        self._available = sys.platform == "darwin"

    def update(self) -> None:
        now = time.time()
        if now - self._last < 3.0:
            return
        self._last = now
        try:
            r = subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to get name of first process whose frontmost is true'],
                capture_output=True, text=True, timeout=2,
            )
            self._app = r.stdout.strip() or self._app
        except Exception:
            pass

    def get(self) -> dict:
        return {"active_app": self._app, "app_switch_count_5min": 0}


def main() -> None:
    cands = [
        "poc/data/checkpoints/day6_ltc/checkpoint_day5.pt",
        "poc/data/checkpoints/day6_ltc/checkpoint_day5_latest.pt",
        "poc/data/checkpoints/day5/checkpoint_day5.pt",
    ]
    ckpt_path = next((p for p in cands if os.path.exists(p)), cands[-1])
    print(f"\n  ⏳ 加载 ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg: LNNTrainConfig = ckpt["config"]
    p2i = ckpt["persona_to_idx"]

    # 跑两个 persona 对比同一段视频流
    for pid in ["vboss_001", "vboss_007"]:
        if pid not in p2i:
            print(f"  ⚠️ {pid} 不在 ckpt"); return

    factory = DefaultLNNFactory()
    state_model = factory.create_state_model(cfg)
    state_model.load_state_dict(ckpt["state_model"])
    state_model.eval()
    pidx_001 = torch.tensor([p2i["vboss_001"]])
    pidx_007 = torch.tensor([p2i["vboss_007"]])

    print(f"  ✅ 模型 + 30 personas 加载完成\n")
    print("  📋 测试流程：跑 60 秒，前 60 帧积累窗口，之后实时输出。")
    print("     按你的节奏做：静坐 / 微笑 / 皱眉 / 张嘴 各 10 秒\n")

    # 采集（v2: 启用 py-feat AU 校准模式）
    use_pyfeat = "--no-pyfeat" not in sys.argv
    face_ext = FaceSignalExtractor(FaceSignalConfig(use_pyfeat=use_pyfeat)); face_ext.open()
    fusion = MultimodalFusion(FusionConfig())
    print(f"  🎭 视觉模式: {'py-feat AU 校准（v2）' if use_pyfeat else '手写公式（v1 旧版）'}")
    vad = SimpleVAD(); has_mic = vad.start()
    app = AppCtx()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 摄像头打不开"); return
    print(f"  📷 摄像头 ✅  🎤 麦克风 {'✅' if has_mic else '❌'}\n")

    seq_len = cfg.sequence_length
    buf: deque[list[float]] = deque(maxlen=seq_len)
    last_t = 0.0
    fc = 0
    history: list[dict] = []
    dim_stats = defaultdict(list)

    try:
        while True:
            now = time.time()
            if now - last_t < 1.0:
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
                continue
            last_t = now
            ret, frame = cap.read()
            if not ret:
                continue

            face = face_ext.capture_signals(frame)
            face_ok = face.get("landmarks_n", 0) > 0
            audio = vad.get() if has_mic else None
            app.update()
            ctx = app.get()

            fi = {"ts": datetime.now().isoformat(), "frame_idx": fc,
                  "face": face, "audio": audio, "context": ctx}
            fused = fusion.fuse(fi)
            buf.append(list(fused.vector))
            fc += 1

            if len(buf) < seq_len:
                if fc % 10 == 0:
                    print(f"  ⏳ 积累 {fc}/{seq_len}", flush=True)
                continue

            v = torch.tensor([list(buf)], dtype=torch.float32)
            with torch.no_grad():
                logits_001 = state_model(v, pidx_001)[0, -1, :]
                logits_007 = state_model(v, pidx_007)[0, -1, :]
                p001 = torch.softmax(logits_001, dim=0).tolist()
                p007 = torch.softmax(logits_007, dim=0).tolist()

            cur = list(buf)[-1]
            for name, val in zip(DIM_NAMES, cur):
                dim_stats[name].append(val)

            top001 = max(enumerate(p001), key=lambda x: x[1])
            top007 = max(enumerate(p007), key=lambda x: x[1])

            print(f"  #{fc:>3} 表情维度: smile={cur[8]:.2f} brow={cur[9]:.2f} cheek={cur[10]:.2f} lip-d={cur[11]:.2f} jaw={cur[12]:.2f} | "
                  f"vboss_001→{_STATE_LABELS[top001[0]]:<22}({top001[1]:.0%})  "
                  f"vboss_007→{_STATE_LABELS[top007[0]]:<22}({top007[1]:.0%})", flush=True)

            history.append({
                "frame": fc, "face_ok": face_ok,
                "vector": cur,
                "top_001": (_STATE_LABELS[top001[0]], top001[1]),
                "top_007": (_STATE_LABELS[top007[0]], top007[1]),
            })
    except KeyboardInterrupt:
        pass
    finally:
        cap.release(); cv2.destroyAllWindows(); face_ext.close(); vad.stop()

    # 统计
    print("\n" + "=" * 80)
    print("📊 18 维特征实测分布（你 60 秒内做的所有动作的统计）")
    print("=" * 80)
    print(f"  {'维度':<8} {'min':<8} {'max':<8} {'mean':<8} {'std':<8} 训练集典型值（合成数据）")
    print("  " + "-" * 76)
    expected = {
        "smile":  "positive 时 ~0.7-0.98 (饱和)",
        "brow":   "negative 时 ~0.5-0.8",
        "cheek":  "positive 时 ~0.3-0.6",
        "lip-d":  "negative 时 ~0.3-0.7",
        "jaw":    "0-0.3",
        "voice":  "speaking 时 = 1.0，否则 0.0",
    }
    for name in DIM_NAMES:
        if not dim_stats[name]:
            continue
        vals = dim_stats[name]
        line = f"  {name:<8} {min(vals):<8.3f} {max(vals):<8.3f} {statistics.mean(vals):<8.3f} {statistics.stdev(vals) if len(vals)>1 else 0:<8.3f}"
        if name in expected:
            line += f" {expected[name]}"
        print(line)

    # 状态分布
    print(f"\n  vboss_001 状态分布:")
    states_001 = defaultdict(int)
    for h in history:
        states_001[h["top_001"][0]] += 1
    for s, c in sorted(states_001.items(), key=lambda x: -x[1]):
        print(f"    {s:<24} {c}帧 ({c/len(history):.0%})")
    print(f"\n  vboss_007 状态分布:")
    states_007 = defaultdict(int)
    for h in history:
        states_007[h["top_007"][0]] += 1
    for s, c in sorted(states_007.items(), key=lambda x: -x[1]):
        print(f"    {s:<24} {c}帧 ({c/len(history):.0%})")

    # 关键诊断
    print("\n" + "=" * 80)
    print("🔬 关键诊断")
    print("=" * 80)
    smile_max = max(dim_stats.get("smile", [0]))
    brow_max = max(dim_stats.get("brow", [0]))
    if smile_max < 0.5:
        print(f"  ⚠️ 你的最大 smile = {smile_max:.2f}，但训练时 positive 状态 smile 通常 0.7+")
        print(f"     根因：mediapipe AU 计算公式对真人微笑不敏感（OOD 输入）")
        print(f"     修复：放大 _AU_SCALE 12.0 → 24.0 或重新训练含真实分布")
    if brow_max < 0.4:
        print(f"  ⚠️ 你的最大 brow_furrow = {brow_max:.2f}，但训练时 negative 状态 brow 通常 0.5+")
        print(f"     根因：同上 OOD")
    diff_count = sum(1 for h in history if h["top_001"][0] != h["top_007"][0])
    print(f"\n  vboss_001 vs vboss_007 不同预测帧数: {diff_count}/{len(history)} ({diff_count/max(len(history),1):.0%})")
    if diff_count / max(len(history), 1) < 0.10:
        print(f"  🟡 个性化差异 < 10%，模型可能未充分学到 per-persona 解码")


if __name__ == "__main__":
    main()
