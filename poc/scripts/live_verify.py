#!/usr/bin/env python3
"""真人实时验证 v3：摄像头 + 麦克风 + 当前应用 → LNN 模型预测你的状态。

v1: 只有摄像头，12 维向量有 4 维锁死 0.5 → 永远偏 neutral。
v2: 补齐三模态（face+mic+app），但 12 维无表情特征 → 不能检测安静的积极。
v3 (Day 6): 18 维特征向量，新增 6 维 AU 表情特征：
  - smile_intensity (AU6+AU12), brow_furrow (AU4), cheek_raise (AU6)
  - lip_corner_depress (AU15), jaw_drop (AU26), mouth_open_ratio

使用方式：
    cd /tmp/lnmem-fix/LnMem
    PYTHONPATH=. .venv-poc/bin/python -u poc/scripts/live_verify.py

操作：
    q / Ctrl+C = 退出
    试试这些动作，看模型反应：
      正常看屏幕工作 → 应该识别为 focused
      闭眼/低头 → 应该识别为 fatigued 或 low_energy
      频繁转头 → 应该识别为 distracted
      微笑/皱眉 → 表情特征变化，可能触发 emotional_positive/negative
      对着电脑说话 → voice 维度变化
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch

from poc.src.capture.face_signals import FaceSignalConfig, FaceSignalExtractor
from poc.src.fusion import MultimodalFusion, FusionConfig
from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.dataset import _STATE_LABELS
from poc.src.lnn.model import DefaultLNNFactory

# ---------- 语音活动检测（VAD）----------

try:
    import sounddevice as sd  # type: ignore
    _SD_AVAILABLE = True
except ImportError:
    sd = None  # type: ignore[assignment]
    _SD_AVAILABLE = False


class SimpleVAD:
    """极简 VAD：监听麦克风音量，判断是否在说话。"""

    def __init__(self, threshold_rms: float = 0.02, window_s: float = 1.0) -> None:
        self._threshold = threshold_rms
        self._window_s = window_s
        self._is_speaking = False
        self._speech_duration = 0.0  # 最近 window 内的说话秒数
        self._rms_history: deque[float] = deque(maxlen=10)
        self._stream = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        if not _SD_AVAILABLE:
            return False
        try:
            self._stream = sd.InputStream(
                channels=1,
                samplerate=16000,
                blocksize=1600,  # 100ms 一块
                callback=self._audio_callback,
            )
            self._stream.start()
            return True
        except Exception as e:
            print(f"  ⚠️  麦克风打开失败: {e}")
            return False

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        rms = float(np.sqrt(np.mean(indata ** 2)))
        with self._lock:
            self._rms_history.append(rms)
            self._is_speaking = rms > self._threshold

    def get_signals(self) -> dict:
        """返回 fusion 需要的 audio 字段。"""
        with self._lock:
            is_speaking = self._is_speaking
            # 最近 N 块里超过阈值的比例 → 近似 utterance duration
            if self._rms_history:
                speaking_ratio = sum(
                    1 for r in self._rms_history if r > self._threshold
                ) / len(self._rms_history)
            else:
                speaking_ratio = 0.0

        return {
            "transcript": "speaking" if is_speaking else "",
            "lang": "zh",
            "duration_s": speaking_ratio * 4.0,  # 映射到 [0, 4s] 范围
        }


# ---------- 当前应用检测（macOS）----------


class AppContextDetector:
    """检测当前前台应用（macOS osascript）。"""

    def __init__(self) -> None:
        self._current_app = ""
        self._app_history: deque[str] = deque(maxlen=40)  # ~5min 内的应用记录
        self._last_check = 0.0
        self._available = False

    def probe(self) -> bool:
        """启动时探测 osascript 是否可用。返回 True = 可用。"""
        if sys.platform != "darwin":
            return False
        try:
            result = subprocess.run(
                ["osascript", "-e", "return \"ok\""],
                capture_output=True,
                text=True,
                timeout=3,
            )
            self._available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            self._available = False
        return self._available

    def update(self) -> None:
        now = time.time()
        if now - self._last_check < 3.0:  # 每 3 秒检测一次（osascript 有开销）
            return
        self._last_check = now
        try:
            result = subprocess.run(
                [
                    "osascript", "-e",
                    'tell application "System Events" to get name of '
                    "first process whose frontmost is true",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            app = result.stdout.strip()
            if app:
                self._current_app = app
                self._app_history.append(app)
        except Exception:
            pass

    def get_signals(self) -> dict:
        """返回 fusion 需要的 context 字段。"""
        # 统计 5min 内应用切换次数
        switches = 0
        prev = None
        for app in self._app_history:
            if prev is not None and app != prev:
                switches += 1
            prev = app

        return {
            "active_app": self._current_app,
            "app_switch_count_5min": switches,
        }


# ---------- 显示 ----------

STATE_DISPLAY: dict[str, tuple[str, str]] = {
    "low_energy":         ("低能量 😴", "\033[34m"),
    "focused":            ("专注 🎯",   "\033[32m"),
    "distracted":         ("分心 🫠",   "\033[33m"),
    "fatigued":           ("疲劳 😩",   "\033[31m"),
    "emotional_positive": ("积极 😊",   "\033[36m"),
    "emotional_negative": ("消极 😞",   "\033[35m"),
    "neutral":            ("中性 😐",   "\033[37m"),
    "transitioning":      ("过渡 🔄",   "\033[90m"),
}
RESET = "\033[0m"
BOLD = "\033[1m"
CLEAR_LINE = "\033[2K\033[G"


def load_model() -> tuple:
    """优先加载 Day 6.5 LTC 训练 checkpoint，回退到 Day 5。"""
    import os
    candidates = [
        "poc/data/checkpoints/day6_ltc/checkpoint_day5.pt",         # 训练完成的最终 ckpt
        "poc/data/checkpoints/day6_ltc/checkpoint_day5_latest.pt",  # 训练中的 incremental ckpt
        "poc/data/checkpoints/day6_ltc/checkpoint_day6.pt",
        "poc/data/checkpoints/day5/checkpoint_day5.pt",             # Day 5 fallback
    ]
    ckpt_path = next((p for p in candidates if os.path.exists(p)), candidates[-1])
    print(f"  ⏳ 加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg: LNNTrainConfig = ckpt["config"]
    p2i: dict[str, int] = ckpt["persona_to_idx"]

    factory = DefaultLNNFactory()
    state_model = factory.create_state_model(cfg)
    state_model.load_state_dict(ckpt["state_model"])
    state_model.eval()
    return state_model, cfg, p2i


def print_header(has_mic: bool, has_app: bool) -> None:
    mic_icon = "✅" if has_mic else "❌"
    app_icon = "✅" if has_app else "❌"
    print(f"\n{'=' * 72}")
    print(f"  {BOLD}LnMem Day 6 LNN — 真人实时验证 v3（三模态 + 表情）{RESET}")
    print(f"  📷 摄像头: ✅   🎤 麦克风: {mic_icon}   💻 应用检测: {app_icon}")
    print(f"{'=' * 72}")
    print(f"  前 60 秒积累信号窗口，之后开始实时预测")
    print(f"  🛑 按 q 或 Ctrl+C 退出")
    print(f"{'=' * 72}\n")


def print_prediction(
    pred_state: str,
    confidence: float,
    top3: list[tuple[str, float]],
    frame_count: int,
    face_ok: bool,
    is_speaking: bool,
    current_app: str,
    raw_face: dict,
) -> None:
    cn, color = STATE_DISPLAY.get(pred_state, (pred_state, "\033[37m"))
    conf_bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))

    face_icon = "👤" if face_ok else "❌"
    mic_icon = "🗣️" if is_speaking else "🤫"

    eye = raw_face.get("eye_closure", 0.0)
    gaze = raw_face.get("gaze_stability", 0.0)
    smile = raw_face.get("smile_intensity", 0.0)
    brow = raw_face.get("brow_furrow", 0.0)

    # 缩短应用名
    app_short = current_app[:12] if current_app else "?"

    # 第 1 行：预测 + 置信度 + 表情
    sys.stdout.write(CLEAR_LINE)
    sys.stdout.write(
        f"  {face_icon}{mic_icon} #{frame_count:>4d}  "
        f"{color}{BOLD}{cn:>10}{RESET}  "
        f"[{conf_bar}] {confidence:.0%}  "
        f"│ 👁️{eye:.2f} 👀{gaze:.2f} 😊{smile:.2f} 😠{brow:.2f} "
        f"📱{app_short}"
    )
    sys.stdout.flush()

    # 第 2 行：Top3
    sys.stdout.write(f"\n{CLEAR_LINE}")
    top3_str = "  ".join(
        f"{STATE_DISPLAY.get(s, (s, ''))[0]}:{p:.0%}" for s, p in top3
    )
    sys.stdout.write(f"  📊 Top3: {top3_str}")
    sys.stdout.write("\033[A")
    sys.stdout.flush()


def main() -> None:
    # 1. 加载模型
    print(f"\n  ⏳ 加载模型（v3 · 18 维）...", end="", flush=True)
    state_model, cfg, p2i = load_model()
    print(f" ✅ ({len(p2i)} 个画像)")

    # 允许命令行 --persona 选 vboss_007（外向）vs vboss_001（内敛）做个性化对比
    import argparse, sys as _sys
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--persona", default="vboss_001",
                         help="vboss_001 (内敛) / vboss_007 (外向) / 等")
    _args, _ = _parser.parse_known_args()
    persona_id = _args.persona
    if persona_id not in p2i:
        print(f"  ⚠️ persona {persona_id} 不在 checkpoint 里，回退到 vboss_001")
        persona_id = "vboss_001"
    pidx = p2i[persona_id]
    print(f"  🎭 使用 persona: {persona_id}（idx={pidx}）")

    # 2. 初始化三模态采集
    face_extractor = FaceSignalExtractor(FaceSignalConfig())
    face_extractor.open()
    fusion = MultimodalFusion(FusionConfig())

    vad = SimpleVAD()
    has_mic = vad.start()

    app_ctx = AppContextDetector()
    has_app = app_ctx.probe()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n  ❌ 无法打开摄像头！")
        print("  macOS: 系统设置 → 隐私与安全性 → 摄像头 → 允许终端/Python")
        vad.stop()
        return

    print_header(has_mic, has_app)

    # 3. 主循环
    seq_len = cfg.sequence_length
    buffer: deque[list[float]] = deque(maxlen=seq_len)
    frame_count = 0
    last_capture_time = 0.0
    predictions_log: list[dict] = []

    try:
        while True:
            now = time.time()
            if now - last_capture_time < 1.0:
                key = cv2.waitKey(50) & 0xFF
                if key == ord("q"):
                    break
                continue

            last_capture_time = now
            ret, frame = cap.read()
            if not ret:
                continue

            # 三模态采集
            face_signals = face_extractor.capture_signals(frame)
            face_ok = face_signals.get("landmarks_n", 0) > 0

            audio_signals = vad.get_signals() if has_mic else None
            app_ctx.update()
            context_signals = app_ctx.get_signals() if has_app else None

            # 融合 18 维（Day 6: +6 表情特征）
            fusion_input = {
                "ts": datetime.now().isoformat(),
                "frame_idx": frame_count,
                "face": face_signals,
                "audio": audio_signals,
                "context": context_signals,
            }
            fused = fusion.fuse(fusion_input)
            buffer.append(list(fused.vector))
            frame_count += 1

            if len(buffer) < seq_len:
                remaining = seq_len - len(buffer)
                mic_status = "🗣️说话中" if (audio_signals and audio_signals.get("transcript")) else "🤫安静"
                app_name = (context_signals or {}).get("active_app", "?")
                sys.stdout.write(CLEAR_LINE)
                sys.stdout.write(
                    f"  ⏳ 积累中 {frame_count}/{seq_len}"
                    f"（{remaining}秒后开始预测）"
                    f"  {'👤' if face_ok else '❌'} {mic_status} 📱{app_name[:15]}"
                )
                sys.stdout.flush()
                continue

            # 推理
            vectors = torch.tensor(
                list(buffer), dtype=torch.float32,
            ).unsqueeze(0)
            pidx_t = torch.tensor([pidx])

            with torch.no_grad():
                logits = state_model(vectors, pidx_t)

            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=0)
            pred_idx = probs.argmax().item()
            pred_state = _STATE_LABELS[pred_idx]
            confidence = probs[pred_idx].item()

            top3_indices = probs.argsort(descending=True)[:3]
            top3 = [
                (_STATE_LABELS[i.item()], probs[i.item()].item())
                for i in top3_indices
            ]

            is_speaking = bool(
                audio_signals and audio_signals.get("transcript")
            )
            current_app = (context_signals or {}).get("active_app", "")

            print_prediction(
                pred_state, confidence, top3,
                frame_count, face_ok, is_speaking, current_app, face_signals,
            )

            predictions_log.append({
                "time": datetime.now().isoformat(),
                "frame": frame_count,
                "pred": pred_state,
                "confidence": confidence,
                "face_ok": face_ok,
                "speaking": is_speaking,
                "app": current_app,
                "vector": list(fused.vector),
            })

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_extractor.close()
        vad.stop()

    # 汇总
    print(f"\n\n{'=' * 72}")
    print(f"  📊 验证汇总（v3 三模态 + 表情）")
    print(f"{'=' * 72}")
    print(f"  总帧数: {frame_count}   有效预测: {len(predictions_log)}")

    if predictions_log:
        from collections import Counter

        counts = Counter(p["pred"] for p in predictions_log)
        print(f"\n  状态分布:")
        for state, count in counts.most_common():
            cn, color = STATE_DISPLAY.get(state, (state, ""))
            pct = count / len(predictions_log)
            bar = "█" * int(pct * 30)
            print(f"    {color}{cn:>10}{RESET}  {bar} {count}帧 ({pct:.0%})")

        avg_conf = sum(p["confidence"] for p in predictions_log) / len(
            predictions_log
        )
        face_rate = sum(
            1 for p in predictions_log if p["face_ok"]
        ) / len(predictions_log)
        speak_rate = sum(
            1 for p in predictions_log if p.get("speaking")
        ) / len(predictions_log)
        print(f"\n  平均置信度: {avg_conf:.1%}")
        print(f"  人脸检测率: {face_rate:.1%}")
        print(f"  说话检测率: {speak_rate:.1%}")

        # 输出最后 5 帧的原始向量供分析
        print(f"\n  最后 5 帧原始向量（18 维）:")
        dims = [
            "eye", "pitch", "yaw", "gaze",
            "fatigue", "attn", "pose", "blink",
            "smile", "brow", "cheek", "lip", "jaw", "mouth",
            "voice", "utt", "ctx", "focus",
        ]
        print(f"    {'帧':>4}  " + "  ".join(f"{d:>6}" for d in dims))
        for p in predictions_log[-5:]:
            v = p["vector"]
            vals = "  ".join(f"{x:6.3f}" for x in v)
            print(f"    {p['frame']:>4}  {vals}  → {p['pred']}")

    print(f"\n  💡 现在试试：")
    print(f"     1. 安静看屏幕 → 应该是 focused")
    print(f"     2. 微笑 → smile_intensity 升高，可能触发 emotional_positive")
    print(f"     3. 皱眉 → brow_furrow 升高，可能触发 emotional_negative")
    print(f"     4. 频繁切换应用 → focus_stability 降低，可能触发 distracted")
    print(f"     5. 闭眼低头 → 应该是 fatigued")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
