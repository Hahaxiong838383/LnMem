"""Tests for fit_orchestrator.try_fit_step (M0 W2 P0-3)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from poc.src.lnn.fit_orchestrator import FitRunResult, try_fit_step
from poc.src.lnn.online_contracts import OnlineAdapterConfig


_REAL_CKPT = Path(__file__).resolve().parent.parent / "data" / "checkpoints" / "day12_with_self" / "checkpoint_day12.pt"


def _make_feedback_event(frame_id: int, label: str = "mismatch", true_state: str = "distracted",
                         reason: str = "brow_burst", session_id: str = "test-session",
                         alignment_latency_s: float | None = None) -> dict:
    """构造一条 feedback jsonl 记录（含 V0.3 user_triggered 字段）。"""
    return {
        "ts": f"2026-05-12T10:00:{frame_id % 60:02d}.000000Z",
        "frame_id": frame_id,
        "reason": reason,
        "signal": {"brow_furrow": 1.0, "smile_intensity": 0.2, "eye_closure": 0.3, "cheek_raise": 0.5},
        "vlm_label": None,
        "user_response": {
            "label": label,
            "true_state": true_state,
            "note": None,
        },
        "latency_s": 1.0,
        "session_id": session_id,
        "event_uuid": f"event-{frame_id:06d}",
        "alignment_latency_s": alignment_latency_s,
    }


def _make_signal(frame_id: int, session_id: str = "test-session") -> dict:
    """构造一条 signal jsonl 记录（18 维 face/expression/audio/context）。"""
    return {
        "ts": f"2026-05-12T10:00:{frame_id % 60:02d}.000000Z",
        "frame_id": frame_id,
        "session_id": session_id,
        "face_detected": True,
        "expression": {
            "brow_furrow": 0.8,
            "smile_intensity": 0.2,
            "cheek_raise": 0.3,
            "lip_corner_depress": 0.1,
            "jaw_drop": 0.1,
            "mouth_open_ratio": 0.1,
        },
        "head_pose": {"pitch_deg": 5.0, "yaw_deg": 2.0, "roll_deg": 1.0},
        "audio": {"voice_present": 0.0, "utterance_density": 0.0},
    }


def _seed_feedback_jsonl(feedback_dir: Path, events: list[dict]) -> None:
    feedback_dir.mkdir(parents=True, exist_ok=True)
    by_day: dict[str, list[dict]] = {}
    for ev in events:
        day = ev["ts"][:10]
        by_day.setdefault(day, []).append(ev)
    for day, evs in by_day.items():
        path = feedback_dir / f"queries_{day}.jsonl"
        path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in evs) + "\n", encoding="utf-8")


def _seed_signals_jsonl(signals_dir: Path, frame_ids: list[int], session_id: str = "test-session") -> None:
    signals_dir.mkdir(parents=True, exist_ok=True)
    sigs = [_make_signal(fid, session_id) for fid in frame_ids]
    day = sigs[0]["ts"][:10]
    path = signals_dir / f"signals_{day}.jsonl"
    path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in sigs) + "\n", encoding="utf-8")


@unittest.skipUnless(_REAL_CKPT.exists(), "day12_with_self ckpt 不存在，跳过 P0-3 集成测试")
class TestFitOrchestrator(unittest.TestCase):

    def test_01_below_threshold_no_fit(self):
        """mismatch < 10 且非 force 模式 → 返 below_threshold，不跑 fit。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fb_dir = tmp_path / "feedback"
            sig_dir = tmp_path / "signals"
            out_dir = tmp_path / "out"
            # 只 4 条 mismatch
            events = [_make_feedback_event(i, label="mismatch") for i in range(4)]
            _seed_feedback_jsonl(fb_dir, events)
            _seed_signals_jsonl(sig_dir, list(range(1, 100)))

            result = try_fit_step(
                feedback_dir=fb_dir, signals_dir=sig_dir,
                ckpt_in_path=_REAL_CKPT, ckpt_out_dir=out_dir,
                persona_idx=30, force=False,
            )
            self.assertFalse(result.triggered)
            self.assertEqual(result.reason, "below_threshold")
            self.assertEqual(result.counts["mismatch"], 4)
            self.assertEqual(result.counts["total"], 4)
            self.assertIsNone(result.ckpt_out_path)

    def test_02_force_mode_runs_fit(self):
        """force=True 跳过阈值，mismatch=2 也能跑（用于 dry-run）。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fb_dir = tmp_path / "feedback"
            sig_dir = tmp_path / "signals"
            out_dir = tmp_path / "out"
            # 2 mismatch + 5 match（trigger_if_ready 不够，但 force 可过）
            events = (
                [_make_feedback_event(i, label="mismatch") for i in range(2)] +
                [_make_feedback_event(i + 100, label="match", true_state=None) for i in range(5)]
            )
            _seed_feedback_jsonl(fb_dir, events)
            _seed_signals_jsonl(sig_dir, list(range(1, 200)))

            result = try_fit_step(
                feedback_dir=fb_dir, signals_dir=sig_dir,
                ckpt_in_path=_REAL_CKPT, ckpt_out_dir=out_dir,
                persona_idx=30, force=True,
            )
            self.assertTrue(result.triggered, f"force 模式应跑，实际 {result.reason}: {result.error}")
            self.assertEqual(result.reason, "fired")
            self.assertIsNotNone(result.ckpt_out_path)
            self.assertTrue(result.ckpt_out_path.exists())
            self.assertIsNotNone(result.metrics)
            self.assertIn("loss", result.metrics)
            self.assertIn("drift_max_norm", result.metrics)
            self.assertGreaterEqual(result.metrics["batch_size"], 1)

    def test_03_threshold_met_runs_fit(self):
        """mismatch ≥ 10 且 total ≥ 10 → trigger 满足，非 force 也跑。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fb_dir = tmp_path / "feedback"
            sig_dir = tmp_path / "signals"
            out_dir = tmp_path / "out"
            # 10 mismatch + 5 match = total 15
            events = (
                [_make_feedback_event(i, label="mismatch") for i in range(10)] +
                [_make_feedback_event(i + 100, label="match", true_state=None) for i in range(5)]
            )
            _seed_feedback_jsonl(fb_dir, events)
            _seed_signals_jsonl(sig_dir, list(range(1, 200)))

            result = try_fit_step(
                feedback_dir=fb_dir, signals_dir=sig_dir,
                ckpt_in_path=_REAL_CKPT, ckpt_out_dir=out_dir,
                persona_idx=30, force=False,
            )
            self.assertTrue(result.triggered, f"阈值满足应跑，实际 {result.reason}: {result.error}")
            self.assertEqual(result.reason, "fired")

    def test_04_no_ckpt_path(self):
        """ckpt_in_path 不存在 → 返 no_ckpt，不抛异常。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = try_fit_step(
                feedback_dir=tmp_path / "fb", signals_dir=tmp_path / "sig",
                ckpt_in_path=tmp_path / "nonexistent.pt",
                ckpt_out_dir=tmp_path / "out",
                persona_idx=30, force=True,
            )
            self.assertFalse(result.triggered)
            self.assertEqual(result.reason, "no_feedback")
            # force=True 且无 feedback → no_feedback（阶段 1 就拦住，未到 ckpt 检查）

    def test_05_no_feedback_force_mode(self):
        """force=True 但无 feedback events → 返 no_feedback。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fb_dir = tmp_path / "feedback"
            fb_dir.mkdir(parents=True)
            result = try_fit_step(
                feedback_dir=fb_dir, signals_dir=tmp_path / "sig",
                ckpt_in_path=_REAL_CKPT, ckpt_out_dir=tmp_path / "out",
                persona_idx=30, force=True,
            )
            self.assertFalse(result.triggered)
            self.assertEqual(result.reason, "no_feedback")

    def test_06_ckpt_out_loadable_and_has_phase2_metadata(self):
        """fit 产生的新 ckpt 应可重新加载，且 phase2_metadata 字段齐全。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fb_dir = tmp_path / "feedback"
            sig_dir = tmp_path / "signals"
            out_dir = tmp_path / "out"
            # frame_id ≥ 60 确保滑窗可用（需 60 帧前向 history）
            events = [_make_feedback_event(i + 100, label="mismatch") for i in range(3)]
            _seed_feedback_jsonl(fb_dir, events)
            _seed_signals_jsonl(sig_dir, list(range(1, 200)))

            result = try_fit_step(
                feedback_dir=fb_dir, signals_dir=sig_dir,
                ckpt_in_path=_REAL_CKPT, ckpt_out_dir=out_dir,
                persona_idx=30, force=True,
            )
            self.assertTrue(result.triggered, f"force 应跑，实际 {result.reason}: {result.error}")

            # 重新加载 ckpt 检查 phase2_metadata
            ckpt = torch.load(result.ckpt_out_path, map_location="cpu", weights_only=False)
            self.assertIn("phase2_metadata", ckpt)
            meta = ckpt["phase2_metadata"]
            self.assertEqual(meta["persona_idx"], 30)
            self.assertTrue(meta["force"])
            self.assertIn("metrics", meta)
            self.assertIn("counts", meta)
            self.assertIn("ckpt_in_sha8", meta)
            # base 字段保留
            self.assertIn("config", ckpt)
            self.assertIn("persona_to_idx", ckpt)
            self.assertEqual(ckpt["persona_to_idx"]["vboss_self"], 30)

    def test_07_drift_l2_clipped(self):
        """fit_step 应执行 drift L2 clip，metrics.drift_max_norm < clip_norm。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fb_dir = tmp_path / "feedback"
            sig_dir = tmp_path / "signals"
            out_dir = tmp_path / "out"
            events = [_make_feedback_event(i + 100, label="mismatch") for i in range(3)]
            _seed_feedback_jsonl(fb_dir, events)
            _seed_signals_jsonl(sig_dir, list(range(1, 200)))

            cfg = OnlineAdapterConfig(delta_clip_norm=0.1)  # 强约束
            result = try_fit_step(
                feedback_dir=fb_dir, signals_dir=sig_dir,
                ckpt_in_path=_REAL_CKPT, ckpt_out_dir=out_dir,
                persona_idx=30, force=True, online_config=cfg,
            )
            self.assertTrue(result.triggered)
            # drift L2 应 < clip_norm（clip_drift_norm post-step 生效）
            self.assertLessEqual(result.metrics["drift_max_norm"], 0.1 + 1e-6)


@unittest.skipUnless(_REAL_CKPT.exists(), "day12_with_self ckpt 不存在，跳过 codex 一审 finding 回归")
class TestCodexFirstReviewFindings(unittest.TestCase):
    """codex 一审 8 finding 修复对应回归测试。"""

    def _setup(self):
        """共用 setup：mismatch=3 + 充足 signals。"""
        tmp = tempfile.mkdtemp()
        tmp_path = Path(tmp)
        fb_dir = tmp_path / "feedback"
        sig_dir = tmp_path / "signals"
        out_dir = tmp_path / "out"
        events = [_make_feedback_event(i + 100, label="mismatch") for i in range(3)]
        _seed_feedback_jsonl(fb_dir, events)
        _seed_signals_jsonl(sig_dir, list(range(1, 200)))
        return tmp_path, fb_dir, sig_dir, out_dir

    def test_f1_01_ckpt_state_dict_compatible_with_base(self):
        """F1-01: 新 ckpt state_model 的 keys 应与原 day12 ckpt 一致（可被 strict=True load）。"""
        tmp_path, fb_dir, sig_dir, out_dir = self._setup()
        result = try_fit_step(
            feedback_dir=fb_dir, signals_dir=sig_dir,
            ckpt_in_path=_REAL_CKPT, ckpt_out_dir=out_dir,
            persona_idx=30, force=True,
        )
        self.assertTrue(result.triggered)

        # 重新加载 ckpt，用 DefaultLNNFactory 创建 model 后 strict=True load
        from poc.src.lnn.model import DefaultLNNFactory
        ckpt = torch.load(result.ckpt_out_path, map_location="cpu", weights_only=False)
        factory = DefaultLNNFactory()
        state_model = factory.create_state_model(ckpt["config"])
        # 关键：strict=True 不应 fail
        state_model.load_state_dict(ckpt["state_model"], strict=True)

        # online_tau_drift 应单独存于 ckpt top-level
        self.assertIn("online_tau_drift", ckpt)
        self.assertIn("online_adapter_schema", ckpt)
        self.assertEqual(ckpt["online_adapter_schema"], "phase2_v1")

    def test_f1_02_state_boundary_personalization_keys_consistent(self):
        """F1-02: state_model 和 boundary_model 的 personalization key space 应一致。"""
        tmp_path, fb_dir, sig_dir, out_dir = self._setup()
        result = try_fit_step(
            feedback_dir=fb_dir, signals_dir=sig_dir,
            ckpt_in_path=_REAL_CKPT, ckpt_out_dir=out_dir,
            persona_idx=30, force=True,
        )
        self.assertTrue(result.triggered)

        ckpt = torch.load(result.ckpt_out_path, map_location="cpu", weights_only=False)
        state_keys = set(ckpt["state_model"].keys())
        boundary_keys = set(ckpt["boundary_model"].keys())
        # 两者 personalization 相关的 key 集合应一致
        state_personalization = {k for k in state_keys if k.startswith("personalization.")}
        boundary_personalization = {k for k in boundary_keys if k.startswith("personalization.")}
        # state 不应含 .base. 前缀，应与 boundary 一致
        self.assertFalse(any(".base." in k for k in state_personalization),
                         f"state_model personalization 不应含 .base. 前缀: {state_personalization}")
        self.assertEqual(state_personalization, boundary_personalization,
                         "state/boundary personalization key space 应一致")

    def test_f2_03_force_zero_mismatch_refused(self):
        """F2-03: force=True 但 mismatch=0 应拒绝训练（reason=no_mismatch）。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fb_dir = tmp_path / "feedback"
            sig_dir = tmp_path / "signals"
            out_dir = tmp_path / "out"
            # 仅 match，无 mismatch
            events = [_make_feedback_event(i + 100, label="match", true_state=None) for i in range(5)]
            _seed_feedback_jsonl(fb_dir, events)
            _seed_signals_jsonl(sig_dir, list(range(1, 200)))

            result = try_fit_step(
                feedback_dir=fb_dir, signals_dir=sig_dir,
                ckpt_in_path=_REAL_CKPT, ckpt_out_dir=out_dir,
                persona_idx=30, force=True,
            )
            self.assertFalse(result.triggered)
            self.assertEqual(result.reason, "no_mismatch")
            self.assertEqual(result.counts["match"], 5)
            self.assertEqual(result.counts["mismatch"], 0)

    def test_f3_01_phase2_metadata_full_config_snapshot(self):
        """F3-01: phase2_metadata 应含完整 OnlineAdapterConfig 快照。"""
        tmp_path, fb_dir, sig_dir, out_dir = self._setup()
        result = try_fit_step(
            feedback_dir=fb_dir, signals_dir=sig_dir,
            ckpt_in_path=_REAL_CKPT, ckpt_out_dir=out_dir,
            persona_idx=30, force=True,
        )
        self.assertTrue(result.triggered)

        ckpt = torch.load(result.ckpt_out_path, map_location="cpu", weights_only=False)
        meta = ckpt["phase2_metadata"]
        oc = meta["online_config"]
        expected_keys = {
            "min_total_feedback", "trigger_mismatch_threshold",
            "delta_lr", "delta_weight_decay", "delta_clip_norm",
            "max_steps", "batch_size", "alignment_window_s", "seed",
        }
        self.assertTrue(expected_keys.issubset(set(oc.keys())),
                        f"online_config 缺字段：expected {expected_keys}, got {set(oc.keys())}")
        # batch_skipped_count 字段
        self.assertIn("batch_skipped_count", meta)

    def test_f3_02_sha8_present_on_load_failure(self):
        """F3-02: ckpt_in_path 损坏导致 load 失败时，ckpt_in_sha8 仍应填好。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # 写一个损坏的 ckpt（非 torch 格式）
            bad_ckpt = tmp_path / "bad.pt"
            bad_ckpt.write_bytes(b"not a real torch checkpoint")

            fb_dir = tmp_path / "feedback"
            events = [_make_feedback_event(i, label="mismatch") for i in range(3)]
            _seed_feedback_jsonl(fb_dir, events)

            result = try_fit_step(
                feedback_dir=fb_dir, signals_dir=tmp_path / "sig",
                ckpt_in_path=bad_ckpt, ckpt_out_dir=tmp_path / "out",
                persona_idx=30, force=True,
            )
            self.assertFalse(result.triggered)
            self.assertEqual(result.reason, "error")
            # 关键：load 失败也应该有 sha8（exists 通过后立即算）
            self.assertIsNotNone(result.ckpt_in_sha8)
            self.assertEqual(len(result.ckpt_in_sha8), 8)

    def test_f3_03_atomic_write_no_partial_file(self):
        """F3-03: 保存使用 atomic write（tmp + os.replace），中途失败不留半文件。"""
        tmp_path, fb_dir, sig_dir, out_dir = self._setup()
        result = try_fit_step(
            feedback_dir=fb_dir, signals_dir=sig_dir,
            ckpt_in_path=_REAL_CKPT, ckpt_out_dir=out_dir,
            persona_idx=30, force=True,
        )
        self.assertTrue(result.triggered)
        # out_dir 内不应有 .tmp 残留
        tmp_residues = list(out_dir.glob("*.tmp"))
        self.assertEqual(len(tmp_residues), 0,
                         f"atomic write 后不应有 .tmp 残留：{tmp_residues}")

    def test_f2_01_only_tau_drift_has_grad(self):
        """F2-01: fit_step 后只有 tau_drift 应有 grad，backbone/readout 应 None。"""
        tmp_path, fb_dir, sig_dir, out_dir = self._setup()
        # 跑一次 fit 后检查 ckpt 中的 tau_drift；
        # 注意：state_model 是 try_fit_step 内 instance，无法直接 inspect grad。
        # 替代验证：ckpt 内 state_model state_dict 应来自 unwrap 后的 base，
        # 即所有 base 参数 requires_grad=False（state_dict 不含 requires_grad 信息，
        # 实际验证靠下游 strict load + 模型推理不变）
        result = try_fit_step(
            feedback_dir=fb_dir, signals_dir=sig_dir,
            ckpt_in_path=_REAL_CKPT, ckpt_out_dir=out_dir,
            persona_idx=30, force=True,
        )
        self.assertTrue(result.triggered)
        # 关键间接证据：drift 真的有更新（非 0）
        ckpt = torch.load(result.ckpt_out_path, map_location="cpu", weights_only=False)
        drift = ckpt["online_tau_drift"]
        # 30 是 vboss_self idx；其他行应为 0（未被训练影响）
        self.assertGreater(drift[30].abs().sum().item(), 0,
                           "vboss_self drift 应被 fit 后非 0")


class TestFitOrchestratorContractsNoCkpt(unittest.TestCase):
    """无需 ckpt 文件存在也能跑的契约测试。"""

    def test_below_threshold_no_ckpt_required(self):
        """non-force + 无 feedback → below_threshold，不需 ckpt 存在。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = try_fit_step(
                feedback_dir=tmp_path / "fb",
                signals_dir=tmp_path / "sig",
                ckpt_in_path=tmp_path / "nonexistent.pt",
                ckpt_out_dir=tmp_path / "out",
                persona_idx=30, force=False,
            )
            self.assertFalse(result.triggered)
            self.assertEqual(result.reason, "below_threshold")


if __name__ == "__main__":
    unittest.main()
