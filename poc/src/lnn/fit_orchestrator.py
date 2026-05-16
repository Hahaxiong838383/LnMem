"""Phase 2 在线微调编排器 — P0-3 工程入口（M0 W2，2026-05-12）。

将 `online_trainer.fit_step` 与 daemon/cron 调度链路解耦，提供单函数入口：

```python
result = try_fit_step(
    feedback_dir=Path.home() / ".lnmem" / "feedback",
    signals_dir=Path.home() / ".lnmem" / "data",
    ckpt_in_path=Path("poc/data/checkpoints/day12_with_self/checkpoint_day12.pt"),
    ckpt_out_dir=Path("poc/data/checkpoints/m0w2_phase2"),
    persona_idx=30,  # vboss_self
    force=False,     # True 时跳 trigger_if_ready 阈值（dry-run）
)
```

返回值 `FitRunResult` 含完整状态：triggered / counts / metrics / ckpt_out_path / error。
全链路 fail-soft（无异常逃出函数），调用方按 reason 分支处理。

LNN-FIRST P-1 合规：本文件只编排已有 `fit_step` 调用，不引入新建模逻辑。
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from poc.src.lnn.feedback_dataset import count_by_label, load_events
from poc.src.lnn.feedback_to_batch import build_training_batch, fill_match_labels_with_argmax
from poc.src.lnn.model import DefaultLNNFactory
from poc.src.lnn.online_adapter import OnlineLoRAAdapter
from poc.src.lnn.online_contracts import (
    FeedbackDataError,
    OnlineAdapterConfig,
    OnlineAdapterError,
)
from poc.src.lnn.online_trainer import fit_step, trigger_if_ready


_REASON_FIRED = "fired"
_REASON_BELOW_THRESHOLD = "below_threshold"
_REASON_NO_FEEDBACK = "no_feedback"
_REASON_NO_SIGNALS = "no_signals"
_REASON_NO_CKPT = "no_ckpt"
_REASON_NO_BATCH = "no_batch"
_REASON_NO_MISMATCH = "no_mismatch"  # F2-03: force=True 但 mismatch=0
_REASON_ADAPTER_MISMATCH = "adapter_mismatch"
_REASON_ERROR = "error"

# F1-01 fix: ckpt schema 版本标识
_CKPT_SCHEMA_VERSION = "phase2_v1"  # state_model 保持 base 结构 + online_tau_drift 单独存


@dataclass(frozen=True)
class FitRunResult:
    """fit_orchestrator.try_fit_step 一次调用的完整结果。"""
    triggered: bool                          # True = fit_step 真跑了，新 ckpt 已落盘
    counts: dict[str, int]                   # match/mismatch/skip/input/total
    reason: str                              # _REASON_*
    metrics: dict[str, float] | None = None  # fit_step 返回的 loss/drift_max_norm/batch_size
    ckpt_out_path: Path | None = None
    ckpt_in_sha8: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _ckpt_sha8(path: Path) -> str:
    """Compute first 8 chars of SHA256 hex digest of file content."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _empty_counts() -> dict[str, int]:
    return {"match": 0, "mismatch": 0, "skip": 0, "input": 0, "total": 0}


def _gather_counts(feedback_dir: Path, *, min_event_ts: str | None = None) -> dict[str, int]:
    """fail-soft 计数 feedback events（无 events 返零计数）。

    F-03 fix (codex 二审 2026-05-16): min_event_ts 透传 load_events 实现增量水位线。
    """
    counts = _empty_counts()
    try:
        events = load_events(feedback_dir, min_event_ts=min_event_ts)
    except FeedbackDataError:
        return counts
    counts.update(count_by_label(events))
    counts["total"] = len(events)
    return counts


def try_fit_step(
    *,
    feedback_dir: Path,
    signals_dir: Path,
    ckpt_in_path: Path,
    ckpt_out_dir: Path,
    persona_idx: int,
    online_config: OnlineAdapterConfig | None = None,
    force: bool = False,
    alignment_window_s: float = 5.0,
    min_event_ts: str | None = None,
) -> FitRunResult:
    """一站式入口：阈值检查 → load_ckpt → build_batch → fit_step → save_ckpt。

    Parameters
    ----------
    feedback_dir : Path
        `~/.lnmem/feedback/` 反馈 jsonl 所在目录
    signals_dir : Path
        `~/.lnmem/data/` signals jsonl 所在目录
    ckpt_in_path : Path
        输入 ckpt（如 `checkpoint_day12.pt`）
    ckpt_out_dir : Path
        输出 ckpt 目录；不存在则创建
    persona_idx : int
        所有反馈视为来自此 persona（PoC 阶段：vboss_self=30）
    online_config : OnlineAdapterConfig | None
        在线学习超参；None 时用默认值
    force : bool
        True 时跳过 trigger_if_ready 阈值检查（dry-run 用，仍要求有可用 batch）
    alignment_window_s : float
        user_triggered 反馈对齐窗口（>此值的样本硬过滤）

    Returns
    -------
    FitRunResult
        triggered=True 代表 fit_step 真跑了且 ckpt 已保存。
        其他情况通过 reason 字段区分（无异常逃出）。
    """
    if online_config is None:
        online_config = OnlineAdapterConfig()

    # === 阶段 1: trigger 阈值检查（force=True 时跳过，但仍统计 counts） ===
    # F-03 fix: min_event_ts 透传 trigger 决策 + counts 与 fit 实际批保持一致
    if not force:
        try:
            ready, counts = trigger_if_ready(feedback_dir, online_config, min_event_ts=min_event_ts)
        except Exception as e:  # noqa: BLE001
            return FitRunResult(
                triggered=False, counts=_empty_counts(),
                reason=_REASON_ERROR, error=f"trigger_if_ready: {e}",
            )
        if not ready:
            return FitRunResult(
                triggered=False, counts=counts,
                reason=_REASON_BELOW_THRESHOLD,
            )
    else:
        counts = _gather_counts(feedback_dir, min_event_ts=min_event_ts)
        if counts["total"] == 0:
            return FitRunResult(
                triggered=False, counts=counts, reason=_REASON_NO_FEEDBACK,
                error="force=True but no feedback events found",
            )
        # F2-03 (codex 一审): force=True 也要求至少 1 mismatch，防"全 match argmax 回填"
        # 在无纠错信号情况下产生权重漂移
        if counts["mismatch"] == 0:
            return FitRunResult(
                triggered=False, counts=counts, reason=_REASON_NO_MISMATCH,
                error="force=True but mismatch=0; refuse to update weights without correction signal",
            )

    # === 阶段 2: 加载 base ckpt ===
    if not ckpt_in_path.exists():
        return FitRunResult(
            triggered=False, counts=counts, reason=_REASON_NO_CKPT,
            error=f"ckpt_in_path not found: {ckpt_in_path}",
        )
    # F3-02 (codex 一审): exists 后先算 sha，让 load 失败路径也能 audit
    ckpt_in_sha = _ckpt_sha8(ckpt_in_path)
    try:
        ckpt = torch.load(ckpt_in_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        factory = DefaultLNNFactory()
        state_model = factory.create_state_model(cfg)
        state_model.load_state_dict(ckpt["state_model"])
    except Exception as e:  # noqa: BLE001
        return FitRunResult(
            triggered=False, counts=counts, reason=_REASON_ERROR,
            error=f"load_ckpt: {e}", ckpt_in_sha8=ckpt_in_sha,
        )

    # === 阶段 3: wrap OnlineLoRAAdapter + freeze 非 tau_drift 参数 ===
    base_adapter = getattr(state_model, "personalization", None)
    if base_adapter is None:
        return FitRunResult(
            triggered=False, counts=counts, reason=_REASON_ADAPTER_MISMATCH,
            error="state_model has no .personalization attr",
            ckpt_in_sha8=ckpt_in_sha,
        )
    try:
        online_adapter = OnlineLoRAAdapter(
            base_adapter=base_adapter,
            online_config=online_config,
        )
    except OnlineAdapterError as e:
        return FitRunResult(
            triggered=False, counts=counts, reason=_REASON_ADAPTER_MISMATCH,
            error=f"OnlineLoRAAdapter init: {e}",
            ckpt_in_sha8=ckpt_in_sha,
        )
    # 关键：替换 state_model 的 personalization，使 fit_step 反传命中 online_adapter
    state_model.personalization = online_adapter

    # F2-01 (codex 一审): 显式 freeze 整个 state_model 除 tau_drift，
    # 防 backward 命中 backbone/readout grad（违反 LNN-FIRST P-1 "只触达 tau_drift"准则）
    for param in state_model.parameters():
        param.requires_grad_(False)
    online_adapter.tau_drift.requires_grad_(True)

    # === 阶段 4: 构造 training batch ===
    try:
        batch = build_training_batch(
            feedback_dir=feedback_dir,
            signals_dir=signals_dir,
            persona_idx=persona_idx,
            window_size=cfg.sequence_length,
            include_skip_input=False,
            alignment_window_s=alignment_window_s,
            min_event_ts=min_event_ts,  # F-03: 增量水位线透传
        )
    except FeedbackDataError as e:
        return FitRunResult(
            triggered=False, counts=counts, reason=_REASON_NO_BATCH,
            error=str(e), ckpt_in_sha8=ckpt_in_sha,
        )
    except Exception as e:  # noqa: BLE001
        return FitRunResult(
            triggered=False, counts=counts, reason=_REASON_ERROR,
            error=f"build_training_batch: {e}", ckpt_in_sha8=ckpt_in_sha,
        )

    if batch.x.shape[0] == 0:
        return FitRunResult(
            triggered=False, counts=counts, reason=_REASON_NO_BATCH,
            error="batch empty after alignment / window filtering",
            ckpt_in_sha8=ckpt_in_sha,
            metadata={"batch_skipped": batch.skipped_count},
        )

    # === 阶段 5: fill match label argmax（响应 codex F1-02 hard assert）===
    try:
        batch = fill_match_labels_with_argmax(batch, state_model)
    except Exception as e:  # noqa: BLE001
        return FitRunResult(
            triggered=False, counts=counts, reason=_REASON_ERROR,
            error=f"fill_match_labels: {e}", ckpt_in_sha8=ckpt_in_sha,
        )

    # === 阶段 6: fit_step（单步 SGD on tau_drift）===
    optim = torch.optim.SGD(
        [online_adapter.tau_drift],
        lr=online_config.delta_lr,
        weight_decay=online_config.delta_weight_decay,
    )
    try:
        metrics = fit_step(
            state_model, online_adapter, optim,
            x=batch.x, persona_idx=batch.persona_idx,
            labels=batch.labels, rewards=batch.rewards,
            config=online_config,
        )
    except Exception as e:  # noqa: BLE001
        # F2-02 (codex 一审): 改 Exception 兜底，torch 路径会抛 RuntimeError（shape/device/nan），
        # 与 fail-soft 全链路语义一致
        return FitRunResult(
            triggered=False, counts=counts, reason=_REASON_ERROR,
            error=f"fit_step: {type(e).__name__}: {e}", ckpt_in_sha8=ckpt_in_sha,
        )

    # === 阶段 7: unwrap + 保存新 ckpt（F1-01/F1-02/F3-01/F3-03 修复） ===
    try:
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)
        # 文件名继续用紧凑格式（避免冒号在文件名引发问题）
        _now_utc = datetime.now(timezone.utc)
        fired_at_filename = _now_utc.strftime("%Y%m%dT%H%M%SZ")
        # F-03 fix (codex 二审 2026-05-16): metadata 中 fired_at_utc 改用标准 ISO8601
        # 原因：cron 读 fired_at_utc 作 min_event_ts 与 events.ts (标准 ISO8601) 字典序比较时，
        # 紧凑格式 "20260516T081409Z" < 标准 "2026-05-16T..." → 全过滤错位（实测 bug）
        fired_at = _now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        ckpt_out_path = ckpt_out_dir / f"checkpoint_phase2_{fired_at_filename}.pt"

        # F1-01 + F1-02 (codex 一审): unwrap state_model.personalization 回 base_adapter
        # 让 state_dict 的 key 与原 day12 ckpt 一致（personalization.embedding.weight ...）
        # 下游 verify_persona_learned 等 strict=True load 兼容；
        # 同时保持 state_model 与 boundary_model 共享 personalization 的契约
        online_tau_drift = online_adapter.tau_drift.detach().cpu().clone()
        state_model.personalization = base_adapter  # unwrap：恢复 base 引用

        ckpt_out = dict(ckpt)  # 浅拷贝 base ckpt 所有字段
        ckpt_out["state_model"] = state_model.state_dict()  # 与原 day12 key 一致
        # F1-01: tau_drift 单独存于 ckpt top-level，下游识别 schema_version 后 wrap
        ckpt_out["online_tau_drift"] = online_tau_drift
        ckpt_out["online_adapter_schema"] = _CKPT_SCHEMA_VERSION

        # F3-01 (codex 一审): metadata 补齐完整 OnlineAdapterConfig 快照 + 数据过滤统计
        ckpt_out["phase2_metadata"] = {
            "fired_at_utc": fired_at,
            "schema_version": _CKPT_SCHEMA_VERSION,
            "ckpt_in_sha8": ckpt_in_sha,
            "ckpt_in_path": str(ckpt_in_path),
            "persona_idx": persona_idx,
            "force": force,
            "counts": counts,
            "metrics": metrics,
            "batch_size": int(batch.x.shape[0]),
            "batch_sources": list(batch.label_sources),
            "batch_skipped_count": int(batch.skipped_count),
            "online_config": {
                "min_total_feedback": online_config.min_total_feedback,
                "trigger_mismatch_threshold": online_config.trigger_mismatch_threshold,
                "delta_lr": online_config.delta_lr,
                "delta_weight_decay": online_config.delta_weight_decay,
                "delta_clip_norm": online_config.delta_clip_norm,
                "max_steps": online_config.max_steps,
                "batch_size": online_config.batch_size,
                "alignment_window_s": online_config.alignment_window_s,
                "seed": online_config.seed,
            },
        }

        # F3-03 (codex 一审): atomic write — tmp + os.replace 防 crash 期间留半文件
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(ckpt_out_dir), delete=False, suffix=".tmp"
        ) as tmpf:
            tmp_path = Path(tmpf.name)
            torch.save(ckpt_out, tmpf)
            tmpf.flush()
            os.fsync(tmpf.fileno())
        os.replace(tmp_path, ckpt_out_path)  # atomic rename
    except Exception as e:  # noqa: BLE001
        # 清理 tmp 文件（如存在）
        if 'tmp_path' in locals() and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return FitRunResult(
            triggered=False, counts=counts, reason=_REASON_ERROR,
            error=f"save_ckpt: {type(e).__name__}: {e}",
            metrics=metrics, ckpt_in_sha8=ckpt_in_sha,
        )

    return FitRunResult(
        triggered=True, counts=counts, reason=_REASON_FIRED,
        metrics=metrics, ckpt_out_path=ckpt_out_path, ckpt_in_sha8=ckpt_in_sha,
        metadata={
            "batch_size": int(batch.x.shape[0]),
            "schema_version": _CKPT_SCHEMA_VERSION,
        },
    )


__all__ = [
    "FitRunResult",
    "try_fit_step",
]
