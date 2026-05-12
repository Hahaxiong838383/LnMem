"""产品定义 §6.3 硬阈值评估器（M0 W2 P-1，2026-05-12）。

合成数据 batch 质量评估器。给出报告 dict，调用方据此决定 batch 是否进训练集。

产品定义 §6.3 硬阈值（prod gate）：
- P0 槽位完整度 ≥ 95%
- 跨 LLM 一致率 ≥ 75%（低于此线整批丢弃）
- 长期轨迹中"虚拟人格漂移率" ≤ 5%（type_2 才检，type_1 不涉及）

PoC 阶段（M0 W2）放宽阈值：
- p0_completeness ≥ 0.80
- scene_diversity ≥ 5/8 场景（**仅 type_1 适用**，type_2/3 应传 skip_scene_check=True）
- cross_llm_consistency 需提供 cross_llm_pairs 才能评估（否则 prod_pass=False，"证据不足"）

F2-04 (codex 一审): persona_diversity 拆双指标
- persona_diversity = 唯一 user_id 数（弱信号，可被自增 id 伪满足）
- persona_behavior_diversity = per-persona slot value jaccard 距离均值（强信号，反映真实区分度）

F4-08 (codex 一审): LLM 自验证偏乐观风险
- generator 同时生 text + ground_truth_slots，可能"语言风格一致致解析一致"
- 真验证应配合 gliner / 规则 baseline parser 做盲审对照（M0 W2 P2 todo）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from contracts.event_tensor import EventTensor
from contracts.schema_v1 import P0_SLOTS, P1_SLOTS, SCHEMA_V1, SlotID


# 产品 §6.3 硬阈值
PROD_P0_COMPLETENESS_THRESHOLD = 0.95
PROD_CROSS_LLM_CONSISTENCY_THRESHOLD = 0.75

# PoC 阶段放宽阈值
POC_P0_COMPLETENESS_THRESHOLD = 0.80
POC_SCENE_DIVERSITY_MIN = 5
POC_CROSS_LLM_CONSISTENCY_THRESHOLD = 0.65


@dataclass(frozen=True)
class BatchQualityReport:
    """合成数据 batch 质量评估报告。"""

    n_total: int
    n_with_p0_full: int      # P0 4 槽全填的事件数
    p0_completeness: float   # = n_with_p0_full / n_total
    avg_p0_filled: float     # 平均 P0 填充数 / 4
    avg_p1_filled: float
    avg_total_filled: float  # 平均总槽位填充 / 20
    scene_diversity: int     # type_1 raw_modalities['_synth_scene'] 唯一数
    scene_breakdown: dict[str, int] = field(default_factory=dict)
    persona_diversity: int = 0       # 唯一 user_id 数（弱信号 — F2-04 codex 一审）
    persona_behavior_diversity: float = 0.0  # F2-04: per-persona slot value jaccard 距离均值（强信号）
    cross_llm_p0_agree_rate: float | None = None  # F1-02 (codex 一审): 跨 LLM P0 一致率
    cross_llm_n_pairs: int = 0
    poc_pass: bool = False
    prod_pass: bool = False
    prod_pass_evidence_complete: bool = False  # F1-02: True 当 cross_llm 数据已提供
    notes: list[str] = field(default_factory=list)


def _persona_behavior_diversity(events: list) -> float:
    """F2-04 (codex 一审): 行为分布差异 — per-persona slot value 集合平均 jaccard 距离。

    若仅有 1 个 persona → 0。若所有 persona 的 slot values 完全相同 → 0。
    若每个 persona 都有完全不同的 slot values → 接近 1.0。
    """
    if len(events) < 2:
        return 0.0
    # 每 persona 收集 slot value tuple 集合
    persona_signatures: dict[str, set] = {}
    for ev in events:
        sig: set = set()
        for sid, sv in ev.slots.items():
            try:
                sig.add((int(sid), str(sv.value)[:40]))
            except (ValueError, TypeError):
                continue
        persona_signatures.setdefault(ev.user_id, set()).update(sig)
    sigs = list(persona_signatures.values())
    if len(sigs) < 2:
        return 0.0
    # 平均 jaccard 距离（1 - intersection/union）
    pair_count = 0
    sum_dist = 0.0
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            inter = len(sigs[i] & sigs[j])
            union = len(sigs[i] | sigs[j])
            if union > 0:
                sum_dist += 1.0 - (inter / union)
                pair_count += 1
    return sum_dist / max(pair_count, 1)


def evaluate_batch(
    events: Iterable[EventTensor],
    *,
    expected_scenes: Optional[set[str]] = None,
    poc_mode: bool = True,
    cross_llm_pairs: Optional[list] = None,
    skip_scene_check: bool = False,
) -> BatchQualityReport:
    """评估一批 EventTensor 的质量。

    Parameters
    ----------
    events : Iterable[EventTensor]
        type_1/2/3 generator 产出的事件批
    expected_scenes : set[str] | None
        期望覆盖的场景集合（type_1 用 SCENES 8 个）。None 时仅计实际覆盖数
    poc_mode : bool
        True = PoC 放宽阈值；False = 产品 §6.3 硬阈值
    """
    events_list = list(events)
    n_total = len(events_list)
    if n_total == 0:
        return BatchQualityReport(
            n_total=0, n_with_p0_full=0, p0_completeness=0.0,
            avg_p0_filled=0.0, avg_p1_filled=0.0, avg_total_filled=0.0,
            scene_diversity=0,
            notes=["empty batch — generator failed for all samples"],
        )

    n_with_p0_full = 0
    sum_p0 = 0
    sum_p1 = 0
    sum_total = 0
    scenes: dict[str, int] = {}
    personas: set[str] = set()

    for ev in events_list:
        present = set(ev.slots.keys())
        p0_filled = len(present & P0_SLOTS)
        p1_filled = len(present & P1_SLOTS)
        total_filled = len(present)
        sum_p0 += p0_filled
        sum_p1 += p1_filled
        sum_total += total_filled
        if p0_filled == 4:
            n_with_p0_full += 1
        # 场景 / persona 多样性
        scene_b = ev.raw_modalities.get("_synth_scene")
        if scene_b is not None:
            scene = scene_b.decode("utf-8") if isinstance(scene_b, bytes) else str(scene_b)
            scenes[scene] = scenes.get(scene, 0) + 1
        personas.add(ev.user_id)

    p0_completeness = n_with_p0_full / n_total
    avg_p0 = sum_p0 / n_total
    avg_p1 = sum_p1 / n_total
    avg_total = sum_total / n_total
    scene_diversity = len(scenes)

    notes: list[str] = []
    if expected_scenes is not None and not skip_scene_check:
        missing = expected_scenes - set(scenes.keys())
        if missing:
            notes.append(f"missing scenes: {sorted(missing)}")

    # F2-04: persona 行为分布差异（不仅 ID 唯一）
    behavior_diversity = _persona_behavior_diversity(events_list)

    # F1-02: cross_llm_consistency（生产门必须，PoC 可选）
    cross_p0_rate: Optional[float] = None
    cross_n = 0
    if cross_llm_pairs:
        cross = cross_llm_consistency(cross_llm_pairs)
        if cross.get("valid"):
            cross_p0_rate = cross["p0_agree_rate"]
            cross_n = cross["n"]

    # 阈值判定（F1-01/F1-02 修复）
    poc_scene_check = (scene_diversity >= POC_SCENE_DIVERSITY_MIN) if not skip_scene_check else True
    poc_pass = (
        p0_completeness >= POC_P0_COMPLETENESS_THRESHOLD
        and poc_scene_check
    )

    # F1-01 + F1-02: prod_pass 必须同时满足 P0 + cross_llm（且 cross 数据齐）
    prod_pass_evidence_complete = cross_p0_rate is not None
    if prod_pass_evidence_complete:
        prod_pass = (
            p0_completeness >= PROD_P0_COMPLETENESS_THRESHOLD
            and cross_p0_rate >= PROD_CROSS_LLM_CONSISTENCY_THRESHOLD
            and (skip_scene_check or scene_diversity >= POC_SCENE_DIVERSITY_MIN)
        )
        if not prod_pass:
            failed = []
            if p0_completeness < PROD_P0_COMPLETENESS_THRESHOLD:
                failed.append(f"p0_completeness={p0_completeness:.3f} < {PROD_P0_COMPLETENESS_THRESHOLD}")
            if cross_p0_rate < PROD_CROSS_LLM_CONSISTENCY_THRESHOLD:
                failed.append(f"cross_llm={cross_p0_rate:.3f} < {PROD_CROSS_LLM_CONSISTENCY_THRESHOLD}")
            notes.append(f"prod_pass FAIL: {'; '.join(failed)}")
    else:
        # F1-02: 无 cross_llm 数据 → 禁止标 prod_pass=True（"证据不足"）
        prod_pass = False
        notes.append(
            "prod_pass=False: cross_llm_pairs not provided "
            "(产品 §6.3 要求一致率 ≥ 0.75，无 second-LLM 数据时禁止标 prod_pass)"
        )

    return BatchQualityReport(
        n_total=n_total,
        n_with_p0_full=n_with_p0_full,
        p0_completeness=p0_completeness,
        avg_p0_filled=avg_p0,
        avg_p1_filled=avg_p1,
        avg_total_filled=avg_total,
        scene_diversity=scene_diversity,
        scene_breakdown=scenes,
        persona_diversity=len(personas),
        persona_behavior_diversity=behavior_diversity,
        cross_llm_p0_agree_rate=cross_p0_rate,
        cross_llm_n_pairs=cross_n,
        prod_pass_evidence_complete=prod_pass_evidence_complete,
        poc_pass=poc_pass,
        prod_pass=prod_pass,
        notes=notes,
    )


def cross_llm_consistency(
    event_pairs: list[tuple[EventTensor, EventTensor]],
) -> dict:
    """跨 LLM 一致率评估（产品定义 §6.3 ≥ 75%）。

    输入：list of (event_main, event_second) — 同 spec 在两个 LLM 解析后的 event 对
    输出：{p0_agree_rate, total_agree_rate, n}
    """
    if not event_pairs:
        return {"valid": False, "reason": "empty_pairs", "p0_agree_rate": 0.0}

    p0_agrees = 0
    p0_total_checks = 0
    total_agrees = 0
    total_checks = 0
    for ev1, ev2 in event_pairs:
        for sid in P0_SLOTS:
            p0_total_checks += 1
            v1 = ev1.get(sid)
            v2 = ev2.get(sid)
            if v1 is None and v2 is None:
                p0_agrees += 1
            elif v1 is not None and v2 is not None and v1.value == v2.value:
                p0_agrees += 1
        # 全槽位（含 P1-P3）
        keys1 = set(ev1.slots.keys())
        keys2 = set(ev2.slots.keys())
        for sid in keys1 | keys2:
            total_checks += 1
            v1 = ev1.get(sid)
            v2 = ev2.get(sid)
            if v1 is None or v2 is None:
                continue
            if v1.value == v2.value:
                total_agrees += 1

    return {
        "valid": True,
        "n": len(event_pairs),
        "p0_agree_rate": p0_agrees / max(p0_total_checks, 1),
        "total_agree_rate": total_agrees / max(total_checks, 1),
    }


__all__ = [
    "BatchQualityReport",
    "evaluate_batch",
    "cross_llm_consistency",
    "PROD_P0_COMPLETENESS_THRESHOLD",
    "PROD_CROSS_LLM_CONSISTENCY_THRESHOLD",
    "POC_P0_COMPLETENESS_THRESHOLD",
    "POC_SCENE_DIVERSITY_MIN",
    "POC_CROSS_LLM_CONSISTENCY_THRESHOLD",
]
