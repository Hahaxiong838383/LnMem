"""poc/src/lnn/extend_persona.py — 扩展 LNN ckpt 增加新画像 slot（如 vboss_self）。

使用场景
--------
day7_v2/checkpoint_day5.pt 含 30 个虚拟画像（vboss_001..030）。新增真人画像
vboss_self（陈川 personalt-test v0.2 自评）需要：

1. embedding shape (30, 8) → (31, 8)，第 31 行用 informed init（HEXACO + motivation）
2. persona_to_idx 加 'vboss_self' → 30
3. config 字段 n_personas: 30 → 31（frozen dataclass 用 dataclasses.replace）

不破坏 base 的训练成果：原 30 行 embedding / tau_projection / FiLM 全保留。

设计原则
--------
- **零下游侵入**：仅修改 ckpt 数据，不改 model.py / personalization.py / contracts.py
- **守纪律**：纯 informed init（自评数据 → 数值），不引入合成 trajectory 或微调（违反
  "不准 fallback 到合成数据微调" 约束）
- **partial 标注**：所有 ⚠️ 字段必须显式落到 metadata，让下游知道哪些维度是占位

informed init 策略（HEXACO + ECR + motivation → 8 维 embedding）
----------------------------------------------------------------
8 维 embedding 的具体含义不是固定的（base.tau_projection 自己学到的），但我们把
真人特质的"显著维度"塞进去，让初始 tau_projection 输出的 τ-bias 不再是 30 行的
"中性平均"，而是带个性化先验的起点。后续真实反馈可在 OnlineLoRAAdapter.tau_drift
基础上继续学习。

具体映射（中心化到 [-0.5, +0.5] 与 base 数值范围吻合）：
- dim 0: HEXACO Honesty-Humility - 0.5
- dim 1: HEXACO Emotionality - 0.5
- dim 2: HEXACO eXtraversion - 0.5
- dim 3: HEXACO Agreeableness - 0.5
- dim 4: HEXACO Conscientiousness - 0.5
- dim 5: HEXACO Openness - 0.5
- dim 6: ECR avoidance - 0.5（dismissive_avoidant 关键识别）
- dim 7: 4 维 motivation 均值 - 0.5

Partial 字段（必须 metadata 显式标注，不影响 embedding 本身但影响下游解读）：
- CAPS if-then patterns: ⚠️ 全空，待真实采集
- 多模态信号签名: ⚠️ confidence=0.3 占位
- Schwartz values 完整 10 维: 仅 top3/bottom3 来自自评，剩余 ⚠️
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

_log = logging.getLogger(__name__)


# ---------- 自评数据解析 ----------


class PersonaParseError(ValueError):
    """vboss markdown 解析失败。"""


def parse_persona_markdown(md_path: Path) -> dict[str, Any]:
    """从 vboss_self.md frontmatter 抽 HEXACO + ECR + motivation。

    Returns
    -------
    {
      "id": str,
      "hexaco": dict[str, float],     # 6 keys
      "attachment": dict[str, Any],
      "motivation": dict[str, float], # 4 keys
      "values_top3": list[str],
      "values_bottom": list[str],
      "partial_fields": list[str],    # 显式标注的 ⚠️ 占位字段
    }
    """
    md_path = Path(md_path)
    if not md_path.exists():
        raise PersonaParseError(f"persona md not found: {md_path}")

    text = md_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise PersonaParseError(f"{md_path} 缺少 YAML frontmatter")

    # 抽 frontmatter（首尾 --- 之间）
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise PersonaParseError(f"{md_path} frontmatter 格式错误")
    fm = parts[1]

    # 不引 yaml 依赖，自己解析需要的字段（PyYAML 在 fresh venv 不一定装）
    # 但 vboss_self.md 字段多嵌套，用 yaml 更稳；先尝试 import yaml
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(fm)
        if not isinstance(data, dict):
            raise PersonaParseError("frontmatter 不是 dict")
    except ImportError as e:
        raise PersonaParseError(
            "PyYAML 未安装，pip install pyyaml；或自行扩展 parser"
        ) from e

    # 校验关键字段
    hexaco = data.get("hexaco")
    if not isinstance(hexaco, dict):
        raise PersonaParseError(f"{md_path} 缺少 hexaco 字段")
    expected_keys = {
        "honesty_humility", "emotionality", "extraversion",
        "agreeableness", "conscientiousness", "openness",
    }
    missing = expected_keys - set(hexaco.keys())
    if missing:
        raise PersonaParseError(f"hexaco 缺字段: {sorted(missing)}")

    attachment = data.get("attachment") or {}
    motivation = data.get("motivation") or {}

    # 抽 ⚠️ partial 字段（显式标注）
    partial_fields: list[str] = []
    if attachment.get("confidence", 1.0) < 0.8:
        partial_fields.append("attachment.confidence_low")
    multimodal = data.get("multimodal_signal_density")
    if data.get("data_source", "").startswith("personalt-test"):
        partial_fields.extend([
            "caps_if_then_patterns_empty",
            "multimodal_signature_confidence_0.3",
            "schwartz_values_full_10dim_partial",
        ])

    return {
        "id": data["id"],
        "hexaco": {k: float(hexaco[k]) for k in expected_keys},
        "attachment": attachment,
        "motivation": {k: float(v) for k, v in motivation.items()},
        "values_top3": data.get("values_top3") or [],
        "values_bottom": data.get("values_bottom") or [],
        "partial_fields": partial_fields,
    }


# ---------- Informed init ----------


def hexaco_to_embedding(
    persona_data: dict[str, Any],
    embed_dim: int = 8,
) -> torch.Tensor:
    """HEXACO + ECR avoidance + motivation mean → 8 维 embedding（centered）.

    Returns
    -------
    embedding : Tensor (embed_dim,)
        值范围 [-0.5, +0.5]，与 base.embedding 数值规模相近（base 由 nn.Embedding
        默认 N(0,1) 初始化后训练）。

    Raises
    ------
    PersonaParseError
        embed_dim != 8（当前固定，未来可扩）
    """
    if embed_dim != 8:
        raise PersonaParseError(
            f"current implementation supports embed_dim=8, got {embed_dim}"
        )

    hexaco = persona_data["hexaco"]
    avoidance = float(persona_data["attachment"].get("avoidance", 0.5))
    motiv = persona_data["motivation"]
    motiv_mean = (
        float(sum(motiv.values()) / len(motiv))
        if motiv else 0.5
    )

    raw = torch.tensor([
        hexaco["honesty_humility"],
        hexaco["emotionality"],
        hexaco["extraversion"],
        hexaco["agreeableness"],
        hexaco["conscientiousness"],
        hexaco["openness"],
        avoidance,
        motiv_mean,
    ], dtype=torch.float32)
    # 中心化到 [-0.5, +0.5]
    return raw - 0.5


# ---------- Ckpt 扩展 ----------


def extend_ckpt_with_persona(
    src_ckpt_path: Path,
    dst_ckpt_path: Path,
    persona_id: str,
    persona_md_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """从 src ckpt 加载 + extend embedding +1 行 + 存到 dst。

    Returns
    -------
    summary : dict
        {persona_idx, embedding_init, partial_fields, n_personas_old/new}
    """
    src_ckpt_path = Path(src_ckpt_path)
    dst_ckpt_path = Path(dst_ckpt_path)
    if not src_ckpt_path.exists():
        raise PersonaParseError(f"src ckpt not found: {src_ckpt_path}")
    if dst_ckpt_path.exists() and not overwrite:
        raise PersonaParseError(
            f"dst ckpt exists, set overwrite=True: {dst_ckpt_path}"
        )

    persona_data = parse_persona_markdown(persona_md_path)
    parsed_id = persona_data["id"]
    if parsed_id != persona_id:
        raise PersonaParseError(
            f"persona md id={parsed_id!r} mismatch arg={persona_id!r}"
        )

    ckpt = torch.load(src_ckpt_path, map_location="cpu", weights_only=False)
    persona_to_idx: dict[str, int] = dict(ckpt["persona_to_idx"])
    if persona_id in persona_to_idx:
        raise PersonaParseError(
            f"persona '{persona_id}' already in ckpt at idx={persona_to_idx[persona_id]}"
        )

    n_old = len(persona_to_idx)
    new_idx = n_old
    persona_to_idx[persona_id] = new_idx

    # 扩 state_model embedding：(n_old, 8) → (n_old+1, 8)
    sm: dict[str, torch.Tensor] = dict(ckpt["state_model"])
    emb_key = "personalization.embedding.weight"
    old_emb = sm[emb_key]
    if old_emb.shape[0] != n_old:
        raise PersonaParseError(
            f"embedding shape mismatch: emb={old_emb.shape}, expected ({n_old}, ?)"
        )
    embed_dim = int(old_emb.shape[1])
    new_row = hexaco_to_embedding(persona_data, embed_dim=embed_dim).to(
        old_emb.dtype
    ).unsqueeze(0)  # (1, embed_dim)
    new_emb = torch.cat([old_emb, new_row], dim=0)  # (n_old+1, embed_dim)
    sm[emb_key] = new_emb

    # 同步 boundary_model（共享 personalization.embedding，结构相同）
    bm: dict[str, torch.Tensor] = dict(ckpt["boundary_model"])
    if emb_key in bm:
        bm[emb_key] = new_emb.clone()

    # frozen LNNTrainConfig: dataclasses.replace 创建新实例
    old_config = ckpt["config"]
    new_config = replace(old_config, n_personas=n_old + 1)

    # 组装 new ckpt
    new_ckpt = dict(ckpt)
    new_ckpt["config"] = new_config
    new_ckpt["persona_to_idx"] = persona_to_idx
    new_ckpt["state_model"] = sm
    new_ckpt["boundary_model"] = bm
    # metadata：partial 字段显式留给下游
    new_ckpt["extension_metadata"] = {
        "extended_persona_id": persona_id,
        "extended_idx": new_idx,
        "init_source": "hexaco_to_embedding(personalt-test v0.2)",
        "init_embedding": new_row.squeeze(0).tolist(),
        "partial_fields": persona_data["partial_fields"],
        "data_source_md": str(persona_md_path),
    }

    dst_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, dst_ckpt_path)

    return {
        "persona_id": persona_id,
        "persona_idx": new_idx,
        "n_personas_old": n_old,
        "n_personas_new": n_old + 1,
        "embedding_init": new_row.squeeze(0).tolist(),
        "partial_fields": persona_data["partial_fields"],
        "dst_path": str(dst_ckpt_path),
    }


__all__ = [
    "PersonaParseError",
    "parse_persona_markdown",
    "hexaco_to_embedding",
    "extend_ckpt_with_persona",
]
