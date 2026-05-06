"""
LnMem PoC 虚拟主人画像校验脚本 v0.1

用法：
    python poc/src/validate_personas.py                       # 校验全部 personas/
    python poc/src/validate_personas.py vboss_001 vboss_002   # 仅校验指定 ID

输出：
    - 单画像质量门：7 条评估标准逐项 PASS/FAIL
    - 集合质量门（仅当跑全部时）：HEXACO 6 维余弦距离矩阵 + 依恋分布 + 覆盖度
    - 失败画像列表 + 修订建议

依赖：pyyaml + numpy（见 poc/requirements-poc.txt）

参考：poc/docs/data-generation-todo.md §4 评估标准
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("⚠️  缺少 pyyaml，运行：pip install pyyaml numpy", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("⚠️  缺少 numpy，运行：pip install pyyaml numpy", file=sys.stderr)
    sys.exit(1)


PERSONAS_DIR = Path(__file__).resolve().parent.parent / "data" / "personas"
HEXACO_KEYS = (
    "honesty_humility",
    "emotionality",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness",
)
ATTACHMENT_TYPES = {
    "secure",
    "anxious",
    "avoidant",
    "fearful_avoidant",
    "secure_with_anxious_tendency",
}
SCHWARTZ_VALUES = {
    # Schwartz PVQ-10 标准 10 维（2026-05-05 v0.9 修订：删除 autonomy，统一用 self-direction
    # 对齐 personalt-test data/values-questions.json source-of-truth）
    "self-direction",
    "stimulation",
    "hedonism",
    "achievement",
    "power",
    "security",
    "conformity",
    "tradition",
    "benevolence",
    "universalism",
}
SIMILARITY_THRESHOLD = 0.4  # HEXACO 6 维欧式距离下限（< 0.4 则差异化不足，老板设）
# Note: HEXACO 6 维都是 [0,1] 正值，余弦距离恒接近 0（向量方向一致）。
# 欧式距离 √Σ(a_i-b_i)² 是更合理的差异度量——直接反映分数差。


def load_persona(path: Path) -> dict[str, Any]:
    """读 markdown 文件，提取 frontmatter + body。"""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        raise ValueError(f"{path.name}: 缺少 YAML frontmatter")
    frontmatter = yaml.safe_load(match.group(1))
    body = match.group(2)
    return {"path": path, "frontmatter": frontmatter, "body": body}


def hexaco_vector(persona: dict[str, Any]) -> np.ndarray:
    """从 frontmatter 提取 HEXACO 6 维向量。"""
    fm = persona["frontmatter"]
    hexaco = fm.get("hexaco") or {}
    missing = [k for k in HEXACO_KEYS if k not in hexaco]
    if missing:
        raise ValueError(f"{persona['path'].name}: HEXACO 缺字段 {missing}")
    return np.array([hexaco[k] for k in HEXACO_KEYS], dtype=float)


def hexaco_distance(a: np.ndarray, b: np.ndarray) -> float:
    """欧式距离 √Σ(a_i-b_i)²。HEXACO 6 维 0-1 上 ≥ 0.4 = 画像差异显著。"""
    return float(np.linalg.norm(a - b))


# ==================== 单画像 7 条评估门 ====================


def gate_frontmatter_complete(persona: dict[str, Any]) -> tuple[bool, str]:
    """门 1：frontmatter 必填字段齐全。"""
    fm = persona["frontmatter"]
    required = ["id", "name", "type_label", "hexaco", "attachment", "motivation",
                "values_top3", "demographics"]
    missing = [k for k in required if k not in fm]
    if missing:
        return False, f"frontmatter 缺字段 {missing}"
    return True, "frontmatter 完整"


def gate_hexaco_in_range(persona: dict[str, Any]) -> tuple[bool, str]:
    """门 2：HEXACO 6 维都在 0.0-1.0。"""
    try:
        vec = hexaco_vector(persona)
    except ValueError as e:
        return False, str(e)
    if (vec < 0).any() or (vec > 1).any():
        return False, f"HEXACO 值越界 {vec.tolist()}"
    return True, f"HEXACO 范围 OK {dict(zip(HEXACO_KEYS, vec.round(2).tolist()))}"


def gate_attachment_valid(persona: dict[str, Any]) -> tuple[bool, str]:
    """门 3：依恋类型在白名单 + 焦虑/回避两维 0-1。"""
    fm = persona["frontmatter"]
    att = fm.get("attachment") or {}
    if att.get("type") not in ATTACHMENT_TYPES:
        return False, f"attachment.type='{att.get('type')}' 不在白名单 {sorted(ATTACHMENT_TYPES)}"
    for k in ("anxiety", "avoidance"):
        v = att.get(k)
        if not isinstance(v, (int, float)) or not 0 <= v <= 1:
            return False, f"attachment.{k}='{v}' 不在 0-1"
    return True, f"依恋 OK type={att['type']} anxiety={att['anxiety']} avoidance={att['avoidance']}"


def gate_values_in_schwartz(persona: dict[str, Any]) -> tuple[bool, str]:
    """门 4：values_top3 + values_bottom 在 Schwartz 10 价值观清单。"""
    fm = persona["frontmatter"]
    top3 = fm.get("values_top3") or []
    bottom = fm.get("values_bottom") or []
    if len(top3) < 3:
        return False, f"values_top3 至少 3 个，当前 {len(top3)}"
    invalid = [v for v in top3 + bottom if v not in SCHWARTZ_VALUES]
    if invalid:
        return False, f"值 {invalid} 不在 Schwartz 10 清单 {sorted(SCHWARTZ_VALUES)}"
    return True, f"价值观 OK top3={top3}"


def gate_if_then_count(persona: dict[str, Any]) -> tuple[bool, str]:
    """门 5：Layer 4 if-then 情境 ≥ 5 个（parse '情境 N' 章节标题）。"""
    body = persona["body"]
    matches = re.findall(r"\*\*情境\s*\d+", body)
    if len(matches) < 5:
        return False, f"if-then 情境只有 {len(matches)} 个，要求 ≥ 5"
    return True, f"if-then 情境 {len(matches)} 个"


def gate_multimodal_signals_complete(persona: dict[str, Any]) -> tuple[bool, str]:
    """门 6：Layer 6 多模态信号 4 类全描述（视觉/疲劳/压力/专注）。"""
    body = persona["body"]
    required = ["视觉基线", "疲劳信号", "压力信号", "专注信号"]
    missing = [k for k in required if k not in body]
    if missing:
        return False, f"多模态信号缺 {missing}"
    return True, "多模态 4 类信号全描述"


def gate_extreme_trait(persona: dict[str, Any]) -> tuple[bool, str]:
    """门 7（信息性）：是否含极端 HEXACO 维度（≤ 0.30 或 ≥ 0.85）。
    不影响 PASS/FAIL，仅记录用于集合覆盖度统计。"""
    vec = hexaco_vector(persona)
    extreme_dims = [k for k, v in zip(HEXACO_KEYS, vec) if v <= 0.30 or v >= 0.85]
    if extreme_dims:
        return True, f"含极端维度 {extreme_dims}"
    return True, "无极端维度（中等画像）"


SINGLE_GATES = [
    ("frontmatter 完整", gate_frontmatter_complete),
    ("HEXACO 范围 0-1", gate_hexaco_in_range),
    ("依恋有效", gate_attachment_valid),
    ("Schwartz 价值观", gate_values_in_schwartz),
    ("if-then ≥ 5", gate_if_then_count),
    ("多模态 4 类信号", gate_multimodal_signals_complete),
    ("极端维度（信息性）", gate_extreme_trait),
]


def validate_single(persona: dict[str, Any]) -> tuple[int, int, list[str]]:
    """跑 7 条门，返回 (passed, total, failures)。"""
    failures = []
    for label, gate in SINGLE_GATES:
        ok, msg = gate(persona)
        status = "✅" if ok else "❌"
        print(f"  {status} {label}: {msg}")
        if not ok:
            failures.append(f"{label}: {msg}")
    passed = len(SINGLE_GATES) - len(failures)
    return passed, len(SINGLE_GATES), failures


# ==================== 集合质量门 ====================


def collection_report(personas: list[dict[str, Any]]) -> None:
    """跑全部画像时输出集合报告。"""
    n = len(personas)
    print("\n" + "=" * 60)
    print(f"📊 集合质量门（共 {n} 个画像）")
    print("=" * 60)

    # HEXACO 欧式距离矩阵
    print("\n### HEXACO 6 维两两欧式距离（≥ 0.4 PASS）")
    vectors = {p["frontmatter"]["id"]: hexaco_vector(p) for p in personas}
    ids = sorted(vectors.keys())
    fail_pairs = []
    for i, id_a in enumerate(ids):
        for id_b in ids[i + 1:]:
            d = hexaco_distance(vectors[id_a], vectors[id_b])
            mark = "✅" if d >= SIMILARITY_THRESHOLD else "❌"
            print(f"  {mark} {id_a} ↔ {id_b}: {d:.3f}")
            if d < SIMILARITY_THRESHOLD:
                fail_pairs.append((id_a, id_b, d))
    if fail_pairs:
        print(f"  ⚠️  {len(fail_pairs)} 对画像差异化不足，建议修订其中之一")

    # 依恋分布
    print("\n### 依恋类型分布")
    type_counts: dict[str, int] = {}
    for p in personas:
        t = p["frontmatter"]["attachment"]["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    secure_like = type_counts.get("secure", 0) + type_counts.get(
        "secure_with_anxious_tendency", 0
    )
    avoidant = type_counts.get("avoidant", 0)
    anxious = type_counts.get("anxious", 0)
    fearful = type_counts.get("fearful_avoidant", 0)
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({100 * c / n:.1f}%)")
    print(f"  门：secure-like ≤ 65%（实际 {100 * secure_like / n:.1f}%）"
          + (" ✅" if secure_like / n <= 0.65 else " ❌"))
    print(f"  门：≥2 回避 + ≥1 焦虑 + ≥1 恐惧-回避（实际 {avoidant}/{anxious}/{fearful}）"
          + (" ✅" if avoidant >= 2 and anxious >= 1 and fearful >= 1 else " ❌"))

    # 极端案例覆盖
    print("\n### 极端 HEXACO 维度覆盖")
    extreme_count = sum(1 for p in personas if any(v <= 0.30 or v >= 0.85
                                                   for v in hexaco_vector(p)))
    print(f"  含极端维度的画像数：{extreme_count} / {n}")

    # similar_to_self 锚点
    print("\n### t-SNE 锚点（similar_to_self: true）")
    self_count = sum(1 for p in personas
                     if p["frontmatter"].get("similar_to_self") is True)
    print(f"  数量：{self_count}（建议 2-3 个）"
          + (" ✅" if 2 <= self_count <= 5 else " ❌"))

    # 职业类型
    print("\n### 职业类型覆盖")
    professions = {p["frontmatter"]["demographics"].get("profession", "?")
                   for p in personas}
    print(f"  种类：{len(professions)}（建议 ≥ 5）"
          + (" ✅" if len(professions) >= 5 else " ❌（继续写画像时补充）"))


# ==================== 主入口 ====================


def main() -> int:
    if not PERSONAS_DIR.exists():
        print(f"⚠️  目录不存在：{PERSONAS_DIR}", file=sys.stderr)
        return 1

    target_ids = sys.argv[1:]
    files = sorted(PERSONAS_DIR.glob("vboss_*.md"))
    if not files:
        print(f"⚠️  无画像文件 ({PERSONAS_DIR}/vboss_*.md)", file=sys.stderr)
        return 1

    if target_ids:
        files = [f for f in files if f.stem in target_ids]
        if not files:
            print(f"⚠️  指定的 ID 都不存在：{target_ids}", file=sys.stderr)
            return 1

    personas: list[dict[str, Any]] = []
    total_pass = total_count = 0
    failed_ids: list[str] = []

    for f in files:
        print(f"\n=== {f.stem} ===")
        try:
            persona = load_persona(f)
        except Exception as e:
            print(f"  ❌ 解析失败：{e}")
            failed_ids.append(f.stem)
            continue
        personas.append(persona)
        passed, total, failures = validate_single(persona)
        total_pass += passed
        total_count += total
        if failures:
            failed_ids.append(f.stem)

    if len(personas) > 1 and not target_ids:
        collection_report(personas)

    print("\n" + "=" * 60)
    print(f"汇总：{len(personas)} 画像，{total_pass}/{total_count} 门通过")
    if failed_ids:
        print(f"建议修订：{', '.join(failed_ids)}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
