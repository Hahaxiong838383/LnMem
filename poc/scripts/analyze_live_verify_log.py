#!/usr/bin/env python3
"""Analyze live_verify log frames.
中文说明：清洗 ANSI 并输出状态/置信度/表情诊断报告。
"""
from __future__ import annotations

import argparse
import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

FRAME_RE = re.compile(r"#\s*(\d+)")
CONF_RE = re.compile(r"(\d+)%")
STATE_SEG_RE = re.compile(r"#\s*\d+\s+(.+?)\s+\[[^\]]*\]\s*\d+%")
ZH_RE = re.compile(r"([\u4e00-\u9fff]{2,4})")
EYE_RE = re.compile(r"👁️?([\d.]+)")
GAZE_RE = re.compile(r"👀([\d.]+)")
SMILE_RE = re.compile(r"😊([\d.]+)")
BROW_RE = re.compile(r"😠([\d.]+)")


@dataclass(frozen=True)
class FrameRecord:
    """Parsed frame row. 中文说明：单帧关键字段。"""

    frame: int
    state_cn: str
    state_en: str
    confidence: float
    eye: float
    gaze: float
    smile: float
    brow: float


def strip_ansi(line: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", line)


def load_state_mapping() -> tuple[dict[str, str], dict[str, str]]:
    """Load STATE_DISPLAY. 中文说明：提取中英 state 映射。"""
    live_verify = Path(__file__).with_name("live_verify.py")
    if not live_verify.exists():
        return {}, {}
    tree = ast.parse(live_verify.read_text(encoding="utf-8"), filename=str(live_verify))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "STATE_DISPLAY" for t in node.targets
        ):
            raw = ast.literal_eval(node.value)
            en_to_cn, cn_to_en = {}, {}
            for state_en, display in raw.items():
                text = str(display[0]) if isinstance(display, tuple) and display else ""
                m = ZH_RE.search(text)
                if m:
                    state_cn = m.group(1)
                    en_to_cn[state_en] = state_cn
                    cn_to_en[state_cn] = state_en
            return en_to_cn, cn_to_en
    return {}, {}


def _extract_float(pattern: re.Pattern[str], text: str) -> float:
    m = pattern.search(text)
    return float(m.group(1)) if m else 0.0


def parse_frame_line(line: str, cn_to_en: dict[str, str]) -> FrameRecord | None:
    """Parse frame first line. 中文说明：仅解析含 #frame 的行。"""
    clean = strip_ansi(line).strip()
    m_frame, m_conf = FRAME_RE.search(clean), CONF_RE.search(clean)
    if not m_frame or not m_conf:
        return None
    m_seg = STATE_SEG_RE.search(clean)
    source = m_seg.group(1) if m_seg else clean[: m_conf.start()]
    m_zh = ZH_RE.search(source)
    if not m_zh:
        return None
    state_cn = m_zh.group(1)
    return FrameRecord(
        frame=int(m_frame.group(1)),
        state_cn=state_cn,
        state_en=cn_to_en.get(state_cn, "unknown"),
        confidence=int(m_conf.group(1)) / 100.0,
        eye=_extract_float(EYE_RE, clean),
        gaze=_extract_float(GAZE_RE, clean),
        smile=_extract_float(SMILE_RE, clean),
        brow=_extract_float(BROW_RE, clean),
    )


def parse_log(path: Path, cn_to_en: dict[str, str]) -> list[FrameRecord]:
    frame_map: dict[int, FrameRecord] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        rec = parse_frame_line(line, cn_to_en)
        if rec:
            frame_map[rec.frame] = rec
    return [frame_map[k] for k in sorted(frame_map)]


def print_report(records: list[FrameRecord], en_to_cn: dict[str, str]) -> None:
    if not records:
        print("未解析到任何帧，请检查 log 路径和格式。")
        print("## Summary")
        print("total_frames=0 | states=0 | lowest_conf_bucket=N/A | emotional_positive=no | smile_peak=0.00")
        return

    counts = Counter(r.state_cn for r in records)
    print("=== 整体统计 ===")
    print(f"总帧数: {len(records)}")
    print(f"平均 confidence: {mean(r.confidence for r in records):.3f} ({mean(r.confidence for r in records):.1%})")
    print(f"smile 均值: {mean(r.smile for r in records):.3f} | brow 均值: {mean(r.brow for r in records):.3f}")
    print("state 频次:")
    for state, cnt in counts.most_common():
        print(f"  - {state}: {cnt}")

    bucket_conf: dict[tuple[int, int], list[float]] = defaultdict(list)
    bucket_sig: dict[tuple[int, int], list[tuple[float, float, float, float]]] = defaultdict(list)
    for r in records:
        key = (((r.frame - 1) // 25) * 25 + 1, ((r.frame - 1) // 25) * 25 + 25)
        bucket_conf[key].append(r.confidence)
        bucket_sig[key].append((r.eye, r.gaze, r.smile, r.brow))

    print("\n=== 置信度时序（每25帧 bucket）===")
    low_bucket, low_min = None, 1.0
    for key in sorted(bucket_conf):
        vals, sigs = bucket_conf[key], bucket_sig[key]
        mn, mx, av = min(vals), max(vals), mean(vals)
        if mn < low_min:
            low_min, low_bucket = mn, (key[0], key[1], mn)
        eye_m, gaze_m = mean(v[0] for v in sigs), mean(v[1] for v in sigs)
        smile_m, brow_m = mean(v[2] for v in sigs), mean(v[3] for v in sigs)
        print(f"  - [{key[0]:>4d}-{key[1]:>4d}] conf={mn:.2f}/{mx:.2f}/{av:.2f} | eye/gaze/smile/brow={eye_m:.2f}/{gaze_m:.2f}/{smile_m:.2f}/{brow_m:.2f}")
    print(f"结论：是否存在 bucket 最低 confidence < 0.50: {'是' if any(min(v) < 0.5 for v in bucket_conf.values()) else '否'}")

    print("\n=== State 切换序列（真切换）===")
    switches: list[tuple[int, FrameRecord, FrameRecord]] = []
    prev = records[0]
    for cur in records[1:]:
        if cur.state_cn != prev.state_cn:
            switches.append((cur.frame, prev, cur))
        prev = cur
    if not switches:
        print("  - 无状态切换")
    else:
        for frame, frm, to in switches:
            print(f"  - frame={frame} {frm.state_cn}->{to.state_cn} (smile={to.smile:.2f}, brow={to.brow:.2f})")

    print("\n=== emotional_positive 命中分析 ===")
    ep_cn = en_to_cn.get("emotional_positive", "积极")
    ep_hits = [r for r in records if r.state_cn == ep_cn or r.state_en == "emotional_positive"]
    if not ep_hits:
        print("  - 未命中 emotional_positive")
    else:
        print(f"  - 命中帧数: {len(ep_hits)}")
        frame_to_smile = {r.frame: r.smile for r in records}
        for r in ep_hits:
            win = [s for f, s in frame_to_smile.items() if r.frame - 10 <= f <= r.frame + 10]
            print(f"  - frame={r.frame} 周围±10帧 smile均值={mean(win) if win else 0.0:.3f}")

    focused_cn = en_to_cn.get("focused", "专注")
    jump = any(frm.state_cn == focused_cn and to.state_cn == ep_cn for _, frm, to in switches)
    low_desc = f"{low_bucket[0]}-{low_bucket[1]}(min={low_bucket[2]:.2f})" if low_bucket else "N/A"
    print(f"focused -> emotional_positive 是否出现: {'是' if jump else '否'}")
    print("\n## Summary")
    print(f"total_frames={len(records)} | states={len(counts)} | lowest_conf_bucket={low_desc} | emotional_positive={'yes' if ep_hits else 'no'} | smile_peak={max(r.smile for r in records):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze live_verify.py realtime log")
    parser.add_argument("log_path", nargs="?", default="/tmp/live_verify.log")
    args = parser.parse_args()
    path = Path(args.log_path)
    if not path.exists():
        raise FileNotFoundError(f"log file not found: {path}")
    en_to_cn, cn_to_en = load_state_mapping()
    print_report(parse_log(path, cn_to_en), en_to_cn)


if __name__ == "__main__":
    main()
