"""Day 4 Phase 3 — 30 画像 × 1 天 MiniMax baseline 生成（5 并发）。

usage:
    cd /tmp/lnmem-fix/LnMem
    MINIMAX_API_KEY=... PYTHONPATH=. .venv-poc/bin/python poc/scripts/run_day4_phase3.py

输出：
    poc/data/synthetic/llm_v2/script_vboss_NNN_day0.json (30 个)
    poc/data/synthetic/llm_v2/signals_vboss_NNN_day0.jsonl (30 个，可选)
    poc/data/synthetic/llm_v2/dataset_report.md
    poc/data/synthetic/llm_v2/_phase3_log.tsv  # 单画像耗时/事件数/coverage
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 让 PYTHONPATH 兜底
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poc.src.data.llm_clients import MiniMaxLLMClient
from poc.src.dataset_report import build_report, write_markdown_report
from poc.src.generate_scripts import generate_day_script, write_script
from poc.src.script_to_signals import script_file_to_signals_file


PERSONAS_DIR = ROOT / "poc/data/personas"
OUT_DIR = ROOT / "poc/data/synthetic/llm_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "_phase3_log.tsv"


def _list_personas() -> list[Path]:
    """vboss_001..vboss_030（不含 vboss_self，因为它是老板真实测评）。"""
    return sorted(p for p in PERSONAS_DIR.glob("vboss_0*.md") if "_self" not in p.name)


def _gen_one(persona_path: Path, client: MiniMaxLLMClient, day_index: int) -> dict:
    pid = persona_path.stem
    t0 = time.time()
    try:
        s = generate_day_script(
            persona_path=persona_path,
            day_index=day_index,
            llm=client,
            target_hours=8.0,
            seed=42 + int(pid.split("_")[-1]),
        )
        out = write_script(s, OUT_DIR)
        elapsed = time.time() - t0
        n_events = len(s.events)
        h = sum(e.duration_s for e in s.events) / 3600.0
        return {"pid": pid, "ok": True, "events": n_events, "h": h, "secs": elapsed, "path": out}
    except Exception as exc:  # noqa: BLE001 — 记到日志，整体跑不挂
        elapsed = time.time() - t0
        return {"pid": pid, "ok": False, "events": 0, "h": 0.0, "secs": elapsed, "err": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    personas = _list_personas()
    if len(personas) < 30:
        print(f"[FATAL] expect >=30 personas, found {len(personas)}", file=sys.stderr)
        return 2

    client = MiniMaxLLMClient(
        model="MiniMax-M2.7-highspeed",
        timeout_s=480.0,
        max_attempts=2,
        max_fillin_rounds=2,
        min_coverage_ratio=0.95,
    )
    print(f"[start] generating {len(personas)} scripts × 1 day → {OUT_DIR}")
    t_start = time.time()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_gen_one, p, client, 0): p for p in personas}
        for fut in as_completed(futs):
            r = fut.result()
            tag = "OK " if r["ok"] else "FAIL"
            print(f"[{tag}] {r['pid']}: {r['events']} events / {r['h']:.2f}h / {r['secs']:.1f}s "
                  + (r.get("err", "") if not r["ok"] else ""))
            rows.append(r)
    wall = time.time() - t_start
    ok_rows = [r for r in rows if r["ok"]]
    fail_rows = [r for r in rows if not r["ok"]]
    print(f"\n[gen wall] {wall:.1f}s ({wall/60:.1f}m)  ok={len(ok_rows)}  fail={len(fail_rows)}")

    # 写日志
    with LOG_PATH.open("w", encoding="utf-8") as fp:
        fp.write("pid\tok\tevents\tcoverage_h\tsecs\terr\n")
        for r in sorted(rows, key=lambda x: x["pid"]):
            fp.write(f"{r['pid']}\t{r['ok']}\t{r['events']}\t{r['h']:.2f}\t{r['secs']:.1f}\t{r.get('err', '')}\n")
    print(f"[log] {LOG_PATH}")

    # 跑 signals + report
    print("\n[signals] converting scripts to 12-dim frames")
    for r in ok_rows:
        sp = OUT_DIR / f"script_{r['pid']}_day0.json"
        gp = OUT_DIR / f"signals_{r['pid']}_day0.jsonl"
        n = script_file_to_signals_file(sp, gp, seed=int(r["pid"].split("_")[-1]))
        print(f"  signals_{r['pid']}: {n} frames")

    print("\n[report] aggregating")
    report = build_report(sorted(OUT_DIR.glob("script_*.json")), PERSONAS_DIR)
    write_markdown_report(report, OUT_DIR / "dataset_report.md")
    n_pairs = sum(1 for a, row in report.hexaco_pairwise_l2.items() for b in row if a < b)
    n_pass = sum(1 for a, row in report.hexaco_pairwise_l2.items() for b, d in row.items() if a < b and d >= 0.4)
    print(f"[report] {report.n_personas} personas / {report.total_duration_h:.2f}h / "
          f"hexaco pairs PASS={n_pass}/{n_pairs}")
    return 0 if not fail_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
