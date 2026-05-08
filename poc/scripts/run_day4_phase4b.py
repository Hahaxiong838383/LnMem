"""Day 4 Phase 4B — 30 画像 × 23 天补足（day_index 7..29 = 5,520h）。

加上 Phase 3 day0 (240h) + Phase 4A day1..6 (1,440h) = 30 天 × 30 画像 × 8h = **7,200h**。

usage:
    cd /tmp/lnmem-fix/LnMem
    MINIMAX_API_KEY=... PYTHONPATH=. .venv-poc/bin/python poc/scripts/run_day4_phase4b.py
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
LOG_PATH = OUT_DIR / "_phase4b_log.tsv"

DAYS_TO_GENERATE = list(range(7, 30))  # day_index 7..29


def _list_personas() -> list[Path]:
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
            seed=42 + int(pid.split("_")[-1]) + day_index * 100,
        )
        out = write_script(s, OUT_DIR)
        elapsed = time.time() - t0
        return {
            "pid": pid,
            "day": day_index,
            "ok": True,
            "events": len(s.events),
            "h": sum(e.duration_s for e in s.events) / 3600.0,
            "secs": elapsed,
            "path": out,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - t0
        return {
            "pid": pid,
            "day": day_index,
            "ok": False,
            "events": 0,
            "h": 0.0,
            "secs": elapsed,
            "err": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    personas = _list_personas()
    if len(personas) < 30:
        print(f"[FATAL] expect >=30 personas, found {len(personas)}", file=sys.stderr)
        return 2

    tasks = [(p, d) for p in personas for d in DAYS_TO_GENERATE]
    print(f"[start] {len(personas)} personas × {len(DAYS_TO_GENERATE)} days = {len(tasks)} tasks", flush=True)

    client = MiniMaxLLMClient(
        model="MiniMax-M2.7-highspeed",
        timeout_s=480.0,
        max_attempts=3,
        max_fillin_rounds=2,
        min_coverage_ratio=0.95,
        first_call_retries=2,
    )

    t_start = time.time()
    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_gen_one, p, client, d): (p, d) for p, d in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            tag = "OK " if r["ok"] else "FAIL"
            print(
                f"[{done}/{len(tasks)}] [{tag}] {r['pid']} d{r['day']}: "
                f"{r['events']} events / {r['h']:.2f}h / {r['secs']:.1f}s "
                + (r.get("err", "") if not r["ok"] else ""),
                flush=True,
            )
            rows.append(r)

    wall = time.time() - t_start
    ok_rows = [r for r in rows if r["ok"]]
    fail_rows = [r for r in rows if not r["ok"]]
    print(f"\n[gen wall] {wall:.1f}s ({wall/60:.1f}m)  ok={len(ok_rows)}  fail={len(fail_rows)}", flush=True)

    with LOG_PATH.open("w", encoding="utf-8") as fp:
        fp.write("pid\tday\tok\tevents\tcoverage_h\tsecs\terr\n")
        for r in sorted(rows, key=lambda x: (x["pid"], x["day"])):
            fp.write(
                f"{r['pid']}\t{r['day']}\t{r['ok']}\t{r['events']}\t"
                f"{r['h']:.2f}\t{r['secs']:.1f}\t{r.get('err', '')}\n"
            )
    print(f"[log] {LOG_PATH}", flush=True)

    print("\n[signals] converting", flush=True)
    for r in ok_rows:
        sp = OUT_DIR / f"script_{r['pid']}_day{r['day']}.json"
        gp = OUT_DIR / f"signals_{r['pid']}_day{r['day']}.jsonl"
        try:
            script_file_to_signals_file(sp, gp, seed=int(r["pid"].split("_")[-1]) + r["day"] * 1000)
        except Exception as exc:  # noqa: BLE001
            print(f"  [signals FAIL] {r['pid']} d{r['day']}: {exc}", flush=True)
    print(f"[signals] done", flush=True)

    print("\n[report] aggregating", flush=True)
    report = build_report(sorted(OUT_DIR.glob("script_*.json")), PERSONAS_DIR)
    write_markdown_report(report, OUT_DIR / "dataset_report.md")
    n_pairs = sum(1 for a, row in report.hexaco_pairwise_l2.items() for b in row if a < b)
    n_pass = sum(
        1 for a, row in report.hexaco_pairwise_l2.items() for b, d in row.items() if a < b and d >= 0.4
    )
    print(
        f"[report] {report.n_personas} personas / {report.total_duration_h:.2f}h / "
        f"hexaco pairs PASS={n_pass}/{n_pairs}",
        flush=True,
    )
    return 0 if not fail_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
