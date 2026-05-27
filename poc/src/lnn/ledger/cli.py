"""CLI for Phase A attribution ledger inspection."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from poc.src.lnn.ledger.sqlite_reader import SQLiteLedgerReader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lnmem-ledger")
    parser.add_argument("--db", type=Path, default=None, help="Override ledger sqlite path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    lineage = subparsers.add_parser("lineage", help="Lookup lineage by ckpt sha8")
    lineage.add_argument("ckpt_sha8")
    lineage.add_argument("--json", action="store_true", dest="as_json")

    listing = subparsers.add_parser("list", help="List recent fit runs")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--json", action="store_true", dest="as_json")

    consistency = subparsers.add_parser("consistency", help="Scan ledger consistency")
    consistency.add_argument("--json", action="store_true", dest="as_json")

    inspect_cmd = subparsers.add_parser("inspect", help="Inspect a fit_run_id")
    inspect_cmd.add_argument("fit_run_id")
    inspect_cmd.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _render_run(reader: SQLiteLedgerReader, fit_run_id: str) -> dict[str, Any] | None:
    fit_run = reader.fit_run_by_id(fit_run_id)
    if fit_run is None:
        return None
    events = reader.events_of_run(fit_run_id)
    delta = reader.delta_of_run(fit_run_id)
    return {
        "fit_run": asdict(fit_run),
        "events": [asdict(event) for event in events],
        "delta_summary": None if delta is None else asdict(delta),
    }


def _print_human_run(payload: dict[str, Any]) -> None:
    fit_run = payload["fit_run"]
    events = payload["events"]
    delta = payload["delta_summary"] or {}
    event_ids = [event["event_uuid"] for event in events]
    print(f"fit_run_id: {fit_run['fit_run_id']}")
    print(
        "ckpt_in_sha8 → ckpt_out_sha8: "
        f"{fit_run['ckpt_in_sha8']} → {fit_run['ckpt_out_sha8']}"
    )
    print(f"fired_at: {fit_run['fired_at_utc']}")
    print(f"persona_idx: {fit_run['persona_idx']}")
    print(f"events ({len(event_ids)}): {event_ids}")
    print(f"delta_l2_total: {delta.get('delta_l2_total')}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    reader = SQLiteLedgerReader(db_path=args.db)

    if args.command == "lineage":
        fit_run = reader.lineage_by_ckpt_out(args.ckpt_sha8)
        if fit_run is not None:
            payload = _render_run(reader, fit_run.fit_run_id)
            if args.as_json:
                print(_json_dump(payload))
            else:
                _print_human_run(payload)
            return 0
        upstream = reader.lineage_by_ckpt_in(args.ckpt_sha8)
        payload = [
            _render_run(reader, run.fit_run_id)
            for run in upstream
        ]
        payload = [item for item in payload if item is not None]
        if not payload:
            print(f"no lineage found for ckpt sha8={args.ckpt_sha8}", file=sys.stderr)
            return 1
        if args.as_json:
            print(_json_dump(payload))
        else:
            for index, item in enumerate(payload):
                if index:
                    print()
                _print_human_run(item)
        return 0

    if args.command == "list":
        payload = [asdict(run) for run in reader.list_fit_runs(limit=args.limit)]
        if args.as_json:
            print(_json_dump(payload))
        else:
            for run in payload:
                print(
                    f"{run['fired_at_utc']} {run['fit_run_id']} "
                    f"{run['ckpt_in_sha8']}→{run['ckpt_out_sha8']} "
                    f"batch={run['batch_size']}"
                )
        return 0

    if args.command == "consistency":
        payload = asdict(reader.consistency_check())
        if args.as_json:
            print(_json_dump(payload))
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0

    if args.command == "inspect":
        payload = _render_run(reader, args.fit_run_id)
        if payload is None:
            print(f"fit_run_id not found: {args.fit_run_id}", file=sys.stderr)
            return 1
        if args.as_json:
            print(_json_dump(payload))
        else:
            _print_human_run(payload)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
