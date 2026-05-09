"""AU diagnostics script. / 实时诊断 AU 提取与校准链路。"""

from __future__ import annotations

import argparse
import csv
import signal
from pathlib import Path
from typing import Any

import cv2  # type: ignore

from poc.src.vision.au_extractor import AUExtractor
from poc.src.vision.feature_calibrator import AUCaliper, map_au_to_emotion

RAW_AUS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12",
    "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU28",
]
EXPR_KEYS = [
    "smile_intensity", "brow_furrow", "cheek_raise",
    "lip_corner_depress", "jaw_drop", "mouth_open_ratio",
]
EMO_KEYS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "contempt"]
PRINT_RAW = ["AU04", "AU06", "AU12", "AU01", "AU02", "AU15", "AU25", "AU26"]
PRINT_CAL = ["AU04", "AU06", "AU12", "AU01"]


def _f2(v: float) -> str:
    return f"{float(v):.2f}"


def _top1(emo: dict[str, float]) -> tuple[str, float]:
    if not emo:
        return "none", 0.0
    k = max(emo, key=lambda x: emo[x])
    return k, float(emo[k])


def _stats_line(name: str, vals: list[float]) -> str:
    return (
        f"{name}: min={min(vals):.2f} mean={(sum(vals)/len(vals)):.2f} "
        f"max={max(vals):.2f} (over {len(vals)} 帧)"
    )


def _calibrate(caliper: AUCaliper, raw: dict[str, float]) -> dict[str, float]:
    if hasattr(caliper, "calibrate"):
        return getattr(caliper, "calibrate")(raw)
    return caliper.step(raw)


def _expr(caliper: AUCaliper, cal: dict[str, float]) -> dict[str, float]:
    if hasattr(caliper, "to_lnn_expression"):
        return caliper.to_lnn_expression(cal)
    return {k: 0.0 for k in EXPR_KEYS}


def _open_csv(path: str | None) -> tuple[Any | None, csv.writer | None]:
    if not path:
        return None, None
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    f = fp.open("w", encoding="utf-8", newline="")
    w = csv.writer(f)
    w.writerow(
        ["frame"]
        + [f"raw_{k}" for k in RAW_AUS]
        + [f"cal_{k}" for k in RAW_AUS]
        + EXPR_KEYS
        + EMO_KEYS
    )
    return f, w


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose AU extraction chain in realtime.")
    parser.add_argument("--save-csv", type=str, default=None, help="Save frame features to CSV.")
    args = parser.parse_args()

    print("loading py-feat...")
    extractor = AUExtractor()
    caliper = AUCaliper()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("camera open failed: cv2.VideoCapture(0)")
        return 1

    csv_file, csv_writer = _open_csv(args.save_csv)
    running = True

    def _stop(_sig: int, _frm: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)

    frame_id = 0
    raw04_stats: list[float] = []
    cal04_stats: list[float] = []
    brow_stats: list[float] = []
    smile_stats: list[float] = []
    cheek_stats: list[float] = []

    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                continue
            frame_id += 1

            au_dict, _ = extractor.extract(frame)
            if not au_dict:
                print(f"[帧 {frame_id}] NO FACE")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            raw = {k: float(au_dict.get(k, 0.0)) for k in RAW_AUS}
            cal_once = _calibrate(caliper, raw)
            cal = {k: float(cal_once.get(k, 0.0)) for k in RAW_AUS}
            expr_once = _expr(caliper, cal)
            expr = {k: float(expr_once.get(k, 0.0)) for k in EXPR_KEYS}
            emo = {k: float(map_au_to_emotion(cal).get(k, 0.0)) for k in EMO_KEYS}

            raw04_stats.append(raw["AU04"])
            cal04_stats.append(cal["AU04"])
            brow_stats.append(expr["brow_furrow"])
            smile_stats.append(expr["smile_intensity"])
            cheek_stats.append(expr["cheek_raise"])

            raw_s = " ".join(f"{k}={_f2(raw[k])}" for k in PRINT_RAW)
            cal_s = " ".join(f"{k}={_f2(cal[k])}" for k in PRINT_CAL)
            emo_k, emo_v = _top1(emo)
            print(
                f"[帧 {frame_id}] raw {raw_s} | cal {cal_s} | "
                f"expr brow={_f2(expr['brow_furrow'])} smile={_f2(expr['smile_intensity'])} "
                f"cheek={_f2(expr['cheek_raise'])} emo top1={emo_k}@{emo_v:.2f}"
            )

            if csv_writer is not None:
                csv_writer.writerow(
                    [frame_id]
                    + [raw[k] for k in RAW_AUS]
                    + [cal[k] for k in RAW_AUS]
                    + [expr[k] for k in EXPR_KEYS]
                    + [emo[k] for k in EMO_KEYS]
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()

    if raw04_stats:
        print(_stats_line("AU04 raw", raw04_stats))
        print(_stats_line("AU04 cal", cal04_stats))
        print(_stats_line("brow_furrow", brow_stats))
        print(_stats_line("smile_intensity", smile_stats))
        print(_stats_line("cheek_raise", cheek_stats))
    else:
        print("No FACE frames collected; stats skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
