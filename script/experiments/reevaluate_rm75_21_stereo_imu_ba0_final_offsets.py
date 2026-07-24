#!/usr/bin/env python3
"""Formally score selected Stereo-IMU BA trajectories at final offsets.

The two episodes without a usable full inertial trajectory remain explicitly
unsupported instead of reporting an APE over their short IMU-covered prefix.
"""

from __future__ import annotations

import argparse
import csv
import json
import runpy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = ROOT / "data/evaluation/workbench/rm75_22_final_robust_20260722/final_merged_20260722_160852"
FINALIZER = ROOT / "script/experiments/finalize_rm75_22_final_robust_results.py"
OPTIMIZER = ROOT / "script/experiments/run_rm75_22_final_robust_optimizer.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--ba-iters", type=int, choices=(0, 10), default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else FINAL_ROOT / f"stereo_imu_ba{args.ba_iters}_final_offset_reevaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    final_rows = read_rows(FINAL_ROOT / "final_21_results.csv")
    offsets = json.loads((FINAL_ROOT / "best_strict_sync_offsets.json").read_text(encoding="utf-8"))

    merge = runpy.run_path(str(FINALIZER), run_name="__stereo_imu_ba0_reevaluation__")
    opt = runpy.run_path(str(OPTIMIZER), run_name="__stereo_imu_ba0_optimizer__")
    fast = runpy.run_path(str(opt["FAST_SCAN_SCRIPT"]), run_name="__stereo_imu_ba0_fast__")
    eval_api = fast["load_eval_api"]()
    records, row_lookup = merge["screen_candidate_records"](opt)
    source_screen = f"ba{args.ba_iters}_fast_screen"
    ba_records = {
        record.episode_key: record
        for record in records
        if record.source_screen == source_screen and record.candidate_name == "stereo_imu"
    }
    smooth_args = SimpleNamespace(position_window=21, position_poly=2, rotation_window=9)
    results: list[dict[str, Any]] = []
    for row in final_rows:
        episode_key = row["episode_key"]
        record = ba_records.get(episode_key)
        if record is None:
            results.append(
                {
                    "episode_key": episode_key,
                    "source_episode": row["source_episode"],
                    "strict_sync_offset_sec": float(offsets[episode_key]),
                    "strict_sync_offset_ms": float(offsets[episode_key]) * 1000.0,
                    "status": "not_run_or_unsupported",
                    "note": f"no usable full Stereo-IMU BA{args.ba_iters} trajectory; do not score partial coverage",
                }
            )
            continue
        record = replace(record, strict_sync_offset_sec=float(offsets[episode_key]))
        print(f"[VERIFY] {episode_key} offset={record.strict_sync_offset_sec * 1000.0:+.3f} ms", flush=True)
        try:
            verified = merge["verify_candidate"](
                rec=record,
                candidate_root=output_dir / "candidate_verification",
                row_lookup=row_lookup,
                opt=opt,
                fast=fast,
                eval_api=eval_api,
                smooth_args=smooth_args,
                resume=args.resume,
            )
            results.append(
                {
                    "episode_key": episode_key,
                    "source_episode": row["source_episode"],
                    "strict_sync_offset_sec": record.strict_sync_offset_sec,
                    "strict_sync_offset_ms": record.strict_sync_offset_sec * 1000.0,
                    "status": "ok",
                    "inertial_coverage": record.inertial_coverage,
                    "ape_translation_se3_mm": verified["formal_ape_translation_se3_mm"],
                    "rpe_translation_eval_mm": verified["formal_rpe_translation_eval_mm"],
                    "ape_rotation_se3_deg": verified["formal_ape_rotation_se3_deg"],
                    "rpe_rotation_eval_deg": verified["formal_rpe_rotation_eval_deg"],
                    "inertial_tum": record.input_reference,
                    "selected_strictsync_csv": verified["selected_strictsync_csv"],
                    "eval_dir": verified["eval_dir"],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "episode_key": episode_key,
                    "source_episode": row["source_episode"],
                    "strict_sync_offset_sec": record.strict_sync_offset_sec,
                    "strict_sync_offset_ms": record.strict_sync_offset_sec * 1000.0,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    results.sort(key=lambda item: item["episode_key"])
    result_csv = output_dir / f"final_21_stereo_imu_ba{args.ba_iters}_new_offset_results.csv"
    write_rows(result_csv, results)
    ok = [row for row in results if row["status"] == "ok"]
    values = [float(row["ape_translation_se3_mm"]) for row in ok]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": f"Stereo-IMU BA{args.ba_iters}, raw inertial trajectory, 21/2/9 smoothing",
        "offset_map": str(FINAL_ROOT / "best_strict_sync_offsets.json"),
        "retained_episodes": len(results),
        "episodes_scored": len(ok),
        "episodes_not_run_or_unsupported": [row["episode_key"] for row in results if row["status"] == "not_run_or_unsupported"],
        "mean_ape_mm": float(np.mean(values)),
        "median_ape_mm": float(np.median(values)),
        "p90_ape_mm": float(np.percentile(values, 90)),
        "max_ape_mm": float(np.max(values)),
        "count_le_10mm": int(sum(value <= 10.0 for value in values)),
        "results_csv": str(result_csv),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        f"# Stereo-IMU BA{args.ba_iters} Re-evaluation With Final Strict-Sync Offsets",
        "",
        "Only native inertial trajectories are scored. Episodes with insufficient inertial coverage are explicitly unsupported.",
        "",
        f"- Scored: `{summary['episodes_scored']}/{summary['retained_episodes']}`",
        f"- Mean APE: `{summary['mean_ape_mm']:.6f} mm`",
        f"- Median APE: `{summary['median_ape_mm']:.6f} mm`",
        f"- P90 APE: `{summary['p90_ape_mm']:.6f} mm`",
        f"- Max APE: `{summary['max_ape_mm']:.6f} mm`",
        f"- APE <= 10 mm: `{summary['count_le_10mm']}/{summary['episodes_scored']}`",
        "",
        "| episode | coverage | final offset ms | APE mm | status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        coverage = row.get("inertial_coverage", "")
        ape = row.get("ape_translation_se3_mm", "")
        report.append(
            f"| {row['episode_key']} | {coverage if coverage == '' else f'{float(coverage):.3f}'} | "
            f"{float(row['strict_sync_offset_ms']):+.3f} | {ape if ape == '' else f'{float(ape):.6f}'} | {row['status']} |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
