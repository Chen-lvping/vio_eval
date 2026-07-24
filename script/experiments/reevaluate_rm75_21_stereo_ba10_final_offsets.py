#!/usr/bin/env python3
"""Formally re-evaluate retained pure Stereo BA10 trajectories using final offsets.

This intentionally reuses the archived colleague Stereo BA10 trajectories.  It
does not rerun ORB-SLAM3 and applies the per-episode offset map selected for
the final 21-episode report before invoking the standard TCP evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = ROOT / "data/evaluation/workbench/rm75_22_final_robust_20260722/final_merged_20260722_160852"
STEREO_ROOT = ROOT / "data/evaluation/workbench/rm75_best_all_gt_20260716"
RESAMPLER = ROOT / "script/postprocess/resample_pose_csv_to_gt_timestamps.py"
EVALUATOR = ROOT / "script/evaluate_vio_tcp_camera_evo.py"
HAND_EYE = ROOT / "data/calibration/handeye_0615/handeye_result.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FINAL_ROOT / "stereo_ba10_final_offset_reevaluation",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_summary(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    metrics = {row["metric"]: float(row["rmse"]) for row in rows if row.get("metric") and row.get("rmse")}
    required = {
        "ape_translation_se3",
        "rpe_translation_5cm",
        "ape_rotation_se3",
        "rpe_rotation_5cm",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise RuntimeError(f"missing metrics in {path}: {missing}")
    return metrics


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def source_pose(source_episode: str) -> Path:
    candidates = sorted(STEREO_ROOT.glob(f"orbslam3_rm75_best_batch_*/{source_episode}/pose_smooth.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one archived pure-stereo pose for {source_episode}, found {candidates}")
    return candidates[0].resolve()


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, int(np.ceil(percentile * len(ordered))) - 1)
    return float(ordered[index])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_csv = FINAL_ROOT / "final_21_results.csv"
    offset_json = FINAL_ROOT / "best_strict_sync_offsets.json"
    if not final_csv.is_file() or not offset_json.is_file():
        raise FileNotFoundError("final 21 result table or strict-sync offset map is missing")

    rows = read_csv(final_csv)
    offsets = {key: float(value) for key, value in json.loads(offset_json.read_text(encoding="utf-8")).items()}
    results: list[dict[str, Any]] = []
    for row in rows:
        episode_key = row["episode_key"]
        if episode_key not in offsets:
            raise KeyError(f"no final strict-sync offset for {episode_key}")
        episode_dir = output_dir / episode_key
        strict_csv = episode_dir / "selected_strictsync.csv"
        eval_dir = episode_dir / "eval"
        summary_csv = eval_dir / "summary.csv"
        source = source_pose(row["source_episode"])
        if not (args.resume and summary_csv.is_file()):
            run(
                [
                    sys.executable, str(RESAMPLER), "--estimate-csv", str(source),
                    "--gt-json", row["ground_truth"], "--output-csv", str(strict_csv),
                    "--time-offset-sec", str(offsets[episode_key]), "--max-gap-sec", "0.05",
                ],
                episode_dir / "resample.log",
            )
            run(
                [
                    sys.executable, str(EVALUATOR), "--estimate", str(strict_csv),
                    "--ground-truth", row["ground_truth"], "--output-dir", str(eval_dir),
                    "--handeye-yaml", str(HAND_EYE), "--calibration-json", row["calibration_json"],
                    "--camera-rig", "stereo_right", "--estimate-frame", "camera",
                    "--time-association", "evo", "--rpe-distance-m", "0.001",
                    "--time-offset-sec", "0", "--t-max-diff-sec", "0.0001",
                ],
                episode_dir / "evaluate.log",
            )
        summary = read_summary(summary_csv)
        results.append(
            {
                "episode_key": episode_key,
                "source_episode": row["source_episode"],
                "strict_sync_offset_sec": offsets[episode_key],
                "strict_sync_offset_ms": offsets[episode_key] * 1000.0,
                "ape_translation_se3_mm": summary["ape_translation_se3"],
                "rpe_translation_eval_mm": summary["rpe_translation_5cm"],
                "ape_rotation_se3_deg": summary["ape_rotation_se3"],
                "rpe_rotation_eval_deg": summary["rpe_rotation_5cm"],
                "matched_trajectory": str(strict_csv),
                "source_pose_smooth": str(source),
            }
        )

    result_csv = output_dir / "final_21_stereo_ba10_new_offset_results.csv"
    write_csv(result_csv, results)
    values = [float(row["ape_translation_se3_mm"]) for row in results]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "episodes": len(results),
        "method": "pure Stereo BA10, archived pose_smooth.csv",
        "offset_map": str(offset_json),
        "mean_ape_mm": float(np.mean(values)),
        "median_ape_mm": float(np.median(values)),
        "p90_ape_mm_nearest_rank": percentile_nearest_rank(values, 0.90),
        "max_ape_mm": float(max(values)),
        "count_le_10mm": int(sum(value <= 10.0 for value in values)),
        "results_csv": str(result_csv),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Pure Stereo BA10 Re-evaluation With Final Strict-Sync Offsets",
        "",
        "The archived pure-stereo trajectories were not rerun. Each was resampled with the final 21-episode offset map and evaluated by the standard TCP evaluator.",
        "",
        f"- Mean APE: `{summary['mean_ape_mm']:.6f} mm`",
        f"- Median APE: `{summary['median_ape_mm']:.6f} mm`",
        f"- P90 APE (nearest rank): `{summary['p90_ape_mm_nearest_rank']:.6f} mm`",
        f"- Max APE: `{summary['max_ape_mm']:.6f} mm`",
        f"- APE <= 10 mm: `{summary['count_le_10mm']}/{summary['episodes']}`",
        "",
        "| episode | final offset ms | APE mm | RPE mm |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in results:
        report.append(
            f"| {row['episode_key']} | {row['strict_sync_offset_ms']:+.3f} | "
            f"{row['ape_translation_se3_mm']:.6f} | {row['rpe_translation_eval_mm']:.6f} |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
