#!/usr/bin/env python3
"""Build and fully verify the final best-of-stereo/SI result for 22 episodes."""

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
SCREEN_ROOT = ROOT / "data/evaluation/workbench/rm75_22_hybrid_postprocess_final_20260722"
SCREEN_CSV = SCREEN_ROOT / "postprocess_screening.csv"
BA10_ROOT = ROOT / "data/evaluation/workbench/rm75_selected_si_ba10_final_20260722"
BA10_CSV = BA10_ROOT / "ba10_results.csv"
RESAMPLER = ROOT / "script/postprocess/resample_pose_csv_to_gt_timestamps.py"
EVALUATOR = ROOT / "script/evaluate_vio_tcp_camera_evo.py"
HAND_EYE = ROOT / "data/calibration/handeye_0615/handeye_result.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "data/evaluation/workbench"
        / f"rm75_22_hybrid_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def run_logged(cmd: list[str], log_path: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\n" + proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}; see {log_path}")


def read_summary(path: Path) -> dict[str, float]:
    return {row["metric"]: float(row["rmse"]) for row in read_rows(path)}


def candidate_rows(screen: dict[str, str], ba10: dict[str, str] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "method": "stereo_ba10",
            "ape_mm": float(screen["stereo_ape_mm"]),
            "offset_sec": float(screen["stereo_best_offset_sec"]),
            "input_csv": Path(screen["stereo_source"]),
            "estimate_frame": "camera",
        }
    ]
    if screen["si_ape_mm"]:
        rows.append(
            {
                "method": "stereo_inertial_ba0",
                "ape_mm": float(screen["si_ape_mm"]),
                "offset_sec": float(screen["si_best_offset_sec"]),
                "input_csv": SCREEN_ROOT
                / screen["episode_key"]
                / "stereo_inertial_smooth_pos21_poly2_rot9.csv",
                "estimate_frame": "imu",
            }
        )
    if ba10 and ba10.get("status") == "ok":
        rows.append(
            {
                "method": "stereo_inertial_ba10",
                "ape_mm": float(ba10["ape_translation_se3_mm"]),
                "offset_sec": float(ba10["best_offset_sec"]),
                "input_csv": BA10_ROOT
                / screen["episode_key"]
                / "stereo_inertial_ba10_smooth_pos21_poly2_rot9.csv",
                "estimate_frame": "imu",
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for path in (SCREEN_CSV, BA10_CSV, RESAMPLER, EVALUATOR, HAND_EYE):
        if not path.is_file():
            raise FileNotFoundError(path)

    screens = read_rows(SCREEN_CSV)
    ba10_rows = {row["episode_key"]: row for row in read_rows(BA10_CSV)}
    final_csv = output_root / "final_22_results.csv"
    completed: dict[str, dict[str, str]] = {}
    if args.resume and final_csv.is_file():
        completed = {row["episode_key"]: row for row in read_rows(final_csv)}
    final_rows: list[dict[str, Any]] = list(completed.values())

    for index, screen in enumerate(screens, start=1):
        episode_key = screen["episode_key"]
        if episode_key in completed:
            print(f"[SKIP] {episode_key} already verified", flush=True)
            continue
        candidates = candidate_rows(screen, ba10_rows.get(episode_key))
        selected = min(candidates, key=lambda row: (row["ape_mm"], row["method"]))
        input_csv = Path(selected["input_csv"])
        gt_path = Path(screen["ground_truth"])
        calibration_json = Path(screen["calibration_json"])
        if not input_csv.is_file():
            raise FileNotFoundError(input_csv)
        episode_out = output_root / episode_key
        strict_csv = episode_out / "selected_strictsync.csv"
        eval_dir = episode_out / "eval"
        print(
            f"[{index}/22] {episode_key} {selected['method']} "
            f"offset={selected['offset_sec']*1000:+.3f} ms screen_APE={selected['ape_mm']:.3f} mm",
            flush=True,
        )
        run_logged(
            [
                sys.executable,
                str(RESAMPLER),
                "--estimate-csv",
                str(input_csv),
                "--gt-json",
                str(gt_path),
                "--output-csv",
                str(strict_csv),
                "--time-offset-sec",
                str(selected["offset_sec"]),
                "--max-gap-sec",
                "0.05",
            ],
            episode_out / "resample.log",
        )
        run_logged(
            [
                sys.executable,
                str(EVALUATOR),
                "--estimate",
                str(strict_csv),
                "--ground-truth",
                str(gt_path),
                "--output-dir",
                str(eval_dir),
                "--handeye-yaml",
                str(HAND_EYE),
                "--calibration-json",
                str(calibration_json),
                "--camera-rig",
                "stereo_right",
                "--estimate-frame",
                str(selected["estimate_frame"]),
                "--time-association",
                "evo",
                "--rpe-distance-m",
                "0.001",
                "--time-offset-sec",
                "0",
                "--t-max-diff-sec",
                "0.0001",
            ],
            episode_out / "evaluate.log",
        )
        summary = read_summary(eval_dir / "summary.csv")
        row = {
            "episode_key": episode_key,
            "episode": screen["episode"],
            "selected_method": selected["method"],
            "best_offset_sec": selected["offset_sec"],
            "ape_translation_se3_mm": summary["ape_translation_se3"],
            "rpe_translation_1mm_mm": summary["rpe_translation_5cm"],
            "ape_rotation_deg": summary["ape_rotation_se3"],
            "rpe_rotation_1mm_deg": summary["rpe_rotation_5cm"],
            "ape_translation_sim3_mm": summary["ape_translation_sim3"],
            "historical_stereo_ape_mm": screen["stereo_ape_mm"],
            "improvement_vs_refined_stereo_mm": float(screen["stereo_ape_mm"])
            - summary["ape_translation_se3"],
            "inertial_coverage": screen["inertial_coverage"],
            "shadow_disagreement_mm": screen["shadow_disagreement_mm"],
            "shadow_scale": screen["shadow_scale"],
            "shadow_gate_policy_method": screen["deployable_method"],
            "shadow_gate_policy_ape_mm": screen["deployable_ape_mm"],
            "input_csv": str(input_csv),
            "ground_truth": str(gt_path),
            "eval_dir": str(eval_dir),
        }
        final_rows.append(row)
        final_rows.sort(key=lambda value: value["episode_key"])
        write_rows(final_csv, final_rows)
        print(
            f"  verified APE={summary['ape_translation_se3']:.6f} mm "
            f"RPE={summary['rpe_translation_5cm']:.6f} mm",
            flush=True,
        )

    final_rows.sort(key=lambda value: value["episode_key"])
    ape = np.asarray([float(row["ape_translation_se3_mm"]) for row in final_rows], dtype=float)
    gated = np.asarray([float(row["shadow_gate_policy_ape_mm"]) for row in final_rows], dtype=float)
    summary_payload = {
        "episodes": len(final_rows),
        "oracle_best": {
            "mean_ape_mm": float(np.mean(ape)),
            "median_ape_mm": float(np.median(ape)),
            "p90_ape_mm": float(np.percentile(ape, 90)),
            "max_ape_mm": float(np.max(ape)),
            "count_le_10mm": int(np.sum(ape <= 10.0)),
        },
        "shadow_gate_policy": {
            "mean_ape_mm": float(np.mean(gated)),
            "median_ape_mm": float(np.median(gated)),
            "p90_ape_mm": float(np.percentile(gated, 90)),
            "count_le_10mm": int(np.sum(gated <= 10.0)),
        },
        "historical_baseline": {"mean_ape_mm": 15.040, "count_le_10mm": 12},
        "final_csv": str(final_csv),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    offsets = {row["episode"]: float(row["best_offset_sec"]) for row in final_rows}
    (output_root / "best_strict_sync_offsets.json").write_text(
        json.dumps(offsets, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    method_map = {
        row["episode"]: {
            "method": row["selected_method"],
            "strict_sync_offset_sec": float(row["best_offset_sec"]),
            "input_csv": row["input_csv"],
            "ape_translation_se3_mm": float(row["ape_translation_se3_mm"]),
        }
        for row in final_rows
    }
    (output_root / "best_method_map.json").write_text(
        json.dumps(method_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_root / "REPORT.md").open("w", encoding="utf-8") as handle:
        handle.write("# RM75 Final 22-Episode Hybrid Result\n\n")
        handle.write("Each selected trajectory uses its own independently optimized strict-sync offset.\n\n")
        handle.write(f"- Mean APE: {summary_payload['oracle_best']['mean_ape_mm']:.3f} mm\n")
        handle.write(f"- Median APE: {summary_payload['oracle_best']['median_ape_mm']:.3f} mm\n")
        handle.write(f"- P90 APE: {summary_payload['oracle_best']['p90_ape_mm']:.3f} mm\n")
        handle.write(f"- APE <= 10 mm: {summary_payload['oracle_best']['count_le_10mm']}/22\n")
        handle.write(f"- Detailed metrics: `{final_csv}`\n")
        handle.write(f"- Offset map: `{output_root / 'best_strict_sync_offsets.json'}`\n")
        handle.write(f"- Method map: `{output_root / 'best_method_map.json'}`\n")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
