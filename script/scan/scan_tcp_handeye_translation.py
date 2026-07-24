#!/usr/bin/env python3
"""Scan TCP hand-eye translation offsets while keeping the rest of the chain fixed."""

from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
TCP_EVAL_PATH = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
DEFAULT_ESTIMATE = REPO_ROOT / "data/gripper_data/episode_20260617_0004/right/pose_data.csv"
DEFAULT_GT = REPO_ROOT / "data/ground_truth/trajectory_samples0617/trajectory_sync_rawpose_004.json"
DEFAULT_HANDEYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/scan_tcp_handeye_translation_0004"


def load_tcp_eval() -> Dict[str, object]:
    return runpy.run_path(str(TCP_EVAL_PATH), run_name="__scan_tcp_handeye_translation__")


def write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mm(value_m: float) -> float:
    return float(value_m) * 1000.0


def metrics_for_offset(
    tcp_eval: Dict[str, object],
    helpers: Dict[str, object],
    estimate: Path,
    ground_truth: Path,
    handeye_matrix: np.ndarray,
    estimate_frame: str,
    time_offset_s: float,
    t_max_diff_s: float,
) -> Dict[str, object]:
    gt_times, gt_pos_all, gt_rot_all = tcp_eval["load_robot_tcp_trajectory"](ground_truth, helpers)
    est_times, est_pos_all, est_rot_all = tcp_eval["load_vio_tcp_trajectory"](
        estimate,
        helpers,
        handeye_matrix,
        estimate_frame,
    )
    times, gt_pos, gt_rot, est_pos, est_rot = tcp_eval["associate_by_nearest_time"](
        gt_times,
        gt_pos_all,
        gt_rot_all,
        est_times,
        est_pos_all,
        est_rot_all,
        time_offset_s,
        t_max_diff_s,
    )
    se3, _, _, trans_errors, rot_errors = helpers["evaluate_alignment"](
        "se3", False, gt_pos, gt_rot, est_pos, est_rot, times, 1.0, 30
    )
    return {
        "matched_samples": int(times.size),
        "matched_duration_s": float(times[-1] - times[0]) if times.size > 1 else 0.0,
        "ape_translation_se3_rmse_mm": mm(se3.translation_metrics_m["rmse"]),
        "ape_translation_se3_mean_mm": mm(se3.translation_metrics_m["mean"]),
        "ape_translation_se3_median_mm": mm(se3.translation_metrics_m["median"]),
        "ape_translation_se3_max_mm": mm(se3.translation_metrics_m["max"]),
        "ape_rotation_se3_rmse_deg": float(se3.rotation_metrics_deg["rmse"]) if se3.rotation_metrics_deg else float("nan"),
        "ape_rotation_se3_mean_deg": float(se3.rotation_metrics_deg["mean"]) if se3.rotation_metrics_deg else float("nan"),
        "ape_rotation_se3_median_deg": float(se3.rotation_metrics_deg["median"]) if se3.rotation_metrics_deg else float("nan"),
        "ape_rotation_se3_max_deg": float(se3.rotation_metrics_deg["max"]) if se3.rotation_metrics_deg else float("nan"),
        "translation_error_p95_mm": mm(float(np.percentile(trans_errors, 95.0))),
        "rotation_error_p95_deg": float(np.percentile(rot_errors, 95.0)),
    }


def parse_offsets_mm(values: Sequence[str]) -> List[float]:
    out: List[float] = []
    seen = set()
    for raw in values:
        value = float(raw)
        for signed in (value, -value):
            key = round(signed, 9)
            if key not in seen:
                out.append(signed)
                seen.add(key)
    if 0.0 not in seen:
        out.append(0.0)
    return sorted(out)


def build_report(
    rows: List[Dict[str, object]],
    baseline: Dict[str, object],
    best: Dict[str, object],
    output_dir: Path,
    handeye_matrix: np.ndarray,
    estimate: Path,
    ground_truth: Path,
    estimate_frame: str,
    time_offset_s: float,
    t_max_diff_s: float,
) -> str:
    lines = [
        "# TCP Hand-Eye Translation Scan",
        "",
        "## Fixed Evaluation Setup",
        "",
        f"- Estimate: `{estimate}`",
        f"- Ground truth: `{ground_truth}`",
        f"- Estimate frame: `{estimate_frame}`",
        f"- Time association: `evo`",
        f"- Fixed `t_offset`: `{time_offset_s:.9f} s`",
        f"- Fixed `t_max_diff`: `{t_max_diff_s:.9f} s`",
        "",
        "## Baseline Hand-Eye Translation",
        "",
        f"- `tx = {handeye_matrix[0, 3] * 1000.0:.3f} mm`",
        f"- `ty = {handeye_matrix[1, 3] * 1000.0:.3f} mm`",
        f"- `tz = {handeye_matrix[2, 3] * 1000.0:.3f} mm`",
        "",
        "## Baseline Metrics",
        "",
        f"- APE translation RMSE: `{baseline['ape_translation_se3_rmse_mm']:.3f} mm`",
        f"- APE rotation RMSE: `{baseline['ape_rotation_se3_rmse_deg']:.3f} deg`",
        "",
        "## Best Scanned Translation Offset",
        "",
        f"- `dtx = {best['delta_tx_mm']:.3f} mm`",
        f"- `dty = {best['delta_ty_mm']:.3f} mm`",
        f"- `dtz = {best['delta_tz_mm']:.3f} mm`",
        f"- New APE translation RMSE: `{best['ape_translation_se3_rmse_mm']:.3f} mm`",
        f"- New APE rotation RMSE: `{best['ape_rotation_se3_rmse_deg']:.3f} deg`",
        f"- Translation improvement: `{baseline['ape_translation_se3_rmse_mm'] - best['ape_translation_se3_rmse_mm']:.3f} mm`",
        f"- Rotation change: `{best['ape_rotation_se3_rmse_deg'] - baseline['ape_rotation_se3_rmse_deg']:.3f} deg`",
        "",
        "## Top 10 Candidates",
        "",
        "| rank | dtx mm | dty mm | dtz mm | APE t RMSE mm | APE r RMSE deg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows[:10], start=1):
        lines.append(
            f"| {idx} | {row['delta_tx_mm']:.1f} | {row['delta_ty_mm']:.1f} | {row['delta_tz_mm']:.1f} | "
            f"{row['ape_translation_se3_rmse_mm']:.3f} | {row['ape_rotation_se3_rmse_deg']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- CSV: `{output_dir / 'scan_results.csv'}`",
            f"- JSON: `{output_dir / 'scan_results.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, default=DEFAULT_ESTIMATE)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--handeye-yaml", type=Path, default=DEFAULT_HANDEYE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--estimate-frame", choices=["imu", "vins_base_link", "camera"], default="vins_base_link")
    parser.add_argument("--time-offset-sec", type=float, default=-6.736420650482241)
    parser.add_argument("--t-max-diff-sec", type=float, default=0.01)
    parser.add_argument(
        "--translation-offsets-mm",
        nargs="+",
        default=["5", "10", "20"],
        help="Absolute offset magnitudes to scan on each axis, in millimeters. Zero is added automatically.",
    )
    args = parser.parse_args()

    if yaml is None:
        raise RuntimeError("PyYAML is required for hand-eye translation scan")

    estimate = args.estimate.expanduser().resolve()
    ground_truth = args.ground_truth.expanduser().resolve()
    handeye_yaml = args.handeye_yaml.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tcp_eval = load_tcp_eval()
    helpers = tcp_eval["load_helpers"]()
    handeye_matrix = tcp_eval["load_tcp_left_camera_transform"](handeye_yaml)
    offsets_mm = parse_offsets_mm(args.translation_offsets_mm)

    rows: List[Dict[str, object]] = []
    for dtx_mm in offsets_mm:
        for dty_mm in offsets_mm:
            for dtz_mm in offsets_mm:
                candidate = handeye_matrix.copy()
                candidate[:3, 3] += np.array([dtx_mm, dty_mm, dtz_mm], dtype=float) * 1e-3
                metrics = metrics_for_offset(
                    tcp_eval,
                    helpers,
                    estimate,
                    ground_truth,
                    candidate,
                    args.estimate_frame,
                    float(args.time_offset_sec),
                    float(args.t_max_diff_sec),
                )
                rows.append(
                    {
                        "delta_tx_mm": float(dtx_mm),
                        "delta_ty_mm": float(dty_mm),
                        "delta_tz_mm": float(dtz_mm),
                        **metrics,
                    }
                )

    rows.sort(key=lambda row: (row["ape_translation_se3_rmse_mm"], abs(row["ape_rotation_se3_rmse_deg"])))
    baseline = next(row for row in rows if row["delta_tx_mm"] == 0.0 and row["delta_ty_mm"] == 0.0 and row["delta_tz_mm"] == 0.0)
    best = rows[0]

    csv_path = output_dir / "scan_results.csv"
    json_path = output_dir / "scan_results.json"
    report_path = output_dir / "REPORT.md"
    write_rows_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        build_report(
            rows,
            baseline,
            best,
            output_dir,
            handeye_matrix,
            estimate,
            ground_truth,
            args.estimate_frame,
            float(args.time_offset_sec),
            float(args.t_max_diff_sec),
        ),
        encoding="utf-8",
    )

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {report_path}")
    print(
        "[BEST]",
        f"dtx={best['delta_tx_mm']:.1f}mm",
        f"dty={best['delta_ty_mm']:.1f}mm",
        f"dtz={best['delta_tz_mm']:.1f}mm",
        f"ape_t={best['ape_translation_se3_rmse_mm']:.3f}mm",
        f"ape_r={best['ape_rotation_se3_rmse_deg']:.3f}deg",
    )
    print(
        "[BASELINE]",
        f"ape_t={baseline['ape_translation_se3_rmse_mm']:.3f}mm",
        f"ape_r={baseline['ape_rotation_se3_rmse_deg']:.3f}deg",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
