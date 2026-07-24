#!/usr/bin/env python3
"""
Fit one SE(3) world alignment on a source VIO/robot TCP dataset and reuse it
unchanged on a target dataset.

This is intentionally different from the usual evo ``--align`` evaluation:
the target trajectory is not re-aligned. The source dataset estimates a rigid
transform from VIO world to robot base, then the target VIO TCP trajectory is
evaluated after applying that same transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TCP_EVAL_SCRIPT = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"

DEFAULT_SOURCE_ESTIMATE = Path("/home/chenlvping/6_data_use/0614 _test/episode_20260614_0239/right/pose_data.csv")
DEFAULT_SOURCE_GT = REPO_ROOT / "data/ground_truth/trajectory_samples0614/trajectory_001.json"
DEFAULT_TARGET_ESTIMATE = REPO_ROOT / "data/gripper_data/episode_20260617_0004/right/pose_data.csv"
DEFAULT_TARGET_GT = REPO_ROOT / "data/ground_truth/trajectory_samples0617/trajectory_sync_rawpose_004.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_0617_0004_fixed_0614_se3"
DEFAULT_HAND_EYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"


def load_tcp_eval() -> Dict[str, object]:
    return runpy.run_path(str(TCP_EVAL_SCRIPT), run_name="__fixed_se3_tcp_eval__")


def load_matched_tcp(
    tcp_eval: Dict[str, object],
    helpers: Dict[str, object],
    estimate: Path,
    ground_truth: Path,
    handeye_yaml: Optional[Path],
    estimate_frame: str,
    max_gap_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_tcp_left_camera = tcp_eval["load_tcp_left_camera_transform"](handeye_yaml)
    gt_times, gt_pos_all, gt_rot_all = tcp_eval["load_robot_tcp_trajectory"](ground_truth, helpers)
    est_times, est_pos_all, est_rot_all = tcp_eval["load_vio_tcp_trajectory"](
        estimate,
        helpers,
        t_tcp_left_camera,
        estimate_frame,
    )
    return tcp_eval["match_ground_truth"](
        gt_times,
        gt_pos_all,
        gt_rot_all,
        est_times,
        est_pos_all,
        est_rot_all,
        helpers,
        max_gap_s,
    )


def transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def apply_se3(
    positions: np.ndarray,
    rotations: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    aligned_pos = (rotation @ positions.T).T + translation
    aligned_rot = np.asarray([rotation @ r for r in rotations], dtype=float)
    return aligned_pos, aligned_rot


def compute_fixed_metrics(
    helpers: Dict[str, object],
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    aligned_pos: np.ndarray,
    aligned_rot: np.ndarray,
    times: np.ndarray,
    rpe_delta_s: float,
    rpe_delta_samples: int,
) -> Dict[str, object]:
    trans_errors = np.linalg.norm(aligned_pos - gt_pos, axis=1)
    rot_errors = np.asarray(
        [helpers["rotation_angle_deg"](gt_rot[i].T @ aligned_rot[i]) for i in range(gt_rot.shape[0])],
        dtype=float,
    )
    rpe_t, rpe_r = helpers["compute_rpe"](
        gt_pos,
        gt_rot,
        aligned_pos,
        aligned_rot,
        times,
        rpe_delta_s,
        rpe_delta_samples,
    )
    gt_path = helpers["path_length"](gt_pos)
    est_path = helpers["path_length"](aligned_pos)
    return {
        "translation_m": helpers["summary_stats"](trans_errors),
        "rotation_deg": helpers["summary_stats"](rot_errors),
        "rpe_translation_m": rpe_t,
        "rpe_rotation_deg": rpe_r,
        "drift": {
            "gt_path_length_m": float(gt_path),
            "aligned_est_path_length_m": float(est_path),
            "path_length_error_m": float(est_path - gt_path),
            "path_length_error_pct": 100.0 * float(est_path - gt_path) / gt_path if gt_path > 1e-12 else float("nan"),
            "final_position_error_m": float(trans_errors[-1]),
            "final_drift_pct_of_path": 100.0 * float(trans_errors[-1]) / gt_path if gt_path > 1e-12 else float("nan"),
        },
        "translation_errors_m": trans_errors,
        "rotation_errors_deg": rot_errors,
    }


def compact_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}


def fmt_float(value: object, scale: float = 1.0) -> str:
    try:
        x = float(value) * scale
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(x):
        return "nan"
    return f"{x:.6f}"


def write_summary(path: Path, source_metrics: Dict[str, object], target_metrics: Dict[str, object]) -> None:
    rows = [
        ("source_fit_ape_translation", source_metrics.get("translation_m"), 1000.0, "mm"),
        ("source_fit_ape_rotation", source_metrics.get("rotation_deg"), 1.0, "deg"),
        ("target_fixed_se3_ape_translation", target_metrics.get("translation_m"), 1000.0, "mm"),
        ("target_fixed_se3_ape_rotation", target_metrics.get("rotation_deg"), 1.0, "deg"),
        ("target_fixed_se3_rpe_translation", target_metrics.get("rpe_translation_m"), 1000.0, "mm"),
        ("target_fixed_se3_rpe_rotation", target_metrics.get("rpe_rotation_deg"), 1.0, "deg"),
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("metric,rmse,mean,median,p95,max,std,unit\n")
        for name, stats, scale, unit in rows:
            stats = stats or {}
            handle.write(
                f"{name},{fmt_float(stats.get('rmse'), scale)},{fmt_float(stats.get('mean'), scale)},"
                f"{fmt_float(stats.get('median'), scale)},{fmt_float(stats.get('p95'), scale)},"
                f"{fmt_float(stats.get('max'), scale)},{fmt_float(stats.get('std'), scale)},{unit}\n"
            )


def write_matched_csv(
    path: Path,
    times: np.ndarray,
    gt_pos: np.ndarray,
    est_pos: np.ndarray,
    aligned_pos: np.ndarray,
    trans_errors: np.ndarray,
    rot_errors: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "gt_x",
                "gt_y",
                "gt_z",
                "estimate_raw_x",
                "estimate_raw_y",
                "estimate_raw_z",
                "estimate_fixed_se3_x",
                "estimate_fixed_se3_y",
                "estimate_fixed_se3_z",
                "fixed_se3_trans_error_m",
                "fixed_se3_rot_error_deg",
            ],
        )
        writer.writeheader()
        for i, t in enumerate(times):
            writer.writerow(
                {
                    "timestamp": f"{float(t):.9f}",
                    "gt_x": f"{gt_pos[i, 0]:.9f}",
                    "gt_y": f"{gt_pos[i, 1]:.9f}",
                    "gt_z": f"{gt_pos[i, 2]:.9f}",
                    "estimate_raw_x": f"{est_pos[i, 0]:.9f}",
                    "estimate_raw_y": f"{est_pos[i, 1]:.9f}",
                    "estimate_raw_z": f"{est_pos[i, 2]:.9f}",
                    "estimate_fixed_se3_x": f"{aligned_pos[i, 0]:.9f}",
                    "estimate_fixed_se3_y": f"{aligned_pos[i, 1]:.9f}",
                    "estimate_fixed_se3_z": f"{aligned_pos[i, 2]:.9f}",
                    "fixed_se3_trans_error_m": f"{trans_errors[i]:.9f}",
                    "fixed_se3_rot_error_deg": f"{rot_errors[i]:.9f}",
                }
            )


def write_report(path: Path, payload: Dict[str, object], matrix_text: str) -> None:
    source = payload["source_fit_metrics"]
    target = payload["target_fixed_se3_metrics"]
    target_drift = target["drift"]
    lines = [
        "# Fixed 0614 SE(3) TCP Evaluation",
        "",
        "## Inputs",
        "",
        f"- Source estimate: `{payload['inputs']['source_estimate']}`",
        f"- Source ground truth: `{payload['inputs']['source_ground_truth']}`",
        f"- Source estimate frame: `{payload['settings']['source_estimate_frame']}`",
        f"- Target estimate: `{payload['inputs']['target_estimate']}`",
        f"- Target ground truth: `{payload['inputs']['target_ground_truth']}`",
        f"- Target estimate frame: `{payload['settings']['target_estimate_frame']}`",
        "",
        "## Fixed Transform",
        "",
        "The matrix below maps source VIO world coordinates into robot base coordinates and is reused unchanged on the target data:",
        "",
        "```text",
        "T_robot_base_from_vio_world =",
        matrix_text,
        "```",
        "",
        "## Results",
        "",
        "| split | metric | RMSE | mean | p95 | max | unit |",
        "|---|---|---:|---:|---:|---:|---|",
        "| source fit | APE translation | "
        f"{source['translation_m']['rmse'] * 1000.0:.3f} | {source['translation_m']['mean'] * 1000.0:.3f} | "
        f"{source['translation_m']['p95'] * 1000.0:.3f} | {source['translation_m']['max'] * 1000.0:.3f} | mm |",
        "| source fit | APE rotation | "
        f"{source['rotation_deg']['rmse']:.3f} | {source['rotation_deg']['mean']:.3f} | "
        f"{source['rotation_deg']['p95']:.3f} | {source['rotation_deg']['max']:.3f} | deg |",
        "| target fixed SE(3) | APE translation | "
        f"{target['translation_m']['rmse'] * 1000.0:.3f} | {target['translation_m']['mean'] * 1000.0:.3f} | "
        f"{target['translation_m']['p95'] * 1000.0:.3f} | {target['translation_m']['max'] * 1000.0:.3f} | mm |",
        "| target fixed SE(3) | APE rotation | "
        f"{target['rotation_deg']['rmse']:.3f} | {target['rotation_deg']['mean']:.3f} | "
        f"{target['rotation_deg']['p95']:.3f} | {target['rotation_deg']['max']:.3f} | deg |",
        "| target fixed SE(3) | RPE translation | "
        f"{target['rpe_translation_m']['rmse'] * 1000.0:.3f} | {target['rpe_translation_m']['mean'] * 1000.0:.3f} | "
        f"{target['rpe_translation_m']['p95'] * 1000.0:.3f} | {target['rpe_translation_m']['max'] * 1000.0:.3f} | mm |",
        "| target fixed SE(3) | RPE rotation | "
        f"{target['rpe_rotation_deg']['rmse']:.3f} | {target['rpe_rotation_deg']['mean']:.3f} | "
        f"{target['rpe_rotation_deg']['p95']:.3f} | {target['rpe_rotation_deg']['max']:.3f} | deg |",
        "",
        "## Coverage",
        "",
        f"- Source matched samples: {payload['coverage']['source_matched_samples']}",
        f"- Target matched samples: {payload['coverage']['target_matched_samples']}",
        f"- Target matched duration: {payload['coverage']['target_matched_duration_s']:.3f} s",
        f"- Target GT path length: {target_drift['gt_path_length_m']:.6f} m",
        f"- Target aligned estimate path length: {target_drift['aligned_est_path_length_m']:.6f} m",
        f"- Target final drift: {target_drift['final_position_error_m'] * 1000.0:.3f} mm",
        "",
        "## Notes",
        "",
        "- Target metrics use the fixed source SE(3); no target-side global alignment or scale correction is fitted.",
        "- RPE uses the configured time/sample interval, so it mainly checks local motion consistency after the fixed frame transform.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-estimate", type=Path, default=DEFAULT_SOURCE_ESTIMATE)
    parser.add_argument("--source-ground-truth", type=Path, default=DEFAULT_SOURCE_GT)
    parser.add_argument("--source-estimate-frame", choices=["imu", "vins_base_link", "camera"], default="imu")
    parser.add_argument("--target-estimate", type=Path, default=DEFAULT_TARGET_ESTIMATE)
    parser.add_argument("--target-ground-truth", type=Path, default=DEFAULT_TARGET_GT)
    parser.add_argument("--target-estimate-frame", choices=["imu", "vins_base_link", "camera"], default="imu")
    parser.add_argument("--handeye-yaml", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-time-gap-ms", type=float, default=80.0)
    parser.add_argument("--rpe-delta-s", type=float, default=1.0)
    parser.add_argument("--rpe-delta-samples", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_estimate = args.source_estimate.expanduser().resolve()
    source_gt = args.source_ground_truth.expanduser().resolve()
    target_estimate = args.target_estimate.expanduser().resolve()
    target_gt = args.target_ground_truth.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    handeye_yaml = args.handeye_yaml.expanduser().resolve() if args.handeye_yaml else None

    tcp_eval = load_tcp_eval()
    helpers = tcp_eval["load_helpers"]()
    max_gap_s = args.max_time_gap_ms * 1e-3

    source_times, source_gt_pos, source_gt_rot, source_est_pos, source_est_rot = load_matched_tcp(
        tcp_eval,
        helpers,
        source_estimate,
        source_gt,
        handeye_yaml,
        args.source_estimate_frame,
        max_gap_s,
    )
    scale, rotation, translation = helpers["align_umeyama"](source_est_pos, source_gt_pos, with_scale=False)
    if abs(float(scale) - 1.0) > 1e-9:
        raise RuntimeError(f"unexpected SE(3) scale {scale}")
    matrix = transform_matrix(rotation, translation)
    source_aligned_pos, source_aligned_rot = apply_se3(source_est_pos, source_est_rot, rotation, translation)
    source_metrics = compute_fixed_metrics(
        helpers,
        source_gt_pos,
        source_gt_rot,
        source_aligned_pos,
        source_aligned_rot,
        source_times,
        args.rpe_delta_s,
        args.rpe_delta_samples,
    )

    target_times, target_gt_pos, target_gt_rot, target_est_pos, target_est_rot = load_matched_tcp(
        tcp_eval,
        helpers,
        target_estimate,
        target_gt,
        handeye_yaml,
        args.target_estimate_frame,
        max_gap_s,
    )
    target_aligned_pos, target_aligned_rot = apply_se3(target_est_pos, target_est_rot, rotation, translation)
    target_metrics = compute_fixed_metrics(
        helpers,
        target_gt_pos,
        target_gt_rot,
        target_aligned_pos,
        target_aligned_rot,
        target_times,
        args.rpe_delta_s,
        args.rpe_delta_samples,
    )

    tcp_eval["write_tum"](output_dir / "gt_tcp.tum", target_times, target_gt_pos, target_gt_rot, helpers)
    tcp_eval["write_tum"](output_dir / "vio_tcp_raw.tum", target_times, target_est_pos, target_est_rot, helpers)
    tcp_eval["write_tum"](output_dir / "vio_tcp_fixed_0614_se3.tum", target_times, target_aligned_pos, target_aligned_rot, helpers)
    write_matched_csv(
        output_dir / "matched_samples.csv",
        target_times,
        target_gt_pos,
        target_est_pos,
        target_aligned_pos,
        target_metrics["translation_errors_m"],
        target_metrics["rotation_errors_deg"],
    )

    matrix_text = tcp_eval["matrix_text"](matrix)
    payload = {
        "inputs": {
            "source_estimate": str(source_estimate),
            "source_ground_truth": str(source_gt),
            "target_estimate": str(target_estimate),
            "target_ground_truth": str(target_gt),
            "handeye_yaml": str(handeye_yaml) if handeye_yaml else None,
        },
        "settings": {
            "source_estimate_frame": args.source_estimate_frame,
            "target_estimate_frame": args.target_estimate_frame,
            "max_time_gap_ms": args.max_time_gap_ms,
            "rpe_delta_s": args.rpe_delta_s,
            "rpe_delta_samples": args.rpe_delta_samples,
        },
        "coverage": {
            "source_matched_samples": int(source_times.size),
            "source_matched_duration_s": float(source_times[-1] - source_times[0]) if source_times.size > 1 else 0.0,
            "target_matched_samples": int(target_times.size),
            "target_matched_duration_s": float(target_times[-1] - target_times[0]) if target_times.size > 1 else 0.0,
            "target_first_timestamp": float(target_times[0]),
            "target_last_timestamp": float(target_times[-1]),
        },
        "fixed_se3": {
            "description": "T_robot_base_from_vio_world fitted on source and applied unchanged to target",
            "matrix": matrix.tolist(),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "scale": 1.0,
        },
        "source_fit_metrics": compact_metrics(source_metrics),
        "target_fixed_se3_metrics": compact_metrics(target_metrics),
        "outputs": {
            "gt_tum": str(output_dir / "gt_tcp.tum"),
            "raw_estimate_tum": str(output_dir / "vio_tcp_raw.tum"),
            "fixed_se3_estimate_tum": str(output_dir / "vio_tcp_fixed_0614_se3.tum"),
            "matched_csv": str(output_dir / "matched_samples.csv"),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "fixed_se3_transform.json").write_text(
        json.dumps(payload["fixed_se3"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(output_dir / "summary.csv", payload["source_fit_metrics"], payload["target_fixed_se3_metrics"])
    write_report(output_dir / "REPORT.md", payload, matrix_text)

    target_t = payload["target_fixed_se3_metrics"]["translation_m"]
    target_r = payload["target_fixed_se3_metrics"]["rotation_deg"]
    print(f"[OK] source matched samples: {source_times.size}")
    print(f"[OK] target matched samples: {target_times.size}")
    print(f"[OK] fixed SE(3) target APE rmse: {target_t['rmse'] * 1000.0:.3f} mm")
    print(f"[OK] fixed SE(3) target rotation rmse: {target_r['rmse']:.3f} deg")
    print(f"[OK] wrote {output_dir / 'summary.csv'}")
    print(f"[OK] wrote {output_dir / 'REPORT.md'}")
    print(f"[OK] wrote {output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
