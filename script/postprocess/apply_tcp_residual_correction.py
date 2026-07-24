#!/usr/bin/env python3
"""
Apply a smooth, episode-calibrated TCP residual correction to a VIO CSV.

This is a trajectory-level backend calibration experiment. It uses the robot
ground truth for the same episode, so its output should be reported as
GT-calibrated post-processing rather than as a pure online VINS result.
"""

from __future__ import annotations

import argparse
import csv
import json
import runpy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
TCP_EVAL = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
VINS_EVAL = REPO_ROOT / "script/evaluate_vins_accuracy.py"


def load_module(path: Path, run_name: str) -> Dict[str, object]:
    return runpy.run_path(str(path), run_name=run_name)


def transform_from_parts(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return list(reader.fieldnames), rows


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_transform_stack(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    transforms = np.repeat(np.eye(4, dtype=float)[None, :, :], positions.shape[0], axis=0)
    transforms[:, :3, :3] = rotations
    transforms[:, :3, 3] = positions
    return transforms


def make_knots(
    times: np.ndarray,
    residual_translation: np.ndarray,
    residual_rotvec: np.ndarray,
    knot_spacing_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if knot_spacing_s <= 0.0:
        return times.copy(), residual_translation.copy(), residual_rotvec.copy()

    start = float(times[0])
    stop = float(times[-1])
    knot_times = np.arange(start, stop + 0.5 * knot_spacing_s, knot_spacing_s, dtype=float)
    if knot_times.size == 0 or abs(knot_times[0] - start) > 1e-12:
        knot_times = np.insert(knot_times, 0, start)
    if knot_times[-1] < stop:
        knot_times = np.append(knot_times, stop)
    else:
        knot_times[-1] = stop

    knot_translation = np.column_stack(
        [np.interp(knot_times, times, residual_translation[:, axis]) for axis in range(3)]
    )
    knot_rotvec = np.column_stack([np.interp(knot_times, times, residual_rotvec[:, axis]) for axis in range(3)])
    return knot_times, knot_translation, knot_rotvec


def interpolate_residual(
    query_times: np.ndarray,
    knot_times: np.ndarray,
    knot_translation: np.ndarray,
    knot_rotvec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(query_times, knot_times[0], knot_times[-1])
    translations = np.column_stack([np.interp(clipped, knot_times, knot_translation[:, axis]) for axis in range(3)])
    rotvecs = np.column_stack([np.interp(clipped, knot_times, knot_rotvec[:, axis]) for axis in range(3)])
    rotations = Rotation.from_rotvec(rotvecs).as_matrix()
    return translations, rotations


def update_velocity_columns(rows: List[Dict[str, str]], times: np.ndarray) -> None:
    if not rows or not all(name in rows[0] for name in ("Vx", "Vy", "Vz", "X", "Y", "Z")):
        return
    positions = np.asarray([[float(row["X"]), float(row["Y"]), float(row["Z"])] for row in rows], dtype=float)
    if positions.shape[0] < 2:
        return
    velocities = np.gradient(positions, times, axis=0)
    for row, velocity in zip(rows, velocities):
        row["Vx"] = f"{velocity[0]:.9g}"
        row["Vy"] = f"{velocity[1]:.9g}"
        row["Vz"] = f"{velocity[2]:.9g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--handeye-yaml", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--estimate-frame", choices=["imu"], default="imu")
    parser.add_argument("--time-offset-sec", type=float, required=True)
    parser.add_argument("--t-max-diff-sec", type=float, default=0.01)
    parser.add_argument(
        "--knot-spacing-sec",
        type=float,
        default=2.0,
        help="Piecewise-linear residual knot spacing. Use 0 for dense residual interpolation.",
    )
    parser.add_argument("--rpe-distance-m", type=float, default=0.05)
    args = parser.parse_args()

    tcp_eval = load_module(TCP_EVAL, "__tcp_eval_residual_correction__")
    helpers = load_module(VINS_EVAL, "__vins_eval_residual_correction__")

    t_tcp_left_camera = tcp_eval["load_tcp_left_camera_transform"](args.handeye_yaml)
    t_left_camera_imu = tcp_eval["T_LEFT_CAMERA_IMU"]
    t_tcp_to_imu = t_tcp_left_camera @ t_left_camera_imu

    gt_times, gt_pos_all, gt_rot_all = tcp_eval["load_robot_tcp_trajectory"](args.ground_truth, helpers)
    est_times, est_tcp_pos_all, est_tcp_rot_all = tcp_eval["load_vio_tcp_trajectory"](
        args.estimate,
        helpers,
        t_tcp_left_camera,
        args.estimate_frame,
    )

    assoc_times, assoc_gt_pos, assoc_gt_rot, assoc_est_pos, assoc_est_rot = tcp_eval["associate_by_nearest_time"](
        gt_times,
        gt_pos_all,
        gt_rot_all,
        est_times,
        est_tcp_pos_all,
        est_tcp_rot_all,
        args.time_offset_sec,
        args.t_max_diff_sec,
    )

    se3, assoc_est_aligned_pos, assoc_est_aligned_rot, assoc_errors, assoc_rot_errors = helpers["evaluate_alignment"](
        "se3",
        False,
        assoc_gt_pos,
        assoc_gt_rot,
        assoc_est_pos,
        assoc_est_rot,
        assoc_times,
        0.0,
        30,
    )
    align_rotation = np.asarray(se3.rotation, dtype=float)
    align_translation = np.asarray(se3.translation, dtype=float)

    residual_rotations = np.asarray(
        [assoc_gt_rot[index] @ assoc_est_aligned_rot[index].T for index in range(assoc_times.size)],
        dtype=float,
    )
    residual_translations = np.asarray(
        [
            assoc_gt_pos[index] - residual_rotations[index] @ assoc_est_aligned_pos[index]
            for index in range(assoc_times.size)
        ],
        dtype=float,
    )
    residual_rotvec = Rotation.from_matrix(residual_rotations).as_rotvec()

    knot_times, knot_translation, knot_rotvec = make_knots(
        assoc_times,
        residual_translations,
        residual_rotvec,
        float(args.knot_spacing_sec),
    )

    shifted_est_times = est_times + float(args.time_offset_sec)
    query_translation, query_rotation = interpolate_residual(
        shifted_est_times,
        knot_times,
        knot_translation,
        knot_rotvec,
    )

    est_aligned_pos_all = (align_rotation @ est_tcp_pos_all.T).T + align_translation
    est_aligned_rot_all = np.asarray([align_rotation @ rotation for rotation in est_tcp_rot_all], dtype=float)
    corrected_tcp_pos_world = np.empty_like(est_tcp_pos_all)
    corrected_tcp_rot_world = np.empty_like(est_tcp_rot_all)
    for index in range(est_times.size):
        corrected_pos_aligned = query_rotation[index] @ est_aligned_pos_all[index] + query_translation[index]
        corrected_rot_aligned = query_rotation[index] @ est_aligned_rot_all[index]
        corrected_tcp_pos_world[index] = align_rotation.T @ (corrected_pos_aligned - align_translation)
        corrected_tcp_rot_world[index] = align_rotation.T @ corrected_rot_aligned

    corrected_tcp = build_transform_stack(corrected_tcp_pos_world, corrected_tcp_rot_world)
    corrected_imu = np.asarray([pose @ t_tcp_to_imu for pose in corrected_tcp], dtype=float)

    fieldnames, rows = read_csv_rows(args.estimate)
    if len(rows) != corrected_imu.shape[0]:
        raise ValueError(f"CSV row count {len(rows)} does not match loaded pose count {corrected_imu.shape[0]}")

    for row, pose in zip(rows, corrected_imu):
        quat = helpers["rot_to_quat_xyzw"](pose[:3, :3])
        row["X"] = f"{pose[0, 3]:.12g}"
        row["Y"] = f"{pose[1, 3]:.12g}"
        row["Z"] = f"{pose[2, 3]:.12g}"
        row["Quat_X"] = f"{quat[0]:.12g}"
        row["Quat_Y"] = f"{quat[1]:.12g}"
        row["Quat_Z"] = f"{quat[2]:.12g}"
        row["Quat_W"] = f"{quat[3]:.12g}"
    update_velocity_columns(rows, est_times)
    write_csv_rows(args.output_csv, fieldnames, rows)

    # Internal sanity check before the heavier evo pass.
    corrected_tcp_pos_assoc = []
    corrected_tcp_rot_assoc = []
    shifted_est = est_times + float(args.time_offset_sec)
    est_start = 0
    for ref_t in assoc_times:
        insert_at = int(np.searchsorted(shifted_est[est_start:], ref_t, side="left")) + est_start
        candidates = []
        if insert_at < shifted_est.size:
            candidates.append(insert_at)
        if insert_at - 1 >= est_start:
            candidates.append(insert_at - 1)
        best_est = min(candidates, key=lambda idx: abs(shifted_est[idx] - ref_t))
        corrected_tcp_pos_assoc.append(corrected_tcp_pos_world[best_est])
        corrected_tcp_rot_assoc.append(corrected_tcp_rot_world[best_est])
        est_start = best_est + 1
    corrected_internal, _, _, corrected_errors, corrected_rot_errors = helpers["evaluate_alignment"](
        "se3",
        False,
        assoc_gt_pos,
        assoc_gt_rot,
        np.asarray(corrected_tcp_pos_assoc, dtype=float),
        np.asarray(corrected_tcp_rot_assoc, dtype=float),
        assoc_times,
        0.0,
        30,
    )

    report = {
        "kind": "gt_calibrated_tcp_residual_correction",
        "estimate": str(args.estimate),
        "ground_truth": str(args.ground_truth),
        "handeye_yaml": str(args.handeye_yaml),
        "output_csv": str(args.output_csv),
        "time_offset_sec": float(args.time_offset_sec),
        "t_max_diff_sec": float(args.t_max_diff_sec),
        "knot_spacing_sec": float(args.knot_spacing_sec),
        "matched_samples": int(assoc_times.size),
        "matched_duration_s": float(assoc_times[-1] - assoc_times[0]),
        "knot_count": int(knot_times.size),
        "pre_correction_internal_ape_rmse_mm": float(se3.translation_metrics_m["rmse"] * 1000.0),
        "post_correction_internal_ape_rmse_mm": float(
            corrected_internal.translation_metrics_m["rmse"] * 1000.0
        ),
        "post_correction_internal_rot_rmse_deg": None
        if not corrected_internal.rotation_metrics_deg
        else float(corrected_internal.rotation_metrics_deg["rmse"]),
        "pre_correction_residual_rmse_mm": float(np.sqrt(np.mean(assoc_errors**2)) * 1000.0),
        "post_correction_residual_rmse_mm": float(np.sqrt(np.mean(corrected_errors**2)) * 1000.0),
        "post_correction_residual_rot_rmse_deg": None
        if corrected_rot_errors is None
        else float(np.sqrt(np.mean(corrected_rot_errors**2))),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[OK] wrote",
        args.output_csv,
        "internal APE",
        f"{report['post_correction_internal_ape_rmse_mm']:.3f} mm",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
