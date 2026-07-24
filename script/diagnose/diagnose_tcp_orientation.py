#!/usr/bin/env python3
"""
Diagnose why TCP orientation error stays large after VIO/robot trajectory alignment.

This script reads already exported TUM trajectories from evaluate_vio_tcp_camera_evo.py:

  * gt_tcp.tum
  * vio_tcp_from_imu_left_camera.tum

It separates three common failure modes:

  * a time offset between robot and VIO trajectories,
  * a fixed world-frame rotation bias,
  * non-constant local orientation drift/convention errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPERS_PATH = REPO_ROOT / "script/evaluate_vins_accuracy.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/tcp_orientation_diagnostics"
DEFAULT_CASES = [
    ("vins", REPO_ROOT / "data/evaluation/core/evo_vio_tcp_0616"),
    ("dynavins", REPO_ROOT / "data/evaluation/core/evo_dynavins_tcp_0616"),
]


def load_helpers() -> Dict[str, object]:
    return runpy.run_path(str(HELPERS_PATH), run_name="__tcp_orientation_diag__")


def load_tum(path: Path, helpers: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: List[float] = []
    positions: List[List[float]] = []
    rotations: List[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            t, x, y, z, qx, qy, qz, qw = map(float, parts[:8])
            times.append(t)
            positions.append([x, y, z])
            rotations.append(helpers["quat_xyzw_to_rot"]([qx, qy, qz, qw]))
    if not times:
        raise RuntimeError(f"no poses loaded from {path}")
    order = np.argsort(np.asarray(times, dtype=float))
    return (
        np.asarray(times, dtype=float)[order],
        np.asarray(positions, dtype=float)[order],
        np.asarray(rotations, dtype=float)[order],
    )


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) * 0.5
    value = max(-1.0, min(1.0, value))
    return math.degrees(math.acos(value))


def rotation_errors_deg(gt_rot: np.ndarray, est_rot: np.ndarray) -> np.ndarray:
    return np.asarray(
        [rotation_angle_deg(r_gt.T @ r_est) for r_gt, r_est in zip(gt_rot, est_rot)],
        dtype=float,
    )


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(math.sqrt(np.mean(values * values)))


def stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {key: float("nan") for key in ["rmse", "mean", "median", "min", "max", "std", "p95"]}
    return {
        "rmse": rmse(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95.0)),
    }


def best_left_rotation(source_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=float)
    for src, dst in zip(source_rot, target_rot):
        matrix += dst @ src.T
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def apply_global_alignment(
    gt_pos: np.ndarray,
    est_pos: np.ndarray,
    est_rot: np.ndarray,
    helpers: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, rotation, translation = helpers["align_umeyama"](est_pos, gt_pos, with_scale=False)
    aligned_pos = (rotation @ est_pos.T).T + translation
    aligned_rot = np.asarray([rotation @ rot for rot in est_rot], dtype=float)
    return aligned_pos, aligned_rot, rotation, translation


def interpolate_tum(
    times: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
    query_times: np.ndarray,
    helpers: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = (query_times >= times[0]) & (query_times <= times[-1])
    queries = query_times[valid]
    out_pos: List[np.ndarray] = []
    out_rot: List[np.ndarray] = []
    quats = np.asarray([helpers["rot_to_quat_xyzw"](rot) for rot in rotations], dtype=float)
    indices = np.searchsorted(times, queries, side="right") - 1
    indices = np.clip(indices, 0, times.size - 2)
    for t, left in zip(queries, indices):
        t0 = times[left]
        t1 = times[left + 1]
        alpha = 0.0 if abs(t1 - t0) < 1e-12 else float((t - t0) / (t1 - t0))
        out_pos.append((1.0 - alpha) * positions[left] + alpha * positions[left + 1])
        out_rot.append(helpers["quat_xyzw_to_rot"](helpers["slerp"](quats[left], quats[left + 1], alpha)))
    return valid, np.asarray(out_pos, dtype=float), np.asarray(out_rot, dtype=float)


def time_offset_scan(
    gt_times: np.ndarray,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_times: np.ndarray,
    est_pos: np.ndarray,
    est_rot: np.ndarray,
    helpers: Dict[str, object],
    offsets: np.ndarray,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for offset in offsets:
        valid, shifted_gt_pos, shifted_gt_rot = interpolate_tum(
            gt_times,
            gt_pos,
            gt_rot,
            est_times + float(offset),
            helpers,
        )
        if shifted_gt_pos.shape[0] < 10:
            continue
        cur_est_pos = est_pos[valid]
        cur_est_rot = est_rot[valid]
        aligned_pos, aligned_rot, _, _ = apply_global_alignment(
            shifted_gt_pos,
            cur_est_pos,
            cur_est_rot,
            helpers,
        )
        trans_mm = np.linalg.norm(aligned_pos - shifted_gt_pos, axis=1) * 1000.0
        rot_deg = rotation_errors_deg(shifted_gt_rot, aligned_rot)
        rows.append(
            {
                "offset_s": float(offset),
                "samples": int(shifted_gt_pos.shape[0]),
                "translation_rmse_mm": rmse(trans_mm),
                "rotation_rmse_deg": rmse(rot_deg),
                "rotation_mean_deg": float(np.mean(rot_deg)),
            }
        )
    return rows


def relative_rotation_errors(
    times: np.ndarray,
    gt_rot: np.ndarray,
    est_rot: np.ndarray,
    delta_s: float,
) -> np.ndarray:
    errors: List[float] = []
    right_indices = np.searchsorted(times, times + delta_s, side="left")
    for left, right in enumerate(right_indices):
        if right >= times.size:
            continue
        d_gt = gt_rot[left].T @ gt_rot[right]
        d_est = est_rot[left].T @ est_rot[right]
        errors.append(rotation_angle_deg(d_gt.T @ d_est))
    return np.asarray(errors, dtype=float)


def downsample(
    times: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples <= 0 or times.size <= max_samples:
        return times, positions, rotations
    indices = np.linspace(0, times.size - 1, max_samples).round().astype(int)
    return times[indices], positions[indices], rotations[indices]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_case(text: str) -> Tuple[str, Path]:
    if ":" not in text:
        path = Path(text)
        return path.name, path
    name, path = text.split(":", 1)
    return name, Path(path)


def diagnose_case(
    name: str,
    case_dir: Path,
    output_dir: Path,
    helpers: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    gt_tum = case_dir / "gt_tcp.tum"
    est_tum = case_dir / "vio_tcp_from_imu_left_camera.tum"
    gt_times, gt_pos, gt_rot = load_tum(gt_tum, helpers)
    est_times, est_pos, est_rot = load_tum(est_tum, helpers)
    if gt_times.size != est_times.size or np.max(np.abs(gt_times - est_times)) > 1e-6:
        valid, gt_pos, gt_rot = interpolate_tum(gt_times, gt_pos, gt_rot, est_times, helpers)
        est_times = est_times[valid]
        est_pos = est_pos[valid]
        est_rot = est_rot[valid]
        gt_times = est_times.copy()

    gt_times, gt_pos, gt_rot = downsample(gt_times, gt_pos, gt_rot, args.max_samples)
    est_times, est_pos, est_rot = downsample(est_times, est_pos, est_rot, args.max_samples)

    aligned_pos, pos_aligned_rot, r_pos, _ = apply_global_alignment(gt_pos, est_pos, est_rot, helpers)
    trans_mm = np.linalg.norm(aligned_pos - gt_pos, axis=1) * 1000.0
    pos_aligned_rot_errors = rotation_errors_deg(gt_rot, pos_aligned_rot)

    r_ori = best_left_rotation(est_rot, gt_rot)
    ori_aligned_rot = np.asarray([r_ori @ rot for rot in est_rot], dtype=float)
    ori_aligned_rot_errors = rotation_errors_deg(gt_rot, ori_aligned_rot)
    align_delta_deg = rotation_angle_deg(r_ori @ r_pos.T)

    rel_rows: List[Dict[str, object]] = []
    for delta_s in args.relative_delta_s:
        rel_errors = relative_rotation_errors(gt_times, gt_rot, est_rot, delta_s)
        rel_stats = stats(rel_errors)
        rel_rows.append(
            {
                "case": name,
                "delta_s": delta_s,
                "pairs": int(rel_errors.size),
                **{f"rotation_{key}_deg": value for key, value in rel_stats.items()},
            }
        )

    offsets = np.arange(args.offset_min_s, args.offset_max_s + args.offset_step_s * 0.5, args.offset_step_s)
    offset_rows = time_offset_scan(gt_times, gt_pos, gt_rot, est_times, est_pos, est_rot, helpers, offsets)
    write_csv(output_dir / f"{name}_time_offset_scan.csv", offset_rows)

    best_by_rot = min(offset_rows, key=lambda row: row["rotation_rmse_deg"])
    best_by_trans = min(offset_rows, key=lambda row: row["translation_rmse_mm"])

    per_pose_rows = []
    sample_count = min(args.save_pose_samples, gt_times.size)
    sample_indices = np.linspace(0, gt_times.size - 1, sample_count).round().astype(int)
    for idx in sample_indices:
        per_pose_rows.append(
            {
                "case": name,
                "index": int(idx),
                "time_s": float(gt_times[idx]),
                "translation_error_mm": float(trans_mm[idx]),
                "rotation_error_pos_align_deg": float(pos_aligned_rot_errors[idx]),
                "rotation_error_ori_align_deg": float(ori_aligned_rot_errors[idx]),
            }
        )
    write_csv(output_dir / f"{name}_pose_error_samples.csv", per_pose_rows)

    return {
        "case": name,
        "case_dir": str(case_dir),
        "samples": int(gt_times.size),
        "duration_s": float(gt_times[-1] - gt_times[0]),
        "translation_rmse_mm": stats(trans_mm)["rmse"],
        "translation_mean_mm": stats(trans_mm)["mean"],
        "position_aligned_rotation_rmse_deg": stats(pos_aligned_rot_errors)["rmse"],
        "position_aligned_rotation_mean_deg": stats(pos_aligned_rot_errors)["mean"],
        "orientation_only_rotation_rmse_deg": stats(ori_aligned_rot_errors)["rmse"],
        "orientation_only_rotation_mean_deg": stats(ori_aligned_rot_errors)["mean"],
        "position_vs_orientation_alignment_delta_deg": align_delta_deg,
        "best_time_offset_by_rotation_s": best_by_rot["offset_s"],
        "best_time_offset_rotation_rmse_deg": best_by_rot["rotation_rmse_deg"],
        "best_time_offset_translation_rmse_mm": best_by_rot["translation_rmse_mm"],
        "best_time_offset_by_translation_s": best_by_trans["offset_s"],
        "best_translation_offset_translation_rmse_mm": best_by_trans["translation_rmse_mm"],
        "relative_rotation": rel_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        help="case name and directory as name:/path/to/evo_output. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=8000)
    parser.add_argument("--offset-min-s", type=float, default=-0.30)
    parser.add_argument("--offset-max-s", type=float, default=0.30)
    parser.add_argument("--offset-step-s", type=float, default=0.01)
    parser.add_argument("--relative-delta-s", type=float, nargs="+", default=[0.05, 0.10, 0.50, 1.00])
    parser.add_argument("--save-pose-samples", type=int, default=200)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    helpers = load_helpers()

    cases = [parse_case(item) for item in args.case] if args.case else DEFAULT_CASES
    summaries = []
    relative_rows: List[Dict[str, object]] = []
    for name, path in cases:
        summary = diagnose_case(name, path.expanduser().resolve(), output_dir, helpers, args)
        relative_rows.extend(summary.pop("relative_rotation"))
        summaries.append(summary)

    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "relative_rotation.csv", relative_rows)
    (output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    report = ["# TCP Orientation Diagnostics", ""]
    report.append("## Summary")
    report.append("")
    report.append("| case | trans RMSE mm | rot RMSE after position align deg | rot RMSE after orientation-only align deg | alignment delta deg | best rot time offset s | best rot RMSE deg |")
    report.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summaries:
        report.append(
            f"| {row['case']} | {row['translation_rmse_mm']:.3f} | "
            f"{row['position_aligned_rotation_rmse_deg']:.3f} | "
            f"{row['orientation_only_rotation_rmse_deg']:.3f} | "
            f"{row['position_vs_orientation_alignment_delta_deg']:.3f} | "
            f"{row['best_time_offset_by_rotation_s']:.3f} | "
            f"{row['best_time_offset_rotation_rmse_deg']:.3f} |"
        )
    report.append("")
    report.append("## How To Read This")
    report.append("")
    report.append("- If the best time offset greatly lowers rotation RMSE, suspect timestamp sync.")
    report.append("- If orientation-only alignment is much better than position alignment, position and attitude disagree about the VINS-world to robot-base rotation.")
    report.append("- If relative rotation errors are still large, the issue is not only one fixed world-frame rotation; check IMU-camera, hand-eye, pose direction, or VIO quaternion convention.")
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"[OK] wrote {output_dir / 'summary.csv'}")
    print(f"[OK] wrote {output_dir / 'relative_rotation.csv'}")
    print(f"[OK] wrote {output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
