#!/usr/bin/env python3
"""Probe robot ground-truth orientation hypotheses using existing trajectory data.

This script does not modify the saved robot JSON. It reuses the current VIO TCP
evaluation chain, then tests whether the large TCP rotation error can be
explained by either:

1. interpreting the robot pose direction as `base_to_gripper` vs
   `gripper_to_base`, or
2. applying one fixed local rotation offset to the robot TCP frame
   (`T_base_tcp_corrected = T_base_tcp @ T_tcp_tcpfix`).

If a single local correction collapses the rotation RMSE while translation stays
at the same level, the issue is consistent with a fixed TCP frame definition or
quaternion-frame convention mismatch. If not, the problem is likely not
explained by one constant correction.
"""

from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VIO_TCP_EVAL_PATH = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
DEFAULT_ESTIMATE = Path("/home/chenlvping/0614 _test/episode_20260614_0239/right/pose_data.csv")
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data/ground_truth/trajectory_samples0614/trajectory_001.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/robot_gt_adjustment_scan"
DEFAULT_HAND_EYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"
DEFAULT_ESTIMATE_TUM = REPO_ROOT / "data/evaluation/core/evo_vio_tcp_0616/vio_tcp_from_imu_left_camera.tum"


def load_vio_tcp_module() -> Dict[str, object]:
    return runpy.run_path(str(VIO_TCP_EVAL_PATH), run_name="__robot_gt_adjustment_scan__")


def load_helpers(vio_tcp_api: Dict[str, object]) -> Dict[str, object]:
    return vio_tcp_api["load_helpers"]()


def normalize_timestamp(value: float) -> float:
    value = float(value)
    av = abs(value)
    if av > 1e17:
        return value * 1e-9
    if av > 1e13:
        return value * 1e-6
    if av > 1e10:
        return value * 1e-3
    return value


def invert_transform(transform: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, :3] = transform[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ transform[:3, 3]
    return out


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) * 0.5
    value = max(-1.0, min(1.0, value))
    return math.degrees(math.acos(value))


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(math.sqrt(np.mean(values * values)))


def stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "rmse": rmse(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def average_rotation(rotations: np.ndarray) -> np.ndarray:
    accumulator = np.sum(rotations, axis=0)
    u, _, vt = np.linalg.svd(accumulator)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def rotation_to_axis_angle(rotation: np.ndarray) -> Tuple[np.ndarray, float]:
    angle_rad = math.radians(rotation_angle_deg(rotation))
    if angle_rad < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    denom = 2.0 * math.sin(angle_rad)
    if abs(denom) < 1e-12:
        # Near pi the axis extraction is numerically awkward; use diagonal terms.
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        if axis.sum() < 1e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        axis = axis / denom
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        axis = axis / norm
    return axis, math.degrees(angle_rad)


def load_robot_tcp_trajectory(
    path: Path,
    helpers: Dict[str, object],
    pose_direction: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples", data)
    times: List[float] = []
    poses: List[np.ndarray] = []
    for sample in samples:
        pose = helpers["transform_from_pose"](sample["position_m"], sample["quaternion_xyzw"])
        if pose_direction == "gripper_to_base":
            pose = invert_transform(pose)
        elif pose_direction != "base_to_gripper":
            raise ValueError(f"unsupported pose direction {pose_direction}")
        times.append(normalize_timestamp(float(sample["timestamp"])))
        poses.append(pose)
    order = np.argsort(np.asarray(times, dtype=float))
    poses_arr = np.asarray(poses, dtype=float)[order]
    return np.asarray(times, dtype=float)[order], poses_arr[:, :3, 3], poses_arr[:, :3, :3]


def load_tum_trajectory(path: Path, helpers: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            times.append(normalize_timestamp(t))
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


def evaluate_direction_case(
    gt_times: np.ndarray,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_times: np.ndarray,
    est_pos: np.ndarray,
    est_rot: np.ndarray,
    helpers: Dict[str, object],
    max_gap_s: float,
) -> Dict[str, object]:
    match_ground_truth = load_vio_tcp_module()["match_ground_truth"]
    times, matched_gt_pos, matched_gt_rot, matched_est_pos, matched_est_rot = match_ground_truth(
        gt_times,
        gt_pos,
        gt_rot,
        est_times,
        est_pos,
        est_rot,
        helpers,
        max_gap_s,
    )
    aligned_pos, aligned_rot, align_rot, align_trans = apply_global_alignment(
        matched_gt_pos,
        matched_est_pos,
        matched_est_rot,
        helpers,
    )
    trans_mm = np.linalg.norm(aligned_pos - matched_gt_pos, axis=1) * 1000.0
    baseline_rot_deg = np.asarray(
        [rotation_angle_deg(r_gt.T @ r_est) for r_gt, r_est in zip(matched_gt_rot, aligned_rot)],
        dtype=float,
    )

    residual_local = np.asarray(
        [r_gt.T @ r_est for r_gt, r_est in zip(matched_gt_rot, aligned_rot)],
        dtype=float,
    )
    best_local_correction = average_rotation(residual_local)
    corrected_gt_rot = np.asarray([r_gt @ best_local_correction for r_gt in matched_gt_rot], dtype=float)
    corrected_rot_deg = np.asarray(
        [rotation_angle_deg(r_gt.T @ r_est) for r_gt, r_est in zip(corrected_gt_rot, aligned_rot)],
        dtype=float,
    )
    residual_after = np.asarray(
        [r_gt.T @ r_est for r_gt, r_est in zip(corrected_gt_rot, aligned_rot)],
        dtype=float,
    )
    residual_after_angles = np.asarray([rotation_angle_deg(r) for r in residual_after], dtype=float)
    axis, angle_deg = rotation_to_axis_angle(best_local_correction)

    return {
        "matched_samples": int(times.size),
        "matched_duration_s": float(times[-1] - times[0]),
        "translation_error_mm": stats(trans_mm),
        "rotation_error_baseline_deg": stats(baseline_rot_deg),
        "rotation_error_after_local_tcp_fix_deg": stats(corrected_rot_deg),
        "estimated_tcp_local_fix": {
            "matrix": best_local_correction.tolist(),
            "angle_deg": angle_deg,
            "axis_xyz": axis.tolist(),
        },
        "residual_after_local_fix_deg": stats(residual_after_angles),
        "position_alignment": {
            "rotation_matrix": align_rot.tolist(),
            "rotation_angle_deg": rotation_angle_deg(align_rot),
            "translation_xyz_m": align_trans.tolist(),
            "translation_norm_m": float(np.linalg.norm(align_trans)),
        },
    }


def build_report(
    estimate_label: str,
    gt_path: Path,
    estimate_frame: str,
    handeye_yaml: Path | None,
    result_rows: List[Dict[str, object]],
) -> str:
    lines = [
        "# Robot GT Adjustment Scan",
        "",
        "## Inputs",
        "",
        f"- VIO estimate: `{estimate_label}`",
        f"- Robot ground truth: `{gt_path}`",
        f"- Estimate frame: `{estimate_frame}`",
        f"- Hand-eye YAML: `{handeye_yaml}`" if handeye_yaml is not None else "- Hand-eye YAML: default from evaluator",
        "",
        "## Interpretation",
        "",
        "- `baseline` is the normal TCP evaluation: align positions with one SE(3) world transform, then measure attitude error.",
        "- `after_local_tcp_fix` keeps translation alignment unchanged, but adds one fixed local rotation to the robot TCP frame.",
        "- If `after_local_tcp_fix` drops sharply, existing data is consistent with a fixed TCP/quaternion frame convention issue.",
        "- If it stays large, existing data does not support a simple constant frame correction.",
        "",
        "## Results",
        "",
        "| pose direction | trans RMSE mm | rot RMSE deg baseline | rot RMSE deg after local fix | estimated local fix angle deg |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            f"| {row['pose_direction']} | "
            f"{row['translation_error_mm']['rmse']:.3f} | "
            f"{row['rotation_error_baseline_deg']['rmse']:.3f} | "
            f"{row['rotation_error_after_local_tcp_fix_deg']['rmse']:.3f} | "
            f"{row['estimated_tcp_local_fix']['angle_deg']:.3f} |"
        )
    lines.append("")
    best = min(result_rows, key=lambda item: item["rotation_error_after_local_tcp_fix_deg"]["rmse"])
    lines.extend(
        [
            "## Best Case",
            "",
            f"- Pose direction: `{best['pose_direction']}`",
            f"- Baseline rotation RMSE: `{best['rotation_error_baseline_deg']['rmse']:.3f} deg`",
            f"- After one local TCP fix: `{best['rotation_error_after_local_tcp_fix_deg']['rmse']:.3f} deg`",
            f"- Estimated local fix angle: `{best['estimated_tcp_local_fix']['angle_deg']:.3f} deg`",
            f"- Estimated local fix axis: `{best['estimated_tcp_local_fix']['axis_xyz']}`",
            "",
            "If this best-case RMSE is still large, existing data cannot rescue the issue with one constant TCP rotation tweak alone.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, default=DEFAULT_ESTIMATE)
    parser.add_argument(
        "--estimate-tum",
        type=Path,
        default=DEFAULT_ESTIMATE_TUM,
        help="precomputed estimated TCP TUM trajectory; if present, bypass raw VIO CSV loading",
    )
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--handeye-yaml", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--estimate-frame", choices=["imu", "vins_base_link"], default="imu")
    parser.add_argument("--max-time-gap-ms", type=float, default=80.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vio_tcp_api = load_vio_tcp_module()
    helpers = load_helpers(vio_tcp_api)
    estimate_tum_path = args.estimate_tum.expanduser().resolve() if args.estimate_tum is not None else None
    if estimate_tum_path is not None and estimate_tum_path.exists():
        est_times, est_pos, est_rot = load_tum_trajectory(estimate_tum_path, helpers)
    else:
        t_tcp_left_camera = vio_tcp_api["load_tcp_left_camera_transform"](args.handeye_yaml.expanduser().resolve())
        est_times, est_pos, est_rot = vio_tcp_api["load_vio_tcp_trajectory"](
            args.estimate.expanduser().resolve(),
            helpers,
            t_tcp_left_camera,
            args.estimate_frame,
        )

    result_rows: List[Dict[str, object]] = []
    for pose_direction in ["base_to_gripper", "gripper_to_base"]:
        gt_times, gt_pos, gt_rot = load_robot_tcp_trajectory(
            args.ground_truth.expanduser().resolve(),
            helpers,
            pose_direction,
        )
        row = evaluate_direction_case(
            gt_times,
            gt_pos,
            gt_rot,
            est_times,
            est_pos,
            est_rot,
            helpers,
            args.max_time_gap_ms * 1e-3,
        )
        row["pose_direction"] = pose_direction
        result_rows.append(row)

    result_rows.sort(key=lambda item: item["rotation_error_after_local_tcp_fix_deg"]["rmse"])
    payload = {
        "estimate": str(args.estimate.expanduser().resolve()),
        "estimate_tum": str(estimate_tum_path) if estimate_tum_path is not None else None,
        "ground_truth": str(args.ground_truth.expanduser().resolve()),
        "estimate_frame": args.estimate_frame,
        "handeye_yaml": str(args.handeye_yaml.expanduser().resolve()),
        "results": result_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(
        build_report(
            str(estimate_tum_path) if estimate_tum_path is not None and estimate_tum_path.exists() else str(args.estimate.expanduser().resolve()),
            args.ground_truth.expanduser().resolve(),
            args.estimate_frame,
            args.handeye_yaml.expanduser().resolve(),
            result_rows,
        ),
        encoding="utf-8",
    )
    print(f"[OK] wrote {output_dir / 'summary.json'}")
    print(f"[OK] wrote {output_dir / 'REPORT.md'}")
    for row in result_rows:
        print(
            row["pose_direction"],
            f"trans_rmse={row['translation_error_mm']['rmse']:.3f}mm",
            f"rot_rmse={row['rotation_error_baseline_deg']['rmse']:.3f}deg",
            f"rot_rmse_after_local_fix={row['rotation_error_after_local_tcp_fix_deg']['rmse']:.3f}deg",
            f"local_fix_angle={row['estimated_tcp_local_fix']['angle_deg']:.3f}deg",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
