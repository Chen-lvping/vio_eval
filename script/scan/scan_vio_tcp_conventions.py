#!/usr/bin/env python3
"""Scan VIO/robot TCP convention choices without rerunning VIO.

This script tests the three most likely convention mistakes:

  * VIO CSV pose direction: T_world_imu vs T_imu_world.
  * camera/IMU extrinsic direction: T_left_camera_imu vs inverse.
  * hand-eye direction: T_tcp_left_camera vs inverse.

For each combination it builds an estimated TCP trajectory, aligns positions
to robot TCP ground truth with SE(3), and reports translation and rotation
errors. A good convention should reduce rotation error without exploding
translation error.
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

DEFAULT_ESTIMATE = Path("/home/chenlvping/6_data_use/0614 _test/episode_20260614_0239/right/pose_data.csv")
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data/ground_truth/trajectory_samples0614/trajectory_001.json"
DEFAULT_CALIBRATION = Path("/home/chenlvping/6_data_use/0614 _test/episode_20260614_0239/calibration.json")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/convention_scan"

T_TCP_LEFT_CAMERA = np.array(
    [
        [0.854672738, -0.422689316, 0.301443615, 0.020087823],
        [0.519166541, 0.694970001, -0.497476432, -0.097333617],
        [0.000783703, 0.581678983, 0.813418064, 0.052300458],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

FALLBACK_T_LEFT_CAMERA_IMU = np.array(
    [
        [-0.999638319, 0.0265241228, -0.00445450377, -0.0022427286],
        [-0.0265235156, -0.999648213, -0.000195318818, 0.0140962508],
        [-0.0044581173, -7.70990737e-05, 0.999990046, -0.0155969206],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def load_helpers() -> Dict[str, object]:
    return runpy.run_path(str(HELPERS_PATH), run_name="__convention_scan__")


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


def transform_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) * 0.5
    value = max(-1.0, min(1.0, value))
    return math.degrees(math.acos(value))


def stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "rmse": float(math.sqrt(np.mean(values * values))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def load_left_camera_imu(calibration_path: Path) -> np.ndarray:
    if not calibration_path.exists():
        return FALLBACK_T_LEFT_CAMERA_IMU.copy()
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    try:
        matrix = data["observation"]["images"]["stereo_right"]["extrinsics"]["T_ic_cam0_to_imu0"]
        return np.asarray(matrix, dtype=float)
    except KeyError:
        return FALLBACK_T_LEFT_CAMERA_IMU.copy()


def load_robot_tcp(path: Path, helpers: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples", data)
    times: List[float] = []
    poses: List[np.ndarray] = []
    for sample in samples:
        timestamp = sample.get("timestamp", sample.get("timestamp_s", sample.get("timestamp_us")))
        if timestamp is None:
            raise KeyError("timestamp_s")
        position = sample.get("position_m", sample.get("position"))
        quat = sample.get("quaternion_xyzw", sample.get("quaternion_wxyz"))
        if position is None or quat is None:
            raise KeyError("position_m/quaternion")
        if isinstance(position, dict):
            position = [position["x"], position["y"], position["z"]]
        times.append(normalize_timestamp(float(timestamp)))
        if "quaternion_wxyz" in sample and "quaternion_xyzw" not in sample:
            quat = [quat["x"], quat["y"], quat["z"], quat["w"]]
        if isinstance(quat, dict):
            if "w" in quat:
                quat = [quat["x"], quat["y"], quat["z"], quat["w"]]
            else:
                quat = [quat["x"], quat["y"], quat["z"], quat["w"]]
        poses.append(helpers["transform_from_pose"](position, quat))
    order = np.argsort(np.asarray(times, dtype=float))
    poses_arr = np.asarray(poses, dtype=float)[order]
    return np.asarray(times, dtype=float)[order], poses_arr[:, :3, 3], poses_arr[:, :3, :3]


def row_value(row: dict, names: Sequence[str]) -> str | None:
    keys = {str(key).strip().lower(): key for key in row.keys()}
    for name in names:
        key = keys.get(name.lower())
        if key is not None and row.get(key) not in (None, ""):
            return str(row[key])
    return None


def load_vio(path: Path, helpers: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    poses: List[np.ndarray] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            ts = row_value(row, ["Timestamp_us", "timestamp_us", "Timestamp_ns", "timestamp_ns", "timestamp", "time", "t"])
            if ts is None:
                ts = str(idx / 30.0)
            xyz = [row_value(row, names) for names in (["X", "x"], ["Y", "y"], ["Z", "z"])]
            quat = [
                row_value(row, ["Quat_X", "qx", "q_x"]),
                row_value(row, ["Quat_Y", "qy", "q_y"]),
                row_value(row, ["Quat_Z", "qz", "q_z"]),
                row_value(row, ["Quat_W", "qw", "q_w"]),
            ]
            if any(value is None for value in xyz + quat):
                continue
            times.append(normalize_timestamp(float(ts)))
            poses.append(helpers["transform_from_pose"]([float(v) for v in xyz], [float(v) for v in quat]))
    order = np.argsort(np.asarray(times, dtype=float))
    return np.asarray(times, dtype=float)[order], np.asarray(poses, dtype=float)[order]


def interpolate_gt(
    gt_times: np.ndarray,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_times: np.ndarray,
    max_gap_s: float,
    helpers: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matched_indices: List[int] = []
    out_pos: List[np.ndarray] = []
    out_rot: List[np.ndarray] = []
    out_times: List[float] = []
    left = 0
    for idx, t in enumerate(est_times):
        while left + 1 < gt_times.size and gt_times[left + 1] <= t:
            left += 1
        if left + 1 >= gt_times.size or t < gt_times[left]:
            continue
        t0 = gt_times[left]
        t1 = gt_times[left + 1]
        if max(abs(t - t0), abs(t1 - t)) > max_gap_s:
            continue
        alpha = 0.0 if abs(t1 - t0) < 1e-12 else float((t - t0) / (t1 - t0))
        q0 = helpers["rot_to_quat_xyzw"](gt_rot[left])
        q1 = helpers["rot_to_quat_xyzw"](gt_rot[left + 1])
        out_times.append(float(t))
        out_pos.append((1.0 - alpha) * gt_pos[left] + alpha * gt_pos[left + 1])
        out_rot.append(helpers["quat_xyzw_to_rot"](helpers["slerp"](q0, q1, alpha)))
        matched_indices.append(idx)
    if not matched_indices:
        raise RuntimeError("no timestamp overlap between VIO and robot trajectory")
    return (
        np.asarray(out_times, dtype=float),
        np.asarray(out_pos, dtype=float),
        np.asarray(out_rot, dtype=float),
        np.asarray(matched_indices, dtype=int),
    )


def apply_chain(
    vio_pose: np.ndarray,
    vio_direction: str,
    camera_imu_direction: str,
    handeye_direction: str,
    t_left_camera_imu: np.ndarray,
    t_tcp_left_camera: np.ndarray,
) -> np.ndarray:
    t_world_imu = vio_pose if vio_direction == "T_world_imu" else invert_transform(vio_pose)

    if camera_imu_direction == "T_left_camera_imu":
        t_imu_left_camera = invert_transform(t_left_camera_imu)
    elif camera_imu_direction == "T_imu_left_camera":
        t_imu_left_camera = t_left_camera_imu
    else:
        raise ValueError(camera_imu_direction)

    if handeye_direction == "T_tcp_left_camera":
        t_left_camera_tcp = invert_transform(t_tcp_left_camera)
    elif handeye_direction == "T_left_camera_tcp":
        t_left_camera_tcp = t_tcp_left_camera
    else:
        raise ValueError(handeye_direction)

    return t_world_imu @ t_imu_left_camera @ t_left_camera_tcp


def evaluate_case(
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_poses: np.ndarray,
    helpers: Dict[str, object],
) -> Dict[str, object]:
    est_pos = est_poses[:, :3, 3]
    est_rot = est_poses[:, :3, :3]
    _, align_rot, align_trans = helpers["align_umeyama"](est_pos, gt_pos, with_scale=False)
    aligned_pos = (align_rot @ est_pos.T).T + align_trans
    aligned_rot = np.asarray([align_rot @ rot for rot in est_rot], dtype=float)
    trans_mm = np.linalg.norm(aligned_pos - gt_pos, axis=1) * 1000.0
    rot_deg = np.asarray([transform_angle_deg(r_gt.T @ r_est) for r_gt, r_est in zip(gt_rot, aligned_rot)])
    trans_stats = stats(trans_mm)
    rot_stats = stats(rot_deg)
    return {
        "ape_translation_rmse_mm": trans_stats["rmse"],
        "ape_translation_mean_mm": trans_stats["mean"],
        "ape_translation_median_mm": trans_stats["median"],
        "ape_rotation_rmse_deg": rot_stats["rmse"],
        "ape_rotation_mean_deg": rot_stats["mean"],
        "ape_rotation_median_deg": rot_stats["median"],
        "align_rotation_angle_deg": transform_angle_deg(align_rot),
        "align_translation_norm_m": float(np.linalg.norm(align_trans)),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, default=DEFAULT_ESTIMATE)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-time-gap-ms", type=float, default=80.0)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    helpers = load_helpers()
    t_left_camera_imu = load_left_camera_imu(args.calibration.expanduser().resolve())
    gt_times, gt_pos_all, gt_rot_all = load_robot_tcp(args.ground_truth.expanduser().resolve(), helpers)
    vio_times, vio_poses_all = load_vio(args.estimate.expanduser().resolve(), helpers)
    times, gt_pos, gt_rot, matched_indices = interpolate_gt(
        gt_times,
        gt_pos_all,
        gt_rot_all,
        vio_times,
        args.max_time_gap_ms * 1e-3,
        helpers,
    )
    vio_poses = vio_poses_all[matched_indices]

    rows: List[Dict[str, object]] = []
    for vio_direction in ["T_world_imu", "T_imu_world"]:
        for camera_imu_direction in ["T_left_camera_imu", "T_imu_left_camera"]:
            for handeye_direction in ["T_tcp_left_camera", "T_left_camera_tcp"]:
                est_poses = np.asarray(
                    [
                        apply_chain(
                            pose,
                            vio_direction,
                            camera_imu_direction,
                            handeye_direction,
                            t_left_camera_imu,
                            T_TCP_LEFT_CAMERA,
                        )
                        for pose in vio_poses
                    ],
                    dtype=float,
                )
                row = {
                    "rank": 0,
                    "vio_direction": vio_direction,
                    "camera_imu_input_meaning": camera_imu_direction,
                    "handeye_input_meaning": handeye_direction,
                    "samples": int(times.size),
                    "duration_s": float(times[-1] - times[0]),
                    **evaluate_case(gt_pos, gt_rot, est_poses, helpers),
                }
                rows.append(row)

    rows.sort(key=lambda item: (float(item["ape_rotation_rmse_deg"]), float(item["ape_translation_rmse_mm"])))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    write_csv(output_dir / "convention_scan.csv", rows)
    payload = {
        "estimate": str(args.estimate.expanduser().resolve()),
        "ground_truth": str(args.ground_truth.expanduser().resolve()),
        "calibration": str(args.calibration.expanduser().resolve()),
        "output_dir": str(output_dir),
        "T_left_camera_imu": t_left_camera_imu.tolist(),
        "T_tcp_left_camera": T_TCP_LEFT_CAMERA.tolist(),
        "rows": rows,
    }
    (output_dir / "convention_scan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    best = rows[0]
    report = [
        "# VIO TCP Convention Scan",
        "",
        "## Best By APE Rotation",
        "",
        f"- VIO direction: `{best['vio_direction']}`",
        f"- Camera/IMU input meaning: `{best['camera_imu_input_meaning']}`",
        f"- Hand-eye input meaning: `{best['handeye_input_meaning']}`",
        f"- APE translation RMSE: `{best['ape_translation_rmse_mm']:.3f} mm`",
        f"- APE rotation RMSE: `{best['ape_rotation_rmse_deg']:.3f} deg`",
        "",
        "## All Cases",
        "",
        "| rank | VIO | camera/IMU | hand-eye | trans RMSE mm | rot RMSE deg |",
        "|---:|---|---|---|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['rank']} | {row['vio_direction']} | {row['camera_imu_input_meaning']} | "
            f"{row['handeye_input_meaning']} | {row['ape_translation_rmse_mm']:.3f} | "
            f"{row['ape_rotation_rmse_deg']:.3f} |"
        )
    report.append("")
    report.append("A valid fix should improve rotation while keeping translation in the same centimeter-level range.")
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"[OK] wrote {output_dir / 'convention_scan.csv'}")
    print(f"[OK] wrote {output_dir / 'REPORT.md'}")
    print(
        "best:",
        best["vio_direction"],
        best["camera_imu_input_meaning"],
        best["handeye_input_meaning"],
        f"trans={best['ape_translation_rmse_mm']:.3f}mm",
        f"rot={best['ape_rotation_rmse_deg']:.3f}deg",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
