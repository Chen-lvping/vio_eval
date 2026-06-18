#!/usr/bin/env python3
"""Scan TCP hand-eye rotation offsets while keeping translation fixed."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
TCP_EVAL_PATH = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
DEFAULT_ESTIMATE = REPO_ROOT / "data/gripper_data2/episode_20260618_0003/right/pose_data.csv"
DEFAULT_GT = REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_3.json"
DEFAULT_HANDEYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/scan_tcp_handeye_rotation_rm75_0003"

T_LEFT_CAMERA_IMU = np.array(
    [
        [-0.999638319, 0.0265241228, -0.00445450377, -0.0022427286],
        [-0.0265235156, -0.999648213, -0.000195318818, 0.0140962508],
        [-0.0044581173, -7.70990737e-05, 0.999990046, -0.0155969206],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

T_VINS_BASE_LINK_IMU = np.array(
    [
        [0.0, 0.0, 1.0, 0.038441],
        [1.0, 0.0, 0.0, 0.040052],
        [0.0, 1.0, 0.0, -0.063843],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def load_tcp_eval() -> Dict[str, object]:
    return runpy.run_path(str(TCP_EVAL_PATH), run_name="__scan_tcp_handeye_rotation__")


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


def first_existing(row: dict, names: Sequence[str]) -> str | None:
    lower = {str(k).strip().lower(): k for k in row.keys()}
    for name in names:
        key = lower.get(name.lower())
        if key is not None and row.get(key) not in (None, ""):
            return str(row[key])
    return None


def write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mm(value_m: float) -> float:
    return float(value_m) * 1000.0


def rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rotation_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rotation_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def euler_delta_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(float(rx_deg))
    ry = math.radians(float(ry_deg))
    rz = math.radians(float(rz_deg))
    return rotation_z(rz) @ rotation_y(ry) @ rotation_x(rx)


def parse_offsets_deg(values: Sequence[str]) -> List[float]:
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


def invert_transform(transform: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, :3] = transform[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ transform[:3, 3]
    return out


def load_vio_raw_trajectory(path: Path, helpers: Dict[str, object], estimate_frame: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: List[float] = []
    poses: List[np.ndarray] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            ts = first_existing(row, ["Timestamp_us", "timestamp_us", "Timestamp_ns", "timestamp_ns", "timestamp", "time", "t"])
            xyz = [first_existing(row, names) for names in (["X", "x"], ["Y", "y"], ["Z", "z"])]
            quat = [
                first_existing(row, ["Quat_X", "qx", "q_x"]),
                first_existing(row, ["Quat_Y", "qy", "q_y"]),
                first_existing(row, ["Quat_Z", "qz", "q_z"]),
                first_existing(row, ["Quat_W", "qw", "q_w"]),
            ]
            if any(v is None for v in xyz + quat):
                continue
            pose = helpers["transform_from_pose"]([float(v) for v in xyz], [float(v) for v in quat])
            if estimate_frame == "vins_base_link":
                pose = pose @ T_VINS_BASE_LINK_IMU
            elif estimate_frame not in ("imu", "camera"):
                raise ValueError(estimate_frame)
            times.append(normalize_timestamp(float(ts)) if ts is not None else float(idx))
            poses.append(pose)
    order = np.argsort(np.asarray(times, dtype=float))
    poses_arr = np.asarray(poses, dtype=float)[order]
    return np.asarray(times, dtype=float)[order], poses_arr[:, :3, 3], poses_arr[:, :3, :3]


def build_tcp_trajectory(
    est_pos_raw: np.ndarray,
    est_rot_raw: np.ndarray,
    estimate_frame: str,
    delta_rot: np.ndarray,
    t_tcp_left_camera: np.ndarray,
    helpers: Dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    t_imu_left_camera = T_LEFT_CAMERA_IMU
    delta = np.eye(4, dtype=float)
    delta[:3, :3] = delta_rot
    t_tcp_left_camera = t_tcp_left_camera @ delta
    t_left_camera_tcp = invert_transform(t_tcp_left_camera)
    t_imu_left_camera_inv = invert_transform(t_imu_left_camera)

    pos_list: List[np.ndarray] = []
    rot_list: List[np.ndarray] = []
    for pos, rot in zip(est_pos_raw, est_rot_raw):
        t_world_raw = np.eye(4, dtype=float)
        t_world_raw[:3, :3] = rot
        t_world_raw[:3, 3] = pos
        if estimate_frame == "imu":
            t_world_tcp = t_world_raw @ t_imu_left_camera_inv @ t_left_camera_tcp
        elif estimate_frame == "vins_base_link":
            t_world_tcp = t_world_raw @ T_VINS_BASE_LINK_IMU @ t_imu_left_camera_inv @ t_left_camera_tcp
        elif estimate_frame == "camera":
            t_world_tcp = t_world_raw @ t_left_camera_tcp
        else:
            raise ValueError(estimate_frame)
        pos_list.append(t_world_tcp[:3, 3])
        rot_list.append(t_world_tcp[:3, :3])
    return np.asarray(pos_list, dtype=float), np.asarray(rot_list, dtype=float)


def metrics_for_offset(
    helpers: Dict[str, object],
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_pos_tcp: np.ndarray,
    est_rot_tcp: np.ndarray,
    times: np.ndarray,
) -> Dict[str, object]:
    se3, _, _, trans_errors, rot_errors = helpers["evaluate_alignment"](
        "se3", False, gt_pos, gt_rot, est_pos_tcp, est_rot_tcp, times, 1.0, 30
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


def associate_once(
    tcp_eval: Dict[str, object],
    helpers: Dict[str, object],
    estimate: Path,
    ground_truth: Path,
    estimate_frame: str,
    time_offset_s: float,
    t_max_diff_s: float,
) -> Dict[str, np.ndarray]:
    gt_times, gt_pos_all, gt_rot_all = tcp_eval["load_robot_tcp_trajectory"](ground_truth, helpers)
    est_times, est_pos_all, est_rot_all = load_vio_raw_trajectory(estimate, helpers, estimate_frame)
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
    return {
        "times": times,
        "gt_pos": gt_pos,
        "gt_rot": gt_rot,
        "est_pos": est_pos,
        "est_rot": est_rot,
    }


def build_report(rows: List[Dict[str, object]], baseline: Dict[str, object], best: Dict[str, object], output_dir: Path) -> str:
    lines = [
        "# TCP Hand-Eye Rotation Scan",
        "",
        f"- Baseline APE t RMSE: `{baseline['ape_translation_se3_rmse_mm']:.3f} mm`",
        f"- Baseline APE r RMSE: `{baseline['ape_rotation_se3_rmse_deg']:.3f} deg`",
        f"- Best delta: `({best['delta_rx_deg']:.1f}, {best['delta_ry_deg']:.1f}, {best['delta_rz_deg']:.1f}) deg`",
        f"- Best APE t RMSE: `{best['ape_translation_se3_rmse_mm']:.3f} mm`",
        f"- Best APE r RMSE: `{best['ape_rotation_se3_rmse_deg']:.3f} deg`",
        "",
        "| rank | drx deg | dry deg | drz deg | APE t RMSE mm | APE r RMSE deg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows[:10], start=1):
        lines.append(
            f"| {idx} | {row['delta_rx_deg']:.1f} | {row['delta_ry_deg']:.1f} | {row['delta_rz_deg']:.1f} | "
            f"{row['ape_translation_se3_rmse_mm']:.3f} | {row['ape_rotation_se3_rmse_deg']:.3f} |"
        )
    lines.append(f"\n- CSV: `{output_dir / 'scan_results.csv'}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, default=DEFAULT_ESTIMATE)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--handeye-yaml", type=Path, default=DEFAULT_HANDEYE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--estimate-frame", choices=["imu", "vins_base_link", "camera"], default="imu")
    parser.add_argument("--time-offset-sec", type=float, default=0.027788415)
    parser.add_argument("--t-max-diff-sec", type=float, default=0.01)
    parser.add_argument("--rotation-offsets-deg", nargs="+", default=["1", "2"], help="Absolute offsets in degrees.")
    args = parser.parse_args()

    if yaml is None:
        raise RuntimeError("PyYAML is required for hand-eye rotation scan")

    estimate = args.estimate.expanduser().resolve()
    ground_truth = args.ground_truth.expanduser().resolve()
    handeye_yaml = args.handeye_yaml.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tcp_eval = load_tcp_eval()
    helpers = tcp_eval["load_helpers"]()
    handeye_matrix = tcp_eval["load_tcp_left_camera_transform"](handeye_yaml)
    offsets_deg = parse_offsets_deg(args.rotation_offsets_deg)

    associated = associate_once(
        tcp_eval,
        helpers,
        estimate,
        ground_truth,
        args.estimate_frame,
        float(args.time_offset_sec),
        float(args.t_max_diff_sec),
    )

    rows: List[Dict[str, object]] = []
    for drx_deg in offsets_deg:
        for dry_deg in offsets_deg:
            for drz_deg in offsets_deg:
                delta_rot = euler_delta_matrix(drx_deg, dry_deg, drz_deg)
                est_pos_tcp, est_rot_tcp = build_tcp_trajectory(
                    associated["est_pos"],
                    associated["est_rot"],
                    args.estimate_frame,
                    delta_rot,
                    handeye_matrix,
                    helpers,
                )
                metrics = metrics_for_offset(
                    helpers,
                    associated["gt_pos"],
                    associated["gt_rot"],
                    est_pos_tcp,
                    est_rot_tcp,
                    associated["times"],
                )
                rows.append({"delta_rx_deg": float(drx_deg), "delta_ry_deg": float(dry_deg), "delta_rz_deg": float(drz_deg), **metrics})

    rows.sort(key=lambda row: (row["ape_translation_se3_rmse_mm"], abs(row["ape_rotation_se3_rmse_deg"])))
    baseline = next(r for r in rows if r["delta_rx_deg"] == 0.0 and r["delta_ry_deg"] == 0.0 and r["delta_rz_deg"] == 0.0)
    best = rows[0]

    (output_dir / "scan_results.csv").write_text("", encoding="utf-8")
    write_rows_csv(output_dir / "scan_results.csv", rows)
    (output_dir / "scan_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(rows, baseline, best, output_dir), encoding="utf-8")

    print(f"[BEST] drx={best['delta_rx_deg']:.1f} dry={best['delta_ry_deg']:.1f} drz={best['delta_rz_deg']:.1f} ape_t={best['ape_translation_se3_rmse_mm']:.3f}mm ape_r={best['ape_rotation_se3_rmse_deg']:.3f}deg")
    print(f"[BASELINE] ape_t={baseline['ape_translation_se3_rmse_mm']:.3f}mm ape_r={baseline['ape_rotation_se3_rmse_deg']:.3f}deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
