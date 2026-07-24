#!/usr/bin/env python3
"""
Evaluate a VIO IMU trajectory against robot TCP ground truth with evo.

Coordinate convention used here:
  * robot JSON pose is T_base_tcp, i.e. TCP child pose in base parent frame.
  * T_tcp_camera is camera child pose in TCP parent frame.
  * VIO CSV pose is usually T_world_imu for raw-pose episodes, or
    T_world_base_link when the estimate-frame is vins_base_link.
  * T_camera_imu is IMU child pose in cam0 parent frame.

Therefore:
  * ground truth TCP pose: T_base_tcp from robot JSON.
  * estimated TCP pose:
      T_world_tcp = T_world_imu
                    @ inverse(T_camera_imu)
                    @ inverse(T_tcp_camera)

    If --estimate-frame camera is used, the CSV is already T_world_camera
    and the script uses:
      T_world_tcp = T_world_camera @ inverse(T_tcp_camera)

evo --align then estimates the rigid transform from VINS world to robot base
and evaluates both TCP trajectories in the same aligned frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_HELPERS = REPO_ROOT / "script/evaluate_vins_accuracy.py"
DEFAULT_ESTIMATE = Path("/home/chenlvping/0614 _test/episode_20260614_0239/right/pose_data.csv")
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data/ground_truth/trajectory_samples0614/trajectory_001.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp"
DEFAULT_HAND_EYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"

T_TCP_CAMERA_DEFAULT = np.array(
    [
        [0.854672738, -0.422689316, 0.301443615, 0.020087823],
        [0.519166541, 0.694970001, -0.497476432, -0.097333617],
        [0.000783703, 0.581678983, 0.813418064, 0.052300458],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

T_CAMERA_IMU_DEFAULT = np.array(
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


def load_helpers() -> Dict[str, object]:
    return runpy.run_path(str(EVAL_HELPERS), run_name="__vio_tcp_camera_eval__")


def load_tcp_camera_transform(path: Path | None) -> np.ndarray:
    if path is None:
        return T_TCP_CAMERA_DEFAULT.copy()
    if yaml is None:
        raise RuntimeError("PyYAML is required for --handeye-yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    matrix = np.asarray(data["result"]["T_cam_to_gripper"], dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{path}: result.T_cam_to_gripper must be 4x4")
    return matrix


def load_camera_imu_transform(calibration_json: Path | None, camera_rig: str | None) -> Tuple[np.ndarray, str, str]:
    if calibration_json is None:
        return T_CAMERA_IMU_DEFAULT.copy(), "built-in default", "cam0"

    data = json.loads(calibration_json.read_text(encoding="utf-8"))
    if camera_rig is None:
        raise ValueError("--camera-rig is required when --calibration-json is used")
    try:
        matrix = np.asarray(data["observation"]["images"][camera_rig]["extrinsics"]["T_ic_cam0_to_imu0"], dtype=float)
    except KeyError as exc:
        raise KeyError(f"{calibration_json}: missing observation.images.{camera_rig}.extrinsics.T_ic_cam0_to_imu0") from exc
    if matrix.shape != (4, 4):
        raise ValueError(f"{calibration_json}: {camera_rig} T_ic_cam0_to_imu0 must be 4x4")
    return matrix, f"{calibration_json}::{camera_rig}.cam0", f"{camera_rig}.cam0"


def _pose_position_xyz(sample: dict) -> List[float]:
    pos = sample.get("position_m", sample.get("position", sample.get("pos")))
    if isinstance(pos, dict):
        lower = {str(k).strip().lower(): v for k, v in pos.items()}
        for keys in (("x", "y", "z"), ("px", "py", "pz")):
            if all(k in lower for k in keys):
                return [float(lower[k]) for k in keys]
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    raise KeyError("position_m")


def _pose_quaternion_xyzw(sample: dict) -> List[float]:
    quat = (
        sample.get("quaternion_xyzw")
        or sample.get("quaternion")
        or sample.get("quat_xyzw")
        or sample.get("quaternion_wxyz")
        or sample.get("quat_wxyz")
    )
    if isinstance(quat, dict):
        lower = {str(k).strip().lower(): v for k, v in quat.items()}
        if all(k in lower for k in ("x", "y", "z", "w")):
            return [float(lower["x"]), float(lower["y"]), float(lower["z"]), float(lower["w"])]
        if all(k in lower for k in ("w", "x", "y", "z")):
            return [float(lower["x"]), float(lower["y"]), float(lower["z"]), float(lower["w"])]
    if isinstance(quat, (list, tuple)) and len(quat) >= 4:
        # Accept either xyzw or wxyz by assuming scalar-last when the source is a list.
        if abs(float(quat[0])) > 0.5 and abs(float(quat[3])) <= 1.0:
            return [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]
        return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
    raise KeyError("quaternion_xyzw")


def load_robot_tcp_trajectory(path: Path, helpers: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples", data)
    times: List[float] = []
    poses: List[np.ndarray] = []
    for sample in samples:
        timestamp = sample.get("timestamp", sample.get("timestamp_s", sample.get("timestamp_us")))
        if timestamp is None:
            raise KeyError("timestamp")
        t_base_tcp = helpers["transform_from_pose"](_pose_position_xyz(sample), _pose_quaternion_xyzw(sample))
        times.append(normalize_timestamp(float(timestamp)))
        poses.append(t_base_tcp)
    order = np.argsort(np.asarray(times, dtype=float))
    poses_arr = np.asarray(poses, dtype=float)[order]
    return np.asarray(times, dtype=float)[order], poses_arr[:, :3, 3], poses_arr[:, :3, :3]


def first_existing(row: dict, names: Sequence[str]) -> Optional[str]:
    lower = {str(k).strip().lower(): k for k in row.keys()}
    for name in names:
        key = lower.get(name.lower())
        if key is not None and row.get(key) not in (None, ""):
            return str(row[key])
    return None


def load_vio_tcp_trajectory(
    path: Path,
    helpers: Dict[str, object],
    t_tcp_camera: np.ndarray,
    t_camera_imu: np.ndarray,
    estimate_frame: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_imu_camera = np.linalg.inv(t_camera_imu)
    t_camera_tcp = np.linalg.inv(t_tcp_camera)
    times: List[float] = []
    poses: List[np.ndarray] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            ts = first_existing(row, ["Timestamp_us", "timestamp_us", "Timestamp_ns", "timestamp_ns", "timestamp", "time", "t"])
            if ts is None:
                ts = str(idx / 30.0)
            xyz = [first_existing(row, names) for names in (["X", "x"], ["Y", "y"], ["Z", "z"])]
            quat = [
                first_existing(row, ["Quat_X", "qx", "q_x"]),
                first_existing(row, ["Quat_Y", "qy", "q_y"]),
                first_existing(row, ["Quat_Z", "qz", "q_z"]),
                first_existing(row, ["Quat_W", "qw", "q_w"]),
            ]
            if any(v is None for v in xyz + quat):
                continue
            t_world_estimate = helpers["transform_from_pose"]([float(v) for v in xyz], [float(v) for v in quat])
            if estimate_frame == "imu":
                t_world_imu = t_world_estimate
                t_world_tcp = t_world_imu @ t_imu_camera @ t_camera_tcp
            elif estimate_frame == "vins_base_link":
                t_world_imu = t_world_estimate @ T_VINS_BASE_LINK_IMU
                t_world_tcp = t_world_imu @ t_imu_camera @ t_camera_tcp
            elif estimate_frame == "camera":
                t_world_camera = t_world_estimate
                t_world_tcp = t_world_camera @ t_camera_tcp
            else:
                raise ValueError(f"unsupported estimate frame {estimate_frame}")
            times.append(normalize_timestamp(float(ts)))
            poses.append(t_world_tcp)
    order = np.argsort(np.asarray(times, dtype=float))
    poses_arr = np.asarray(poses, dtype=float)[order]
    return np.asarray(times, dtype=float)[order], poses_arr[:, :3, 3], poses_arr[:, :3, :3]


def slerp_rotations(rotations: np.ndarray, helpers: Dict[str, object], left: int, alpha: float) -> np.ndarray:
    q0 = helpers["rot_to_quat_xyzw"](rotations[left])
    q1 = helpers["rot_to_quat_xyzw"](rotations[left + 1])
    q = helpers["slerp"](q0, q1, alpha)
    return helpers["quat_xyzw_to_rot"](q)


def match_ground_truth(
    gt_times: np.ndarray,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_times: np.ndarray,
    est_pos: np.ndarray,
    est_rot: np.ndarray,
    helpers: Dict[str, object],
    max_gap_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out_times: List[float] = []
    out_gt_pos: List[np.ndarray] = []
    out_gt_rot: List[np.ndarray] = []
    out_est_pos: List[np.ndarray] = []
    out_est_rot: List[np.ndarray] = []
    left = 0
    for idx, t in enumerate(est_times):
        while left + 1 < gt_times.size and gt_times[left + 1] <= t:
            left += 1
        if left + 1 >= gt_times.size or t < gt_times[left]:
            continue
        t0, t1 = gt_times[left], gt_times[left + 1]
        if max(abs(t - t0), abs(t1 - t)) > max_gap_s:
            continue
        alpha = 0.0 if abs(t1 - t0) <= 1e-12 else float((t - t0) / (t1 - t0))
        out_times.append(float(t))
        out_gt_pos.append((1.0 - alpha) * gt_pos[left] + alpha * gt_pos[left + 1])
        out_gt_rot.append(slerp_rotations(gt_rot, helpers, left, alpha))
        out_est_pos.append(est_pos[idx])
        out_est_rot.append(est_rot[idx])
    if not out_times:
        raise RuntimeError("no timestamp overlap between VIO and robot trajectory")
    return (
        np.asarray(out_times, dtype=float),
        np.asarray(out_gt_pos, dtype=float),
        np.asarray(out_gt_rot, dtype=float),
        np.asarray(out_est_pos, dtype=float),
        np.asarray(out_est_rot, dtype=float),
    )


def associate_by_nearest_time(
    ref_times: np.ndarray,
    ref_pos: np.ndarray,
    ref_rot: np.ndarray,
    est_times: np.ndarray,
    est_pos: np.ndarray,
    est_rot: np.ndarray,
    time_offset_s: float,
    max_diff_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shifted_est = np.asarray(est_times, dtype=float) + float(time_offset_s)
    out_times: List[float] = []
    out_ref_pos: List[np.ndarray] = []
    out_ref_rot: List[np.ndarray] = []
    out_est_pos: List[np.ndarray] = []
    out_est_rot: List[np.ndarray] = []

    est_start = 0
    for ref_idx, ref_t in enumerate(ref_times):
        insert_at = int(np.searchsorted(shifted_est[est_start:], ref_t, side="left")) + est_start
        candidates: List[int] = []
        if insert_at < shifted_est.size:
            candidates.append(insert_at)
        if insert_at - 1 >= est_start:
            candidates.append(insert_at - 1)
        if not candidates:
            continue
        best_est = min(candidates, key=lambda idx: abs(shifted_est[idx] - ref_t))
        if abs(shifted_est[best_est] - ref_t) > max_diff_s:
            continue
        out_times.append(float(ref_t))
        out_ref_pos.append(ref_pos[ref_idx])
        out_ref_rot.append(ref_rot[ref_idx])
        out_est_pos.append(est_pos[best_est])
        out_est_rot.append(est_rot[best_est])
        est_start = best_est + 1

    if not out_times:
        raise RuntimeError("no timestamp association between reference and estimate trajectory")
    return (
        np.asarray(out_times, dtype=float),
        np.asarray(out_ref_pos, dtype=float),
        np.asarray(out_ref_rot, dtype=float),
        np.asarray(out_est_pos, dtype=float),
        np.asarray(out_est_rot, dtype=float),
    )


def write_tum(path: Path, times: np.ndarray, positions: np.ndarray, rotations: np.ndarray, helpers: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for t, pos, rot in zip(times, positions, rotations):
            q = helpers["rot_to_quat_xyzw"](rot)
            handle.write(
                f"{t:.9f} {pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )


def run_cmd(cmd: Sequence[str], log_path: Path) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nsee {log_path}")
    return proc.stdout.strip()


def parse_evo_stdout(text: str) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for raw in text.splitlines():
        parts = raw.strip().split()
        if len(parts) == 2:
            try:
                stats[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return stats


def matrix_text(matrix: np.ndarray) -> str:
    rows = []
    for row in np.asarray(matrix, dtype=float):
        rows.append("[" + ", ".join(f"{value: .9f}" for value in row) + "]")
    return "[\n  " + ",\n  ".join(rows) + "\n]"


def evo_command_text(command: Sequence[str]) -> str:
    return " ".join(f'"{item}"' if " " in item else item for item in command)


def compute_internal_se3_metrics(gt_pos, gt_rot, est_pos, est_rot, times, helpers: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    se3, _, _, trans_errors, rot_errors = helpers["evaluate_alignment"](
        "se3", False, gt_pos, gt_rot, est_pos, est_rot, times, 1.0, 30
    )
    return {
        "translation_m": se3.translation_metrics_m,
        "rotation_deg": se3.rotation_metrics_deg or {},
        "rpe_translation_m": se3.rpe_translation_metrics_m or {},
        "rpe_rotation_deg": se3.rpe_rotation_metrics_deg or {},
        "drift": se3.drift,
    }


def default_time_offset_guess(gt_times: np.ndarray, est_times: np.ndarray) -> float:
    if gt_times.size == 0 or est_times.size == 0:
        return 0.0
    start_guess = float(gt_times[0] - est_times[0])
    end_guess = float(gt_times[-1] - est_times[-1])
    return 0.5 * (start_guess + end_guess)


def scan_evo_time_offset(
    gt_times: np.ndarray,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_times: np.ndarray,
    est_pos: np.ndarray,
    est_rot: np.ndarray,
    helpers: Dict[str, object],
    center_offset_s: float,
    span_s: float,
    step_s: float,
    max_diff_s: float,
    score_metric: str,
    min_samples: int,
) -> Tuple[float, List[Dict[str, float]]]:
    offsets = np.arange(center_offset_s - span_s, center_offset_s + span_s + 0.5 * step_s, step_s)
    rows: List[Dict[str, float]] = []
    best: Optional[Tuple[float, int, float]] = None

    for offset in offsets:
        try:
            assoc_times, assoc_gt_pos, assoc_gt_rot, assoc_est_pos, assoc_est_rot = associate_by_nearest_time(
                gt_times,
                gt_pos,
                gt_rot,
                est_times,
                est_pos,
                est_rot,
                float(offset),
                max_diff_s,
            )
        except RuntimeError:
            continue
        if assoc_times.size < min_samples:
            continue
        se3, _, _, _, _ = helpers["evaluate_alignment"](
            "se3", False, assoc_gt_pos, assoc_gt_rot, assoc_est_pos, assoc_est_rot, assoc_times, 1.0, 30
        )
        trans_rmse_mm = float(se3.translation_metrics_m["rmse"] * 1000.0)
        rot_rmse_deg = float(se3.rotation_metrics_deg["rmse"]) if se3.rotation_metrics_deg else float("nan")
        duration_s = float(assoc_times[-1] - assoc_times[0]) if assoc_times.size > 1 else 0.0
        row = {
            "offset_s": float(offset),
            "samples": int(assoc_times.size),
            "duration_s": duration_s,
            "translation_rmse_mm": trans_rmse_mm,
            "rotation_rmse_deg": rot_rmse_deg,
            "first_time_s": float(assoc_times[0]),
            "last_time_s": float(assoc_times[-1]),
        }
        rows.append(row)
        score_value = trans_rmse_mm if score_metric == "translation" else rot_rmse_deg
        candidate = (score_value, -int(assoc_times.size), float(offset))
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise RuntimeError("time-offset auto scan found no valid association window")
    return best[2], rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, default=DEFAULT_ESTIMATE)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--handeye-yaml",
        type=Path,
        default=DEFAULT_HAND_EYE,
        help="hand-eye YAML. result.T_cam_to_gripper is used as T_tcp_camera for this TCP-chain check.",
    )
    parser.add_argument("--calibration-json", type=Path, default=None, help="Episode calibration.json used to resolve rig-specific cam0->imu0 extrinsics")
    parser.add_argument("--camera-rig", type=str, default=None, help="Rig key under calibration observation.images, e.g. stereo_right")
    parser.add_argument(
        "--estimate-frame",
        choices=["imu", "vins_base_link", "camera"],
        default="imu",
        help="semantic frame stored in the estimate CSV. Use camera for cam0 pose CSV, imu for old 0614 chain, or vins_base_link for VINS-Fusion base_link output.",
    )
    parser.add_argument(
        "--time-association",
        choices=["interpolate", "evo"],
        default="interpolate",
        help="interpolate: interpolate GT to estimate timestamps before exporting TUM; evo: export raw timestamps and let evo associate with --t_offset/--t_max_diff.",
    )
    parser.add_argument("--max-time-gap-ms", type=float, default=80.0)
    parser.add_argument("--time-offset-sec", type=float, default=0.0, help="Offset added to estimate timestamps for evo-style association")
    parser.add_argument("--t-max-diff-sec", type=float, default=0.01, help="Max timestamp difference for evo-style association")
    parser.add_argument("--time-offset-auto", action="store_true", help="Auto-scan the best evo time offset around a center value")
    parser.add_argument("--time-offset-auto-span-sec", type=float, default=2.0, help="Search +/- span around the auto-scan center")
    parser.add_argument("--time-offset-auto-step-sec", type=float, default=0.01, help="Step size for auto-scan")
    parser.add_argument(
        "--time-offset-auto-score",
        choices=["translation", "rotation"],
        default="translation",
        help="Metric minimized during evo time-offset auto scan",
    )
    parser.add_argument("--time-offset-auto-min-samples", type=int, default=50, help="Minimum associated samples required for a scan candidate")
    parser.add_argument("--rpe-distance-m", type=float, default=0.05)
    args = parser.parse_args()

    estimate_path = args.estimate.expanduser().resolve()
    gt_path = args.ground_truth.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    helpers = load_helpers()
    t_tcp_camera = load_tcp_camera_transform(args.handeye_yaml.expanduser().resolve() if args.handeye_yaml else None)
    calibration_json = args.calibration_json.expanduser().resolve() if args.calibration_json else None
    t_camera_imu, camera_imu_source, camera_label = load_camera_imu_transform(calibration_json, args.camera_rig)
    gt_times, gt_pos_all, gt_rot_all = load_robot_tcp_trajectory(gt_path, helpers)
    est_times, est_pos_all, est_rot_all = load_vio_tcp_trajectory(
        estimate_path,
        helpers,
        t_tcp_camera,
        t_camera_imu,
        args.estimate_frame,
    )
    auto_time_offset: Optional[Dict[str, object]] = None
    if args.time_association == "evo" and args.time_offset_auto:
        center_offset_s = args.time_offset_sec
        if abs(center_offset_s) <= 1e-12:
            center_offset_s = default_time_offset_guess(gt_times, est_times)
        best_offset_s, scan_rows = scan_evo_time_offset(
            gt_times,
            gt_pos_all,
            gt_rot_all,
            est_times,
            est_pos_all,
            est_rot_all,
            helpers,
            center_offset_s,
            float(args.time_offset_auto_span_sec),
            float(args.time_offset_auto_step_sec),
            float(args.t_max_diff_sec),
            args.time_offset_auto_score,
            int(args.time_offset_auto_min_samples),
        )
        args.time_offset_sec = float(best_offset_s)
        auto_time_offset = {
            "enabled": True,
            "center_offset_s": float(center_offset_s),
            "chosen_offset_s": float(best_offset_s),
            "span_s": float(args.time_offset_auto_span_sec),
            "step_s": float(args.time_offset_auto_step_sec),
            "score": args.time_offset_auto_score,
            "min_samples": int(args.time_offset_auto_min_samples),
            "rows": scan_rows,
        }

    if args.time_association == "interpolate":
        times, gt_pos, gt_rot, est_pos, est_rot = match_ground_truth(
            gt_times,
            gt_pos_all,
            gt_rot_all,
            est_times,
            est_pos_all,
            est_rot_all,
            helpers,
            args.max_time_gap_ms * 1e-3,
        )
        tum_gt_times = times
        tum_est_times = times
    else:
        times, gt_pos, gt_rot, est_pos, est_rot = associate_by_nearest_time(
            gt_times,
            gt_pos_all,
            gt_rot_all,
            est_times,
            est_pos_all,
            est_rot_all,
            args.time_offset_sec,
            args.t_max_diff_sec,
        )
        tum_gt_times = gt_times
        tum_est_times = est_times
    associated_sample_count = int(times.size)
    associated_duration_s = float(times[-1] - times[0]) if times.size > 1 else 0.0

    gt_tum = output_dir / "gt_tcp.tum"
    est_tum = output_dir / "vio_tcp_from_imu_cam0.tum"
    gt_matched_tum = output_dir / "gt_tcp_matched.tum"
    est_matched_tum = output_dir / "vio_tcp_matched.tum"
    if args.time_association == "interpolate":
        write_tum(gt_tum, tum_gt_times, gt_pos, gt_rot, helpers)
        write_tum(est_tum, tum_est_times, est_pos, est_rot, helpers)
    else:
        write_tum(gt_tum, tum_gt_times, gt_pos_all, gt_rot_all, helpers)
        write_tum(est_tum, tum_est_times, est_pos_all, est_rot_all, helpers)
    write_tum(gt_matched_tum, times, gt_pos, gt_rot, helpers)
    write_tum(est_matched_tum, times, est_pos, est_rot, helpers)

    evo_ape = str(Path.home() / ".local/bin/evo_ape")
    evo_rpe = str(Path.home() / ".local/bin/evo_rpe")
    common = ["tum", str(gt_tum), str(est_tum), "--no_warnings", "-a"]
    if args.time_association == "evo":
        common.extend(["--t_offset", str(args.time_offset_sec), "--t_max_diff", str(args.t_max_diff_sec)])
    commands = [
        (
            "ape_translation_se3",
            [evo_ape, *common, "-r", "trans_part", "--change_unit", "mm", "--save_results", str(output_dir / "ape_translation_se3.zip")],
        ),
        (
            "ape_rotation_se3",
            [evo_ape, *common, "-r", "angle_deg", "--save_results", str(output_dir / "ape_rotation_se3.zip")],
        ),
        (
            "ape_translation_sim3",
            [evo_ape, *common, "-s", "-r", "trans_part", "--change_unit", "mm", "--save_results", str(output_dir / "ape_translation_sim3.zip")],
        ),
        (
            "rpe_translation_5cm",
            [
                evo_rpe,
                *common,
                "-r",
                "trans_part",
                "-d",
                str(args.rpe_distance_m),
                "-u",
                "m",
                "--pairs_from_reference",
                "--change_unit",
                "mm",
                "--save_results",
                str(output_dir / "rpe_translation_5cm.zip"),
            ],
        ),
        (
            "rpe_rotation_5cm",
            [
                evo_rpe,
                *common,
                "-r",
                "angle_deg",
                "-d",
                str(args.rpe_distance_m),
                "-u",
                "m",
                "--pairs_from_reference",
                "--save_results",
                str(output_dir / "rpe_rotation_5cm.zip"),
            ],
        ),
    ]

    outputs: Dict[str, str] = {}
    command_texts: Dict[str, str] = {}
    for name, command in commands:
        command_texts[name] = evo_command_text(command)
        try:
            outputs[name] = run_cmd(command, log_dir / f"{name}.log")
        except RuntimeError:
            if name.startswith("rpe_"):
                print(f"[WARNING] RPE skipped ({name})", flush=True)
                outputs[name] = "[skipped]"
                continue
            raise

    internal = compute_internal_se3_metrics(gt_pos, gt_rot, est_pos, est_rot, times, helpers)
    payload = {
        "estimate": str(estimate_path),
        "ground_truth": str(gt_path),
        "gt_tum": str(gt_tum),
        "estimate_tum": str(est_tum),
        "matched_gt_tum": str(gt_matched_tum),
        "matched_estimate_tum": str(est_matched_tum),
        "matched_samples": associated_sample_count,
        "matched_duration_s": associated_duration_s,
        "evo": {name: parse_evo_stdout(text) for name, text in outputs.items()},
        "evo_commands": command_texts,
        "internal_se3": internal,
        "transforms": {
            "T_tcp_camera": t_tcp_camera.tolist(),
            "T_camera_imu": t_camera_imu.tolist(),
            "T_imu_camera": np.linalg.inv(t_camera_imu).tolist(),
            "T_camera_tcp": np.linalg.inv(t_tcp_camera).tolist(),
            "T_vins_base_link_imu": T_VINS_BASE_LINK_IMU.tolist(),
        },
        "camera_label": camera_label,
        "camera_imu_source": camera_imu_source,
        "camera_rig": args.camera_rig or "",
        "calibration_json": str(calibration_json) if calibration_json else "",
        "estimate_frame": args.estimate_frame,
        "time_association": args.time_association,
        "time_offset_sec": args.time_offset_sec,
        "t_max_diff_sec": args.t_max_diff_sec,
        "time_offset_auto": auto_time_offset,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if auto_time_offset:
        scan_rows = auto_time_offset["rows"]
        with (output_dir / "time_offset_scan.csv").open("w", encoding="utf-8") as handle:
            handle.write("offset_s,samples,duration_s,translation_rmse_mm,rotation_rmse_deg,first_time_s,last_time_s\n")
            for row in scan_rows:
                handle.write(
                    f"{row['offset_s']:.9f},{row['samples']},{row['duration_s']:.9f},"
                    f"{row['translation_rmse_mm']:.6f},{row['rotation_rmse_deg']:.6f},"
                    f"{row['first_time_s']:.9f},{row['last_time_s']:.9f}\n"
                )
        (output_dir / "time_offset_scan.json").write_text(json.dumps(scan_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output_dir / "summary.csv").open("w", encoding="utf-8") as handle:
        handle.write("metric,rmse,mean,median,min,max,std,unit\n")
        for name, stats in payload["evo"].items():
            unit = "mm" if "translation" in name else "deg"
            handle.write(
                f"{name},{stats.get('rmse', float('nan')):.6f},{stats.get('mean', float('nan')):.6f},"
                f"{stats.get('median', float('nan')):.6f},{stats.get('min', float('nan')):.6f},"
                f"{stats.get('max', float('nan')):.6f},{stats.get('std', float('nan')):.6f},{unit}\n"
            )

    with (output_dir / "REPORT.md").open("w", encoding="utf-8") as handle:
        handle.write("# VIO TCP Trajectory Accuracy Evaluation\n\n")
        handle.write("## Actual Inputs And Outputs\n\n")
        handle.write(f"- VIO trajectory: `{estimate_path}`\n")
        handle.write(f"- VIO estimate frame: `{args.estimate_frame}`\n")
        handle.write(f"- Robot ground truth: `{gt_path}`\n")
        handle.write(f"- Output directory: `{output_dir}`\n")
        handle.write(f"- Reference TUM: `{gt_tum}`\n")
        handle.write(f"- Estimated TUM: `{est_tum}`\n\n")
        handle.write(f"- Matched reference TUM: `{gt_matched_tum}`\n")
        handle.write(f"- Matched estimated TUM: `{est_matched_tum}`\n\n")
        handle.write("## Coordinate Chain\n\n")
        if args.estimate_frame == "camera":
            handle.write(f"The VIO CSV stores the `{camera_label}` pose in the VINS world frame:\n\n")
            handle.write("```text\nT_world_camera\n```\n\n")
        elif args.estimate_frame == "imu":
            handle.write("The VIO CSV stores the IMU pose in the VINS world frame:\n\n")
            handle.write("```text\nT_world_imu\n```\n\n")
        else:
            handle.write("The VIO CSV stores VINS-Fusion `base_link` pose in the VINS world frame. The script first recovers IMU pose:\n\n")
            handle.write("```text\nT_world_imu = T_world_base_link @ T_base_link_imu\n```\n\n")
        handle.write("The robot JSON stores the TCP pose in the robot base frame:\n\n")
        handle.write("```text\nT_base_tcp\n```\n\n")
        handle.write(f"The hand-eye calibration is used as `T_tcp_camera`, with `tcp` as parent and `{camera_label}` as child:\n\n")
        handle.write("```text\nT_tcp_camera =\n")
        handle.write(matrix_text(t_tcp_camera))
        handle.write("\n```\n\n")
        handle.write(
            f"The camera-IMU extrinsic is used as `T_camera_imu`, with `{camera_label}` as parent and `imu` as child.\n"
            f"Source: `{camera_imu_source}`\n\n"
        )
        handle.write("```text\nT_camera_imu =\n")
        handle.write(matrix_text(t_camera_imu))
        handle.write("\n```\n\n")
        if args.estimate_frame == "vins_base_link":
            handle.write("The VINS-Fusion `base_link` to IMU transform is:\n\n")
            handle.write("```text\nT_base_link_imu =\n")
            handle.write(matrix_text(T_VINS_BASE_LINK_IMU))
            handle.write("\n```\n\n")
        handle.write("Therefore the VIO-derived TCP trajectory is:\n\n")
        handle.write("```text\n")
        if args.estimate_frame == "camera":
            handle.write("T_world_tcp     = T_world_camera @ inverse(T_tcp_camera)\n")
        else:
            if args.estimate_frame == "vins_base_link":
                handle.write("T_world_imu     = T_world_base_link @ T_base_link_imu\n")
            handle.write("T_world_camera  = T_world_imu @ inverse(T_camera_imu)\n")
            handle.write("T_world_tcp     = T_world_camera @ inverse(T_tcp_camera)\n")
            handle.write("                = T_world_imu @ inverse(T_camera_imu) @ inverse(T_tcp_camera)\n")
        handle.write("```\n\n")
        handle.write("The robot TCP trajectory is already:\n\n")
        handle.write("```text\nT_base_tcp\n```\n\n")
        handle.write("evo `--align` estimates the rigid transform from VINS world to robot base, then computes the TCP trajectory error in one common frame.\n\n")
        handle.write("## Time Association\n\n")
        if args.time_association == "interpolate":
            handle.write("The robot trajectory is interpolated to the VIO timestamps before exporting the TUM files.\n")
            handle.write(f"The interpolation bracket threshold is `{args.max_time_gap_ms:.1f} ms`.\n\n")
        else:
            handle.write("Both TCP trajectories keep their raw timestamps in the exported TUM files.\n")
            handle.write("evo performs the timestamp association directly using the configured time offset and max difference.\n\n")
            if auto_time_offset:
                handle.write("- Time offset was selected automatically by scan.\n")
                handle.write(f"- Auto-scan center: `{auto_time_offset['center_offset_s']:.9f} s`\n")
                handle.write(f"- Auto-scan span: `+/- {auto_time_offset['span_s']:.9f} s`\n")
                handle.write(f"- Auto-scan step: `{auto_time_offset['step_s']:.9f} s`\n")
                handle.write(f"- Auto-scan score: `{auto_time_offset['score']}`\n")
                handle.write(f"- Auto-scan min samples: `{auto_time_offset['min_samples']}`\n")
                handle.write("- Auto-scan files: `time_offset_scan.csv`, `time_offset_scan.json`\n\n")
            handle.write(f"- evo `--t_offset`: `{args.time_offset_sec:.9f} s`\n")
            handle.write(f"- evo `--t_max_diff`: `{args.t_max_diff_sec:.9f} s`\n\n")
        handle.write(f"- Matched samples: {times.size}\n")
        handle.write(f"- Matched duration: {payload['matched_duration_s']:.3f} s\n")
        handle.write(f"- First matched timestamp: {float(times[0]):.9f} s\n")
        handle.write(f"- Last matched timestamp: {float(times[-1]):.9f} s\n\n")
        handle.write("## Evo Commands\n\n")
        for name, command in command_texts.items():
            handle.write(f"### {name}\n\n")
            handle.write("```bash\n")
            handle.write(command)
            handle.write("\n```\n\n")
        handle.write("## Results\n\n")
        handle.write("| metric | RMSE | mean | median | min | max | std | unit |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for name, stats in payload["evo"].items():
            unit = "mm" if "translation" in name else "deg"
            handle.write(
                f"| {name} | {stats.get('rmse', float('nan')):.3f} | {stats.get('mean', float('nan')):.3f} | "
                f"{stats.get('median', float('nan')):.3f} | {stats.get('min', float('nan')):.3f} | "
                f"{stats.get('max', float('nan')):.3f} | {stats.get('std', float('nan')):.3f} | {unit} |\n"
            )
        handle.write("\n## Notes\n\n")
        handle.write("- `ape_translation_se3` is absolute TCP position error after SE(3) alignment, without scale correction.\n")
        handle.write("- `ape_translation_sim3` adds evo scale correction and is useful for diagnosing scale drift.\n")
        handle.write("- `ape_rotation_se3` is absolute TCP attitude error after the same global SE(3) alignment.\n")
        handle.write("- `rpe_translation_5cm` and `rpe_rotation_5cm` measure local relative motion over a 5 cm reference-trajectory interval.\n")
        handle.write("- Large rotation error with centimeter-level translation usually means orientation frame convention still deserves a separate check.\n")

    for name, text in outputs.items():
        (output_dir / f"{name}.txt").write_text(text + "\n", encoding="utf-8")

    print(f"[OK] wrote {output_dir / 'summary.csv'}")
    print(f"[OK] wrote {output_dir / 'REPORT.md'}")
    print(f"[OK] wrote {output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
