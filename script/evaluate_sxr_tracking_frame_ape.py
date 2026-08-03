#!/usr/bin/env python3
"""Compare an SXR tracking-VIO trajectory at calibrated rig reference points.

Basalt exports the tracking-left-camera trajectory.  The device calibration
places tracking, RGB, and controller cameras under the same IMU.  This tool
uses those rigid transforms to evaluate the same VIO result at each available
physical reference point against ``head_pose.csv``.

The conversion is a right-side body-frame transform, so it is not redundant
with evo's global SE(3) alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from run_sxr_basalt import matrix_to_xyzw
from run_sxr_orb_stereo_baseline import build_viewer, evaluate


FRAME_CHOICES = ("tracking_cam0", "imu", "rgb_cam0", "ctrl_cam0")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path, help="Basalt tracking run directory.")
    parser.add_argument(
        "--target-frames",
        nargs="+",
        choices=FRAME_CHOICES,
        default=list(FRAME_CHOICES),
        help="Calibrated reference points to compare. Defaults to all candidates.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--camera-quaternion-order",
        choices=("wxyz", "xyzw"),
        default="wxyz",
        help="Quaternion order in current SXR CSV camera_params files.",
    )
    parser.add_argument(
        "--normalize-start",
        action="store_true",
        help="Also export a comparison where each trajectory's first pose is the identity.",
    )
    return parser.parse_args(argv)


def inverse(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return result


def quat_to_matrix(values: Sequence[float], order: str) -> np.ndarray:
    if order == "xyzw":
        x, y, z, w = (float(value) for value in values)
    else:
        w, x, y, z = (float(value) for value in values)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def matrix_from_xyzw_pose(pose: dict[str, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_to_matrix((pose["qx"], pose["qy"], pose["qz"], pose["qw"]), "xyzw")
    transform[:3, 3] = (float(pose["px"]), float(pose["py"]), float(pose["pz"]))
    return transform


def matrix_from_camera(camera: dict, order: str) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_to_matrix(camera["extrinsics"]["rotation"], order)
    transform[:3, 3] = np.asarray(camera["extrinsics"]["position"], dtype=np.float64)
    return transform


def tum_pose(line: str) -> tuple[float, np.ndarray] | None:
    fields = line.split()
    if len(fields) != 8 or fields[0].startswith("#"):
        return None
    timestamp, x, y, z, qx, qy, qz, qw = (float(value) for value in fields)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_to_matrix((qx, qy, qz, qw), "xyzw")
    transform[:3, 3] = (x, y, z)
    return timestamp, transform


def read_tum(path: Path) -> Iterable[tuple[float, np.ndarray]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            decoded = tum_pose(line)
            if decoded is not None:
                yield decoded


def write_tum(path: Path, poses: Iterable[tuple[float, np.ndarray]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for timestamp, transform in poses:
            qx, qy, qz, qw = matrix_to_xyzw(transform)
            handle.write(
                f"{timestamp:.9f} {transform[0, 3]:.9f} {transform[1, 3]:.9f} {transform[2, 3]:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )
            count += 1
    return count


def write_head_pose_reference(source: Path, destination: Path) -> int:
    count = 0
    with source.open(newline="", encoding="utf-8") as input_handle, destination.open("w", encoding="utf-8") as output_handle:
        for row in csv.DictReader(input_handle):
            output_handle.write(
                f"{int(row['timestamp_ns']) * 1e-9:.9f} {float(row['pos_x']):.9f} {float(row['pos_y']):.9f} {float(row['pos_z']):.9f} "
                f"{float(row['quat_x']):.9f} {float(row['quat_y']):.9f} {float(row['quat_z']):.9f} {float(row['quat_w']):.9f}\n"
            )
            count += 1
    return count


def normalize_tum_start(source: Path, destination: Path) -> int:
    poses = list(read_tum(source))
    if len(poses) < 3:
        raise ValueError(f"expected at least three poses in {source}")
    initial_inverse = inverse(poses[0][1])
    return write_tum(destination, ((timestamp, initial_inverse @ transform) for timestamp, transform in poses))


def evaluate_start_aligned(reference: Path, estimate: Path, output: Path) -> None:
    """Evaluate trajectories with coincident first poses and no later SE(3) fit."""
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "evo_ape",
        "tum",
        str(reference),
        str(estimate),
        "--pose_relation",
        "trans_part",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log = output / "ape_translation_start_aligned.log"
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"evo_ape failed; see {log}")


def target_transforms(episode: Path, run_dir: Path, order: str) -> dict[str, np.ndarray]:
    calibration = json.loads((run_dir / "basalt_sxr_csv_kb4_calib.json").read_text(encoding="utf-8"))
    t_i_tracking = matrix_from_xyzw_pose(calibration["value0"]["T_imu_cam"][0])
    transforms = {"tracking_cam0": t_i_tracking, "imu": np.eye(4, dtype=np.float64)}
    for stream, name in (("rgb", "rgb_cam0"), ("ctrl", "ctrl_cam0")):
        factory = json.loads((episode / f"camera_params_{stream}.json").read_text(encoding="utf-8"))
        transforms[name] = matrix_from_camera(factory["cameras"][0], order)
    return transforms


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    episode, run_dir = args.episode_dir.expanduser().resolve(), args.run_dir.expanduser().resolve()
    source = run_dir / "basalt_stereo_vio_cam0.tum"
    if not source.is_file():
        raise FileNotFoundError(f"missing Basalt tracking trajectory: {source}")
    if not (episode / "head_pose.csv").is_file():
        raise FileNotFoundError(f"missing head_pose.csv under {episode}")
    output = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "frame_ape"
    output.mkdir(parents=True, exist_ok=True)
    reference = output / "head_pose_reference.tum"
    reference_count = write_head_pose_reference(episode / "head_pose.csv", reference)
    transforms = target_transforms(episode, run_dir, args.camera_quaternion_order)
    poses = list(read_tum(source))
    if len(poses) < 3:
        raise ValueError(f"expected at least three poses in {source}")
    t_tracking_imu = inverse(transforms["tracking_cam0"])
    report: dict[str, object] = {
        "source_trajectory": str(source),
        "reference": str(reference),
        "source_frame": "tracking_cam0",
        "reference_frame": "device head_pose (spatial transform to IMU is not supplied by recorder)",
        "reference_poses": reference_count,
        "trajectory_poses": len(poses),
        "camera_quaternion_order": args.camera_quaternion_order,
        "formula": "T_world_target = T_world_tracking_cam0 * inverse(T_imu_tracking_cam0) * T_imu_target",
        "targets": {},
    }
    for target in args.target_frames:
        destination = output / f"basalt_stereo_vio_{target}.tum"
        t_i_target = transforms[target]
        count = write_tum(destination, ((stamp, pose @ t_tracking_imu @ t_i_target) for stamp, pose in poses))
        target_output = output / target
        evaluate(reference, destination, target_output / "evaluation")
        build_viewer(
            reference,
            destination,
            target_output / "trajectory_viewer",
        )
        normalized: dict[str, str | int] | None = None
        if args.normalize_start:
            normalized_dir = target_output / "start_normalized"
            normalized_dir.mkdir(parents=True, exist_ok=True)
            normalized_reference = normalized_dir / "head_pose_start_normalized.tum"
            normalized_estimate = normalized_dir / f"basalt_stereo_vio_{target}_start_normalized.tum"
            normalized_reference_count = normalize_tum_start(reference, normalized_reference)
            normalized_estimate_count = normalize_tum_start(destination, normalized_estimate)
            evaluate(normalized_reference, normalized_estimate, normalized_dir / "evaluation")
            evaluate_start_aligned(normalized_reference, normalized_estimate, normalized_dir / "evaluation")
            build_viewer(
                normalized_reference,
                normalized_estimate,
                normalized_dir / "trajectory_viewer",
            )
            normalized = {
                "reference": str(normalized_reference),
                "reference_poses": normalized_reference_count,
                "trajectory": str(normalized_estimate),
                "trajectory_poses": normalized_estimate_count,
                "viewer": str(normalized_dir / "trajectory_viewer/index.html"),
                "ape_start_aligned_log": str(normalized_dir / "evaluation/ape_translation_start_aligned.log"),
            }
        report["targets"][target] = {
            "trajectory": str(destination),
            "poses": count,
            "T_imu_target": transforms[target].tolist(),
            "ape_log": str(target_output / "evaluation/ape_translation.log"),
            "rpe_log": str(target_output / "evaluation/rpe_translation.log"),
            "viewer": str(target_output / "trajectory_viewer/index.html"),
            "start_normalized": normalized,
        }
    (output / "frame_transform_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] evaluated {len(args.target_frames)} calibrated target frames -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
