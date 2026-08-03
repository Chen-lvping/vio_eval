#!/usr/bin/env python3
"""Validate SXR stereo geometry and camera-to-IMU timing for one CSV episode."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import av
import cv2
import numpy as np


OFFSET_KEYS = {"rgb": "rgb-left", "tracking": "trackingA", "ctrl": "ctrl-trackingA"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--stream", choices=tuple(OFFSET_KEYS), default="tracking")
    parser.add_argument("--camera-quaternion-order", choices=("xyzw", "wxyz"), default="xyzw")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=6000)
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args(argv)


def inverse(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return result


def pose(camera: dict[str, Any], order: str) -> np.ndarray:
    """Decode factory T_imu_cam; IMU is parent and camera is child."""
    values = np.asarray(camera["extrinsics"]["rotation"], dtype=np.float64)
    if order == "xyzw":
        x, y, z, w = values
    elif order == "wxyz":
        w, x, y, z = values
    else:
        raise ValueError(f"unsupported quaternion order: {order}")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("zero-norm camera quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    result[:3, 3] = np.asarray(camera["extrinsics"]["position"], dtype=np.float64)
    return result


def kb_unproject(pixel: tuple[float, float], camera: dict[str, Any]) -> np.ndarray:
    intrinsics = camera["intrinsics"]
    x = (pixel[0] - float(intrinsics["centerX"])) / float(intrinsics["focalX"])
    y = (pixel[1] - float(intrinsics["centerY"])) / float(intrinsics["focalY"])
    distorted = math.hypot(x, y)
    if distorted < 1e-12:
        return np.array((0.0, 0.0, 1.0), dtype=np.float64)
    k1, k2, k3, k4 = (float(value) for value in intrinsics["radialDistortion"][:4])
    theta = distorted
    for _ in range(12):
        theta2 = theta * theta
        polynomial = theta * (1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))))
        derivative = 1 + theta2 * (3 * k1 + theta2 * (5 * k2 + theta2 * (7 * k3 + 9 * theta2 * k4)))
        theta -= (polynomial - distorted) / derivative
    scale = math.sin(theta) / distorted
    return np.array((x * scale, y * scale, math.cos(theta)), dtype=np.float64)


def kb_project(point: np.ndarray, camera: dict[str, Any]) -> np.ndarray | None:
    norm = float(np.linalg.norm(point))
    if norm < 1e-12 or point[2] <= 0:
        return None
    x, y, z = point / norm
    radius = math.hypot(float(x), float(y))
    intrinsics = camera["intrinsics"]
    if radius < 1e-12:
        return np.array((float(intrinsics["centerX"]), float(intrinsics["centerY"])))
    theta = math.atan2(radius, float(z))
    k1, k2, k3, k4 = (float(value) for value in intrinsics["radialDistortion"][:4])
    theta2 = theta * theta
    distorted = theta * (1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))))
    return np.array((
        float(intrinsics["focalX"]) * distorted * float(x) / radius + float(intrinsics["centerX"]),
        float(intrinsics["focalY"]) * distorted * float(y) / radius + float(intrinsics["centerY"]),
    ))


def read_pair(video: Path, index_wanted: int) -> tuple[np.ndarray, np.ndarray]:
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index == index_wanted:
                image = frame.to_ndarray(format="gray")
                if image.shape[1] % 2:
                    raise ValueError(f"expected side-by-side stereo, got width={image.shape[1]}")
                return image[:, : image.shape[1] // 2], image[:, image.shape[1] // 2 :]
    raise ValueError(f"frame {index_wanted} is unavailable in {video}")


def score_transform(left: np.ndarray, right: np.ndarray, cam0: dict[str, Any], cam1: dict[str, Any], t_left_right: np.ndarray, features: int) -> dict[str, float | int | None]:
    extractor = cv2.ORB_create(nfeatures=features, fastThreshold=7)
    left_keys, left_desc = extractor.detectAndCompute(left, None)
    right_keys, right_desc = extractor.detectAndCompute(right, None)
    if left_desc is None or right_desc is None:
        return {"features_left": len(left_keys), "features_right": len(right_keys), "ratio_matches": 0, "positive_depth": 0, "inlier_reprojections": 0, "median_reprojection_px": None}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(left_desc, right_desc, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
    rotation, translation = t_left_right[:3, :3], t_left_right[:3, 3]
    positive, errors = 0, []
    for match in matches:
        ray_left = kb_unproject(left_keys[match.queryIdx].pt, cam0)
        ray_right = rotation @ kb_unproject(right_keys[match.trainIdx].pt, cam1)
        depth, *_ = np.linalg.lstsq(np.column_stack((ray_left, -ray_right)), translation, rcond=None)
        if depth[0] <= 0.03 or depth[1] <= 0.03:
            continue
        positive += 1
        point = 0.5 * (depth[0] * ray_left + translation + depth[1] * ray_right)
        predicted_left = kb_project(point, cam0)
        predicted_right = kb_project(rotation.T @ (point - translation), cam1)
        if predicted_left is None or predicted_right is None:
            continue
        error = max(float(np.linalg.norm(predicted_left - left_keys[match.queryIdx].pt)), float(np.linalg.norm(predicted_right - right_keys[match.trainIdx].pt)))
        if error < 3.0:
            errors.append(error)
    return {"features_left": len(left_keys), "features_right": len(right_keys), "ratio_matches": len(matches), "positive_depth": positive, "inlier_reprojections": len(errors), "median_reprojection_px": float(np.median(errors)) if errors else None}


def stamps(path: Path, field: str) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [int(row[field]) for row in csv.DictReader(handle)]


def rate_hz(values: list[int]) -> float | None:
    delta = np.diff(np.asarray(values, dtype=np.int64))
    delta = delta[delta > 0]
    return float(1e9 / np.median(delta)) if len(delta) else None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    episode = args.episode_dir.expanduser().resolve()
    required = (f"{args.stream}.mp4", f"{args.stream}_metainfo.csv", "accel.csv", "gyro.csv", "imu_calibration.json", f"camera_params_{args.stream}.json")
    missing = [name for name in required if not (episode / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} in {episode}")
    camera_data = json.loads((episode / f"camera_params_{args.stream}.json").read_text(encoding="utf-8"))
    imu_data = json.loads((episode / "imu_calibration.json").read_text(encoding="utf-8"))
    cam0, cam1 = camera_data["cameras"]
    raw0, raw1 = pose(cam0, args.camera_quaternion_order), pose(cam1, args.camera_quaternion_order)
    t_left_right = inverse(raw0) @ raw1
    left, right = read_pair(episode / f"{args.stream}.mp4", args.frame_index)
    forward = score_transform(left, right, cam0, cam1, t_left_right, args.max_features)
    reverse = score_transform(left, right, cam0, cam1, inverse(t_left_right), args.max_features)
    frame_stamps = stamps(episode / f"{args.stream}_metainfo.csv", "mid_exposure_utc_ns")
    accel_stamps, gyro_stamps = stamps(episode / "accel.csv", "timestamp_ns"), stamps(episode / "gyro.csv", "timestamp_ns")
    offset_key = OFFSET_KEYS[args.stream]
    offset_s = float(imu_data["imu"]["time_alignment_s"]["cameras"][offset_key])
    aligned = np.asarray(frame_stamps, dtype=np.int64) + round(offset_s * 1e9)
    imu_min, imu_max = min(accel_stamps[0], gyro_stamps[0]), max(accel_stamps[-1], gyro_stamps[-1])
    overlap = float(np.mean((aligned >= imu_min) & (aligned <= imu_max)))
    passed = int(forward["positive_depth"]) >= 20 and int(forward["positive_depth"]) >= 3 * max(1, int(reverse["positive_depth"])) and overlap >= 0.98
    report = {
        "episode": str(episode), "stream": args.stream, "result": "PASS" if passed else "REVIEW_REQUIRED",
        "factory_pose_semantics": f"T_imu_cam (IMU parent, camera child), quaternion order {args.camera_quaternion_order}",
        "camera_model": "KB4 / Kannala-Brandt 4 coefficients", "geometry_frame_index": args.frame_index,
        "validated_stereo_transform": {"formula": "inverse(T_imu_left) @ T_imu_right", "baseline_m": float(np.linalg.norm(t_left_right[:3, 3])), "forward": forward, "reverse": reverse},
        "timing": {"camera_offset_key": offset_key, "factory_camera_to_imu_offset_s": offset_s, "camera_effective_hz": rate_hz(frame_stamps), "accel_effective_hz": rate_hz(accel_stamps), "gyro_effective_hz": rate_hz(gyro_stamps), "camera_imu_overlap_fraction": overlap},
    }
    destination = args.log_file.expanduser().resolve() if args.log_file else episode / f"{args.stream}_calibration_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[LOG] {destination}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
