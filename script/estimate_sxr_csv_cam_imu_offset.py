#!/usr/bin/env python3
"""Estimate SXR RGB-to-IMU timing from visual and gyro angular speeds.

This diagnostic deliberately does not read ``head_pose.csv``. It derives
relative camera rotations from consecutive left RGB images, then scans the
correlation between their angular speeds and the gyroscope magnitude. It is a
configuration check, not a trajectory evaluation or a replacement for a
dedicated calibration recording.
"""

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

from check_sxr_csv_calibration import kb_unproject, pose


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--stream", choices=("rgb", "tracking", "ctrl"), default="tracking", help="Stereo stream; tracking is the localization rig.")
    parser.add_argument("--camera-quaternion-order", choices=("xyzw", "wxyz"), default="xyzw", help="SXR tracking extrinsics use xyzw.")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=2500)
    parser.add_argument("--offset-span-ms", type=float, default=30.0)
    parser.add_argument("--offset-step-ms", type=float, default=0.5)
    return parser.parse_args(argv)


def read_csv(path: Path, timestamp: str, fields: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    stamps, values = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamps.append(int(row[timestamp]) * 1e-9)
            values.append([float(row[field]) for field in fields])
    return np.asarray(stamps, dtype=np.float64), np.asarray(values, dtype=np.float64)


def relative_rotation(first: np.ndarray, second: np.ndarray, camera: dict[str, Any], extractor: cv2.ORB) -> tuple[np.ndarray | None, int, int]:
    keys0, desc0 = extractor.detectAndCompute(first, None)
    keys1, desc1 = extractor.detectAndCompute(second, None)
    if desc0 is None or desc1 is None:
        return None, 0, 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc0, desc1, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    points0, points1 = [], []
    for match in matches:
        ray0 = kb_unproject(keys0[match.queryIdx].pt, camera)
        ray1 = kb_unproject(keys1[match.trainIdx].pt, camera)
        # Essential-matrix estimation uses a normalized pinhole plane. Keep
        # the central forward-facing part of this fisheye image where that
        # projection remains well-conditioned.
        if min(ray0[2], ray1[2]) < 0.5:
            continue
        points0.append((ray0[0] / ray0[2], ray0[1] / ray0[2]))
        points1.append((ray1[0] / ray1[2], ray1[1] / ray1[2]))
    if len(points0) < 12:
        return None, len(matches), 0
    points0_array, points1_array = np.asarray(points0, dtype=np.float64), np.asarray(points1, dtype=np.float64)
    essential, mask = cv2.findEssentialMat(points0_array, points1_array, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=0.003)
    if essential is None or mask is None:
        return None, len(matches), 0
    _count, rotation, _translation, pose_mask = cv2.recoverPose(essential[:3, :3], points0_array, points1_array, np.eye(3), mask=mask)
    inliers = int(np.count_nonzero(pose_mask))
    if inliers < 12:
        return None, len(matches), inliers
    # recoverPose returns R_C1_to_C2. For a body angular velocity expressed
    # in the first camera frame, the corresponding camera-coordinate vector
    # is -log(R_C1_to_C2) / dt.
    return -cv2.Rodrigues(rotation)[0].reshape(3), len(matches), inliers


def visual_speeds(video: Path, stamps: np.ndarray, camera: dict[str, Any], stride: int, features: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    extractor = cv2.ORB_create(nfeatures=features, fastThreshold=10)
    times, speeds, records = [], [], []
    previous: tuple[int, np.ndarray] | None = None
    container = av.open(str(video))
    try:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index >= len(stamps):
                break
            if index % stride:
                continue
            image = frame.to_ndarray(format="gray")[:, : frame.width // 2]
            if previous is None:
                previous = (index, image)
                continue
            index0, first = previous
            rotation_vector, matches, inliers = relative_rotation(first, image, camera, extractor)
            interval = float(stamps[index] - stamps[index0])
            record: dict[str, float | int] = {"frame0": index0, "frame1": index, "matches": matches, "inliers": inliers, "dt_s": interval}
            if rotation_vector is not None and interval > 1e-6:
                times.append(0.5 * (stamps[index0] + stamps[index]))
                speeds.append(rotation_vector / interval)
                record["angular_speed_rad_s"] = float(np.linalg.norm(rotation_vector) / interval)
            records.append(record)
            previous = (index, image)
    finally:
        container.close()
    return np.asarray(times), np.asarray(speeds, dtype=np.float64), records


def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 12 or np.std(first) < 1e-8 or np.std(second) < 1e-8:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def corrected_gyro(samples: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    imu = calibration["imu"]
    bias = np.asarray(imu["bias"]["gyroscope_rads"], dtype=np.float64)
    scale = np.asarray(imu["scale_factor"]["gyroscope"], dtype=np.float64)
    n0, n1, n2 = imu["nonorthogonality"]["gyroscope"]
    nonorthogonal = np.array(((1.0, n0, n1), (0.0, 1.0, n2), (0.0, 0.0, 1.0)), dtype=np.float64)
    return (nonorthogonal @ np.diag(1.0 + scale) @ (samples - bias).T).T


def interpolate_vectors(query: np.ndarray, stamps: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(query, stamps, values[:, axis], left=np.nan, right=np.nan) for axis in range(3)])


def rotation_from_vectors(imu_vectors: np.ndarray, camera_vectors: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit R_camera_imu and return rotation, mean cosine, and angular RMSE."""
    cross = imu_vectors.T @ camera_vectors
    left, _singular, right_t = np.linalg.svd(cross)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right_t)
    # imu_rows @ q ~= camera_rows, hence R_camera_imu = q.T.
    rotation = (left @ correction @ right_t).T
    predicted = (rotation @ imu_vectors.T).T
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(camera_vectors, axis=1)
    cosine = np.clip(np.sum(predicted * camera_vectors, axis=1) / np.maximum(denominator, 1e-12), -1.0, 1.0)
    angles = np.arccos(cosine)
    return rotation, float(np.mean(cosine)), float(np.sqrt(np.mean(angles * angles)))


def rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(first @ second.T) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.frame_stride < 1 or args.max_features < 100 or args.offset_span_ms <= 0 or args.offset_step_ms <= 0:
        raise ValueError("invalid stride, feature count, or offset scan")
    episode = args.episode_dir.expanduser().resolve()
    stream = args.stream
    required = (f"{stream}.mp4", f"{stream}_metainfo.csv", "gyro.csv", f"camera_params_{stream}.json", "imu_calibration.json")
    missing = [name for name in required if not (episode / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {episode}")
    frame_stamps, _ = read_csv(episode / f"{stream}_metainfo.csv", "mid_exposure_utc_ns", ("frame_index",))
    gyro_stamps, gyro = read_csv(episode / "gyro.csv", "timestamp_ns", ("x", "y", "z"))
    camera = json.loads((episode / f"camera_params_{stream}.json").read_text(encoding="utf-8"))["cameras"][0]
    calibration = json.loads((episode / "imu_calibration.json").read_text(encoding="utf-8"))
    offset_key = {"rgb": "rgb-left", "tracking": "trackingA", "ctrl": "ctrl-trackingA"}[stream]
    factory_offset = float(calibration["imu"]["time_alignment_s"]["cameras"][offset_key])
    visual_times, visual_vectors, records = visual_speeds(episode / f"{stream}.mp4", frame_stamps, camera, args.frame_stride, args.max_features)
    visual_rates = np.linalg.norm(visual_vectors, axis=1)
    gyro_corrected = corrected_gyro(gyro, calibration)
    gyro_rates = np.linalg.norm(gyro_corrected, axis=1)
    offsets = np.arange(-args.offset_span_ms, args.offset_span_ms + 0.5 * args.offset_step_ms, args.offset_step_ms) * 1e-3
    rows = []
    for offset in offsets:
        sampled = np.interp(visual_times + offset, gyro_stamps, gyro_rates, left=np.nan, right=np.nan)
        valid = np.isfinite(sampled)
        rows.append({"camera_to_imu_offset_s": float(offset), "correlation": correlation(visual_rates[valid], sampled[valid]), "samples": int(valid.sum())})
    ranked = sorted((row for row in rows if row["correlation"] is not None), key=lambda row: float(row["correlation"]), reverse=True)
    if not ranked:
        raise RuntimeError("not enough visual-rotation estimates for a timing correlation")
    nearest_factory = min(rows, key=lambda row: abs(float(row["camera_to_imu_offset_s"]) - factory_offset))
    # Factory pose is T_imu_cam. Visual angular velocity is in camera axes,
    # so gyro vectors in IMU axes require R_cam_imu = R_imu_cam.T.
    factory_rotation = pose(camera, args.camera_quaternion_order)[:3, :3].T
    sampled_gyro = interpolate_vectors(visual_times + factory_offset, gyro_stamps, gyro_corrected)
    valid = np.isfinite(sampled_gyro).all(axis=1)
    # Small rotations are dominated by essential-matrix noise. Retain samples
    # with meaningful visual and inertial excitation for the extrinsic check.
    valid &= visual_rates > 0.08
    valid &= np.linalg.norm(sampled_gyro, axis=1) > 0.08
    if int(valid.sum()) < 20:
        raise RuntimeError("not enough excited visual/IMU pairs for extrinsic rotation validation")
    fitted_rotation, fitted_cosine, fitted_angular_rmse = rotation_from_vectors(sampled_gyro[valid], visual_vectors[valid])
    factory_prediction = (factory_rotation @ sampled_gyro[valid].T).T
    factory_denominator = np.linalg.norm(factory_prediction, axis=1) * np.linalg.norm(visual_vectors[valid], axis=1)
    factory_cosine = np.clip(np.sum(factory_prediction * visual_vectors[valid], axis=1) / np.maximum(factory_denominator, 1e-12), -1.0, 1.0)
    report = {
        "episode": str(episode), "stream": stream,
        "method": "visual relative rotation magnitude versus raw gyro magnitude; head_pose not read",
        "factory_camera_to_imu_offset_s": factory_offset, "factory_offset_key": offset_key,
        "best_camera_to_imu_offset_s": ranked[0]["camera_to_imu_offset_s"],
        "best_correlation": ranked[0]["correlation"],
        "factory_nearest_scan_offset_s": nearest_factory["camera_to_imu_offset_s"],
        "factory_nearest_correlation": nearest_factory["correlation"],
        "visual_rotation_estimates": int(len(visual_rates)),
        "visual_pair_attempts": len(records),
        "extrinsic_rotation_validation": {
            "excited_pairs": int(valid.sum()),
            "factory_rotation_camera_from_imu": factory_rotation.tolist(),
            "factory_pose_semantics": f"T_imu_cam (IMU parent, camera child), quaternion order {args.camera_quaternion_order}",
            "fitted_rotation_camera_from_imu": fitted_rotation.tolist(),
            "factory_to_visual_mean_cosine": float(np.mean(factory_cosine)),
            "fitted_to_visual_mean_cosine": fitted_cosine,
            "fitted_to_visual_angular_rmse_deg": math.degrees(fitted_angular_rmse),
            "factory_vs_fitted_rotation_deg": rotation_angle_deg(factory_rotation, fitted_rotation),
        },
        "visual_pair_records": records,
        "offset_scan": rows,
        "caveat": "Use only a pronounced and repeatable peak to change timing. This method does not use the reference trajectory.",
    }
    output = args.log_file.expanduser().resolve() if args.log_file else episode / "cam_imu_timing_diagnostic.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("visual_pair_records", "offset_scan")}, indent=2))
    print(f"[LOG] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
