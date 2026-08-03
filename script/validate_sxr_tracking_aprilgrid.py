#!/usr/bin/env python3
"""Validate SXR tracking-camera/IMU semantics from an AprilGrid recording.

The SXR factory files encode a camera pose with IMU as parent and quaternion
order xyzw.  This checker uses only the tracking stereo images, the known
AprilGrid geometry, and gyro CSV to validate the conversion into the OpenCV
optical frame expected by open-source VIO.  ``head_pose.csv`` is never read.
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
from pupil_apriltags import Detector

from check_sxr_csv_calibration import kb_unproject


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tag-family", default="tag16h5")
    parser.add_argument("--tag-rows", type=int, default=6)
    parser.add_argument("--tag-cols", type=int, default=6)
    parser.add_argument("--tag-size", type=float, default=0.055, help="Tag edge length in metres.")
    parser.add_argument("--tag-spacing", type=float, default=0.3, help="Gap/tag-size ratio.")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--offset-span-ms", type=float, default=30.0)
    parser.add_argument("--offset-step-ms", type=float, default=0.5)
    return parser.parse_args(argv)


def quat_xyzw_to_matrix(values: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in values)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-9:
        raise ValueError("zero-norm factory camera quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
                     (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
                     (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y))), dtype=np.float64)


def factory_t_imu_camera(camera: dict[str, Any]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_xyzw_to_matrix(camera["extrinsics"]["rotation"])
    transform[:3, 3] = np.asarray(camera["extrinsics"]["position"], dtype=np.float64)
    return transform


def inverse(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return result


def tag_corners(tag_id: int, cols: int, size: float, spacing: float) -> np.ndarray:
    pitch = size * (1.0 + spacing)
    row, col = divmod(tag_id, cols)
    x, y = col * pitch, row * pitch
    # Pupil AprilTags corners are top-left, top-right, bottom-right, bottom-left.
    return np.array(((x, y, 0.0), (x + size, y, 0.0),
                     (x + size, y + size, 0.0), (x, y + size, 0.0)), dtype=np.float64)


def normalized_corners(tags: Sequence[Any], camera: dict[str, Any], rows: int, cols: int, size: float, spacing: float) -> tuple[np.ndarray, np.ndarray, int]:
    object_points, image_points, accepted = [], [], 0
    maximum_id = rows * cols
    for tag in tags:
        if not 0 <= int(tag.tag_id) < maximum_id:
            continue
        rays = np.asarray([kb_unproject(tuple(corner), camera) for corner in tag.corners], dtype=np.float64)
        if np.any(rays[:, 2] < 0.15):
            continue
        object_points.extend(tag_corners(int(tag.tag_id), cols, size, spacing))
        image_points.extend(rays[:, :2] / rays[:, 2:3])
        accepted += 1
    return np.asarray(object_points, dtype=np.float64), np.asarray(image_points, dtype=np.float64), accepted


def pose_from_tags(tags: Sequence[Any], camera: dict[str, Any], rows: int, cols: int, size: float, spacing: float) -> tuple[np.ndarray | None, dict[str, float | int]]:
    obj, img, count = normalized_corners(tags, camera, rows, cols, size, spacing)
    detail: dict[str, float | int] = {"tags": count, "corners": int(len(obj))}
    # Two known AprilGrid cells provide eight coplanar corners and remove the
    # single-tag planar-pose ambiguity while retaining useful close-up frames.
    if count < 2:
        return None, detail
    success, rvec, _translation, inliers = cv2.solvePnPRansac(
        obj, img, np.eye(3), None, iterationsCount=200, reprojectionError=0.008,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 8:
        detail["pnp_inliers"] = 0 if inliers is None else int(len(inliers))
        return None, detail
    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(obj[inliers.reshape(-1)], rvec, _translation, np.eye(3), None)
    error = np.linalg.norm(projected.reshape(-1, 2) - img[inliers.reshape(-1)], axis=1)
    detail.update({"pnp_inliers": int(len(inliers)), "pnp_median_normalized_error": float(np.median(error))})
    return rotation, detail


def read_timestamps(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        return np.asarray([int(row["mid_exposure_utc_ns"]) * 1e-9 for row in csv.DictReader(handle)], dtype=np.float64)


def read_gyro(path: Path, calibration: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    stamps, values = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamps.append(int(row["timestamp_ns"]) * 1e-9)
            values.append((float(row["x"]), float(row["y"]), float(row["z"])))
    imu = calibration["imu"]
    bias = np.asarray(imu["bias"]["gyroscope_rads"], dtype=np.float64)
    scale = np.asarray(imu["scale_factor"]["gyroscope"], dtype=np.float64)
    n0, n1, n2 = imu["nonorthogonality"]["gyroscope"]
    correction = np.array(((1.0, n0, n1), (0.0, 1.0, n2), (0.0, 0.0, 1.0))) @ np.diag(1.0 + scale)
    return np.asarray(stamps), (correction @ (np.asarray(values) - bias).T).T


def interpolate(query: np.ndarray, stamps: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(query, stamps, values[:, index], left=np.nan, right=np.nan) for index in range(3)])


def fit_rotation(imu_vectors: np.ndarray, camera_vectors: np.ndarray) -> tuple[np.ndarray, float, float]:
    cross = imu_vectors.T @ camera_vectors
    left, _singular, right_t = np.linalg.svd(cross)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right_t)
    rotation = (left @ correction @ right_t).T  # R_camera_optical_from_imu
    prediction = (rotation @ imu_vectors.T).T
    cosine = np.clip(np.sum(prediction * camera_vectors, axis=1) / np.maximum(np.linalg.norm(prediction, axis=1) * np.linalg.norm(camera_vectors, axis=1), 1e-12), -1.0, 1.0)
    angles = np.arccos(cosine)
    return rotation, float(np.mean(cosine)), math.degrees(float(np.sqrt(np.mean(angles * angles))))


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip((np.trace(first @ second.T) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(float(math.acos(cosine)))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tag_rows < 2 or args.tag_cols < 2 or args.tag_size <= 0 or args.tag_spacing < 0 or args.frame_stride < 1:
        raise ValueError("invalid AprilGrid geometry or frame stride")
    episode = args.episode_dir.expanduser().resolve()
    required = ("tracking.mp4", "tracking_metainfo.csv", "camera_params_tracking.json", "imu_calibration.json", "gyro.csv")
    missing = [name for name in required if not (episode / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {episode}")
    image = json.loads((episode / "camera_params_tracking.json").read_text(encoding="utf-8"))
    calibration = json.loads((episode / "imu_calibration.json").read_text(encoding="utf-8"))
    camera = image["cameras"][0]
    stamps = read_timestamps(episode / "tracking_metainfo.csv")
    gyro_stamps, gyro = read_gyro(episode / "gyro.csv", calibration)
    detector = Detector(families=args.tag_family, nthreads=1, quad_decimate=1.0, refine_edges=1)
    poses: list[tuple[int, np.ndarray]] = []
    records: list[dict[str, float | int]] = []
    ids: set[int] = set()
    container = av.open(str(episode / "tracking.mp4"))
    try:
        for index, frame in enumerate(container.decode(video=0)):
            if index >= len(stamps):
                break
            if index % args.frame_stride:
                continue
            gray = frame.to_ndarray(format="gray")[:, : frame.width // 2]
            tags = detector.detect(gray, estimate_tag_pose=False)
            ids.update(int(tag.tag_id) for tag in tags if 0 <= int(tag.tag_id) < args.tag_rows * args.tag_cols)
            pose, detail = pose_from_tags(tags, camera, args.tag_rows, args.tag_cols, args.tag_size, args.tag_spacing)
            detail["frame"] = index
            records.append(detail)
            if pose is not None:
                poses.append((index, pose))
    finally:
        container.close()
    visual_times, visual_vectors = [], []
    for (index0, rotation0), (index1, rotation1) in zip(poses, poses[1:]):
        dt = float(stamps[index1] - stamps[index0])
        if dt <= 1e-6:
            continue
        # R maps coordinates from camera0 to camera1. The minus log is the
        # body angular velocity expressed in the first optical camera frame.
        relative = rotation1 @ rotation0.T
        visual_vectors.append(-cv2.Rodrigues(relative)[0].reshape(3) / dt)
        visual_times.append(0.5 * (stamps[index0] + stamps[index1]))
    output = args.output.expanduser().resolve()
    if len(visual_vectors) < 30:
        report = {
            "result": "REVIEW_TARGET_LAYOUT",
            "episode": str(episode), "stream": "tracking", "head_pose_read": False,
            "target_requested": {"family": args.tag_family, "tagRows": args.tag_rows, "tagCols": args.tag_cols, "tagSize_m": args.tag_size, "tagSpacing_ratio": args.tag_spacing},
            "factory_pose_semantics": "T_imu_camera (IMU parent, camera child), quaternion order xyzw",
            "observation": {"sampled_frames": len(records), "pnp_pose_frames": len(poses), "unique_tags": sorted(ids), "unique_tag_count": len(ids)},
            "blocking_reason": (
                f"{args.tag_family} detections expose {len(ids)} IDs while the requested grid has "
                f"{args.tag_rows * args.tag_cols} cells. The target family/layout (ID-to-cell mapping) "
                "must be supplied before its corners can be used for metric PnP calibration."
            ),
            "required_input": "The AprilGrid YAML/PDF source or its tag family plus exact ID layout. Do not infer a row-major 6x6 layout from this recording.",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"[LOG] {output}")
        return 2
    visual_times_array, visual = np.asarray(visual_times), np.asarray(visual_vectors)
    factory_offset = float(calibration["imu"]["time_alignment_s"]["cameras"]["trackingA"])
    visual_rates, gyro_rates = np.linalg.norm(visual, axis=1), np.linalg.norm(gyro, axis=1)
    offsets = np.arange(-args.offset_span_ms, args.offset_span_ms + 0.5 * args.offset_step_ms, args.offset_step_ms) * 1e-3
    scan = []
    for offset in offsets:
        sampled = np.interp(visual_times_array + offset, gyro_stamps, gyro_rates, left=np.nan, right=np.nan)
        valid = np.isfinite(sampled)
        score = float(np.corrcoef(visual_rates[valid], sampled[valid])[0, 1]) if valid.sum() >= 12 else None
        scan.append({"camera_to_imu_offset_s": float(offset), "correlation": score, "samples": int(valid.sum())})
    best = max((row for row in scan if row["correlation"] is not None), key=lambda row: float(row["correlation"]))
    nearest = min(scan, key=lambda row: abs(float(row["camera_to_imu_offset_s"]) - factory_offset))
    sampled_gyro = interpolate(visual_times_array + factory_offset, gyro_stamps, gyro)
    active = np.isfinite(sampled_gyro).all(axis=1) & (visual_rates > 0.08) & (np.linalg.norm(sampled_gyro, axis=1) > 0.08)
    if active.sum() < 30:
        raise RuntimeError("not enough excited AprilGrid/gyro pairs")
    fitted, fitted_cosine, fitted_rmse = fit_rotation(sampled_gyro[active], visual[active])
    raw_t_i_c = factory_t_imu_camera(camera)
    factory_optical_from_imu = raw_t_i_c[:3, :3].T
    factory_prediction = (factory_optical_from_imu @ sampled_gyro[active].T).T
    factory_cosine = np.mean(np.sum(factory_prediction * visual[active], axis=1) / np.maximum(np.linalg.norm(factory_prediction, axis=1) * np.linalg.norm(visual[active], axis=1), 1e-12))
    report = {
        "result": "REVIEW_OPTICAL_AXIS_CONVERSION",
        "episode": str(episode),
        "stream": "tracking", "head_pose_read": False,
        "target": {"family": args.tag_family, "tagRows": args.tag_rows, "tagCols": args.tag_cols, "tagSize_m": args.tag_size, "tagSpacing_ratio": args.tag_spacing},
        "factory_pose_semantics": "T_imu_camera (IMU parent, camera child), quaternion order xyzw",
        "factory_time_offset_s": factory_offset,
        "observation": {"sampled_frames": len(records), "pnp_pose_frames": len(poses), "relative_rotations": len(visual), "unique_tags": sorted(ids), "unique_tag_count": len(ids), "median_pnp_inliers": float(np.median([item.get("pnp_inliers", 0) for item in records]))},
        "timing": {"best_offset_s": best["camera_to_imu_offset_s"], "best_correlation": best["correlation"], "factory_nearest_offset_s": nearest["camera_to_imu_offset_s"], "factory_nearest_correlation": nearest["correlation"]},
        "rotation": {"excited_pairs": int(active.sum()), "factory_optical_from_imu": factory_optical_from_imu.tolist(), "factory_mean_cosine": float(factory_cosine), "fitted_optical_from_imu": fitted.tolist(), "fitted_mean_cosine": fitted_cosine, "fitted_angular_rmse_deg": fitted_rmse, "factory_vs_fitted_deg": rotation_distance_deg(factory_optical_from_imu, fitted)},
        "recommended_vio_input": {"T_imu_tracking_left_translation_m": raw_t_i_c[:3, 3].tolist(), "T_tracking_left_optical_from_imu_rotation": fitted.tolist(), "status": "Do not deploy automatically: run Basalt/Kalibr joint optimization and holdout validation first."},
        "caveat": "This solves the optical-axis conversion diagnostic from the calibration recording; full camera-IMU calibration still needs joint optimization to validate translation and refine time offset.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[LOG] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
