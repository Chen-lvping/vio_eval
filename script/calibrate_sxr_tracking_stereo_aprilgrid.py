#!/usr/bin/env python3
"""Calibrate the SXR tracking stereo transform from common AprilTag corners.

Each tag is a self-contained planar target.  Matching the same detected tag
between both cameras estimates their relative pose without assuming that every
cell of the AprilGrid is visible or that tag IDs are row-major.  head_pose is
not read.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import av
import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation

from check_sxr_csv_calibration import kb_unproject
from run_sxr_csv_basalt import factory_t_imu_camera, inverse


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--tag-family", default="tag16h5")
    parser.add_argument("--tag-size", type=float, default=0.055)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--camera-quaternion-order", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument("--maximum-rotation-deviation-deg", type=float, default=2.0)
    parser.add_argument("--maximum-translation-deviation-m", type=float, default=0.01)
    return parser.parse_args(argv)


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(first @ second.T) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def tag_pose(corners: np.ndarray, camera: dict[str, Any], size: float) -> np.ndarray | None:
    rays = np.asarray([kb_unproject(tuple(corner), camera) for corner in corners], dtype=np.float64)
    if np.any(rays[:, 2] < 0.15):
        return None
    normalized = rays[:, :2] / rays[:, 2:3]
    half = size * 0.5
    # Tag corners from pupil_apriltags are top-left, top-right, bottom-right,
    # bottom-left.  This centered square ordering is required by IPPE_SQUARE.
    object_points = np.array(((-half, half, 0.0), (half, half, 0.0), (half, -half, 0.0), (-half, -half, 0.0)), dtype=np.float64)
    ok, rvec, translation = cv2.solvePnP(object_points, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or translation[2, 0] <= 0.0:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3], transform[:3, 3] = rotation, translation[:, 0]
    return transform


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tag_size <= 0.0 or args.frame_stride < 1:
        raise ValueError("invalid tag size or frame stride")
    episode, output = args.episode_dir.expanduser().resolve(), args.output_json.expanduser().resolve()
    required = ("tracking.mp4", "camera_params_tracking.json")
    missing = [name for name in required if not (episode / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {episode}")
    cameras = json.loads((episode / "camera_params_tracking.json").read_text(encoding="utf-8"))["cameras"]
    left_detector = Detector(families=args.tag_family, nthreads=1, quad_decimate=1.0, refine_edges=1)
    right_detector = Detector(families=args.tag_family, nthreads=1, quad_decimate=1.0, refine_edges=1)
    transforms: list[np.ndarray] = []
    records: list[dict[str, int]] = []
    container = av.open(str(episode / "tracking.mp4"))
    try:
        for frame_index, frame in enumerate(container.decode(container.streams.video[0])):
            if frame_index % args.frame_stride:
                continue
            image = frame.to_ndarray(format="gray")
            width = image.shape[1] // 2
            left_image = np.ascontiguousarray(image[:, :width])
            right_image = np.ascontiguousarray(image[:, width:])
            left = {int(tag.tag_id): np.asarray(tag.corners, dtype=np.float64).copy() for tag in left_detector.detect(left_image)}
            right = {int(tag.tag_id): np.asarray(tag.corners, dtype=np.float64).copy() for tag in right_detector.detect(right_image)}
            common = sorted(set(left).intersection(right))
            accepted = 0
            for tag_id in common:
                camera0_from_tag = tag_pose(left[tag_id], cameras[0], args.tag_size)
                camera1_from_tag = tag_pose(right[tag_id], cameras[1], args.tag_size)
                if camera0_from_tag is None or camera1_from_tag is None:
                    continue
                transforms.append(camera1_from_tag @ inverse(camera0_from_tag))
                accepted += 1
            records.append({"frame_index": frame_index, "left_tags": len(left), "right_tags": len(right), "common_tags": len(common), "accepted_tags": accepted})
    finally:
        container.close()
    if len(transforms) < 20:
        raise RuntimeError(f"only {len(transforms)} common-tag stereo poses")
    rotations = np.asarray([transform[:3, :3] for transform in transforms])
    translations = np.asarray([transform[:3, 3] for transform in transforms])
    provisional_rotation = Rotation.from_matrix(rotations).mean().as_matrix()
    provisional_translation = np.median(translations, axis=0)
    rotation_errors = np.asarray([angle_deg(rotation, provisional_rotation) for rotation in rotations])
    translation_errors = np.linalg.norm(translations - provisional_translation, axis=1)
    keep = (rotation_errors <= args.maximum_rotation_deviation_deg) & (translation_errors <= args.maximum_translation_deviation_m)
    if int(keep.sum()) < 20:
        raise RuntimeError(f"only {int(keep.sum())} geometrically consistent stereo poses")
    fitted_rotation = Rotation.from_matrix(rotations[keep]).mean().as_matrix()
    fitted_translation = np.median(translations[keep], axis=0)
    fitted = np.eye(4, dtype=np.float64)
    fitted[:3, :3], fitted[:3, 3] = fitted_rotation, fitted_translation
    factory0 = factory_t_imu_camera(cameras[0], "imu-to-camera", args.camera_quaternion_order)
    factory1 = factory_t_imu_camera(cameras[1], "imu-to-camera", args.camera_quaternion_order)
    factory = inverse(factory1) @ factory0
    baseline_m = float(np.linalg.norm(factory[:3, 3]))
    payload = {
        "method": "per-tag KB4-undistorted IPPE stereo pose; head_pose not read",
        "episode": str(episode), "tag_family": args.tag_family, "tag_size_m": args.tag_size,
        "camera_quaternion_order": args.camera_quaternion_order,
        "factory_camera1_from_camera0": factory.tolist(), "fitted_camera1_from_camera0": fitted.tolist(),
        "factory_baseline_m": baseline_m, "fitted_baseline_m": float(np.linalg.norm(fitted_translation)),
        "factory_to_fitted_rotation_deg": angle_deg(factory[:3, :3], fitted_rotation),
        "raw_common_tag_poses": len(transforms), "consistent_common_tag_poses": int(keep.sum()),
        "rotation_error_median_deg": float(np.median(rotation_errors[keep])),
        "translation_error_median_m": float(np.median(translation_errors[keep])), "frame_records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "frame_records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
