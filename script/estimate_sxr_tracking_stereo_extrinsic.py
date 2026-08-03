#!/usr/bin/env python3
"""Estimate a fixed tracking-stereo transform without reading head_pose."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from run_sxr_csv_basalt import factory_t_imu_camera, inverse


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--frame-stride", type=int, default=40)
    parser.add_argument("--minimum-inliers", type=int, default=60)
    parser.add_argument("--maximum-rotation-deviation-deg", type=float, default=5.0)
    parser.add_argument("--nfeatures", type=int, default=2600)
    parser.add_argument("--camera-quaternion-order", choices=("wxyz", "xyzw"), default="wxyz")
    return parser.parse_args(argv)


def camera_matrix(camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = camera["intrinsics"]
    matrix = np.array(((float(intrinsics["focalX"]), 0.0, float(intrinsics["centerX"])), (0.0, float(intrinsics["focalY"]), float(intrinsics["centerY"])), (0.0, 0.0, 1.0)), dtype=np.float64)
    return matrix, np.asarray(intrinsics["radialDistortion"][:4], dtype=np.float64)


def angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(first @ second.T) - 1.0) * 0.5, -1.0, 1.0))))


def ratio_matches(first: np.ndarray, second: np.ndarray) -> list[cv2.DMatch]:
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(first, second, k=2)
    return [best for best, next_best in pairs if best.distance < 0.75 * next_best.distance]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.frame_stride < 1 or args.minimum_inliers < 12 or args.nfeatures < 100:
        raise ValueError("invalid stereo-calibration settings")
    episode, output = args.episode_dir.expanduser().resolve(), args.output_json.expanduser().resolve()
    if not (episode / "tracking.mp4").is_file() or not (episode / "camera_params_tracking.json").is_file():
        raise FileNotFoundError("episode lacks tracking.mp4 or camera_params_tracking.json")
    cameras = json.loads((episode / "camera_params_tracking.json").read_text(encoding="utf-8"))["cameras"]
    k0, d0 = camera_matrix(cameras[0]); k1, d1 = camera_matrix(cameras[1])
    t0 = factory_t_imu_camera(cameras[0], "imu-to-camera", args.camera_quaternion_order)
    t1 = factory_t_imu_camera(cameras[1], "imu-to-camera", args.camera_quaternion_order)
    factory = inverse(t1) @ t0
    baseline_m = float(np.linalg.norm(factory[:3, 3]))
    detector = cv2.ORB_create(nfeatures=args.nfeatures, fastThreshold=8)
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    records: list[dict[str, int]] = []
    container = av.open(str(episode / "tracking.mp4"))
    try:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index % args.frame_stride:
                continue
            image = frame.to_ndarray(format="gray"); width = image.shape[1] // 2
            keys0, desc0 = detector.detectAndCompute(image[:, :width], None)
            keys1, desc1 = detector.detectAndCompute(image[:, width:], None)
            if desc0 is None or desc1 is None:
                continue
            matches = ratio_matches(desc0, desc1)
            if len(matches) < args.minimum_inliers:
                continue
            points0 = np.asarray([keys0[match.queryIdx].pt for match in matches], dtype=np.float64).reshape(-1, 1, 2)
            points1 = np.asarray([keys1[match.trainIdx].pt for match in matches], dtype=np.float64).reshape(-1, 1, 2)
            normalized0 = cv2.fisheye.undistortPoints(points0, k0, d0).reshape(-1, 2)
            normalized1 = cv2.fisheye.undistortPoints(points1, k1, d1).reshape(-1, 2)
            essential, mask = cv2.findEssentialMat(normalized0, normalized1, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=0.008)
            if essential is None or mask is None:
                continue
            inliers, rotation, translation, _ = cv2.recoverPose(essential[:3, :3], normalized0, normalized1, np.eye(3), mask=mask)
            if inliers >= args.minimum_inliers:
                rotations.append(rotation); translations.append(translation[:, 0])
                records.append({"frame_index": index, "matches": len(matches), "inliers": int(inliers)})
    finally:
        container.close()
    if len(rotations) < 4:
        raise RuntimeError(f"only {len(rotations)} valid stereo geometry estimates")
    provisional = Rotation.from_matrix(np.asarray(rotations)).mean().as_matrix()
    keep = np.asarray([angle_deg(rotation, provisional) <= args.maximum_rotation_deviation_deg for rotation in rotations])
    if int(keep.sum()) < 4:
        raise RuntimeError("too few consistent stereo geometry estimates")
    fitted_rotation = Rotation.from_matrix(np.asarray(rotations)[keep]).mean().as_matrix()
    directions = np.asarray(translations)[keep]
    directions[directions @ directions[0] < 0.0] *= -1.0
    fitted_translation = np.mean(directions, axis=0)
    fitted_translation *= baseline_m / float(np.linalg.norm(fitted_translation))
    fitted = np.eye(4, dtype=np.float64); fitted[:3, :3] = fitted_rotation; fitted[:3, 3] = fitted_translation
    translation_cosine = float(np.clip(factory[:3, 3] @ fitted_translation / (baseline_m * baseline_m), -1.0, 1.0))
    payload = {
        "method": "KB4-undistorted stereo essential-matrix RANSAC; head_pose not read",
        "episode": str(episode), "camera_quaternion_order": args.camera_quaternion_order,
        "factory_camera1_from_camera0": factory.tolist(), "fitted_camera1_from_camera0": fitted.tolist(),
        "factory_baseline_m": baseline_m, "factory_to_fitted_rotation_deg": angle_deg(factory[:3, :3], fitted_rotation),
        "factory_to_fitted_translation_direction_deg": math.degrees(math.acos(translation_cosine)),
        "sampled_valid_pairs": len(rotations), "consistent_pairs": int(keep.sum()), "pair_records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "pair_records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
