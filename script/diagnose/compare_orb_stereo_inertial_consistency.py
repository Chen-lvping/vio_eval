#!/usr/bin/env python3
"""Compare ORB stereo-inertial and stereo trajectories without ground truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def normalize_time(value: float) -> float:
    value = float(value)
    if abs(value) > 1e17:
        return value * 1e-9
    if abs(value) > 1e13:
        return value * 1e-6
    if abs(value) > 1e10:
        return value * 1e-3
    return value


def quat_to_rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    q = np.asarray([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    poses: list[np.ndarray] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        values = [float(item) for item in text.split()]
        if len(values) < 8:
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = quat_to_rotation(*values[4:8])
        pose[:3, 3] = values[1:4]
        times.append(normalize_time(values[0]))
        poses.append(pose)
    if not poses:
        raise ValueError(f"trajectory has no valid TUM poses: {path}")
    order = np.argsort(np.asarray(times))
    return np.asarray(times, dtype=np.float64)[order], np.asarray(poses)[order]


def load_tbc(path: Path) -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"cannot open ORB settings: {path}")
    try:
        for key in ("Tbc", "IMU.T_b_c1"):
            node = fs.getNode(key)
            if not node.empty():
                matrix = np.asarray(node.mat(), dtype=np.float64)
                if matrix.shape == (4, 4):
                    return matrix
    finally:
        fs.release()
    raise KeyError(f"Tbc/IMU.T_b_c1 missing from {path}")


def associate(
    ref_times: np.ndarray,
    est_times: np.ndarray,
    max_diff: float,
) -> tuple[np.ndarray, np.ndarray]:
    ref_indices: list[int] = []
    est_indices: list[int] = []
    for est_idx, timestamp in enumerate(est_times):
        right = int(np.searchsorted(ref_times, timestamp))
        candidates = [idx for idx in (right - 1, right) if 0 <= idx < ref_times.size]
        if not candidates:
            continue
        ref_idx = min(candidates, key=lambda idx: abs(ref_times[idx] - timestamp))
        if abs(ref_times[ref_idx] - timestamp) <= max_diff:
            ref_indices.append(ref_idx)
            est_indices.append(est_idx)
    if len(ref_indices) < 3:
        raise ValueError(f"only {len(ref_indices)} associated poses")
    return np.asarray(ref_indices), np.asarray(est_indices)


def rigid_alignment(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ref_center = reference.mean(axis=0)
    est_center = estimate.mean(axis=0)
    ref_zero = reference - ref_center
    est_zero = estimate - est_center
    u, singular, vt = np.linalg.svd(est_zero.T @ ref_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = ref_center - rotation @ est_center
    denom = float(np.sum(est_zero * est_zero))
    sim3_scale = float(np.sum(singular) / denom) if denom > 0.0 else float("nan")
    return rotation, translation, sim3_scale


def rotation_angle_deg(matrix: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inertial-tum", type=Path, required=True)
    parser.add_argument("--stereo-tum", type=Path, required=True)
    parser.add_argument("--inertial-settings", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--t-max-diff", type=float, default=0.01)
    args = parser.parse_args()

    imu_times, imu_poses = load_tum(args.inertial_tum)
    stereo_times, stereo_poses = load_tum(args.stereo_tum)
    t_body_camera = load_tbc(args.inertial_settings)
    imu_camera_poses = imu_poses @ t_body_camera
    ref_idx, est_idx = associate(stereo_times, imu_times, args.t_max_diff)
    reference = stereo_poses[ref_idx]
    estimate = imu_camera_poses[est_idx]
    align_rotation, align_translation, sim3_scale = rigid_alignment(
        reference[:, :3, 3], estimate[:, :3, 3]
    )
    aligned_positions = (align_rotation @ estimate[:, :3, 3].T).T + align_translation
    translation_errors = np.linalg.norm(reference[:, :3, 3] - aligned_positions, axis=1)
    rotation_errors = np.asarray(
        [
            rotation_angle_deg(ref[:3, :3].T @ align_rotation @ est[:3, :3])
            for ref, est in zip(reference, estimate)
        ]
    )
    payload = {
        "inertial_tum": str(args.inertial_tum.resolve()),
        "stereo_tum": str(args.stereo_tum.resolve()),
        "inertial_settings": str(args.inertial_settings.resolve()),
        "matched_poses": int(ref_idx.size),
        "translation_rmse_m": float(np.sqrt(np.mean(translation_errors**2))),
        "translation_rmse_mm": float(np.sqrt(np.mean(translation_errors**2)) * 1000.0),
        "translation_median_mm": float(np.median(translation_errors) * 1000.0),
        "translation_max_mm": float(np.max(translation_errors) * 1000.0),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rotation_errors**2))),
        "sim3_scale_stereo_over_inertial": sim3_scale,
        "t_max_diff_sec": float(args.t_max_diff),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
