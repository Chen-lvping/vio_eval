#!/usr/bin/env python3
"""Blind smoothing for pose CSV trajectories.

This is intentionally independent from COLMAP. It smooths only the estimated
trajectory and can optionally shift output timestamps for evaluation copies.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R


POSE_CSV_COLUMNS = ["Timestamp_us", "Vx", "Vy", "Vz", "X", "Y", "Z", "Quat_X", "Quat_Y", "Quat_Z", "Quat_W"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--position-window", type=int, default=21)
    parser.add_argument("--position-poly", type=int, default=2)
    parser.add_argument("--rotation-window", type=int, default=9)
    parser.add_argument("--timestamp-offset-sec", type=float, default=0.0)
    parser.add_argument("--copy-only", action="store_true", help="Only rewrite timestamps/velocities; do not smooth pose values.")
    return parser.parse_args()


def normalize_quat(values: Iterable[float]) -> np.ndarray:
    quat = np.asarray(list(values), dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if quat.shape != (4,) or norm <= 0.0:
        raise ValueError("invalid quaternion")
    quat /= norm
    return -quat if quat[3] < 0 else quat


def load_pose_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    stamps: List[int] = []
    positions: List[List[float]] = []
    quats: List[np.ndarray] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stamps.append(int(round(float(row["Timestamp_us"]))))
            positions.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
            quats.append(normalize_quat([float(row["Quat_X"]), float(row["Quat_Y"]), float(row["Quat_Z"]), float(row["Quat_W"])]))
    if not stamps:
        raise ValueError(f"{path} contains no pose rows")
    return np.asarray(stamps, dtype=np.int64), np.asarray(positions, dtype=np.float64), np.asarray(quats, dtype=np.float64)


def valid_odd_window(requested: int, size: int, poly: int) -> int:
    if size <= poly + 1:
        return 0
    window = max(poly + 2, requested)
    if window % 2 == 0:
        window += 1
    max_window = size if size % 2 == 1 else size - 1
    window = min(window, max_window)
    if window <= poly:
        return 0
    return window


def smooth_positions(positions: np.ndarray, window: int, poly: int) -> np.ndarray:
    actual = valid_odd_window(window, len(positions), poly)
    if actual == 0:
        return positions.copy()
    return np.vstack([savgol_filter(positions[:, axis], actual, poly, mode="interp") for axis in range(3)]).T


def make_quat_signs_continuous(quats: np.ndarray) -> np.ndarray:
    out = quats.copy()
    for idx in range(1, len(out)):
        if float(np.dot(out[idx - 1], out[idx])) < 0.0:
            out[idx] *= -1.0
    return out


def smooth_rotations(quats: np.ndarray, window: int) -> np.ndarray:
    actual = valid_odd_window(window, len(quats), 0)
    if actual == 0:
        return make_quat_signs_continuous(quats)
    rots = R.from_quat(make_quat_signs_continuous(quats))
    half = actual // 2
    out = []
    for idx in range(len(rots)):
        start = max(0, idx - half)
        end = min(len(rots), idx + half + 1)
        center = rots[idx]
        rel = (center.inv() * rots[start:end]).as_rotvec()
        quat = (center * R.from_rotvec(rel.mean(axis=0))).as_quat()
        if quat[3] < 0:
            quat = -quat
        out.append(quat)
    return np.asarray(out, dtype=np.float64)


def compute_velocities(stamps_us: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if len(stamps_us) == 1:
        return np.zeros((1, 3), dtype=np.float64)
    velocities = []
    times_s = stamps_us.astype(np.float64) * 1e-6
    for idx in range(len(stamps_us)):
        prev_idx = max(0, idx - 1)
        next_idx = min(len(stamps_us) - 1, idx + 1)
        dt = float(times_s[next_idx] - times_s[prev_idx])
        velocities.append(np.zeros(3, dtype=np.float64) if dt <= 0.0 else (positions[next_idx] - positions[prev_idx]) / dt)
    return np.asarray(velocities, dtype=np.float64)


def write_pose_csv(path: Path, stamps_us: np.ndarray, positions: np.ndarray, quats: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    velocities = compute_velocities(stamps_us, positions)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(POSE_CSV_COLUMNS)
        for stamp_us, velocity, position, quat in zip(stamps_us, velocities, positions, quats):
            writer.writerow(
                [
                    int(stamp_us),
                    f"{velocity[0]:.9f}",
                    f"{velocity[1]:.9f}",
                    f"{velocity[2]:.9f}",
                    f"{position[0]:.9f}",
                    f"{position[1]:.9f}",
                    f"{position[2]:.9f}",
                    f"{quat[0]:.9f}",
                    f"{quat[1]:.9f}",
                    f"{quat[2]:.9f}",
                    f"{quat[3]:.9f}",
                ]
            )


def main() -> int:
    args = parse_args()
    stamps_us, positions, quats = load_pose_csv(args.input_csv.expanduser())
    if args.copy_only:
        smoothed_positions = positions.copy()
        smoothed_quats = make_quat_signs_continuous(quats)
    else:
        smoothed_positions = smooth_positions(positions, args.position_window, args.position_poly)
        smoothed_quats = smooth_rotations(quats, args.rotation_window)
    shifted_stamps = stamps_us + int(round(args.timestamp_offset_sec * 1_000_000.0))
    order = np.argsort(shifted_stamps)
    write_pose_csv(args.output_csv.expanduser(), shifted_stamps[order], smoothed_positions[order], smoothed_quats[order])
    print(f"[Smooth] wrote {len(stamps_us)} poses: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
