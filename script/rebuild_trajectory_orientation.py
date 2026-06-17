#!/usr/bin/env python3
"""Rebuild trajectory quaternions from saved raw_pose with different orientation modes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


def rotation_vector_to_quaternion(rx: float, ry: float, rz: float) -> List[float]:
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    sin_half = math.sin(angle / 2.0)
    cos_half = math.cos(angle / 2.0)
    scale = sin_half / angle
    return [rx * scale, ry * scale, rz * scale, cos_half]


def axis_rotation(axis: str, angle: float) -> List[List[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == "x":
        return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    if axis == "y":
        return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    if axis == "z":
        return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    raise ValueError(f"unsupported axis {axis}")


def matmul3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [
        [sum(a[row][idx] * b[idx][col] for idx in range(3)) for col in range(3)]
        for row in range(3)
    ]


def matrix_to_quaternion_xyzw(rotation: List[List[float]]) -> List[float]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2][1] - rotation[1][2]) / scale
        qy = (rotation[0][2] - rotation[2][0]) / scale
        qz = (rotation[1][0] - rotation[0][1]) / scale
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
        qw = (rotation[2][1] - rotation[1][2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0][1] + rotation[1][0]) / scale
        qz = (rotation[0][2] + rotation[2][0]) / scale
    elif rotation[1][1] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
        qw = (rotation[0][2] - rotation[2][0]) / scale
        qx = (rotation[0][1] + rotation[1][0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1][2] + rotation[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
        qw = (rotation[1][0] - rotation[0][1]) / scale
        qx = (rotation[0][2] + rotation[2][0]) / scale
        qy = (rotation[1][2] + rotation[2][1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("quaternion norm is too small")
    return [qx / norm, qy / norm, qz / norm, qw / norm]


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
    rotation = matmul3(
        matmul3(axis_rotation("z", yaw), axis_rotation("y", pitch)),
        axis_rotation("x", roll),
    )
    return matrix_to_quaternion_xyzw(rotation)


def pose_orientation_to_quaternion(rx: float, ry: float, rz: float, orientation_mode: str) -> List[float]:
    if orientation_mode == "rpy":
        return rpy_to_quaternion(rx, ry, rz)
    if orientation_mode == "rotvec":
        return rotation_vector_to_quaternion(rx, ry, rz)
    raise ValueError(f"unsupported orientation mode {orientation_mode}")


def quat_angle_deg(q0: List[float], q1: List[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(q0, q1))
    dot = max(-1.0, min(1.0, abs(dot)))
    return math.degrees(2.0 * math.acos(dot))


def rebuild(payload: Dict[str, object], mode: str) -> Dict[str, object]:
    samples = payload.get("samples", [])
    rebuilt_samples = []
    diffs_deg = []
    for sample in samples:
        raw_pose = sample.get("raw_pose")
        if raw_pose is None or len(raw_pose) < 6:
            raise ValueError("all samples must contain raw_pose[0:6]")
        rx, ry, rz = [float(v) for v in raw_pose[3:6]]
        rebuilt_quat = pose_orientation_to_quaternion(rx, ry, rz, mode)
        rebuilt = dict(sample)
        rebuilt["quaternion_xyzw"] = rebuilt_quat
        rebuilt["orientation_mode"] = mode
        if "quaternion_xyzw" in sample:
            diffs_deg.append(quat_angle_deg(sample["quaternion_xyzw"], rebuilt_quat))
        rebuilt_samples.append(rebuilt)

    meta = dict(payload.get("meta", {}))
    meta["orientation_mode"] = mode
    meta["rebuilt_from_raw_pose"] = True
    meta["source_file_orientation_mode"] = payload.get("meta", {}).get("orientation_mode")
    if diffs_deg:
        meta["source_vs_rebuilt_quaternion_angle_mean_deg"] = sum(diffs_deg) / len(diffs_deg)
        meta["source_vs_rebuilt_quaternion_angle_max_deg"] = max(diffs_deg)
    return {"meta": meta, "samples": rebuilt_samples}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="trajectory JSON with raw_pose")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["rpy", "rotvec"],
        default=["rpy", "rotvec"],
        help="orientation variants to export",
    )
    parser.add_argument("--output-dir", type=Path, help="defaults to input parent directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else input_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    for mode in args.modes:
        rebuilt = rebuild(payload, mode)
        output_path = output_dir / f"{input_path.stem}_{mode}.json"
        output_path.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=2), encoding="utf-8")
        meta = rebuilt["meta"]
        print(
            f"[OK] wrote {output_path} "
            f"(mean_delta_deg={meta.get('source_vs_rebuilt_quaternion_angle_mean_deg', float('nan')):.6f}, "
            f"max_delta_deg={meta.get('source_vs_rebuilt_quaternion_angle_max_deg', float('nan')):.6f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
