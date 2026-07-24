#!/usr/bin/env python3
"""Derive one equivalent RPY triple from trajectory quaternions.

This does not recover original raw_pose. It only computes one Euler-angle
parameterization consistent with the saved quaternion under the same
Rz(yaw) * Ry(pitch) * Rx(roll) convention used by record_trajectory.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


def normalize_quaternion_xyzw(q: List[float]) -> List[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm < 1e-12:
        raise ValueError("invalid quaternion norm")
    return [float(v) / norm for v in q]


def quat_xyzw_to_rot(q: List[float]) -> List[List[float]]:
    x, y, z, w = normalize_quaternion_xyzw(q)
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def rot_to_equivalent_rpy(rotation: List[List[float]]) -> List[float]:
    """Inverse of R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    r20 = float(rotation[2][0])
    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    cp = math.cos(pitch)

    if abs(cp) > 1e-9:
        roll = math.atan2(float(rotation[2][1]), float(rotation[2][2]))
        yaw = math.atan2(float(rotation[1][0]), float(rotation[0][0]))
    else:
        # Gimbal lock: pick roll = 0 and absorb the ambiguity into yaw.
        roll = 0.0
        yaw = math.atan2(-float(rotation[0][1]), float(rotation[1][1]))
    return [roll, pitch, yaw]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="trajectory JSON with quaternion_xyzw")
    parser.add_argument("--output", type=Path, help="default: sibling *_equivalent_rpy.json")
    parser.add_argument("--csv", type=Path, help="optional CSV output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if len(samples) < 1:
        raise ValueError(f"{input_path} contains no samples")

    out_samples = []
    for sample in samples:
        q = [float(v) for v in sample["quaternion_xyzw"]]
        rpy = rot_to_equivalent_rpy(quat_xyzw_to_rot(q))
        rebuilt = dict(sample)
        rebuilt["equivalent_rpy"] = rpy
        out_samples.append(rebuilt)

    meta = dict(payload.get("meta", {}))
    meta["equivalent_rpy_convention"] = "Rz(yaw) * Ry(pitch) * Rx(roll)"
    meta["equivalent_rpy_is_not_raw_pose"] = True

    out_payload = {"meta": meta, "samples": out_samples}
    output_path = args.output.expanduser().resolve() if args.output else input_path.with_name(f"{input_path.stem}_equivalent_rpy.json")
    output_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.csv:
        csv_path = args.csv.expanduser().resolve()
        import csv

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "timestamp", "roll", "pitch", "yaw"])
            for i, sample in enumerate(out_samples):
                r, p, y = sample["equivalent_rpy"]
                writer.writerow([i, sample["timestamp"], r, p, y])

    print(f"[OK] wrote {output_path}")
    if args.csv:
        print(f"[OK] wrote {args.csv.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
