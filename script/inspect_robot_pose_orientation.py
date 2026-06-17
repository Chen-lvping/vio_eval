#!/usr/bin/env python3
"""
Inspect how the robot SDK pose[3:6] should be interpreted.

The script reads rm_get_current_arm_state() repeatedly and prints the same raw
orientation as both RPY and rotation-vector. It also reports each mode's
relative rotation from the first sample, which is useful when you manually
rotate the TCP around one axis.
"""

from __future__ import annotations

from Robotic_Arm.rm_robot_interface import *
import argparse
import json
import math
import time
from pathlib import Path


def matmul3(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)] for r in range(3)]


def transpose3(a):
    return [[a[c][r] for c in range(3)] for r in range(3)]


def axis_rotation(axis, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == "x":
        return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    if axis == "y":
        return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    if axis == "z":
        return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    raise ValueError(axis)


def rpy_to_rotation(roll, pitch, yaw):
    return matmul3(matmul3(axis_rotation("z", yaw), axis_rotation("y", pitch)), axis_rotation("x", roll))


def rotvec_to_rotation(rx, ry, rz):
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x, y, z = rx / angle, ry / angle, rz / angle
    c = math.cos(angle)
    s = math.sin(angle)
    v = 1.0 - c
    return [
        [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
        [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
        [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
    ]


def matrix_to_quaternion_xyzw(rotation):
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
    return [qx / norm, qy / norm, qz / norm, qw / norm]


def rotation_angle_deg(rotation):
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    value = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(value))


def relative_angle_deg(reference, current):
    return rotation_angle_deg(matmul3(transpose3(reference), current))


def fmt(values, digits=6):
    return "[" + ", ".join(f"{float(v): .{digits}f}" for v in values) + "]"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.1.18")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--freq", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 means run until Ctrl+C")
    parser.add_argument("--output", type=Path, help="optional JSON log path")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.freq <= 0:
        raise ValueError("--freq must be positive")

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(args.ip, args.port)
    print(f"[INFO] connected robot arm id={getattr(handle, 'id', 'unknown')}")
    print("[INFO] Keep the first sample still, then rotate TCP around one axis.")
    print("[INFO] Compare rel_rpy_deg and rel_rotvec_deg with the motion you actually made.")

    first_rpy = None
    first_rotvec = None
    samples = []
    period = 1.0 / args.freq
    start = time.monotonic()
    try:
        while True:
            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break
            status, state_data = arm.rm_get_current_arm_state()
            if status != 0:
                print(f"[WARN] read failed: status={status}")
                time.sleep(period)
                continue
            pose = [float(v) for v in state_data["pose"]]
            raw = pose[3:6]
            rpy_r = rpy_to_rotation(*raw)
            rotvec_r = rotvec_to_rotation(*raw)
            if first_rpy is None:
                first_rpy = rpy_r
                first_rotvec = rotvec_r
            rel_rpy = relative_angle_deg(first_rpy, rpy_r)
            rel_rotvec = relative_angle_deg(first_rotvec, rotvec_r)
            q_rpy = matrix_to_quaternion_xyzw(rpy_r)
            q_rotvec = matrix_to_quaternion_xyzw(rotvec_r)
            sample = {
                "timestamp": time.time(),
                "pose": pose,
                "position_m": pose[:3],
                "orientation_raw": raw,
                "quaternion_rpy_xyzw": q_rpy,
                "quaternion_rotvec_xyzw": q_rotvec,
                "rel_rpy_deg": rel_rpy,
                "rel_rotvec_deg": rel_rotvec,
            }
            samples.append(sample)
            print(
                f"pos={fmt(pose[:3], 5)} raw={fmt(raw, 5)} "
                f"rel_rpy_deg={rel_rpy:7.3f} rel_rotvec_deg={rel_rotvec:7.3f} "
                f"q_rpy={fmt(q_rpy, 4)} q_rotvec={fmt(q_rotvec, 4)}",
                flush=True,
            )
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[INFO] stopped")
    finally:
        arm.rm_delete_robot_arm()
        print("[INFO] disconnected robot arm")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"samples": samples}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] wrote {args.output}")


if __name__ == "__main__":
    main()
