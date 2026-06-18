#!/usr/bin/env python3
"""Record RM-75-B end-effector pose trajectory with quaternion orientation."""

import argparse
from ctypes import byref
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple


DEFAULT_IP = "192.168.1.18"
DEFAULT_PORT = 8080
DEFAULT_RATE_HZ = 50.0


def _load_sdk():
    if sys.version_info < (3, 9):
        raise RuntimeError(
            "RealMan Robotic_Arm Python API2 requires Python 3.9 or newer. "
            f"Current interpreter is {sys.version.split()[0]}."
        )

    try:
        from Robotic_Arm.rm_robot_interface import (
            RoboticArm,
            rm_current_arm_state_t,
            rm_get_current_arm_state,
            rm_thread_mode_e,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import RealMan SDK. Install it with: pip install Robotic_Arm"
        ) from exc

    return RoboticArm, rm_thread_mode_e, rm_current_arm_state_t, rm_get_current_arm_state


def _read_current_arm_state_direct(arm: Any, state_cls: Any, get_state_func: Any) -> Tuple[int, Any]:
    state = state_cls()
    ret = get_state_func(arm.handle, byref(state))
    return ret, state


def _state_to_pose_sample(state: Any, timestamp_s: float) -> Dict[str, Any]:
    pose = state.pose
    position = pose.position
    quat = pose.quaternion

    return {
        "timestamp_s": timestamp_s,
        "position_m": {
            "x": float(position.x),
            "y": float(position.y),
            "z": float(position.z),
        },
        "quaternion_wxyz": {
            "w": float(quat.w),
            "x": float(quat.x),
            "y": float(quat.y),
            "z": float(quat.z),
        },
    }


def _read_pose_sample(arm: Any, state_cls: Any, get_state_func: Any) -> Dict[str, Any]:
    ret, state = _read_current_arm_state_direct(arm, state_cls, get_state_func)
    sample_time_s = time.time()
    if ret != 0:
        raise RuntimeError(f"rm_get_current_arm_state failed, return code: {ret}")
    return _state_to_pose_sample(state, sample_time_s)


def record_trajectory(ip: str, port: int, rate_hz: float) -> Dict[str, Any]:
    if rate_hz <= 0:
        raise ValueError("--rate-hz must be greater than 0")

    RoboticArm, rm_thread_mode_e, state_cls, get_state_func = _load_sdk()

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(ip, port)
    if getattr(handle, "id", -1) == -1:
        raise RuntimeError(f"Failed to connect to robot controller at {ip}:{port}")

    samples: List[Dict[str, Any]] = []
    start_time_s = time.time()
    start_monotonic_s = time.perf_counter()
    period_s = 1.0 / rate_hz

    try:
        while True:
            samples.append(_read_pose_sample(arm, state_cls, get_state_func))
            next_sample_s = start_monotonic_s + len(samples) * period_s
            sleep_s = next_sample_s - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass
    finally:
        arm.rm_delete_robot_arm()

    end_time_s = time.time()
    actual_duration_s = end_time_s - start_time_s
    actual_rate_hz = len(samples) / actual_duration_s if actual_duration_s > 0 else 0.0

    return {
        "model": "RM-75-B",
        "frame": "current work frame / current tool TCP",
        "ip": ip,
        "port": port,
        "target_rate_hz": rate_hz,
        "actual_rate_hz": actual_rate_hz,
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "duration_s": actual_duration_s,
        "sample_count": len(samples),
        "timestamp_source": "host_after_rm_get_current_arm_state",
        "samples": samples,
    }


def save_trajectory(path: str, trajectory: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record RM-75-B end-effector pose trajectory via RealMan Python API2."
    )
    parser.add_argument("-o", "--output", required=True, help="Output trajectory JSON file path")
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"Robot controller IP, default: {DEFAULT_IP}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Robot controller port, default: {DEFAULT_PORT}")
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ, help=f"Sampling rate in Hz, default: {DEFAULT_RATE_HZ:g}")
    args = parser.parse_args()

    try:
        result = record_trajectory(args.ip, args.port, args.rate_hz)
        save_trajectory(args.output, result)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Saved {result['sample_count']} samples to {args.output} "
        f"(target {result['target_rate_hz']:.3f} Hz, actual {result['actual_rate_hz']:.3f} Hz)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())