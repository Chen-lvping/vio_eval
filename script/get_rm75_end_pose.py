#!/usr/bin/env python3
"""Read the current RM-75-B end-effector pose with quaternion orientation."""

import argparse
import json
import sys
import time
from typing import Any, Dict, List


def _load_sdk():
    if sys.version_info < (3, 9):
        raise RuntimeError(
            "RealMan Robotic_Arm Python API2 requires Python 3.9 or newer. "
            f"Current interpreter is {sys.version.split()[0]}."
        )

    try:
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import RealMan SDK. Install it with: pip install Robotic_Arm"
        ) from exc

    return RoboticArm, rm_thread_mode_e


def _as_float_list(values: Any, expected_len: int, name: str) -> List[float]:
    try:
        result = [float(v) for v in values]
    except TypeError as exc:
        raise ValueError(f"{name} is not a sequence: {values!r}") from exc

    if len(result) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values, got {len(result)}")
    return result


def _extract_pose_euler(state: Dict[str, Any]) -> List[float]:
    """Return [x, y, z, rx, ry, rz] from the SDK state dictionary."""
    pose = state.get("pose")
    if isinstance(pose, dict):
        position = pose.get("position", {})
        euler = pose.get("euler", {})
        return [
            float(position["x"]),
            float(position["y"]),
            float(position["z"]),
            float(euler["rx"]),
            float(euler["ry"]),
            float(euler["rz"]),
        ]

    return _as_float_list(pose, 6, "state['pose']")


def read_end_pose(ip: str, port: int) -> Dict[str, Any]:
    RoboticArm, rm_thread_mode_e = _load_sdk()

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(ip, port)
    if getattr(handle, "id", -1) == -1:
        raise RuntimeError(f"Failed to connect to robot controller at {ip}:{port}")

    try:
        request_time_s = time.time()
        ret, state = arm.rm_get_current_arm_state()
        sample_time_s = time.time()
        if ret != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed, return code: {ret}")

        pose_euler = _extract_pose_euler(state)
        quat_wxyz = _as_float_list(arm.rm_algo_euler2quaternion(pose_euler[3:]), 4, "quaternion")

        return {
            "model": "RM-75-B",
            "frame": "current work frame / current tool TCP",
            "timestamp_s": sample_time_s,
            "timestamp_source": "host_after_rm_get_current_arm_state",
            "api_call_duration_s": sample_time_s - request_time_s,
            "position_m": {
                "x": pose_euler[0],
                "y": pose_euler[1],
                "z": pose_euler[2],
            },
            "quaternion_wxyz": {
                "w": quat_wxyz[0],
                "x": quat_wxyz[1],
                "y": quat_wxyz[2],
                "z": quat_wxyz[3],
            },
            "euler_rad": {
                "rx": pose_euler[3],
                "ry": pose_euler[4],
                "rz": pose_euler[5],
            },
            "joint_deg": state.get("joint"),
            "raw_state": state,
        }
    finally:
        arm.rm_delete_robot_arm()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Get current RM-75-B end-effector pose via RealMan Python API2."
    )
    parser.add_argument("ip", help="Robot controller IP, for example 192.168.1.18")
    parser.add_argument("--port", type=int, default=8080, help="Robot controller port, default: 8080")
    args = parser.parse_args()

    try:
        result = read_end_pose(args.ip, args.port)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
