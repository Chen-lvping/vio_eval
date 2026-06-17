#!/usr/bin/env python3
"""
Record a continuous trajectory of the robot arm end-effector.

Usage:
    python record_trajectory.py                         # record until Ctrl+C
    python record_trajectory.py --freq 50               # 50 Hz sampling
    python record_trajectory.py --duration 10           # record for 10 seconds
    python record_trajectory.py --output path/to/file   # custom output path

Start the script, move the robot arm freely, then stop with Ctrl+C (or wait
for --duration).  The full trajectory (timestamp, position, orientation) is
saved as a single JSON file.
"""

from Robotic_Arm.rm_robot_interface import *
import argparse
import json
import math
import os
import signal
import sys
import time


def rotation_vector_to_quaternion(rx, ry, rz):
    """Convert a rotation vector to a quaternion in xyzw order."""
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]

    sin_half = math.sin(angle / 2.0)
    cos_half = math.cos(angle / 2.0)
    scale = sin_half / angle
    return [rx * scale, ry * scale, rz * scale, cos_half]


def matrix_to_quaternion_xyzw(rotation):
    """Convert a 3x3 rotation matrix to a quaternion in xyzw order."""
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


def matmul3(a, b):
    return [
        [sum(a[row][idx] * b[idx][col] for idx in range(3)) for col in range(3)]
        for row in range(3)
    ]


def axis_rotation(axis, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == "x":
        return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    if axis == "y":
        return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    if axis == "z":
        return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    raise ValueError(f"unsupported axis {axis}")


def rpy_to_quaternion(roll, pitch, yaw):
    """Convert roll-pitch-yaw radians to quaternion using Rz(yaw) * Ry(pitch) * Rx(roll)."""
    rotation = matmul3(
        matmul3(axis_rotation("z", yaw), axis_rotation("y", pitch)),
        axis_rotation("x", roll),
    )
    return matrix_to_quaternion_xyzw(rotation)


def pose_orientation_to_quaternion(rx, ry, rz, orientation_mode):
    if orientation_mode == "rpy":
        return rpy_to_quaternion(rx, ry, rz)
    if orientation_mode == "rotvec":
        return rotation_vector_to_quaternion(rx, ry, rz)
    raise ValueError(f"unsupported orientation mode {orientation_mode}")


class TrajectoryRecorder:
    """Connect to the robot arm and record timed pose samples."""

    def __init__(self, ip, port, orientation_mode):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(ip, port)
        self.orientation_mode = orientation_mode
        print(f"[INFO] connected robot arm id={getattr(self.handle, 'id', 'unknown')}")
        print(f"[INFO] interpreting pose[3:6] as {orientation_mode}")
        self._samples = []

    def cleanup(self):
        self.arm.rm_delete_robot_arm()
        print("[INFO] disconnected robot arm")

    def read_pose(self):
        """Return a dict with timestamp, position, quaternion, and raw pose."""
        timestamp_before = time.time()
        monotonic_before = time.monotonic()
        status, state_data = self.arm.rm_get_current_arm_state()
        timestamp_after = time.time()
        monotonic_after = time.monotonic()
        if status != 0:
            raise RuntimeError(f"failed to get arm state, status={status}")

        pose = state_data["pose"]
        position = [float(pose[0]), float(pose[1]), float(pose[2])]
        orientation_raw = [float(pose[3]), float(pose[4]), float(pose[5])]
        quaternion = pose_orientation_to_quaternion(
            float(pose[3]),
            float(pose[4]),
            float(pose[5]),
            self.orientation_mode,
        )
        return {
            # Use the midpoint of the SDK call as the best available sample time.
            "timestamp": 0.5 * (timestamp_before + timestamp_after),
            "timestamp_before": timestamp_before,
            "timestamp_after": timestamp_after,
            "monotonic_before": monotonic_before,
            "monotonic_after": monotonic_after,
            "read_duration_sec": monotonic_after - monotonic_before,
            "position_m": position,
            "quaternion_xyzw": quaternion,
            "orientation_raw": orientation_raw,
            "orientation_mode": self.orientation_mode,
            "raw_pose": [float(v) for v in pose],
        }

    def record(self, frequency_hz, duration_sec, progress_interval_sec):
        """
        Record trajectory samples at *frequency_hz* until *duration_sec* elapses
        (or until stop() is called via signal handler).
        """
        period = 1.0 / frequency_hz
        self._samples.clear()
        self._stopped = False
        sample_count = 0

        next_progress = time.monotonic() + progress_interval_sec
        start_wall = time.monotonic()

        print(f"[INFO] recording at {frequency_hz} Hz (period={period*1000:.1f} ms)")
        if duration_sec > 0:
            print(f"[INFO] will stop automatically after {duration_sec} s")
        else:
            print("[INFO] press Ctrl+C to stop recording")
        print("[INFO] recording ...")

        while not self._stopped:
            elapsed = time.monotonic() - start_wall
            if duration_sec > 0 and elapsed >= duration_sec:
                print(f"[INFO] reached duration limit ({duration_sec} s)")
                break

            loop_start = time.monotonic()
            try:
                sample = self.read_pose()
            except RuntimeError as exc:
                print(f"[WARN] read failed: {exc}", file=sys.stderr)
                time.sleep(period)
                continue

            self._samples.append(sample)
            sample_count += 1

            if loop_start >= next_progress:
                print(
                    f"  recorded {sample_count} samples, elapsed {elapsed:.1f} s",
                    flush=True,
                )
                next_progress = loop_start + progress_interval_sec

            # Sleep for the remainder of the period (if any)
            worked = time.monotonic() - loop_start
            remaining = period - worked
            if remaining > 0:
                time.sleep(remaining)

        real_duration = time.monotonic() - start_wall
        actual_freq = sample_count / real_duration if real_duration > 0 else 0.0
        print(f"[INFO] recording finished: {sample_count} samples in {real_duration:.2f} s"
              f" ({actual_freq:.1f} Hz actual)")

        return {
            "num_samples": sample_count,
            "duration_sec": real_duration,
            "actual_frequency_hz": actual_freq,
            "samples": self._samples,
        }

    def stop(self):
        """Signal the recording loop to exit at the next iteration."""
        self._stopped = True


def save_trajectory(result, output_path, meta_info):
    """Write the trajectory result to a JSON file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    clean_samples = []
    for s in result["samples"]:
        sample = {
            "timestamp": s["timestamp"],
            "position_m": s["position_m"],
            "quaternion_xyzw": s["quaternion_xyzw"],
            "orientation_raw": s["orientation_raw"],
            "orientation_mode": s["orientation_mode"],
            "read_duration_sec": s["read_duration_sec"],
        }
        sample["raw_pose"] = s["raw_pose"]
        if meta_info.get("save_timing_details", False):
            sample["timestamp_before"] = s["timestamp_before"]
            sample["timestamp_after"] = s["timestamp_after"]
            sample["monotonic_before"] = s["monotonic_before"]
            sample["monotonic_after"] = s["monotonic_after"]
        clean_samples.append(sample)

    payload = {
        "meta": {
            "description": "recorded arm end-effector trajectory",
            "frame": meta_info.get("frame", "base_to_tcp"),
            "num_samples": result["num_samples"],
            "duration_sec": round(result["duration_sec"], 6),
            "actual_frequency_hz": round(result["actual_frequency_hz"], 3),
            "target_frequency_hz": meta_info.get("target_frequency_hz"),
            "target_duration_sec": meta_info.get("target_duration_sec"),
            "orientation_mode": meta_info.get("orientation_mode"),
            "timestamp_policy": "midpoint_of_rm_get_current_arm_state_call",
            "raw_pose_saved": True,
        },
        "samples": clean_samples,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[INFO] saved trajectory ({len(clean_samples)} samples) to {output_path}")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[INFO] file size: {file_size_mb:.2f} MB")


def default_output_path(output_dir):
    """Generate a numbered output path that doesn't overwrite existing files."""
    os.makedirs(output_dir, exist_ok=True)
    pattern = "trajectory_{:03d}.json"
    idx = 1
    while os.path.exists(os.path.join(output_dir, pattern.format(idx))):
        idx += 1
    return os.path.join(output_dir, pattern.format(idx))


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Record a continuous end-effector trajectory of the robot arm."
    )
    parser.add_argument("--ip", default="192.168.1.18", help="robot controller ip")
    parser.add_argument("--port", type=int, default=8080, help="robot controller port")
    parser.add_argument(
        "--freq",
        type=float,
        default=20.0,
        help="sampling frequency in Hz (default 20)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="recording duration in seconds (0 = unlimited, stop with Ctrl+C)",
    )
    parser.add_argument(
        "--frame",
        default="base_to_tcp",
        help="pose frame label saved in the output meta",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=["rpy", "rotvec"],
        default="rpy",
        help="interpret pose[3:6] as roll/pitch/yaw radians (default) or rotation vector",
    )
    parser.add_argument(
        "--save-timing-details",
        action="store_true",
        help="save timestamp_before/after and monotonic_before/after for each SDK read",
    )
    parser.add_argument(
        "--output",
        help="output JSON path (default: vio_eval/data/ground_truth/trajectory_samples/trajectory_NNN.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "ground_truth",
            "trajectory_samples",
        ),
        help="directory for auto-named trajectory outputs",
    )
    parser.add_argument(
        "--progress",
        type=float,
        default=5.0,
        help="print progress every N seconds (default 5)",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.freq <= 0:
        raise ValueError("--freq must be positive")
    if args.duration < 0:
        raise ValueError("--duration must be non-negative")
    if args.progress <= 0:
        raise ValueError("--progress must be positive")

    recorder = TrajectoryRecorder(args.ip, args.port, args.orientation_mode)

    # Install signal handler for graceful Ctrl+C stop.
    def signal_handler(signum, frame):
        print("\n[INFO] received SIGINT, stopping recording ...")
        recorder.stop()

    original_sigint = signal.signal(signal.SIGINT, signal_handler)

    try:
        result = recorder.record(
            frequency_hz=args.freq,
            duration_sec=args.duration,
            progress_interval_sec=args.progress,
        )
    finally:
        # Restore default SIGINT behaviour.
        signal.signal(signal.SIGINT, original_sigint)
        recorder.cleanup()

    if result["num_samples"] == 0:
        print("[WARN] no samples recorded, nothing saved", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or default_output_path(args.output_dir)
    meta_info = {
        "frame": args.frame,
        "target_frequency_hz": args.freq,
        "target_duration_sec": args.duration if args.duration > 0 else None,
        "orientation_mode": args.orientation_mode,
        "save_timing_details": args.save_timing_details,
    }
    save_trajectory(result, output_path, meta_info)


if __name__ == "__main__":
    main()
