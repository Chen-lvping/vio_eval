from Robotic_Arm.rm_robot_interface import *
import argparse
import json
import math
import os
import re
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

    return normalize_quaternion([qx, qy, qz, qw])


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
    """Convert RealMan rx,ry,rz Euler angles using Rz(yaw) * Ry(pitch) * Rx(roll)."""
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


def normalize_quaternion(quaternion):
    norm = math.sqrt(sum(v * v for v in quaternion))
    if norm < 1e-12:
        raise ValueError("quaternion norm is too small")
    return [v / norm for v in quaternion]


def quaternion_dot(q1, q2):
    return sum(a * b for a, b in zip(q1, q2))


def average_quaternions(quaternions):
    """Average quaternions with hemisphere alignment."""
    if not quaternions:
        raise ValueError("no quaternions to average")

    reference = normalize_quaternion(quaternions[0])
    accumulator = [0.0, 0.0, 0.0, 0.0]
    aligned = []

    for quaternion in quaternions:
        current = normalize_quaternion(quaternion)
        if quaternion_dot(current, reference) < 0.0:
            current = [-v for v in current]
        aligned.append(current)
        for idx, value in enumerate(current):
            accumulator[idx] += value

    mean_quaternion = normalize_quaternion(accumulator)
    return mean_quaternion, aligned


def compute_position_stats(positions):
    count = len(positions)
    mean = [sum(p[idx] for p in positions) / count for idx in range(3)]
    std = []
    for idx in range(3):
        variance = sum((p[idx] - mean[idx]) ** 2 for p in positions) / count
        std.append(math.sqrt(variance))
    return mean, std


def compute_orientation_errors_deg(quaternions, mean_quaternion):
    errors = []
    for quaternion in quaternions:
        dot = abs(quaternion_dot(quaternion, mean_quaternion))
        dot = max(-1.0, min(1.0, dot))
        errors.append(math.degrees(2.0 * math.acos(dot)))
    mean_error = sum(errors) / len(errors)
    max_error = max(errors)
    return errors, mean_error, max_error


class StaticPoseCollector:
    def __init__(self, ip, port, orientation_mode):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(ip, port)
        self.orientation_mode = orientation_mode
        print(f"[INFO] connected robot arm id={getattr(self.handle, 'id', 'unknown')}")
        print("[INFO] this script only reads pose and does not command robot motion")
        print(f"[INFO] interpreting pose[3:6] as {orientation_mode}")

    def cleanup(self):
        self.arm.rm_delete_robot_arm()
        print("[INFO] disconnected robot arm")

    def read_pose(self):
        status, state_data = self.arm.rm_get_current_arm_state()
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
            "timestamp": time.time(),
            "position_m": position,
            "quaternion_xyzw": quaternion,
            "orientation_raw": orientation_raw,
            "orientation_mode": self.orientation_mode,
            "raw_pose": [float(value) for value in pose],
        }

    def capture_average_pose(self, sample_count, interval_sec, settle_sec, frame_name):
        if settle_sec > 0.0:
            print(f"[INFO] waiting {settle_sec:.2f}s for settling")
            time.sleep(settle_sec)

        samples = []
        for idx in range(sample_count):
            sample = self.read_pose()
            samples.append(sample)
            position = sample["position_m"]
            print(
                "[SAMPLE {:02d}/{:02d}] t={:.6f} pos=({:.6f}, {:.6f}, {:.6f})".format(
                    idx + 1,
                    sample_count,
                    sample["timestamp"],
                    position[0],
                    position[1],
                    position[2],
                )
            )
            if idx + 1 < sample_count:
                time.sleep(interval_sec)

        positions = [sample["position_m"] for sample in samples]
        quaternions = [sample["quaternion_xyzw"] for sample in samples]

        mean_position, std_position = compute_position_stats(positions)
        mean_quaternion, aligned_quaternions = average_quaternions(quaternions)
        _, mean_angle_error_deg, max_angle_error_deg = compute_orientation_errors_deg(
            aligned_quaternions,
            mean_quaternion,
        )

        return {
            "sample_id": None,
            "timestamp_start": samples[0]["timestamp"],
            "timestamp_end": samples[-1]["timestamp"],
            "timestamp_mean": sum(sample["timestamp"] for sample in samples) / len(samples),
            "frame": frame_name,
            "sample_count": sample_count,
            "sample_interval_sec": interval_sec,
            "settle_sec": settle_sec,
            "position_m": mean_position,
            "quaternion_xyzw": mean_quaternion,
            "orientation_mode": self.orientation_mode,
            "position_std_m": std_position,
            "orientation_mean_error_deg": mean_angle_error_deg,
            "orientation_max_error_deg": max_angle_error_deg,
            "samples": samples,
        }


def save_result(result, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(f"[INFO] saved result to {output_path}")


def detect_existing_capture_indices(output_dir):
    if not os.path.isdir(output_dir):
        return []

    pattern = re.compile(r"^static_pose_(\d+)\.json$")
    indices = []
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def next_capture_index(output_dir):
    indices = detect_existing_capture_indices(output_dir)
    return (indices[-1] + 1) if indices else 1


def default_output_path(output_dir, capture_index):
    return os.path.join(output_dir, f"static_pose_{capture_index:03d}.json")


def summary_output_path(output_dir):
    return os.path.join(output_dir, "static_pose_summary.json")


def build_summary_entry(result, output_path):
    return {
        "sample_id": result["sample_id"],
        "file": os.path.basename(output_path),
        "timestamp_mean": result["timestamp_mean"],
        "frame": result["frame"],
        "position_m": result["position_m"],
        "quaternion_xyzw": result["quaternion_xyzw"],
        "orientation_mode": result["orientation_mode"],
        "position_std_m": result["position_std_m"],
        "orientation_mean_error_deg": result["orientation_mean_error_deg"],
        "orientation_max_error_deg": result["orientation_max_error_deg"],
    }


def update_summary_file(output_dir, summary_entry):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = summary_output_path(output_dir)
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as file:
            summary_data = json.load(file)
    else:
        summary_data = {"captures": []}

    captures = [
        item
        for item in summary_data.get("captures", [])
        if item.get("sample_id") != summary_entry["sample_id"]
    ]
    captures.append(summary_entry)
    captures.sort(key=lambda item: item["sample_id"])
    summary_data["captures"] = captures

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary_data, file, indent=2, ensure_ascii=False)

    return summary_path


def resolve_output_path(explicit_output, output_dir, capture_index, interactive_mode):
    if not explicit_output:
        return default_output_path(output_dir, capture_index)

    if not interactive_mode or capture_index == 1:
        return explicit_output

    base, ext = os.path.splitext(explicit_output)
    suffix = f"_{capture_index:02d}"
    return f"{base}{suffix}{ext or '.json'}"


def print_summary(result):
    pos = result["position_m"]
    quat = result["quaternion_xyzw"]
    pos_std = result["position_std_m"]
    print("")
    print("[RESULT] average pose")
    print(
        "position_m         = [{:.6f}, {:.6f}, {:.6f}]".format(
            pos[0], pos[1], pos[2]
        )
    )
    print(
        "quaternion_xyzw    = [{:.8f}, {:.8f}, {:.8f}, {:.8f}]".format(
            quat[0], quat[1], quat[2], quat[3]
        )
    )
    print(
        "position_std_m     = [{:.6e}, {:.6e}, {:.6e}]".format(
            pos_std[0], pos_std[1], pos_std[2]
        )
    )
    print(
        "orientation_err_deg = mean {:.6f}, max {:.6f}".format(
            result["orientation_mean_error_deg"],
            result["orientation_max_error_deg"],
        )
    )


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Capture an averaged static end-effector pose without moving the robot arm."
    )
    parser.add_argument("--ip", default="192.168.1.18", help="robot controller ip")
    parser.add_argument("--port", type=int, default=8080, help="robot controller port")
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="number of samples to average after the robot has settled",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="seconds between samples",
    )
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=1.0,
        help="extra wait time after you confirm the robot is stable",
    )
    parser.add_argument(
        "--frame",
        default="base_to_tcp",
        help="pose frame label to save, for example base_to_tcp or base_to_flange",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=["rpy", "rotvec"],
        default="rpy",
        help="interpret pose[3:6] as RealMan Euler RPY or as a rotation vector",
    )
    parser.add_argument(
        "--output",
        help="json path for a single capture; defaults to vio_eval/data/calibration/static_pose_samples/",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "calibration",
            "static_pose_samples",
        ),
        help="directory for auto-named json outputs",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="capture one result immediately without interactive prompts",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.interval < 0.0:
        raise ValueError("--interval must be non-negative")
    if args.settle_sec < 0.0:
        raise ValueError("--settle-sec must be non-negative")

    collector = StaticPoseCollector(args.ip, args.port, args.orientation_mode)
    auto_capture_index = next_capture_index(args.output_dir) if not args.output else 1

    try:
        while True:
            if not args.once:
                print("")
                print("Press Enter after the robot is fully still to capture one averaged pose.")
                print("Input q then Enter to quit.")
                user_input = input("> ").strip().lower()
                if user_input == "q":
                    break

            capture_index = auto_capture_index
            print(
                f"[INFO] capture #{capture_index}: {args.samples} samples, interval {args.interval:.3f}s"
            )
            result = collector.capture_average_pose(
                sample_count=args.samples,
                interval_sec=args.interval,
                settle_sec=args.settle_sec,
                frame_name=args.frame,
            )
            result["sample_id"] = capture_index
            print_summary(result)

            output_path = resolve_output_path(
                explicit_output=args.output,
                output_dir=args.output_dir,
                capture_index=capture_index,
                interactive_mode=not args.once,
            )
            save_result(result, output_path)
            if not args.output:
                summary_path = update_summary_file(
                    args.output_dir,
                    build_summary_entry(result, output_path),
                )
                print(f"[INFO] updated summary {summary_path}")
            print(f"[OK] capture #{capture_index} completed")
            print("[INFO] press Enter when the next pose is fully still")
            auto_capture_index += 1

            if args.once:
                break

            # Avoid overwriting when --output is omitted and the next capture starts quickly.
            time.sleep(0.2)

    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()
