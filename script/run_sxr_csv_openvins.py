#!/usr/bin/env python3
"""Run OpenVINS stereo-inertial on an SXR DatasetRecorder CSV episode.

The CSV recorder stores side-by-side video plus independent IMU CSV files,
whereas the older OpenVINS adapter expects an MCAP recording. This adapter
creates a short-lived ROS bag inside the existing OpenVINS Docker image and
evaluates its native IMU/body pose only after the VIO run completes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from trajectory_scale_gate import require_reasonable_scale

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "openvins-noetic-ready:latest"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stream", choices=("rgb", "tracking", "ctrl"), default="tracking")
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--profile", choices=("robust", "default", "static-bootstrap"), default="robust")
    parser.add_argument("--docker-image", default=DEFAULT_IMAGE)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--keep-bag", action="store_true")
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--extrinsics-convention",
        choices=("imu-to-camera", "camera-to-imu"),
        default="imu-to-camera",
        help="Factory camera pose convention; tracking calibration is imu-to-camera.",
    )
    parser.add_argument(
        "--camera-quaternion-order",
        choices=("wxyz", "xyzw"),
        default="xyzw",
        help="Quaternion order in camera_params_<stream>.json.",
    )
    parser.add_argument(
        "--camera-from-imu-rotation-json",
        type=Path,
        default=None,
        help="Optional validated camera-from-IMU rotation override, matching the Basalt tracking setup.",
    )
    return parser.parse_args(argv)


def inverse(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def quat_to_matrix(values: Sequence[float], order: str) -> np.ndarray:
    if order == "wxyz":
        w, x, y, z = (float(value) for value in values)
    else:
        x, y, z, w = (float(value) for value in values)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("zero-norm camera extrinsic quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def factory_t_imu_camera(camera: dict[str, Any], convention: str, quaternion_order: str) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_to_matrix(camera["extrinsics"]["rotation"], quaternion_order)
    transform[:3, 3] = np.asarray(camera["extrinsics"]["position"], dtype=np.float64)
    return transform if convention == "imu-to-camera" else inverse(transform)


def load_camera_from_imu_rotation(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    rotation = np.asarray(payload["extrinsic_rotation_validation"]["fitted_rotation_camera_from_imu"], dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"invalid camera-from-IMU rotation in {path}")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-3 or np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1e-3:
        raise ValueError(f"camera-from-IMU rotation in {path} is not proper")
    return rotation


def camera_time_offset_sec(calibration: dict[str, Any], stream: str, override: float | None = None) -> float:
    names = {"rgb": "rgb-left", "tracking": "trackingA", "ctrl": "ctrl-trackingA"}
    factory = float(calibration["imu"]["time_alignment_s"]["cameras"][names[stream]])
    return factory if override is None else override


def read_vector_csv(path: Path) -> list[tuple[int, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            (int(row["timestamp_ns"]), np.asarray((float(row["x"]), float(row["y"]), float(row["z"])), dtype=np.float64))
            for row in csv.DictReader(handle)
        ]


def corrected_imu(episode: Path, calibration: dict[str, Any]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    imu = calibration["imu"]
    corrected: list[tuple[int, np.ndarray, np.ndarray]] = []
    accel, gyro = read_vector_csv(episode / "accel.csv"), read_vector_csv(episode / "gyro.csv")
    acc_bias = np.asarray(imu["bias"]["accelerometer_mps2"], dtype=np.float64)
    gyr_bias = np.asarray(imu["bias"]["gyroscope_rads"], dtype=np.float64)
    def matrix(kind: str) -> np.ndarray:
        n0, n1, n2 = imu["nonorthogonality"][kind]
        scale = np.asarray(imu["scale_factor"][kind], dtype=np.float64)
        return np.array(((1.0, n0, n1), (0.0, 1.0, n2), (0.0, 0.0, 1.0)), dtype=np.float64) @ np.diag(1.0 + scale)
    acc_matrix, gyr_matrix = matrix("accelerometer"), matrix("gyroscope")
    index = 0
    for timestamp, angular_rate in gyro:
        while index + 1 < len(accel) and abs(accel[index + 1][0] - timestamp) <= abs(accel[index][0] - timestamp):
            index += 1
        if abs(accel[index][0] - timestamp) <= 2_000_000:
            corrected.append((timestamp, acc_matrix @ (accel[index][1] - acc_bias), gyr_matrix @ (angular_rate - gyr_bias)))
    if len(corrected) < 100:
        raise ValueError("fewer than 100 synchronized IMU samples")
    return corrected


def write_reference(source: Path, destination: Path) -> int:
    count = 0
    with source.open(newline="", encoding="utf-8") as input_handle, destination.open("w", encoding="utf-8") as output_handle:
        for row in csv.DictReader(input_handle):
            output_handle.write(
                f"{int(row['timestamp_ns']) * 1e-9:.9f} {float(row['pos_x']):.9f} {float(row['pos_y']):.9f} {float(row['pos_z']):.9f} "
                f"{float(row['quat_x']):.9f} {float(row['quat_y']):.9f} {float(row['quat_z']):.9f} {float(row['quat_w']):.9f}\n"
            )
            count += 1
    return count


def evaluate(reference: Path, estimate: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    require_reasonable_scale(reference, estimate, output / "scale_gate.json")
    for filename, command in {
        "ape_translation.log": ["evo_ape", "tum", str(reference), str(estimate), "--align", "--pose_relation", "trans_part"],
        "rpe_translation.log": ["evo_rpe", "tum", str(reference), str(estimate), "--align", "--pose_relation", "trans_part", "--delta", "1", "--delta_unit", "f"],
    }.items():
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (output / filename).write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{' '.join(command[:2])} failed; see {output / filename}")


def write_matrix(handle: Any, value: np.ndarray) -> None:
    for row in value:
        handle.write("    - [" + ", ".join(f"{float(item):.15g}" for item in row) + "]\n")


def camera_transforms(
    image: dict[str, Any],
    convention: str,
    quaternion_order: str,
    camera_from_imu_rotation: np.ndarray | None,
) -> list[np.ndarray]:
    """Return the camera-to-IMU transforms expected by OpenVINS YAML."""
    transforms = [factory_t_imu_camera(camera, convention, quaternion_order) for camera in image["cameras"]]
    if camera_from_imu_rotation is not None:
        if len(transforms) != 2:
            raise ValueError("the visual-IMU rotation override requires a stereo rig")
        stereo = inverse(transforms[0]) @ transforms[1]
        corrected = transforms[0].copy()
        corrected[:3, :3] = camera_from_imu_rotation.T
        transforms = [corrected, corrected @ stereo]
    # OpenVINS names this field T_imu_cam, but its parser treats it as T_CtoI.
    # Factory CSV calibration instead gives T_imu_to_camera.
    return [inverse(transform) for transform in transforms]


def write_config(
    output: Path,
    image: dict[str, Any],
    imu_calibration: dict[str, Any],
    stream: str,
    scale: float,
    profile: str,
    convention: str,
    quaternion_order: str,
    imu_rate_hz: float,
    camera_from_imu_rotation: np.ndarray | None,
) -> Path:
    config_dir = output / "openvins_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    dynamic_initialization = profile != "static-bootstrap"
    estimator = config_dir / "estimator_config.yaml"
    estimator.write_text(
        "%YAML:1.0\n"
        "verbosity: \"INFO\"\n"
        "use_fej: true\n"
        "integration: \"rk4\"\n"
        "use_stereo: true\n"
        "max_cameras: 2\n"
        "calib_cam_extrinsics: false\n"
        "calib_cam_intrinsics: false\n"
        "calib_cam_timeoffset: false\n"
        "calib_imu_intrinsics: false\n"
        "calib_imu_g_sensitivity: false\n"
        "max_clones: 11\n"
        "max_slam: 50\n"
        "max_slam_in_update: 25\n"
        "max_msckf_in_update: 40\n"
        "dt_slam_delay: 1\n"
        "gravity_mag: 9.81\n"
        "feat_rep_msckf: \"GLOBAL_3D\"\n"
        "feat_rep_slam: \"ANCHORED_MSCKF_INVERSE_DEPTH\"\n"
        "feat_rep_aruco: \"ANCHORED_MSCKF_INVERSE_DEPTH\"\n"
        "try_zupt: true\n"
        "zupt_chi2_multipler: 0.5\n"
        "zupt_max_velocity: 0.1\n"
        "zupt_noise_multiplier: 10.0\n"
        "zupt_max_disparity: 0.5\n"
        "zupt_only_at_beginning: true\n"
        "init_window_time: 2.0\n"
        "init_imu_thresh: 0.8\n"
        "init_max_disparity: 15\n"
        "init_max_features: 80\n"
        f"init_dyn_use: {str(dynamic_initialization).lower()}\n"
        "init_dyn_mle_opt_calib: false\n"
        "init_dyn_mle_max_iter: 50\n"
        "init_dyn_mle_max_time: 0.05\n"
        "init_dyn_mle_max_threads: 6\n"
        "init_dyn_num_pose: 6\n"
        "init_dyn_min_deg: 5\n"
        "init_dyn_inflation_ori: 10\n"
        "init_dyn_inflation_vel: 100\n"
        "init_dyn_inflation_bg: 100\n"
        "init_dyn_inflation_ba: 100\n"
        "init_dyn_min_rec_cond: 1e-12\n"
        "init_dyn_bias_g: [0.0, 0.0, 0.0]\n"
        "init_dyn_bias_a: [0.0, 0.0, 0.0]\n"
        "save_total_state: true\n"
        "record_timing_information: false\n"
        "record_timing_filepath: \"/tmp/sxr_openvins_timing.txt\"\n"
        "filepath_est: \"/output/openvins_estimate.txt\"\n"
        "filepath_std: \"/output/openvins_estimate_std.txt\"\n"
        "filepath_gt: \"/output/openvins_groundtruth.txt\"\n"
        "use_klt: true\n"
        "num_pts: 420\n"
        "fast_threshold: 10\n"
        "grid_x: 8\n"
        "grid_y: 5\n"
        "min_px_dist: 8\n"
        "knn_ratio: 0.70\n"
        "track_frequency: 30.0\n"
        "downsample_cameras: false\n"
        "num_opencv_threads: 4\n"
        "histogram_method: \"NONE\"\n"
        "use_aruco: false\n"
        "num_aruco: 1024\n"
        "downsize_aruco: true\n"
        "up_msckf_sigma_px: 1.0\n"
        "up_msckf_chi2_multipler: 1.0\n"
        "up_slam_sigma_px: 1.0\n"
        "up_slam_chi2_multipler: 1.0\n"
        "up_aruco_sigma_px: 1.0\n"
        "up_aruco_chi2_multipler: 1.0\n"
        "use_mask: false\n"
        "relative_config_imu: \"kalibr_imu_chain.yaml\"\n"
        "relative_config_imucam: \"kalibr_imucam_chain.yaml\"\n",
        encoding="utf-8",
    )
    transforms = camera_transforms(image, convention, quaternion_order, camera_from_imu_rotation)
    cameras = image["cameras"]
    time_offset = camera_time_offset_sec(imu_calibration, stream, None)
    with (config_dir / "kalibr_imucam_chain.yaml").open("w", encoding="utf-8") as handle:
        handle.write("%YAML:1.0\n\n")
        for index, (camera, transform) in enumerate(zip(cameras, transforms)):
            intrinsics = camera["intrinsics"]
            distortion = ", ".join(f"{float(value):.15g}" for value in intrinsics["radialDistortion"][:4])
            values = ", ".join(
                f"{float(intrinsics[key]) * scale:.15g}" for key in ("focalX", "focalY", "centerX", "centerY")
            )
            handle.write(f"cam{index}:\n  T_imu_cam:\n")
            write_matrix(handle, transform)
            handle.write(
                f"  cam_overlaps: [{1 - index}]\n"
                "  camera_model: pinhole\n"
                f"  distortion_coeffs: [{distortion}]\n"
                "  distortion_model: equidistant\n"
                f"  intrinsics: [{values}]\n"
                f"  resolution: [{round(int(camera['width']) * scale)}, {round(int(camera['height']) * scale)}]\n"
                f"  rostopic: /sxr/cam{index}\n"
                f"  timeshift_cam_imu: {time_offset:.15g}\n"
            )
    noise = imu_calibration["noise"]
    with (config_dir / "kalibr_imu_chain.yaml").open("w", encoding="utf-8") as handle:
        handle.write("%YAML:1.0\n\nimu0:\n  T_i_b:\n")
        write_matrix(handle, np.eye(4))
        handle.write(
            f"  accelerometer_noise_density: {float(noise['accel_noise_std_mps2'][0]) / math.sqrt(imu_rate_hz):.15g}\n"
            f"  accelerometer_random_walk: {float(noise['accel_bias_std_mps2'][0]):.15g}\n"
            f"  gyroscope_noise_density: {float(noise['gyro_noise_std_rads'][0]) / math.sqrt(imu_rate_hz):.15g}\n"
            f"  gyroscope_random_walk: {float(noise['gyro_bias_std_rads'][0]):.15g}\n"
            "  rostopic: /sxr/imu\n"
            "  time_offset: 0.0\n"
            f"  update_rate: {imu_rate_hz:.15g}\n"
            "  model: \"kalibr\"\n"
        )
        for name in ("Tw", "R_IMUtoGYRO", "Ta", "R_IMUtoACC"):
            handle.write(f"  {name}:\n")
            write_matrix(handle, np.eye(3))
        handle.write("  Tg:\n    - [0.0, 0.0, 0.0]\n    - [0.0, 0.0, 0.0]\n    - [0.0, 0.0, 0.0]\n")
    return config_dir


def load_frames(episode: Path, stream: str, samples: list[tuple[int, np.ndarray, np.ndarray]], offset_ns: int) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with (episode / f"{stream}_metainfo.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index, timestamp = int(row["frame_index"]), int(row["mid_exposure_utc_ns"])
            if samples[0][0] <= timestamp + offset_ns <= samples[-1][0]:
                rows.append((index, timestamp))
    if len(rows) < 12:
        raise ValueError("fewer than twelve camera pairs overlap the IMU timeline")
    return rows


def export_bag(
    episode: Path,
    output: Path,
    stream: str,
    scale: float,
    samples: list[tuple[int, np.ndarray, np.ndarray]],
    frames: list[tuple[int, int]],
    width: int,
    height: int,
) -> int:
    import rosbag
    import rospy
    from sensor_msgs.msg import Image, Imu

    bag = output / "sxr_csv_openvins_input.bag"
    wanted = dict(frames)
    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(episode / f"{stream}.mp4"), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert decoder.stdout is not None
    written, imu_index, channels = 0, 0, 3
    frame_bytes = width * height * 2 * channels
    try:
        with rosbag.Bag(str(bag), "w", compression="lz4") as handle:
            def write_imu(timestamp: int, acceleration: np.ndarray, angular_rate: np.ndarray) -> None:
                stamp = rospy.Time(timestamp // 1_000_000_000, timestamp % 1_000_000_000)
                message = Imu()
                message.header.stamp, message.header.frame_id = stamp, "imu0"
                message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z = acceleration
                message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z = angular_rate
                handle.write("/sxr/imu", message, t=stamp)

            for index in range(max(wanted) + 1):
                raw = decoder.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    break
                timestamp = wanted.get(index)
                if timestamp is None:
                    continue
                while imu_index < len(samples) and samples[imu_index][0] <= timestamp:
                    write_imu(*samples[imu_index])
                    imu_index += 1
                image = np.frombuffer(raw, dtype=np.uint8).reshape((height, width * 2, channels))
                stamp = rospy.Time(timestamp // 1_000_000_000, timestamp % 1_000_000_000)
                for side, half in enumerate((image[:, :width], image[:, width:])):
                    gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
                    if scale != 1.0:
                        gray = cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
                    message = Image()
                    message.header.stamp, message.header.frame_id = stamp, f"cam{side}"
                    message.height, message.width = gray.shape
                    message.encoding, message.step, message.data = "mono8", gray.shape[1], gray.tobytes()
                    handle.write(f"/sxr/cam{side}", message, t=stamp)
                written += 1
            while imu_index < len(samples):
                write_imu(*samples[imu_index])
                imu_index += 1
    finally:
        decoder.stdout.close()
        decoder.wait(timeout=60)
    if written != len(frames):
        raise RuntimeError(f"decoded {written}/{len(frames)} requested stereo pairs")
    return written


def inside(args: argparse.Namespace) -> int:
    episode, output = args.episode_dir.resolve(), args.output_dir.resolve()
    image = json.loads((episode / f"camera_params_{args.stream}.json").read_text(encoding="utf-8"))
    imu_calibration = json.loads((episode / "imu_calibration.json").read_text(encoding="utf-8"))
    samples = corrected_imu(episode, imu_calibration)
    rate = 1e9 / float(np.median(np.diff(np.asarray([stamp for stamp, _, _ in samples], dtype=np.int64))))
    offset = round(camera_time_offset_sec(imu_calibration, args.stream, None) * 1e9)
    frames = load_frames(episode, args.stream, samples, offset)
    rotation = load_camera_from_imu_rotation(args.camera_from_imu_rotation_json)
    config_dir = write_config(
        output, image, imu_calibration, args.stream, args.image_scale, args.profile, args.extrinsics_convention,
        args.camera_quaternion_order, rate, rotation,
    )
    pairs = export_bag(
        episode, output, args.stream, args.image_scale, samples, frames,
        int(image["cameras"][0]["width"]), int(image["cameras"][0]["height"]),
    )
    write_reference(episode / "head_pose.csv", output / "head_pose_reference.tum")
    name = "sxr_csv_" + output.name.replace("-", "_")
    destination = Path("/root/openvins_ws/src/open_vins/config") / name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(config_dir, destination)
    shell = (
        "set -e; source /opt/ros/noetic/setup.bash; source /root/openvins_ws/devel/setup.bash; "
        "roscore >/tmp/sxr_openvins_roscore.log 2>&1 & R=$!; "
        "trap 'rosnode kill /ov_msckf >/dev/null 2>&1 || true; kill $R >/dev/null 2>&1 || true' EXIT; "
        "sleep 2; rosparam set /use_sim_time true; "
        # Running through roslaunch aborts in this image before subscribing, with
        # boost::lock_error. The same node and YAML run correctly when invoked
        # directly, so keep roslaunch out of the process hierarchy.
        f"/root/openvins_ws/devel/.private/ov_msckf/lib/ov_msckf/run_subscribe_msckf /root/openvins_ws/src/open_vins/config/{name}/estimator_config.yaml __name:=ov_msckf >/tmp/sxr_openvins_node.log 2>&1 & O=$!; "
        "sleep 3; if ! kill -0 $O >/dev/null 2>&1; then cat /tmp/sxr_openvins_node.log; exit 1; fi; "
        "rostopic echo -p /ov_msckf/poseimu > /output/openvins_poseimu.csv 2>/tmp/sxr_openvins_pose_echo.log & P=$!; "
        "rosbag play --clock /output/sxr_csv_openvins_input.bag; sleep 3; "
        "if ! kill -0 $O >/dev/null 2>&1; then kill $P >/dev/null 2>&1 || true; cat /tmp/sxr_openvins_node.log; exit 1; fi; "
        "kill $P >/dev/null 2>&1 || true; rosnode kill /ov_msckf >/dev/null 2>&1 || true; wait $O || true; "
        "cat /tmp/sxr_openvins_node.log"
    )
    result = subprocess.run(["bash", "-lc", shell], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (output / "openvins.log").write_text(result.stdout, encoding="utf-8")
    provenance = {
        "algorithm": "OpenVINS MSCKF stereo-inertial", "stream": args.stream,
        "image_scale": args.image_scale, "profile": args.profile, "image_pairs": pairs,
        "imu_samples": len(samples), "imu_rate_hz": rate,
        "camera_timestamp_offset_sec": camera_time_offset_sec(imu_calibration, args.stream, None),
        "extrinsics_convention": args.extrinsics_convention,
        "camera_quaternion_order": args.camera_quaternion_order,
        "camera_imu_rotation_source": "factory" if args.camera_from_imu_rotation_json is None else str(args.camera_from_imu_rotation_json),
        "returncode": result.returncode,
        "head_pose_note": "Device-provided reference, not independently verified ground truth.",
    }
    (output / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if not args.keep_bag:
        (output / "sxr_csv_openvins_input.bag").unlink(missing_ok=True)
    return result.returncode


def find_column(names: list[str], candidates: tuple[str, ...]) -> str:
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for name in names:
        if any(name.lower().endswith(candidate) for candidate in candidates):
            return name
    raise KeyError(f"missing one of {candidates} in OpenVINS pose CSV")


def export_openvins_imu_tum(source: Path, destination: Path) -> int:
    count = 0
    with source.open(newline="", encoding="utf-8") as input_handle, destination.open("w", encoding="utf-8") as output_handle:
        reader = csv.DictReader(input_handle)
        if not reader.fieldnames:
            raise ValueError(f"OpenVINS did not write a pose CSV: {source}")
        names = [name.strip() for name in reader.fieldnames]
        timestamp = find_column(names, ("%time", "time", "header.stamp"))
        x = find_column(names, ("pose.position.x", "position.x"))
        y = find_column(names, ("pose.position.y", "position.y"))
        z = find_column(names, ("pose.position.z", "position.z"))
        qx = find_column(names, ("pose.orientation.x", "orientation.x"))
        qy = find_column(names, ("pose.orientation.y", "orientation.y"))
        qz = find_column(names, ("pose.orientation.z", "orientation.z"))
        qw = find_column(names, ("pose.orientation.w", "orientation.w"))
        for row in reader:
            values = [float(row[key]) for key in (timestamp, x, y, z, qx, qy, qz, qw)]
            if not np.isfinite(values).all():
                continue
            time = values[0]
            if time > 1e12:
                time *= 1e-9
            quaternion = np.asarray(values[4:], dtype=np.float64)
            norm = float(np.linalg.norm(quaternion))
            if norm < 1e-12:
                continue
            quaternion /= norm
            output_handle.write(
                f"{time:.9f} {values[1]:.9f} {values[2]:.9f} {values[3]:.9f} "
                f"{quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} {quaternion[3]:.9f}\n"
            )
            count += 1
    return count


def build_viewer(reference: Path, estimate: Path, output: Path) -> None:
    subprocess.run([
        "python3", str(ROOT / "script/visualize/visualize_trajectory_pair.py"),
        "--ref", str(reference), "--est", str(estimate), "--output-dir", str(output),
        "--ref-name", "SXR device reference (head_pose)",
        "--est-name", "OpenVINS MSCKF stereo-inertial (IMU/body)",
        "--title", "SXR OpenVINS IMU/body vs device reference",
        "--default-mode", "se3",
    ], check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.2 <= args.image_scale <= 1.0:
        raise ValueError("--image-scale must be in [0.2, 1.0]")
    if args.inside:
        return inside(args)
    episode, output = args.episode_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    required = ("accel.csv", "gyro.csv", "head_pose.csv", f"{args.stream}.mp4", f"{args.stream}_metainfo.csv", f"camera_params_{args.stream}.json", "imu_calibration.json")
    missing = [name for name in required if not (episode / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {episode}")
    output.mkdir(parents=True, exist_ok=True)
    rotation_arg = ""
    if args.camera_from_imu_rotation_json is not None:
        rotation_path = args.camera_from_imu_rotation_json.expanduser().resolve()
        try:
            rotation_arg = f" --camera-from-imu-rotation-json /workspace/{rotation_path.relative_to(ROOT)}"
        except ValueError as error:
            raise ValueError("--camera-from-imu-rotation-json must be under the repository root") from error
    command = [
        "docker", "run", "--rm", "--net=host", "--ipc=host",
        "--mount", f"type=bind,source={ROOT},target=/workspace,readonly",
        "--mount", f"type=bind,source={episode},target=/input,readonly",
        "--mount", f"type=bind,source={output},target=/output",
        args.docker_image, "bash", "-lc",
        "source /opt/ros/noetic/setup.bash && source /root/openvins_ws/devel/setup.bash && "
        "python3 /workspace/script/run_sxr_csv_openvins.py --inside --episode-dir /input --output-dir /output "
        f"--stream {args.stream} --image-scale {args.image_scale} --profile {args.profile} "
        f"--extrinsics-convention {args.extrinsics_convention} --camera-quaternion-order {args.camera_quaternion_order}"
        f"{rotation_arg}" + (" --keep-bag" if args.keep_bag else ""),
    ]
    (output / "docker_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    status = subprocess.run(command, timeout=args.timeout_sec, check=False).returncode
    if status:
        raise RuntimeError(f"OpenVINS container exited {status}; see {output / 'openvins.log'}")
    trajectory = output / "openvins_imu_trajectory.tum"
    poses = export_openvins_imu_tum(output / "openvins_poseimu.csv", trajectory)
    if poses < 3:
        raise RuntimeError(f"OpenVINS exported only {poses} IMU poses; see {output / 'openvins.log'}")
    reference = output / "head_pose_reference.tum"
    evaluate(reference, trajectory, output / "evaluation")
    build_viewer(reference, trajectory, output / "evaluation/trajectory_viewer")
    print(f"[OK] {episode.name}: {poses} OpenVINS IMU poses -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
