#!/usr/bin/env python3
"""Run Basalt stereo-inertial VIO on an SXR DatasetRecorder CSV episode.

The recorder stores a side-by-side stereo stream, exposure-midpoint camera
timestamps, approximately 1 kHz IMU CSVs, and factory calibration.  This
adapter writes a temporary EuRoC dataset for Basalt and only uses head_pose as
an evaluation reference after the estimate is complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import av
import cv2
import numpy as np

from run_sxr_basalt import matrix_to_xyzw, transform_imu_to_cam0
from run_sxr_csv_vinsfusion import (
    imu_correction,
    join_imu,
    quat_to_matrix,
    read_vector_csv,
    write_reference,
)
from run_sxr_orb_stereo_baseline import ROOT, evaluate


DEFAULT_BASALT = ROOT / "third_party/basalt_release/release/bin/basalt_vio"
DEFAULT_CONFIG = ROOT / "third_party/basalt/data/euroc_config.json"
DEFAULT_TEMPLATE = ROOT / "third_party/basalt/data/t265_kb4_calib.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stream", choices=("rgb", "tracking", "ctrl"), default="tracking")
    parser.add_argument(
        "--evaluation-frame",
        choices=("imu", "tracking_cam0"),
        default="imu",
        help=(
            "Basalt trajectory frame used for the default APE/RPE and viewer. "
            "The native Basalt trajectory is the IMU/body pose; tracking_cam0 "
            "is retained as a calibrated diagnostic conversion."
        ),
    )
    parser.add_argument("--basalt-bin", type=Path, default=DEFAULT_BASALT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a Basalt JSON config value; repeat for multiple values.",
    )
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument(
        "--image-export",
        choices=("auto", "pyav", "ffmpeg"),
        default="auto",
        help="Image-export backend. auto uses the validated PyAV path; FFmpeg is a lower-memory fallback.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=75, choices=range(50, 101))
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument(
        "--cam-time-offset-sec",
        type=float,
        default=None,
        help=(
            "Override the factory camera-to-IMU clock offset in seconds. "
            "By default the stream-specific value in imu_calibration.json is used."
        ),
    )
    parser.add_argument(
        "--extrinsics-convention",
        choices=("imu-to-camera", "camera-to-imu"),
        default="imu-to-camera",
        help=(
            "Factory pose semantics. SXR exports T_imu_camera: IMU is the "
            "parent frame and the camera is the child frame."
        ),
    )
    parser.add_argument(
        "--camera-quaternion-order",
        choices=("wxyz", "xyzw"),
        default="xyzw",
        help="Factory camera extrinsic quaternion order. SXR tracking calibration is xyzw.",
    )
    parser.add_argument(
        "--camera-from-imu-rotation-json",
        type=Path,
        default=None,
        help=(
            "Optional independent visual-IMU rotation diagnostic JSON. Its "
            "fitted camera-from-IMU rotation replaces only the rig-to-IMU "
            "rotation while preserving Basalt's validated stereo transform."
        ),
    )
    parser.add_argument(
        "--stereo-camera1-from-camera0-json",
        type=Path,
        default=None,
        help="Optional visual stereo-rig calibration JSON from estimate_sxr_tracking_stereo_extrinsic.py.",
    )
    return parser.parse_args(argv)


def resolve_config(source: Path, output: Path, overrides: Sequence[str]) -> Path:
    """Materialize the exact Basalt config used by this run."""
    data = json.loads(source.read_text(encoding="utf-8"))
    target = data["value0"] if isinstance(data.get("value0"), dict) else data
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"invalid --config-override {item!r}; expected KEY=VALUE")
        key, raw_value = item.split("=", 1)
        if not key:
            raise ValueError(f"invalid --config-override {item!r}; key is empty")
        try:
            value: Any = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        target[key] = value
    if not overrides:
        return source
    resolved = output / "basalt_config_resolved.json"
    resolved.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return resolved


def inverse(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def factory_t_imu_camera(camera: dict[str, Any], convention: str, quaternion_order: str) -> np.ndarray:
    """Return T_imu_camera for Basalt's T_i_c calibration field."""
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quat_to_matrix(camera["extrinsics"]["rotation"], quaternion_order)
    result[:3, 3] = np.asarray(camera["extrinsics"]["position"], dtype=np.float64)
    return result if convention == "imu-to-camera" else inverse(result)


def corrected_imu(episode: Path, calibration: dict[str, Any]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    imu = calibration["imu"]
    acc_bias, acc_matrix = imu_correction(imu, "accelerometer", "accelerometer_mps2")
    gyr_bias, gyr_matrix = imu_correction(imu, "gyroscope", "gyroscope_rads")
    return [
        (stamp, acc_matrix @ (acceleration - acc_bias), gyr_matrix @ (angular_rate - gyr_bias))
        for stamp, acceleration, angular_rate in join_imu(
            read_vector_csv(episode / "accel.csv"), read_vector_csv(episode / "gyro.csv")
        )
    ]


def camera_time_offset_sec(imu: dict[str, Any], stream: str, override: float | None) -> float:
    offset_names = {"rgb": "rgb-left", "tracking": "trackingA", "ctrl": "ctrl-trackingA"}
    factory_offset = float(imu["imu"]["time_alignment_s"]["cameras"][offset_names[stream]])
    return factory_offset if override is None else override


def load_camera_from_imu_rotation(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    try:
        rotation = np.asarray(
            payload["extrinsic_rotation_validation"]["fitted_rotation_camera_from_imu"], dtype=np.float64
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} lacks extrinsic_rotation_validation.fitted_rotation_camera_from_imu") from error
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{path} has an invalid camera-from-IMU rotation")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-3 or np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1e-3:
        raise ValueError(f"{path} camera-from-IMU rotation is not a proper rotation")
    return rotation


def load_stereo_camera1_from_camera0(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    try:
        transform = np.asarray(payload["fitted_camera1_from_camera0"], dtype=np.float64)
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} lacks fitted_camera1_from_camera0") from error
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError(f"{path} has an invalid camera1-from-camera0 transform")
    if abs(float(np.linalg.det(transform[:3, :3])) - 1.0) > 1e-3 or np.linalg.norm(transform[:3, :3].T @ transform[:3, :3] - np.eye(3)) > 1e-3:
        raise ValueError(f"{path} camera1-from-camera0 rotation is not proper")
    return transform


def export_native_imu_trajectory(source: Path, destination: Path) -> int:
    """Export Basalt's native T_w_imu trajectory in normal TUM formatting."""
    count = 0
    with source.open(encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip() or line.startswith("#"):
                continue
            values = line.replace(",", " ").split()
            if len(values) != 8:
                continue
            timestamp, x, y, z, qx, qy, qz, qw = map(float, values)
            quaternion = np.asarray((qx, qy, qz, qw), dtype=np.float64)
            norm = float(np.linalg.norm(quaternion))
            if not np.isfinite((timestamp, x, y, z)).all() or not np.isfinite(quaternion).all() or norm < 1e-12:
                raise ValueError(f"invalid Basalt pose in {source}")
            qx, qy, qz, qw = quaternion / norm
            dst.write(f"{timestamp:.9f} {x:.9f} {y:.9f} {z:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
            count += 1
    return count


def build_basalt_viewer(reference: Path, estimate: Path, output: Path, evaluation_frame: str) -> None:
    frame_name = "IMU/body" if evaluation_frame == "imu" else "tracking cam0"
    subprocess.run([
        "python3", str(ROOT / "script/visualize/visualize_trajectory_pair.py"),
        "--ref", str(reference), "--est", str(estimate), "--output-dir", str(output),
        "--ref-name", "SXR device reference (head_pose)",
        "--est-name", f"Basalt stereo-inertial ({frame_name})",
        "--title", f"SXR Basalt stereo-inertial {frame_name} vs device reference",
        "--default-mode", "se3",
    ], check=True)


def load_frames(episode: Path, stream: str, samples: list[tuple[int, np.ndarray, np.ndarray]], stride: int, maximum: int, offset_ns: int) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with (episode / f"{stream}_metainfo.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # Basalt's online VIO path leaves cam_time_offset_ns unused.  Move
            # the source timestamp onto the IMU clock here instead of relying
            # on the calibration field, whose definition is t_imu=t_raw+offset.
            rows.append((int(row["frame_index"]), int(row["mid_exposure_utc_ns"]) + offset_ns))
    rows = [row for row in rows if samples[0][0] <= row[1] <= samples[-1][0]][::stride]
    if maximum:
        rows = rows[:maximum]
    if len(rows) < 3:
        raise ValueError("fewer than three camera pairs overlap the IMU")
    return rows


def write_calibration(path: Path, image: dict[str, Any], imu: dict[str, Any], scale: float, convention: str, quaternion_order: str, imu_rate_hz: float, camera_from_imu_rotation: np.ndarray | None, stereo_camera1_from_camera0: np.ndarray | None) -> np.ndarray:
    data = json.loads(DEFAULT_TEMPLATE.read_text(encoding="utf-8"))
    target = data["value0"]
    cameras = image["cameras"]
    target["intrinsics"], target["T_imu_cam"] = [], []
    transforms: list[np.ndarray] = []
    for camera in cameras:
        intrinsics = camera["intrinsics"]
        distortion = intrinsics["radialDistortion"][:4]
        target["intrinsics"].append({"camera_type": "kb4", "intrinsics": {
            "fx": float(intrinsics["focalX"]) * scale,
            "fy": float(intrinsics["focalY"]) * scale,
            "cx": float(intrinsics["centerX"]) * scale,
            "cy": float(intrinsics["centerY"]) * scale,
            "k1": float(distortion[0]), "k2": float(distortion[1]),
            "k3": float(distortion[2]), "k4": float(distortion[3]),
        }})
        transforms.append(factory_t_imu_camera(camera, convention, quaternion_order))
    if camera_from_imu_rotation is not None or stereo_camera1_from_camera0 is not None:
        if len(transforms) != 2:
            raise ValueError("camera overrides currently require a stereo rig")
        # Basalt stores T_i_c. Preserve T_c0_c1 exactly, because it was
        # independently validated from real stereo correspondences.
        stereo = inverse(transforms[0]) @ transforms[1]
        if stereo_camera1_from_camera0 is not None:
            stereo = inverse(stereo_camera1_from_camera0)
        corrected_cam0 = transforms[0].copy()
        if camera_from_imu_rotation is not None:
            corrected_cam0[:3, :3] = camera_from_imu_rotation.T
        transforms = [corrected_cam0, corrected_cam0 @ stereo]
    for transform in transforms:
        qx, qy, qz, qw = matrix_to_xyzw(transform)
        target["T_imu_cam"].append({"px": float(transform[0, 3]), "py": float(transform[1, 3]), "pz": float(transform[2, 3]), "qx": qx, "qy": qy, "qz": qz, "qw": qw})
    width, height = int(cameras[0]["width"]), int(cameras[0]["height"])
    target["resolution"] = [[round(width * scale), round(height * scale)] for _ in range(2)]
    target["imu_update_rate"] = imu_rate_hz
    noise = imu["noise"]
    target["accel_noise_std"] = list(map(float, noise["accel_noise_std_mps2"]))
    target["gyro_noise_std"] = list(map(float, noise["gyro_noise_std_rads"]))
    target["accel_bias_std"] = list(map(float, noise["accel_bias_std_mps2"]))
    target["gyro_bias_std"] = list(map(float, noise["gyro_bias_std_rads"]))
    # The raw EuRoC timestamps written by this adapter are already aligned to
    # the IMU clock, see load_frames(). Keep Basalt's field zero to prevent a
    # future Basalt version from applying the offset twice.
    target["cam_time_offset_ns"] = 0
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return transforms[0]


def write_euroc_dataset(temp: Path, episode: Path, stream: str, frames: list[tuple[int, int]], samples: list[tuple[int, np.ndarray, np.ndarray]], width: int, height: int, scale: float, quality: int) -> int:
    cam0, cam1, imu = temp / "mav0/cam0", temp / "mav0/cam1", temp / "mav0/imu0"
    for directory in (cam0 / "data", cam1 / "data", imu):
        directory.mkdir(parents=True, exist_ok=True)
    wanted, rows = dict(frames), []
    container = av.open(str(episode / f"{stream}.mp4"))
    for frame_index, frame in enumerate(container.decode(container.streams.video[0])):
        if frame_index > max(wanted):
            break
        timestamp = wanted.get(frame_index)
        if timestamp is None:
            continue
        image = frame.to_ndarray(format="bgr24")
        if image.shape[:2] != (height, width * 2):
            raise ValueError(f"unexpected {stream} frame shape {image.shape[1]}x{image.shape[0]}")
        filename = f"{timestamp}.jpg"
        for destination, half in ((cam0 / "data", image[:, :width]), (cam1 / "data", image[:, width:])):
            if scale != 1.0:
                half = cv2.resize(half, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", half, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            encoded.tofile(str(destination / filename))
        rows.append((timestamp, filename))
    container.close()
    if len(rows) != len(frames):
        raise RuntimeError(f"decoded {len(rows)}/{len(frames)} requested stereo pairs")
    for camera in (cam0, cam1):
        with (camera / "data.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["#timestamp [ns]", "filename"])
            writer.writerows(rows)
    with (imu / "data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["#timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"])
        for stamp, acceleration, angular_rate in samples:
            writer.writerow([stamp, *angular_rate, *acceleration])
    return len(rows)


def write_euroc_dataset_ffmpeg(temp: Path, episode: Path, stream: str, frames: list[tuple[int, int]], samples: list[tuple[int, np.ndarray, np.ndarray]], width: int, height: int, scale: float, quality: int) -> int:
    """Export a side-by-side stream without Python per-frame decode overhead."""
    cam0, cam1, imu = temp / "mav0/cam0", temp / "mav0/cam1", temp / "mav0/imu0"
    for directory in (cam0 / "data", cam1 / "data", imu):
        directory.mkdir(parents=True, exist_ok=True)
    scaled_width, scaled_height = round(width * scale), round(height * scale)
    filters = (
        f"[0:v]crop={width}:{height}:0:0,scale={scaled_width}:{scaled_height}:flags=area[left];"
        f"[0:v]crop={width}:{height}:{width}:0,scale={scaled_width}:{scaled_height}:flags=area[right]"
    )
    # FFmpeg's qscale is inverse quality. q=5 is close to OpenCV's q75 while
    # avoiding a materially different visual frontend input.
    command = [
        "ffmpeg", "-y", "-v", "error", "-threads", "1", "-filter_threads", "1",
        "-i", str(episode / f"{stream}.mp4"), "-filter_complex", filters,
        "-map", "[left]", "-frames:v", str(max(index for index, _ in frames) + 1),
        "-q:v", "5", str(cam0 / "data/%06d.jpg"),
        "-map", "[right]", "-frames:v", str(max(index for index, _ in frames) + 1),
        "-q:v", "5", str(cam1 / "data/%06d.jpg"),
    ]
    subprocess.run(command, check=True, timeout=900)
    rows = [(timestamp, f"{index + 1:06d}.jpg") for index, timestamp in frames]
    for camera in (cam0, cam1):
        with (camera / "data.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["#timestamp [ns]", "filename"])
            writer.writerows(rows)
    with (imu / "data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["#timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"])
        for stamp, acceleration, angular_rate in samples:
            writer.writerow([stamp, *angular_rate, *acceleration])
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.1 <= args.image_scale <= 1.0 or args.frame_stride < 1 or args.max_pairs < 0:
        raise ValueError("invalid image scale, frame stride, or pair limit")
    episode, output = args.episode_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    required = ("accel.csv", "gyro.csv", "head_pose.csv", "imu_calibration.json", f"{args.stream}.mp4", f"{args.stream}_metainfo.csv", f"camera_params_{args.stream}.json")
    missing = [name for name in required if not (episode / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {episode}")
    binary, config = args.basalt_bin.expanduser().resolve(), args.config.expanduser().resolve()
    if not binary.is_file() or not config.is_file():
        raise FileNotFoundError("Basalt executable or configuration file missing")
    output.mkdir(parents=True, exist_ok=True)
    config = resolve_config(config, output, args.config_override)
    image = json.loads((episode / f"camera_params_{args.stream}.json").read_text(encoding="utf-8"))
    extrinsics_convention = args.extrinsics_convention
    image_export = "pyav" if args.image_export == "auto" else args.image_export
    imu_calibration = json.loads((episode / "imu_calibration.json").read_text(encoding="utf-8"))
    camera_from_imu_rotation = load_camera_from_imu_rotation(args.camera_from_imu_rotation_json)
    stereo_camera1_from_camera0 = load_stereo_camera1_from_camera0(args.stereo_camera1_from_camera0_json)
    samples = corrected_imu(episode, imu_calibration)
    imu_rate_hz = 1e9 / float(np.median(np.diff(np.asarray([stamp for stamp, _, _ in samples], dtype=np.int64))))
    cam_time_offset_sec = camera_time_offset_sec(imu_calibration, args.stream, args.cam_time_offset_sec)
    frames = load_frames(
        episode, args.stream, samples, args.frame_stride, args.max_pairs,
        round(cam_time_offset_sec * 1e9),
    )
    reference = output / "head_pose_reference.tum"; write_reference(episode / "head_pose.csv", reference)
    calib_file = output / "basalt_sxr_csv_kb4_calib.json"
    t_i_c0 = write_calibration(
        calib_file, image, imu_calibration, args.image_scale,
        extrinsics_convention, args.camera_quaternion_order, imu_rate_hz, camera_from_imu_rotation,
        stereo_camera1_from_camera0,
    )
    temporary = tempfile.TemporaryDirectory(prefix=f"sxr_csv_basalt_{episode.name}_", dir="/tmp")
    temp = output / "euroc_export" if args.keep_intermediate else Path(temporary.name)
    try:
        export = write_euroc_dataset_ffmpeg if image_export == "ffmpeg" else write_euroc_dataset
        pairs = export(
            temp, episode, args.stream, frames, samples,
            int(image["cameras"][0]["width"]), int(image["cameras"][0]["height"]),
            args.image_scale, args.jpeg_quality,
        )
        command = [str(binary), "--dataset-path", str(temp), "--dataset-type", "euroc", "--cam-calib", str(calib_file), "--config-path", str(config), "--show-gui", "0", "--save-trajectory", "tum", "--num-threads", str(args.num_threads)]
        environment = dict(os.environ)
        release_lib = binary.parent.parent / "lib"
        if release_lib.is_dir():
            environment["LD_LIBRARY_PATH"] = str(release_lib) + ":" + environment.get("LD_LIBRARY_PATH", "")
        with (output / "basalt.log").open("w", encoding="utf-8") as handle:
            handle.write("COMMAND: " + " ".join(command) + "\n")
            status = subprocess.run(command, cwd=output, stdout=handle, stderr=subprocess.STDOUT, timeout=args.timeout_sec, check=False, env=environment).returncode
        if status:
            raise RuntimeError(f"Basalt exited {status}; see {output / 'basalt.log'}")
    finally:
        temporary.cleanup()
    raw_trajectory = output / "trajectory.txt"
    imu_trajectory = output / "basalt_stereo_vio_imu.tum"
    tracking_cam0_trajectory = output / "basalt_stereo_vio_tracking_cam0.tum"
    imu_poses = export_native_imu_trajectory(raw_trajectory, imu_trajectory)
    tracking_cam0_poses = transform_imu_to_cam0(raw_trajectory, tracking_cam0_trajectory, t_i_c0)
    if imu_poses < 3 or tracking_cam0_poses < 3:
        raise RuntimeError(
            f"Basalt output only {imu_poses} IMU poses and {tracking_cam0_poses} tracking-camera poses"
        )
    estimate = imu_trajectory if args.evaluation_frame == "imu" else tracking_cam0_trajectory
    evaluate(reference, estimate, output / "evaluation")
    build_basalt_viewer(reference, estimate, output / "evaluation/trajectory_viewer", args.evaluation_frame)
    (output / "run_summary.json").write_text(json.dumps({
        "algorithm": "Basalt stereo-inertial", "episode": str(episode), "stream": args.stream,
        "basalt_config": str(config),
        "basalt_config_overrides": list(args.config_override),
        "evaluation_frame": args.evaluation_frame,
        "evaluation_trajectory": str(estimate),
        "trajectory_imu": str(imu_trajectory),
        "trajectory_imu_poses": imu_poses,
        "trajectory_imu_semantics": "Basalt native T_world_imu/body pose",
        "trajectory_tracking_cam0": str(tracking_cam0_trajectory),
        "trajectory_tracking_cam0_poses": tracking_cam0_poses,
        "trajectory_tracking_cam0_semantics": "T_world_imu * T_imu_tracking_cam0; diagnostic calibrated camera pose",
        "image_pairs": pairs, "image_scale": args.image_scale,
        "frame_stride": args.frame_stride, "imu_rate_hz": imu_rate_hz, "factory_imu_correction": True,
        "image_export": image_export,
        "extrinsics_convention": extrinsics_convention,
        "extrinsics_convention_source": args.extrinsics_convention,
        "camera_quaternion_order": args.camera_quaternion_order,
        "camera_imu_rotation_source": "factory" if args.camera_from_imu_rotation_json is None else str(args.camera_from_imu_rotation_json.expanduser().resolve()),
        "stereo_camera1_from_camera0_source": "factory" if args.stereo_camera1_from_camera0_json is None else str(args.stereo_camera1_from_camera0_json.expanduser().resolve()),
        "camera_timestamp_offset_sec": cam_time_offset_sec,
        "basalt_cam_time_offset_ns": 0,
        "cam_time_offset_source": "factory" if args.cam_time_offset_sec is None else "override",
        "head_pose_note": "Device-provided reference, not independently verified ground truth.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {episode.name}: {imu_poses}/{pairs} IMU poses -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
