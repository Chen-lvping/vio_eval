#!/usr/bin/env python3
"""Run VINS-Fusion on an SXR DatasetRecorder CSV episode.

The device-side DatasetRecorder format contains side-by-side stereo MP4s,
mid-exposure timestamps, raw 1 kHz IMU samples, factory calibration, and a
device-provided head-pose reference. This adapter moves images to the IMU
clock domain and evaluates only with SE(3) alignment.
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

import numpy as np

from run_sxr_vinsfusion import (
    DEFAULT_VINS_ROOT,
    csv_to_tum,
    evaluate,
    opencv_matrix,
    runtime_command,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_FROM_IMU_ROTATION = ROOT / (
    "data/evaluation/workbench/"
    "sxr_csv_20260730_205558_tracking_aprilgrid_preflight_v1/"
    "cam_imu_timing_extrinsic_validation.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stream", choices=("rgb", "tracking", "ctrl"), default="rgb")
    parser.add_argument(
        "--evaluation-frame",
        choices=("imu", "cam0"),
        default="imu",
        help="VINS output used for the default APE/RPE and viewer. pose_data is the body/IMU state.",
    )
    parser.add_argument("--image-scale", type=float, default=0.5)
    parser.add_argument("--vins-root", type=Path, default=DEFAULT_VINS_ROOT)
    parser.add_argument("--runtime", choices=("docker", "native"), default="docker")
    parser.add_argument("--docker-image", default="vio-eval-ov2slam:noetic")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument(
        "--extrinsics-convention",
        choices=("imu-to-camera", "camera-to-imu"),
        default="imu-to-camera",
        help="Factory pose semantics. SXR exports T_imu_camera with IMU as parent.",
    )
    parser.add_argument(
        "--camera-quaternion-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Factory camera extrinsic quaternion order. SXR exports xyzw.",
    )
    parser.add_argument(
        "--camera-from-imu-rotation-json",
        type=Path,
        default=None,
        help="Optional independent camera-from-IMU rotation validation JSON.",
    )
    parser.add_argument(
        "--cam-time-offset-sec",
        type=float,
        default=None,
        help=(
            "Override the factory camera-to-IMU clock offset. The adapter "
            "moves camera timestamps into the IMU clock domain, matching the "
            "Basalt SXR baseline, then writes VINS td=0."
        ),
    )
    return parser.parse_args(argv)


def inverse(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def quat_to_matrix(quaternion: Sequence[float], order: str) -> np.ndarray:
    if order == "xyzw":
        x, y, z, w = (float(value) for value in quaternion)
    elif order == "wxyz":
        w, x, y, z = (float(value) for value in quaternion)
    else:
        raise ValueError(f"unsupported quaternion order: {order}")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-9:
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


def pose_matrix(camera: dict[str, Any], convention: str, quaternion_order: str) -> np.ndarray:
    """Return VINS body_T_cam (T_imu_camera) from factory pose fields."""
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quat_to_matrix(camera["extrinsics"]["rotation"], quaternion_order)
    result[:3, 3] = np.asarray(camera["extrinsics"]["position"], dtype=np.float64)
    return result if convention == "imu-to-camera" else inverse(result)


def load_camera_from_imu_rotation(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    try:
        rotation = np.asarray(
            payload["extrinsic_rotation_validation"]["fitted_rotation_camera_from_imu"], dtype=np.float64
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} lacks a fitted camera-from-IMU rotation") from error
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{path} has an invalid camera-from-IMU rotation")
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-3:
        raise ValueError(f"{path} camera-from-IMU rotation is not proper")
    return rotation


def camera_yaml(path: Path, camera: dict[str, Any], scale: float, name: str) -> None:
    intrinsics = camera["intrinsics"]
    distortion = camera["intrinsics"]["radialDistortion"][:4]
    path.write_text(
        "%YAML:1.0\n---\n"
        "model_type: KANNALA_BRANDT\n"
        f"camera_name: {name}\n"
        f"image_width: {round(int(camera['width']) * scale)}\n"
        f"image_height: {round(int(camera['height']) * scale)}\n"
        "projection_parameters:\n"
        f"   k2: {float(distortion[0]):.15g}\n"
        f"   k3: {float(distortion[1]):.15g}\n"
        f"   k4: {float(distortion[2]):.15g}\n"
        f"   k5: {float(distortion[3]):.15g}\n"
        f"   mu: {float(intrinsics['focalX']) * scale:.15g}\n"
        f"   mv: {float(intrinsics['focalY']) * scale:.15g}\n"
        f"   u0: {float(intrinsics['centerX']) * scale:.15g}\n"
        f"   v0: {float(intrinsics['centerY']) * scale:.15g}\n",
        encoding="utf-8",
    )


def imu_correction(imu: dict[str, Any], kind: str, bias_key: str) -> tuple[np.ndarray, np.ndarray]:
    bias = np.asarray(imu["bias"][bias_key], dtype=np.float64)
    scale = np.asarray(imu["scale_factor"][kind], dtype=np.float64)
    n0, n1, n2 = imu["nonorthogonality"][kind]
    nonorthogonal = np.array(((1.0, n0, n1), (0.0, 1.0, n2), (0.0, 0.0, 1.0)), dtype=np.float64)
    return bias, nonorthogonal @ np.diag(1.0 + scale)


def read_vector_csv(path: Path) -> list[tuple[int, np.ndarray]]:
    values: list[tuple[int, np.ndarray]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values.append((int(row["timestamp_ns"]), np.array((float(row["x"]), float(row["y"]), float(row["z"])), dtype=np.float64)))
    return values


def join_imu(accel: list[tuple[int, np.ndarray]], gyro: list[tuple[int, np.ndarray]]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    output: list[tuple[int, np.ndarray, np.ndarray]] = []
    index = 0
    for stamp, omega in gyro:
        while index + 1 < len(accel) and abs(accel[index + 1][0] - stamp) <= abs(accel[index][0] - stamp):
            index += 1
        if abs(accel[index][0] - stamp) <= 2_000_000:
            output.append((stamp, accel[index][1], omega))
    if len(output) < 100:
        raise ValueError("fewer than 100 synchronized IMU samples")
    return output


def write_reference(source: Path, destination: Path) -> int:
    count = 0
    with source.open(newline="", encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for row in csv.DictReader(src):
            dst.write(
                f"{int(row['timestamp_ns']) * 1e-9:.9f} {float(row['pos_x']):.9f} {float(row['pos_y']):.9f} {float(row['pos_z']):.9f} "
                f"{float(row['quat_x']):.9f} {float(row['quat_y']):.9f} {float(row['quat_z']):.9f} {float(row['quat_w']):.9f}\n"
            )
            count += 1
    return count


def export_inputs(
    episode: Path,
    output: Path,
    stream: str,
    calibration: dict[str, Any],
    camera_offset_ns: int,
) -> tuple[Path, Path, Path, int, float]:
    frame_csv = episode / f"{stream}_metainfo.csv"
    frames: list[tuple[int, int]] = []
    with frame_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # Put images and IMU in the same clock domain before invoking the
            # estimator. This is the same convention used by the Basalt
            # baseline and keeps VINS' td parameter unambiguous.
            frames.append((int(row["frame_index"]), int(row["mid_exposure_utc_ns"]) + camera_offset_ns))
    accel = read_vector_csv(episode / "accel.csv")
    gyro = read_vector_csv(episode / "gyro.csv")
    imu = calibration["imu"]
    acc_bias, acc_matrix = imu_correction(imu, "accelerometer", "accelerometer_mps2")
    gyr_bias, gyr_matrix = imu_correction(imu, "gyroscope", "gyroscope_rads")
    corrected = [(t, acc_matrix @ (a - acc_bias), gyr_matrix @ (w - gyr_bias)) for t, a, w in join_imu(accel, gyro)]
    frames = [(index, stamp) for index, stamp in frames if corrected[0][0] <= stamp <= corrected[-1][0]]
    if len(frames) < 12:
        raise ValueError(f"only {len(frames)} {stream} frames overlap IMU samples")
    intervals_ns = np.diff(np.asarray([stamp for stamp, _, _ in corrected], dtype=np.int64))
    positive_intervals = intervals_ns[intervals_ns > 0]
    if len(positive_intervals) == 0:
        raise ValueError("IMU timestamps are not strictly increasing")
    imu_rate_hz = 1e9 / float(np.median(positive_intervals))
    camera_csv, imu_csv = output / "camera_timestamps.csv", output / "imu.csv"
    reference = output / "head_pose_reference.tum"
    with camera_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("frame_index", "timestamp_ns")); writer.writerows(frames)
    with imu_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("timestamp_ns", "gx", "gy", "gz", "ax", "ay", "az"))
        writer.writerows((stamp, *omega.tolist(), *acceleration.tolist()) for stamp, acceleration, omega in corrected)
    write_reference(episode / "head_pose.csv", reference)
    return camera_csv, imu_csv, reference, len(frames), imu_rate_hz


def write_config(output: Path, image: dict[str, Any], imu_calibration: dict[str, Any], stream: str, scale: float, convention: str, quaternion_order: str, imu_rate_hz: float, camera_from_imu_rotation: np.ndarray | None) -> Path:
    cameras = image["cameras"]
    cam0, cam1 = cameras[0], cameras[1]
    camera_yaml(output / "cam0_kb4.yaml", cam0, scale, f"sxr_{stream}_left")
    camera_yaml(output / "cam1_kb4.yaml", cam1, scale, f"sxr_{stream}_right")
    body_cam0 = pose_matrix(cam0, convention, quaternion_order)
    body_cam1 = pose_matrix(cam1, convention, quaternion_order)
    if camera_from_imu_rotation is not None:
        # Preserve the factory stereo rig and replace only the independently
        # validated left-camera-to-IMU rotation.
        stereo = inverse(body_cam0) @ body_cam1
        body_cam0[:3, :3] = camera_from_imu_rotation.T
        body_cam1 = body_cam0 @ stereo
    noise, imu = imu_calibration["noise"], imu_calibration["imu"]
    rate = imu_rate_hz
    config = output / "sxr_csv_vinsfusion_stereo_imu.yaml"
    config.write_text(
        "%YAML:1.0\n\n"
        "imu: 1\nnum_of_cam: 2\n\n"
        "imu_topic: /sxr/imu\nimage0_topic: /sxr/cam0\nimage1_topic: /sxr/cam1\n"
        f"output_path: {str(output)!r}\n"
        "cam0_calib: cam0_kb4.yaml\ncam1_calib: cam1_kb4.yaml\n"
        f"image_width: {round(int(cam0['width']) * scale)}\nimage_height: {round(int(cam0['height']) * scale)}\n\n"
        "base_to_imu: [0, 0, 0, 0, 0, 0]\nestimate_extrinsic: 0\n"
        + opencv_matrix("body_T_cam0", body_cam0)
        + opencv_matrix("body_T_cam1", body_cam1)
        + "\nmultiple_thread: 0\n"
        "max_cnt: 250\nmin_dist: 15\nfreq: 0\nF_threshold: 1.0\nshow_track: 0\nflow_back: 1\nequalize: 1\n"
        "max_solver_time: 0.08\nmax_num_iterations: 10\nkeyframe_parallax: 8.0\n"
        # This VINS-Fusion implementation applies dt in its discrete
        # preintegration covariance propagation. The recorder fields are
        # per-sample standard deviations, so use them directly. Dividing by
        # sqrt(rate) double-counts dt and produces an ill-conditioned IMU
        # information matrix.
        f"acc_n: {float(noise['accel_noise_std_mps2'][0]):.15g}\n"
        f"gyr_n: {float(noise['gyro_noise_std_rads'][0]):.15g}\n"
        f"acc_w: {float(noise['accel_bias_std_mps2'][0]):.15g}\n"
        f"gyr_w: {float(noise['gyro_bias_std_rads'][0]):.15g}\n"
        "g_norm: 9.81\nestimate_td: 0\n"
        "# Camera CSV timestamps were already shifted into the IMU clock.\ntd: 0\n"
        "load_previous_pose_graph: 0\n"
        f"pose_graph_save_path: {str(output / 'pose_graph')!r}\n"
        "save_image: 0\nsave_vio_pose: 1\n"
        f"vio_pose_save_path: {str(output)!r}\n",
        encoding="utf-8",
    )
    return config


def make_vertical_video(source: Path, destination: Path, width: int, height: int, scale: float) -> None:
    target_width, target_height = round(width * scale), round(height * scale)
    if target_width % 2 or target_height % 2:
        raise ValueError("--image-scale must result in even dimensions")
    filters = (
        f"[0:v]crop={width}:{height}:0:0,scale={target_width}:{target_height}:flags=lanczos[left];"
        f"[0:v]crop={width}:{height}:{width}:0,scale={target_width}:{target_height}:flags=lanczos[right];"
        "[left][right]vstack=inputs=2"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-filter_complex", filters, "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", str(destination)],
        check=True,
        timeout=900,
    )


def build_vins_viewer(reference: Path, estimate: Path, output: Path, evaluation_frame: str) -> None:
    frame_name = "IMU/body" if evaluation_frame == "imu" else "cam0"
    subprocess.run([
        "python3", str(ROOT / "script/visualize/visualize_trajectory_pair.py"),
        "--ref", str(reference), "--est", str(estimate), "--output-dir", str(output),
        "--ref-name", "SXR device reference (head_pose)",
        "--est-name", f"VINS-Fusion stereo-inertial ({frame_name})",
        "--title", f"SXR VINS-Fusion stereo-inertial {frame_name} vs device reference",
        "--default-mode", "se3",
    ], check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.2 <= args.image_scale <= 1.0:
        raise ValueError("--image-scale must be in [0.2, 1.0]")
    episode = args.episode_dir.expanduser().resolve()
    required = ("accel.csv", "gyro.csv", "head_pose.csv", f"{args.stream}.mp4", f"{args.stream}_metainfo.csv", f"camera_params_{args.stream}.json", "imu_calibration.json")
    missing = [name for name in required if not (episode / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {episode}")
    output = args.output_dir.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    binary = args.vins_root.expanduser().resolve() / "devel/lib/vins/vins_offline_raw_runner"
    if not binary.is_file():
        raise FileNotFoundError(f"VINS offline runner not found: {binary}")
    image = json.loads((episode / f"camera_params_{args.stream}.json").read_text(encoding="utf-8"))
    imu_calibration = json.loads((episode / "imu_calibration.json").read_text(encoding="utf-8"))
    alignment_keys = {"rgb": "rgb-left", "tracking": "trackingA", "ctrl": "ctrl-trackingA"}
    factory_offset_sec = float(imu_calibration["imu"]["time_alignment_s"]["cameras"][alignment_keys[args.stream]])
    camera_offset_sec = factory_offset_sec if args.cam_time_offset_sec is None else args.cam_time_offset_sec
    camera_csv, imu_csv, reference, frames, imu_rate_hz = export_inputs(
        episode, output, args.stream, imu_calibration, round(camera_offset_sec * 1e9)
    )
    camera_rotation = load_camera_from_imu_rotation(args.camera_from_imu_rotation_json)
    config = write_config(output, image, imu_calibration, args.stream, args.image_scale, args.extrinsics_convention, args.camera_quaternion_order, imu_rate_hz, camera_rotation)
    temporary = output / ".vinsfusion_temp"; shutil.rmtree(temporary, ignore_errors=True); temporary.mkdir()
    try:
        left = image["cameras"][0]
        video = temporary / "stereo_top_bottom.mp4"
        make_vertical_video(episode / f"{args.stream}.mp4", video, int(left["width"]), int(left["height"]), args.image_scale)
        command = runtime_command(args, binary, config, camera_csv, imu_csv, video)
        with (output / "vinsfusion.log").open("w", encoding="utf-8") as handle:
            handle.write("COMMAND: " + " ".join(command) + "\n\n")
            status = subprocess.run(command, cwd=output, stdout=handle, stderr=subprocess.STDOUT, timeout=args.timeout_sec, check=False).returncode
        if status:
            raise RuntimeError(f"VINS-Fusion exited {status}; see {output / 'vinsfusion.log'}")
    finally:
        if not args.keep_intermediate:
            shutil.rmtree(temporary, ignore_errors=True)
    raw_imu, raw_cam0 = output / "pose_data.csv", output / "pose_data_cam0.csv"
    imu_trajectory = output / "vinsfusion_imu_trajectory.tum"
    cam0_trajectory = output / "vinsfusion_cam0_trajectory.tum"
    missing_outputs = [path.name for path in (raw_imu, raw_cam0) if not path.is_file()]
    if missing_outputs:
        raise RuntimeError(f"VINS produced no {missing_outputs}; see {output / 'vinsfusion.log'}")
    imu_poses = csv_to_tum(raw_imu, imu_trajectory)
    cam0_poses = csv_to_tum(raw_cam0, cam0_trajectory)
    if imu_poses < 3 or cam0_poses < 3:
        raise RuntimeError(f"VINS produced only {imu_poses} IMU poses and {cam0_poses} cam0 poses; see {output / 'vinsfusion.log'}")
    estimate = imu_trajectory if args.evaluation_frame == "imu" else cam0_trajectory
    evaluate(reference, estimate, output / "evaluation")
    build_vins_viewer(reference, estimate, output / "evaluation/trajectory_viewer", args.evaluation_frame)
    (output / "run_summary.json").write_text(json.dumps({
        "algorithm": "VINS-Fusion stereo-inertial", "episode": str(episode), "stream": args.stream,
        "evaluation_frame": args.evaluation_frame,
        "evaluation_trajectory": str(estimate),
        "trajectory_imu": str(imu_trajectory), "trajectory_imu_poses": imu_poses,
        "trajectory_imu_semantics": "VINS-Fusion pose_data body/IMU state",
        "trajectory_cam0": str(cam0_trajectory), "trajectory_cam0_poses": cam0_poses,
        "trajectory_cam0_semantics": "VINS-Fusion pose_data_cam0 calibrated camera state",
        "input_stereo_pairs": frames, "image_scale": args.image_scale,
        "imu_rate_hz": imu_rate_hz,
        "extrinsics_convention": args.extrinsics_convention,
        "camera_quaternion_order": args.camera_quaternion_order,
        "camera_from_imu_rotation_source": str(args.camera_from_imu_rotation_json) if args.camera_from_imu_rotation_json else None,
        "camera_timestamp_offset_sec": camera_offset_sec,
        "vins_td_sec": 0.0,
        "cam_time_offset_source": "factory" if args.cam_time_offset_sec is None else "override",
        "factory_imu_correction": True,
        "head_pose_note": "Device-provided reference, not independently verified ground truth.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {episode.name}: {imu_poses} IMU poses from {frames} stereo pairs -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
