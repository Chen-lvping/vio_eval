#!/usr/bin/env python3
"""Run the locally-built VINS-Fusion stereo-inertial estimator on an SXR episode.

The SXR recorder stores the two RGB cameras side-by-side in ``rgb.mp4``.
The local VINS offline runner consumes top-bottom stereo frames, so this
adapter creates a short-lived, scaled top-bottom video.  It preserves the
recorded frame/IMU clock, applies the factory IMU correction, derives KB4
camera models and camera-to-IMU extrinsics, and packages trajectory metrics
and an interactive comparison viewer.

``head_pose`` is used only as a device-provided reference for repeatable
comparisons; it is not labelled independent ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from trajectory_scale_gate import require_reasonable_scale
import cv2

from check_episode_data import CheckResult, inspect_mcap, resolve_data_dir
from run_sxr_openvins import join_imu, records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VINS_ROOT = Path("/home/chenlvping/0_SLAM/vinsfusion_ws")
RGB_WIDTH, RGB_HEIGHT = 4656, 1748


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vins-root", type=Path, default=DEFAULT_VINS_ROOT)
    parser.add_argument("--image-scale", type=float, default=0.5)
    parser.add_argument("--rectify", action="store_true", help="rectify KB4 stereo to a common pinhole pair before VINS")
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--runtime", choices=("docker", "native"), default="docker")
    parser.add_argument("--docker-image", default="vins-noetic-kalibr-ceres:latest")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    return parser.parse_args(argv)


def inverse(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def opencv_matrix(name: str, value: np.ndarray) -> str:
    data = ", ".join(f"{float(x):.15g}" for x in value.reshape(-1))
    return f"{name}: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: d\n   data: [{data}]\n"


def write_camera(path: Path, camera: dict[str, Any], scale: float) -> None:
    intr = camera["intrinsics"]["2328x1748"]
    coefficients = camera["distortion_coeffs"][:4]
    path.write_text(
        "%YAML:1.0\n---\n"
        "model_type: KANNALA_BRANDT\n"
        "camera_name: sxr_rgb\n"
        f"image_width: {round(2328 * scale)}\n"
        f"image_height: {round(1748 * scale)}\n"
        "projection_parameters:\n"
        f"   k2: {float(coefficients[0]):.15g}\n"
        f"   k3: {float(coefficients[1]):.15g}\n"
        f"   k4: {float(coefficients[2]):.15g}\n"
        f"   k5: {float(coefficients[3]):.15g}\n"
        f"   mu: {float(intr['fx']) * scale:.15g}\n"
        f"   mv: {float(intr['fy']) * scale:.15g}\n"
        f"   u0: {float(intr['ppx']) * scale:.15g}\n"
        f"   v0: {float(intr['ppy']) * scale:.15g}\n",
        encoding="utf-8",
    )


def write_pinhole_camera(path: Path, projection: np.ndarray, width: int, height: int) -> None:
    path.write_text(
        "%YAML:1.0\n---\nmodel_type: PINHOLE\ncamera_name: sxr_rgb_rectified\n"
        f"image_width: {width}\nimage_height: {height}\n"
        "distortion_parameters:\n   k1: 0\n   k2: 0\n   p1: 0\n   p2: 0\n"
        "projection_parameters:\n"
        f"   fx: {float(projection[0, 0]):.15g}\n   fy: {float(projection[1, 1]):.15g}\n"
        f"   cx: {float(projection[0, 2]):.15g}\n   cy: {float(projection[1, 2]):.15g}\n",
        encoding="utf-8",
    )


def rectification(calibration: dict[str, Any], scale: float) -> dict[str, Any]:
    """Build output-sized KB4 rectification maps and rectified camera poses."""
    rgb = calibration["observation"]["images"]["rgb"]
    width, height = round(2328 * scale), round(1748 * scale)
    matrices, distortions = [], []
    for name in ("cam0", "cam1"):
        intr = rgb[name]["intrinsics"]["2328x1748"]
        matrices.append(np.array(((intr["fx"], 0, intr["ppx"]), (0, intr["fy"], intr["ppy"]), (0, 0, 1)), dtype=np.float64))
        distortions.append(np.asarray(rgb[name]["distortion_coeffs"][:4], dtype=np.float64).reshape(4, 1))
    extr = rgb["extrinsics"]
    source0 = np.asarray(extr["T_ic_imu0_cam0"], dtype=np.float64)
    source1 = np.asarray(extr["T_ic_imu0_cam1"], dtype=np.float64)
    # Transform from physical cam0 coordinates to physical cam1 coordinates.
    cam1_cam0 = source1 @ inverse(source0)
    r0, r1, p0, p1, _q = cv2.fisheye.stereoRectify(
        matrices[0], distortions[0], matrices[1], distortions[1], (2328, 1748),
        cam1_cam0[:3, :3], cam1_cam0[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=(width, height), balance=0.0, fov_scale=1.0,
    )
    maps = [
        cv2.fisheye.initUndistortRectifyMap(matrices[0], distortions[0], r0, p0[:, :3], (width, height), cv2.CV_32FC1),
        cv2.fisheye.initUndistortRectifyMap(matrices[1], distortions[1], r1, p1[:, :3], (width, height), cv2.CV_32FC1),
    ]
    def pose(original_cam_to_imu: np.ndarray, rectification_rotation: np.ndarray) -> np.ndarray:
        rectified_to_original = np.eye(4, dtype=np.float64); rectified_to_original[:3, :3] = rectification_rotation.T
        return inverse(original_cam_to_imu) @ rectified_to_original
    return {
        "width": width, "height": height, "maps": maps, "projection": (p0, p1),
        "body_cam": (pose(source0, r0), pose(source1, r1)),
    }


def write_config(output: Path, calibration: dict[str, Any], scale: float, rectified: dict[str, Any] | None) -> Path:
    rgb = calibration["observation"]["images"]["rgb"]
    imu = calibration["observation"]["imu"]
    if rectified is None:
        write_camera(output / "cam0_kb4.yaml", rgb["cam0"], scale)
        write_camera(output / "cam1_kb4.yaml", rgb["cam1"], scale)
    else:
        write_pinhole_camera(output / "cam0_pinhole_rectified.yaml", rectified["projection"][0], rectified["width"], rectified["height"])
        write_pinhole_camera(output / "cam1_pinhole_rectified.yaml", rectified["projection"][1], rectified["width"], rectified["height"])
    extrinsics = rgb["extrinsics"]
    # SXR T_ic_imu0_cam* maps IMU coordinates into camera coordinates.  VINS
    # RIC/TIC are used as camera coordinates -> IMU/body coordinates.
    body_cam0 = inverse(np.asarray(extrinsics["T_ic_imu0_cam0"], dtype=np.float64)) if rectified is None else rectified["body_cam"][0]
    body_cam1 = inverse(np.asarray(extrinsics["T_ic_imu0_cam1"], dtype=np.float64)) if rectified is None else rectified["body_cam"][1]
    noise, rate = imu["noise"], 1013.8
    timing = imu.get("time_alignment_s", {}).get("cameras", {})
    config = output / "sxr_vinsfusion_stereo_imu.yaml"
    config.write_text(
        "%YAML:1.0\n\n"
        "# Generated from the episode's calibration.json.\n"
        "imu: 1\nnum_of_cam: 2\n\n"
        "imu_topic: /sxr/imu\nimage0_topic: /sxr/cam0\nimage1_topic: /sxr/cam1\n"
        f"output_path: {str(output)!r}\n"
        + ("cam0_calib: cam0_kb4.yaml\ncam1_calib: cam1_kb4.yaml\n" if rectified is None else "cam0_calib: cam0_pinhole_rectified.yaml\ncam1_calib: cam1_pinhole_rectified.yaml\n")
        + f"image_width: {round(2328 * scale) if rectified is None else rectified['width']}\nimage_height: {round(1748 * scale) if rectified is None else rectified['height']}\n\n"
        "# Identity: VINS output body is the IMU frame.\n"
        "base_to_imu: [0, 0, 0, 0, 0, 0]\n"
        "estimate_extrinsic: 0\n"
        + opencv_matrix("body_T_cam0", body_cam0)
        + opencv_matrix("body_T_cam1", body_cam1)
        + "\nmultiple_thread: 0\n\n"
        "# Offline quality profile: allow denser tracking than real-time defaults.\n"
        "max_cnt: 250\nmin_dist: 15\nfreq: 0\nF_threshold: 1.0\n"
        "show_track: 0\nflow_back: 1\nequalize: 1\n\n"
        "max_solver_time: 0.08\nmax_num_iterations: 10\nkeyframe_parallax: 8.0\n\n"
        "# Sample std -> continuous-time noise density using the measured IMU rate.\n"
        f"acc_n: {float(noise['accel_noise_std_mps2'][0]) / math.sqrt(rate):.15g}\n"
        f"gyr_n: {float(noise['gyro_noise_std_rads'][0]) / math.sqrt(rate):.15g}\n"
        f"acc_w: {float(noise['accel_bias_std_mps2'][0]):.15g}\n"
        f"gyr_w: {float(noise['gyro_bias_std_rads'][0]):.15g}\n"
        "g_norm: 9.81\n\n"
        "estimate_td: 0\n"
        f"# VINS convention: t_camera + td = t_imu.\ntd: {float(timing.get('rgb-left', 0.0)):.15g}\n\n"
        "load_previous_pose_graph: 0\n"
        f"pose_graph_save_path: {str(output / 'pose_graph')!r}\n"
        "save_image: 0\nsave_vio_pose: 1\n"
        f"vio_pose_save_path: {str(output)!r}\n",
        encoding="utf-8",
    )
    return config


def correction(imu: dict[str, Any], kind: str, bias_key: str) -> tuple[np.ndarray, np.ndarray]:
    bias = np.asarray(imu["bias"][bias_key], dtype=np.float64)
    scale = np.asarray(imu["scale_factor"][kind], dtype=np.float64)
    n0, n1, n2 = imu["nonorthogonality"][kind]
    nonorthogonal = np.array(((1.0, n0, n1), (0.0, 1.0, n2), (0.0, 0.0, 1.0)), dtype=np.float64)
    return bias, nonorthogonal @ np.diag(1.0 + scale)


def export_inputs(episode: Path, output: Path, calibration: dict[str, Any]) -> tuple[Path, Path, Path, int]:
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    frames, accel, gyro, _head = records(episode / "sensor.mcap")
    head, _hand, _channels, _schemas = inspect_mcap(episode / "sensor.mcap", CheckResult())
    if not frames or not head.poses:
        raise ValueError("rgb_metainfo or head_pose missing")
    try:
        start_us = next(x["start_offset_us"] for x in metadata["head_pose_details"] if x.get("name") == "head_pose")
    except (KeyError, StopIteration) as error:
        raise KeyError("metadata.head_pose_details.head_pose.start_offset_us missing") from error
    imu_samples = join_imu(accel, gyro)
    imu_cfg = calibration["observation"]["imu"]
    acc_bias, acc_matrix = correction(imu_cfg, "accelerometer", "accelerometer_mps2")
    gyr_bias, gyr_matrix = correction(imu_cfg, "gyroscope", "gyroscope_rads")
    corrected = [(t, acc_matrix @ (a - acc_bias), gyr_matrix @ (w - gyr_bias)) for t, a, w in imu_samples]
    aligned = [(index, head.poses[0][0] + middle - int(start_us) * 1000) for index, middle in frames]
    aligned = [(index, stamp) for index, stamp in aligned if corrected[0][0] <= stamp <= corrected[-1][0]]
    if len(aligned) < 12:
        raise ValueError(f"only {len(aligned)} RGB frames overlap the IMU time range")
    camera_csv, imu_csv, reference = output / "camera_timestamps.csv", output / "imu.csv", output / "head_pose_reference.tum"
    with camera_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("frame_index", "timestamp_ns")); writer.writerows(aligned)
    with imu_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("timestamp_ns", "gx", "gy", "gz", "ax", "ay", "az"))
        writer.writerows((t, *w.tolist(), *a.tolist()) for t, a, w in corrected)
    with reference.open("w", encoding="utf-8") as handle:
        for timestamp, position, quaternion in head.poses:
            handle.write(f"{timestamp * 1e-9:.9f} {position[0]:.9f} {position[1]:.9f} {position[2]:.9f} {quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} {quaternion[3]:.9f}\n")
    return camera_csv, imu_csv, reference, len(aligned)


def make_vertical_video(source: Path, destination: Path, scale: float, rectified: dict[str, Any] | None) -> None:
    width, height = round(2328 * scale), round(1748 * scale)
    if width % 2 or height % 2:
        raise ValueError("--image-scale must produce even stereo dimensions")
    if rectified is None:
        filters = f"[0:v]crop=2328:1748:0:0,scale={width}:{height}:flags=lanczos[left];[0:v]crop=2328:1748:2328:0,scale={width}:{height}:flags=lanczos[right];[left][right]vstack=inputs=2"
        command = ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-filter_complex", filters, "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", str(destination)]
        subprocess.run(command, check=True, timeout=900)
        return
    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(source), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height * 2}", "-framerate", "60", "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", str(destination)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert decoder.stdout is not None and encoder.stdin is not None
    frame_bytes = RGB_WIDTH * RGB_HEIGHT * 3
    try:
        while raw := decoder.stdout.read(frame_bytes):
            if len(raw) != frame_bytes:
                raise RuntimeError("truncated RGB video frame while rectifying")
            image = np.frombuffer(raw, dtype=np.uint8).reshape(RGB_HEIGHT, RGB_WIDTH, 3)
            left = cv2.remap(image[:, :2328], *rectified["maps"][0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            right = cv2.remap(image[:, 2328:], *rectified["maps"][1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            encoder.stdin.write(np.vstack((left, right)).tobytes())
    finally:
        encoder.stdin.close(); decoder.stdout.close()
    decoder_status = decoder.wait(timeout=120); encoder_status = encoder.wait(timeout=120)
    if decoder_status or encoder_status:
        raise RuntimeError(f"rectified video conversion failed (decoder={decoder_status}, encoder={encoder_status})")


def csv_to_tum(source: Path, destination: Path) -> int:
    count = 0
    with source.open(newline="", encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for row in csv.DictReader(src):
            try:
                dst.write(f"{int(row['Timestamp_us']) * 1e-6:.9f} {float(row['X']):.9f} {float(row['Y']):.9f} {float(row['Z']):.9f} {float(row['Quat_X']):.9f} {float(row['Quat_Y']):.9f} {float(row['Quat_Z']):.9f} {float(row['Quat_W']):.9f}\n")
                count += 1
            except (KeyError, TypeError, ValueError):
                continue
    return count


def evaluate(reference: Path, estimate: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    require_reasonable_scale(reference, estimate, out / "scale_gate.json")
    for filename, command in {
        "ape_translation.log": ["evo_ape", "tum", str(reference), str(estimate), "--align", "--pose_relation", "trans_part"],
        "rpe_translation.log": ["evo_rpe", "tum", str(reference), str(estimate), "--align", "--pose_relation", "trans_part", "--delta", "1", "--delta_unit", "f"],
    }.items():
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (out / filename).write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{' '.join(command[:2])} failed; see {out / filename}")


def build_viewer(reference: Path, estimate: Path, out: Path) -> None:
    command = ["python3", str(ROOT / "script/visualize/visualize_trajectory_pair.py"), "--ref", str(reference), "--est", str(estimate), "--output-dir", str(out), "--ref-name", "SXR device reference (head_pose)", "--est-name", "VINS-Fusion stereo-inertial (cam0)", "--title", "SXR VINS-Fusion vs device reference"]
    subprocess.run(command, check=True)


def runtime_command(args: argparse.Namespace, binary: Path, config: Path, camera_csv: Path, imu_csv: Path, video: Path) -> list[str]:
    native = [str(binary), str(config), str(camera_csv), str(imu_csv), str(video)]
    if args.runtime == "native":
        return native
    # The VINS workspace was compiled in ROS1 Noetic.  Mount paths are kept
    # identical, so calibration files and generated CSVs need no copying.
    setup = args.vins_root.expanduser().resolve() / "devel/setup.bash"
    roscore_log = config.parent / "roscore.log"
    runner = " ".join(shlex.quote(value) for value in native)
    shell = (
        "set -e; source /opt/ros/noetic/setup.bash; source " + shlex.quote(str(setup))
        + "; export ROS_MASTER_URI=http://127.0.0.1:11311; roscore > " + shlex.quote(str(roscore_log))
        + " 2>&1 & master=$!; trap 'kill $master >/dev/null 2>&1 || true' EXIT; sleep 1; "
        + runner
    )
    return ["docker", "run", "--rm", "--ipc", "host", "-v", "/home/chenlvping:/home/chenlvping:rw", "-w", str(output_parent(config)), args.docker_image, "bash", "-lc", shell]


def output_parent(config: Path) -> Path:
    """Return a mounted working directory for the Docker-side VINS process."""
    parent = config.parent
    if not str(parent).startswith("/home/chenlvping/"):
        raise ValueError("Docker VINS runtime requires --output-dir below /home/chenlvping")
    return parent


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.2 <= args.image_scale <= 1.0:
        raise ValueError("--image-scale must be in [0.2, 1.0]")
    _, episode = resolve_data_dir(args.episode_dir)
    output = args.output_dir.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    binary = args.vins_root.expanduser().resolve() / "devel/lib/vins/vins_offline_raw_runner"
    if not binary.is_file():
        raise FileNotFoundError(f"VINS offline runner not found: {binary}")
    calibration = json.loads((episode / "calibration.json").read_text(encoding="utf-8"))
    rectified = rectification(calibration, args.image_scale) if args.rectify else None
    config = write_config(output, calibration, args.image_scale, rectified)
    camera_csv, imu_csv, reference, input_frames = export_inputs(episode, output, calibration)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.keep_intermediate:
            work = output / "intermediate"; work.mkdir(exist_ok=True)
        else:
            # Docker has the home tree mounted but not the host's /tmp.  Keep
            # the short-lived video under this dedicated output subdirectory.
            work = output / ".vinsfusion_temp"; shutil.rmtree(work, ignore_errors=True); work.mkdir()
        video = work / "stereo_top_bottom.mp4"
        make_vertical_video(episode / "rgb.mp4", video, args.image_scale, rectified)
        command = runtime_command(args, binary, config, camera_csv, imu_csv, video)
        with (output / "vinsfusion.log").open("w", encoding="utf-8") as handle:
            handle.write("COMMAND: " + " ".join(command) + "\n\n")
            status = subprocess.run(command, cwd=output, stdout=handle, stderr=subprocess.STDOUT, timeout=args.timeout_sec, check=False).returncode
        if status:
            raise RuntimeError(f"VINS-Fusion exited {status}; see {output / 'vinsfusion.log'}")
    finally:
        if temporary is not None:
            temporary.cleanup()
        elif not args.keep_intermediate:
            shutil.rmtree(output / ".vinsfusion_temp", ignore_errors=True)
    raw = output / "pose_data_cam0.csv"
    estimate = output / "vinsfusion_cam0_trajectory.tum"
    if not raw.is_file():
        raise RuntimeError(f"VINS produced no {raw.name}; see {output / 'vinsfusion.log'}")
    pose_count = csv_to_tum(raw, estimate)
    if pose_count < 3:
        raise RuntimeError(f"VINS produced only {pose_count} usable cam0 poses; see {output / 'vinsfusion.log'}")
    evaluate(reference, estimate, output / "evaluation")
    build_viewer(reference, estimate, output / "evaluation/trajectory_viewer")
    (output / "run_summary.json").write_text(json.dumps({
        "algorithm": "VINS-Fusion stereo-inertial", "episode": str(episode), "input_rgb_pairs": input_frames,
        "trajectory_cam0_poses": pose_count, "image_scale": args.image_scale, "rectified_pinhole_input": args.rectify,
        "head_pose_note": "Device-provided reference, not independently verified ground truth.",
        "factory_imu_correction": True, "camera_to_imu_extrinsics": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {episode.name}: {pose_count} cam0 poses from {input_frames} stereo pairs -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
