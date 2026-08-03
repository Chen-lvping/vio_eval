#!/usr/bin/env python3
"""Run Basalt stereo VIO on one offline SXR Ego episode.

The adapter exports the SXR side-by-side RGB and synchronized IMU stream into
the EuRoC layout expected by Basalt.  It keeps the factory KB4 calibration and
IMU correction, and evaluates cam0 against the device ``head_pose`` reference.
That reference is a device-provided reference trajectory, not independent GT.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import av
import cv2
import numpy as np

from check_episode_data import CheckResult, inspect_mcap, resolve_data_dir
from run_sxr_orb_stereo_baseline import (
    ROOT,
    head_start_offset_ns,
    inv_se3,
    write_reference,
)
from run_sxr_orb_stereo_inertial import correct_imu, read_sensor_records


DEFAULT_BASALT = ROOT / "third_party/basalt/build/release/basalt_vio"
DEFAULT_CONFIG = ROOT / "third_party/basalt/data/euroc_config.json"
DEFAULT_CALIB_TEMPLATE = ROOT / "third_party/basalt/data/t265_kb4_calib.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basalt-bin", type=Path, default=DEFAULT_BASALT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--jpeg-quality", type=int, default=100, choices=range(50, 101))
    parser.add_argument("--image-scale", type=float, default=1.0,
                        help="Uniform image and intrinsics scale in (0, 1].")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Process at most this many stereo pairs (0 means the full episode).",
    )
    parser.add_argument("--imu-time-offset-sec", type=float, default=None,
                        help="camera timestamp plus this value equals the IMU clock; defaults to rgb-left factory value")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--keep-intermediate", action="store_true")
    return parser.parse_args(argv)


def build_basalt_viewer(reference: Path, estimate: Path, output: Path) -> None:
    subprocess.run([
        "python3", str(ROOT / "script/visualize/visualize_trajectory_pair.py"),
        "--ref", str(reference), "--est", str(estimate), "--output-dir", str(output),
        "--ref-name", "SXR device reference (head_pose)",
        "--est-name", "Basalt stereo VIO (cam0)",
        "--title", "SXR Basalt Stereo VIO vs device reference",
    ], check=True)


def matrix_to_xyzw(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Numerically stable rotation-matrix to xyzw quaternion conversion."""
    m = matrix[:3, :3]
    trace = float(np.trace(m))
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw, qx = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        qy, qz = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw, qx = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        qy, qz = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw, qx = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        qy, qz = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def write_calibration(path: Path, calibration: dict, offset_ns: int, image_scale: float) -> np.ndarray:
    """Write Basalt's cereal JSON and return its cam0-to-IMU transform."""
    data = json.loads(DEFAULT_CALIB_TEMPLATE.read_text(encoding="utf-8"))
    target = data["value0"]
    rgb = calibration["observation"]["images"]["rgb"]
    imu = calibration["observation"]["imu"]
    target["intrinsics"] = []
    target["T_imu_cam"] = []
    t_i_c: list[np.ndarray] = []
    for name in ("cam0", "cam1"):
        cam = rgb[name]
        intr = cam["intrinsics"]["2328x1748"]
        distortion = cam["distortion_coeffs"][:4]
        target["intrinsics"].append({"camera_type": "kb4", "intrinsics": {
            "fx": float(intr["fx"]) * image_scale, "fy": float(intr["fy"]) * image_scale,
            "cx": float(intr["ppx"]) * image_scale, "cy": float(intr["ppy"]) * image_scale,
            "k1": float(distortion[0]), "k2": float(distortion[1]),
            "k3": float(distortion[2]), "k4": float(distortion[3]),
        }})
        # SXR stores IMU -> camera. Basalt's T_imu_cam is camera -> IMU.
        camera_to_imu = inv_se3(np.asarray(rgb["extrinsics"][f"T_ic_imu0_{name}"], dtype=np.float64))
        t_i_c.append(camera_to_imu)
        qx, qy, qz, qw = matrix_to_xyzw(camera_to_imu)
        target["T_imu_cam"].append({"px": float(camera_to_imu[0, 3]), "py": float(camera_to_imu[1, 3]),
                                    "pz": float(camera_to_imu[2, 3]), "qx": qx, "qy": qy, "qz": qz, "qw": qw})
    noise = imu["noise"]
    target["resolution"] = [[round(2328 * image_scale), round(1748 * image_scale)] for _ in range(2)]
    target["imu_update_rate"] = 1013.8
    target["accel_noise_std"] = list(map(float, noise["accel_noise_std_mps2"]))
    target["gyro_noise_std"] = list(map(float, noise["gyro_noise_std_rads"]))
    target["accel_bias_std"] = list(map(float, noise["accel_bias_std_mps2"]))
    target["gyro_bias_std"] = list(map(float, noise["gyro_bias_std_rads"]))
    target["cam_time_offset_ns"] = int(offset_ns)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return t_i_c[0]


def write_euroc_dataset(temp: Path, episode: Path, aligned: list[tuple[int, int]], imu_samples: list[tuple[int, np.ndarray, np.ndarray]], jpeg_quality: int, image_scale: float) -> int:
    left, right = temp / "mav0/cam0", temp / "mav0/cam1"
    for folder in (left / "data", right / "data", temp / "mav0/imu0"):
        folder.mkdir(parents=True, exist_ok=True)
    wanted = dict(aligned)
    rows: list[tuple[int, str]] = []
    container = av.open(str(episode / "rgb.mp4"))
    for frame_index, frame in enumerate(container.decode(container.streams.video[0])):
        if frame_index > max(wanted):
            break
        timestamp = wanted.get(frame_index)
        if timestamp is None:
            continue
        image = frame.to_ndarray(format="bgr24")
        if image.shape[:2] != (1748, 4656):
            raise ValueError(f"unexpected RGB size {image.shape[1]}x{image.shape[0]}")
        # Basalt's image loader uses the filename suffix to select a decoder.
        # Keep the JPEG payload and extension consistent.
        filename = f"{timestamp}.jpg"
        for destination, half in ((left / "data", image[:, :2328]), (right / "data", image[:, 2328:])):
            if image_scale != 1.0:
                half = cv2.resize(half, (round(half.shape[1] * image_scale), round(half.shape[0] * image_scale)), interpolation=cv2.INTER_AREA)
            # Basalt accepts JPEG by file signature; color is kept for imread.
            ok, encoded = cv2.imencode(".jpg", half, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            encoded.tofile(str(destination / filename))
        rows.append((timestamp, filename))
    container.close()
    if len(rows) != len(aligned):
        raise RuntimeError(f"wrote {len(rows)} pairs, expected {len(aligned)}")
    for camera in (left, right):
        with (camera / "data.csv").open("w", newline="", encoding="utf-8") as handle:
            # Basalt's EuRoC reader treats the filename field literally.
            # Use LF, otherwise csv's default CRLF leaves a trailing '\\r'.
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["#timestamp [ns]", "filename"])
            writer.writerows(rows)
    with (temp / "mav0/imu0/data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["#timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]",
                         "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"])
        for timestamp, acceleration, angular_rate in imu_samples:
            writer.writerow([timestamp, *angular_rate, *acceleration])
    return len(rows)


def transform_imu_to_cam0(source: Path, destination: Path, t_i_c0: np.ndarray) -> int:
    count = 0
    with source.open(encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip() or line.startswith("#"):
                continue
            values = line.replace(",", " ").split()
            if len(values) != 8:
                continue
            timestamp, x, y, z, qx, qy, qz, qw = map(float, values)
            quat = np.array([qx, qy, qz, qw], dtype=np.float64)
            quat /= np.linalg.norm(quat)
            qx, qy, qz, qw = quat
            rotation = np.array(((1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)),
                                 (2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)),
                                 (2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy))), dtype=np.float64)
            t_w_i = np.eye(4); t_w_i[:3, :3] = rotation; t_w_i[:3, 3] = (x, y, z)
            t_w_c = t_w_i @ t_i_c0
            ox, oy, oz, ow = matrix_to_xyzw(t_w_c)
            dst.write(f"{timestamp:.9f} {t_w_c[0,3]:.9f} {t_w_c[1,3]:.9f} {t_w_c[2,3]:.9f} {ox:.9f} {oy:.9f} {oz:.9f} {ow:.9f}\n")
            count += 1
    return count


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.1 <= args.image_scale <= 1.0:
        raise ValueError("--image-scale must be in [0.1, 1.0]")
    _root, episode = resolve_data_dir(args.episode_dir)
    output = args.output_dir.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    binary = args.basalt_bin.expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"Basalt executable missing: {binary}; build third_party/basalt first")
    calibration = json.loads((episode / "calibration.json").read_text(encoding="utf-8"))
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    frames, accel, gyro = read_sensor_records(episode / "sensor.mcap")
    head, _hand, _channels, _schemas = inspect_mcap(episode / "sensor.mcap", CheckResult())
    if len(frames) < 3 or len(head.poses) < 2:
        raise ValueError("episode lacks RGB frames or head_pose")
    samples = correct_imu(calibration, accel, gyro)
    camera_offset = args.imu_time_offset_sec
    if camera_offset is None:
        camera_offset = float(calibration["observation"]["imu"]["time_alignment_s"]["cameras"]["rgb-left"])
    offset_ns = int(round(camera_offset * 1e9))
    head_start = min(item[0] for item in head.poses)
    aligned = [(index, head_start + mid - head_start_offset_ns(metadata)) for index, mid in frames]
    imu_start, imu_end = samples[0][0], samples[-1][0]
    aligned = [(index, timestamp) for index, timestamp in aligned if imu_start <= timestamp + offset_ns <= imu_end]
    if args.max_pairs:
        if args.max_pairs < 3:
            raise ValueError("--max-pairs must be at least 3")
        aligned = aligned[:args.max_pairs]
    if len(aligned) < 3:
        raise ValueError("fewer than three image pairs overlap IMU after factory offset")
    reference = output / "head_pose_reference.tum"; write_reference(reference, head.poses)
    calib_file = output / "basalt_sxr_kb4_calib.json"
    t_i_c0 = write_calibration(calib_file, calibration, offset_ns, args.image_scale)
    config_file = args.config.expanduser().resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Basalt config missing: {config_file}")
    args.temp_root.mkdir(parents=True, exist_ok=True)
    persistent = output / "euroc_export" if args.keep_intermediate else None
    temp_context = tempfile.TemporaryDirectory(prefix=f"sxr_basalt_{episode.name}_", dir=args.temp_root) if persistent is None else None
    temp = persistent if persistent else Path(temp_context.name)
    try:
        image_pairs = write_euroc_dataset(temp, episode, aligned, samples, args.jpeg_quality, args.image_scale)
        command = [str(binary), "--dataset-path", str(temp), "--dataset-type", "euroc", "--cam-calib", str(calib_file),
                   "--config-path", str(config_file), "--show-gui", "0", "--save-trajectory", "tum"]
        if args.num_threads:
            command += ["--num-threads", str(args.num_threads)]
        with (output / "basalt.log").open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + " ".join(command) + "\n")
            log.write(f"CAMERA_TO_IMU_OFFSET_SEC: {camera_offset:.9f}\n")
            environment = None
            # Upstream binary releases keep libbasalt beside bin rather than
            # registering it system-wide. Source builds do not need this path.
            release_lib = binary.parent.parent / "lib"
            if release_lib.is_dir():
                environment = dict(os.environ)
                environment["LD_LIBRARY_PATH"] = str(release_lib) + ":" + environment.get("LD_LIBRARY_PATH", "")
            status = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout_sec, check=False, env=environment).returncode
        if status:
            raise RuntimeError(f"Basalt exited {status}; see {output / 'basalt.log'}")
    finally:
        if temp_context is not None:
            temp_context.cleanup()
    raw = output / "trajectory.txt"; estimate = output / "basalt_stereo_vio_cam0.tum"
    count = transform_imu_to_cam0(raw, estimate, t_i_c0)
    if count < 3:
        raise RuntimeError(f"Basalt output only has {count} poses")
    from run_sxr_orb_stereo_baseline import evaluate
    evaluate(reference, estimate, output / "evaluation")
    build_basalt_viewer(reference, estimate, output / "evaluation/trajectory_viewer")
    (output / "run_summary.txt").write_text(
        f"episode={episode}\nmode=Basalt stereo VIO (KB4 + factory IMU)\nimage_pairs={image_pairs}\ntrajectory_poses={count}\n"
        f"imu_samples={len(samples)}\ncam_time_offset_sec={camera_offset:.9f}\njpeg_quality={args.jpeg_quality}\nimage_scale={args.image_scale}\nfactory_imu_correction=True\n"
        "output_pose=cam0 (Basalt IMU pose transformed with factory extrinsic)\nreference=device head_pose (not independent GT)\n",
        encoding="utf-8")
    print(f"[OK] {episode.name}: {count}/{image_pairs} poses -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
