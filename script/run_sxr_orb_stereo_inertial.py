#!/usr/bin/env python3
"""Run the local ORB-SLAM3 stereo-inertial pipeline on one SXR Ego episode.

The SXR recorder uses side-by-side RGB and separate accelerometer/gyroscope
topics in ``sensor.mcap``.  This adapter exports a temporary EuRoC layout,
applies the factory IMU correction from ``calibration.json``, and evaluates
the resulting camera trajectory against the device-provided ``head_pose``
reference.  The reference is not independent ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import av
import cv2
import numpy as np

from check_episode_data import CheckResult, inspect_mcap, iter_mcap_messages, resolve_data_dir
from run_sxr_orb_stereo_baseline import (
    ROOT,
    RGB_META_FORMAT,
    build_viewer,
    convert_euroc_to_tum,
    evaluate,
    head_start_offset_ns,
    inv_se3,
    matrix_yaml,
    write_reference,
    write_settings,
)


DEFAULT_ORB_ROOT = ROOT / ".orbslam3_determinism_kfcull"
VEC3_FORMAT = "<q3f"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--orb-root", type=Path, default=DEFAULT_ORB_ROOT)
    parser.add_argument("--binary-name", default="stereo_inertial_euroc")
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--jpeg-quality", type=int, default=100, choices=range(50, 101))
    parser.add_argument("--nfeatures", type=int, default=3000)
    parser.add_argument("--ini-th-fast", type=int, default=12)
    parser.add_argument("--min-th-fast", type=int, default=3)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=0.95,
                        help="minimum trajectory/image-pair coverage for a passing run")
    parser.add_argument("--allow-partial", action="store_true",
                        help="keep a partial trajectory as a successful process exit (diagnostics only)")
    parser.add_argument(
        "--imu-time-offset-sec",
        type=float,
        default=None,
        help="IMU minus rgb-left timestamp. Defaults to calibration.rgb-left.",
    )
    parser.add_argument("--timeout-sec", type=int, default=1800)
    return parser.parse_args(argv)


def read_sensor_records(path: Path) -> tuple[list[tuple[int, int]], list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]]]:
    frames: list[tuple[int, int]] = []
    accel: list[tuple[int, np.ndarray]] = []
    gyro: list[tuple[int, np.ndarray]] = []
    for _schema, channel, message in iter_mcap_messages(path):
        topic = str(getattr(channel, "topic", "")).strip("/")
        payload = bytes(getattr(message, "data", b""))
        if topic == "rgb_metainfo" and len(payload) >= struct.calcsize(RGB_META_FORMAT):
            _start, mid_utc_ns, _pts_us, frame_index, _id, _duration, _gain = struct.unpack_from(RGB_META_FORMAT, payload)
            frames.append((int(frame_index), int(mid_utc_ns)))
        elif topic in {"imu/accel", "imu/gyro"} and len(payload) >= struct.calcsize(VEC3_FORMAT):
            timestamp, x, y, z = struct.unpack_from(VEC3_FORMAT, payload)
            if timestamp > 0:
                (accel if topic.endswith("accel") else gyro).append((int(timestamp), np.array((x, y, z), dtype=np.float64)))
    return sorted(frames), sorted(accel), sorted(gyro)


def correct_imu(calibration: dict, accel: list[tuple[int, np.ndarray]], gyro: list[tuple[int, np.ndarray]]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    imu = calibration["observation"]["imu"]

    def calibration_matrix(kind: str, bias_key: str) -> tuple[np.ndarray, np.ndarray]:
        bias = np.asarray(imu["bias"][bias_key], dtype=np.float64)
        scale = np.asarray(imu["scale_factor"][kind], dtype=np.float64)
        n0, n1, n2 = imu["nonorthogonality"][kind]
        nonorthogonal = np.array(((1.0, n0, n1), (0.0, 1.0, n2), (0.0, 0.0, 1.0)), dtype=np.float64)
        return bias, nonorthogonal @ np.diag(1.0 + scale)

    accel_bias, accel_matrix = calibration_matrix("accelerometer", "accelerometer_mps2")
    gyro_bias, gyro_matrix = calibration_matrix("gyroscope", "gyroscope_rads")
    result: list[tuple[int, np.ndarray, np.ndarray]] = []
    index = 0
    for timestamp, omega in gyro:
        while index + 1 < len(accel) and abs(accel[index + 1][0] - timestamp) <= abs(accel[index][0] - timestamp):
            index += 1
        if abs(accel[index][0] - timestamp) <= 2_000_000:
            result.append((timestamp, accel_matrix @ (accel[index][1] - accel_bias), gyro_matrix @ (omega - gyro_bias)))
    if len(result) < 100:
        raise ValueError("fewer than 100 matched IMU samples")
    return result


def append_imu_settings(path: Path, calibration: dict) -> float:
    observation = calibration["observation"]
    imu = observation["imu"]
    rgb = observation["images"]["rgb"]
    # SXR's stored transform maps IMU coordinates into cam0 coordinates.
    # ORB-SLAM3 IMU.T_b_c1 is camera coordinates into the IMU/body frame.
    t_b_c0 = inv_se3(np.asarray(rgb["extrinsics"]["T_ic_imu0_cam0"], dtype=np.float64))
    rate = 1013.8
    noise = imu["noise"]
    lines = ["", *matrix_yaml("IMU.T_b_c1", t_b_c0)]
    lines.extend([
        "", f"IMU.Frequency: {rate:.12g}",
        f"IMU.NoiseGyro: {float(noise['gyro_noise_std_rads'][0]) / np.sqrt(rate):.12g}",
        f"IMU.NoiseAcc: {float(noise['accel_noise_std_mps2'][0]) / np.sqrt(rate):.12g}",
        f"IMU.GyroWalk: {float(noise['gyro_bias_std_rads'][0]):.12g}",
        f"IMU.AccWalk: {float(noise['accel_bias_std_mps2'][0]):.12g}",
        "IMU.InsertKFsWhenLost: 1",
    ])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return float(imu.get("time_alignment_s", {}).get("cameras", {}).get("rgb-left", 0.0))


def write_imu_csv(path: Path, samples: list[tuple[int, np.ndarray, np.ndarray]], camera_offset_ns: int, start_ns: int, end_ns: int) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["#timestamp [ns]", "gx", "gy", "gz", "ax", "ay", "az"])
        for timestamp, acceleration, angular_rate in samples:
            adjusted = timestamp - camera_offset_ns
            if start_ns <= adjusted <= end_ns:
                writer.writerow([adjusted, *angular_rate, *acceleration])
                count += 1
    if count < 100:
        raise ValueError("insufficient IMU coverage after timestamp alignment")
    return count


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < args.min_coverage <= 1.0:
        raise ValueError("--min-coverage must be in (0, 1]")
    _root, episode = resolve_data_dir(args.episode_dir)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((episode / "calibration.json").read_text(encoding="utf-8"))
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    frames, accel, gyro = read_sensor_records(episode / "sensor.mcap")
    head, _hand, _channels, _schemas = inspect_mcap(episode / "sensor.mcap", CheckResult())
    if len(frames) < 3 or len(head.poses) < 2:
        raise ValueError("episode lacks RGB frames or head_pose")
    imu = correct_imu(calibration, accel, gyro)
    head_start = min(pose[0] for pose in head.poses)
    offset = head_start_offset_ns(metadata)
    aligned = [(index, head_start + mid - offset) for index, mid in frames]
    camera_offset = args.imu_time_offset_sec
    if camera_offset is None:
        camera_offset = float(calibration["observation"]["imu"].get("time_alignment_s", {}).get("cameras", {}).get("rgb-left", 0.0))
    offset_ns = int(round(camera_offset * 1e9))
    imu_camera_times = [timestamp - offset_ns for timestamp, _accel, _gyro in imu]
    aligned = [(index, timestamp) for index, timestamp in aligned if imu_camera_times[0] <= timestamp <= imu_camera_times[-1]]
    if len(aligned) < 3:
        raise ValueError("fewer than three RGB frames overlap the IMU timeline")

    settings = output / "orbslam3_sxr_stereo_inertial.yaml"
    write_settings(settings, calibration, args.nfeatures, args.ini_th_fast, args.min_th_fast)
    default_offset = append_imu_settings(settings, calibration)
    if args.imu_time_offset_sec is None:
        camera_offset = default_offset
        offset_ns = int(round(camera_offset * 1e9))
    reference = output / "head_pose_reference.tum"
    write_reference(reference, head.poses)
    args.temp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"sxr_imu_{episode.name}_", dir=args.temp_root) as temp_name:
        temp = Path(temp_name)
        left, right = temp / "mav0/cam0/data", temp / "mav0/cam1/data"
        left.mkdir(parents=True); right.mkdir(parents=True)
        (temp / "mav0/imu0").mkdir(parents=True)
        wanted = dict(aligned)
        container = av.open(str(episode / "rgb.mp4"))
        written = 0
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None
        for frame_index, frame in enumerate(container.decode(container.streams.video[0])):
            timestamp = wanted.get(frame_index)
            if timestamp is None:
                continue
            image = frame.to_ndarray(format="bgr24")
            if image.shape[:2] != (1748, 4656):
                raise ValueError(f"unexpected RGB size {image.shape[1]}x{image.shape[0]}")
            for destination, half in ((left, image[:, :2328]), (right, image[:, 2328:])):
                if clahe is not None:
                    enhanced = clahe.apply(cv2.cvtColor(half, cv2.COLOR_BGR2GRAY))
                    half = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                ok, encoded = cv2.imencode(".jpg", half, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not ok:
                    raise RuntimeError("JPEG encode failed")
                encoded.tofile(str(destination / f"{timestamp}.png"))
            written += 1
        container.close()
        if written != len(aligned):
            raise RuntimeError(f"wrote {written} stereo pairs, expected {len(aligned)}")
        (temp / "times.txt").write_text("\n".join(str(timestamp) for _index, timestamp in aligned) + "\n", encoding="utf-8")
        imu_count = write_imu_csv(temp / "mav0/imu0/data.csv", imu, offset_ns, aligned[0][1] - 5_000_000, aligned[-1][1] + 5_000_000)
        orb_root = args.orb_root.expanduser().resolve()
        binary = orb_root / "Examples/Stereo-Inertial" / args.binary_name
        vocabulary = orb_root / "Vocabulary/ORBvoc.txt"
        if not binary.is_file() or not vocabulary.is_file():
            raise FileNotFoundError(f"ORB binary/vocabulary missing: {binary}, {vocabulary}")
        command = [str(binary), str(vocabulary), str(settings), str(temp), str(temp / "times.txt"), "sxr_orb_stereo_inertial"]
        environment = os.environ.copy()
        environment.setdefault("ORB_SLAM3_FINAL_BA_ITERS", "10")
        with (output / "orbslam3.log").open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + " ".join(command) + "\n")
            log.write(f"IMU_TIME_OFFSET_SEC: {camera_offset:.9f}\n\n")
            status = subprocess.run(command, cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_sec, check=False).returncode
        if status:
            raise RuntimeError(f"ORB-SLAM3 exited {status}; see {output / 'orbslam3.log'}")
    estimate = output / "orb_stereo_inertial_trajectory.tum"
    count = convert_euroc_to_tum(output / "f_sxr_orb_stereo_inertial.txt", estimate)
    if count < 3:
        raise RuntimeError(f"ORB output only has {count} poses")
    coverage = count / len(aligned)
    evaluate(reference, estimate, output / "evaluation")
    build_viewer(reference, estimate, output / "evaluation/trajectory_viewer")
    (output / "run_summary.txt").write_text(
        f"episode={episode}\nmode=ORB-SLAM3 Stereo-Inertial\nimage_pairs={len(aligned)}\ntrajectory_poses={count}\n"
        f"imu_samples={imu_count}\nimu_time_offset_sec={camera_offset:.9f}\njpeg_quality={args.jpeg_quality}\n"
        f"orb_features={args.nfeatures}\nfast_thresholds={args.ini_th_fast}/{args.min_th_fast}\nclahe={args.clahe}\n"
        f"trajectory_coverage={coverage:.6f}\nfactory_imu_correction=True\ntemporary_euroc_export=cleaned\n",
        encoding="utf-8",
    )
    if coverage < args.min_coverage:
        message = (f"partial ORB Stereo-Inertial trajectory: {count}/{len(aligned)} "
                   f"({coverage:.1%}) is below --min-coverage {args.min_coverage:.1%}")
        if not args.allow_partial:
            raise RuntimeError(message + f"; diagnostics retained in {output}")
        print(f"[PARTIAL] {message} -> {output}")
        return 0
    print(f"[OK] {episode.name}: {count}/{len(aligned)} poses, {imu_count} IMU samples -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
