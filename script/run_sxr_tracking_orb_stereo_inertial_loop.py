#!/usr/bin/env python3
"""Run loop-closing ORB-SLAM3 stereo-inertial on an SXR CSV tracking episode."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import av
import cv2
import numpy as np

from run_sxr_csv_basalt import (
    camera_time_offset_sec,
    corrected_imu,
    factory_t_imu_camera,
    inverse,
    load_camera_from_imu_rotation,
    load_frames,
    load_stereo_camera1_from_camera0,
)
from run_sxr_csv_orb_stereo import convert_euroc, write_settings
from run_sxr_csv_vinsfusion import write_reference
from run_sxr_orb_stereo_baseline import ROOT, build_viewer, evaluate


DEFAULT_ORB_ROOT = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_clean")
DEFAULT_ROTATION = ROOT / (
    "data/evaluation/workbench/"
    "sxr_csv_20260730_205558_tracking_aprilgrid_preflight_v1/"
    "cam_imu_timing_extrinsic_validation.json"
)
DEFAULT_STEREO_CALIBRATION = ROOT / (
    "data/evaluation/workbench/"
    "sxr_tracking_stereo_aprilgrid_calibration_20260731_104007.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--orb-root", type=Path, default=DEFAULT_ORB_ROOT)
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit exported pairs; 0 uses the full episode.")
    parser.add_argument("--nfeatures", type=int, default=3500)
    parser.add_argument("--ini-th-fast", type=int, default=10)
    parser.add_argument("--min-th-fast", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--imu-fast-init", type=int, choices=(0, 1), default=0)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument(
        "--camera-from-imu-rotation-json",
        type=Path,
        default=DEFAULT_ROTATION,
        help="Independently validated camera-from-IMU rotation used by the Basalt baseline.",
    )
    parser.add_argument(
        "--stereo-camera1-from-camera0-json",
        type=Path,
        default=DEFAULT_STEREO_CALIBRATION,
        help="Independently measured AprilGrid stereo calibration used for ORB stereo initialization.",
    )
    return parser.parse_args(argv)


def matrix_yaml(name: str, value: np.ndarray) -> str:
    values = ", ".join(f"{float(item):.12g}" for item in value.reshape(-1))
    return (
        f"{name}: !!opencv-matrix\n"
        "  rows: 4\n"
        "  cols: 4\n"
        "  dt: f\n"
        f"  data: [{values}]\n"
    )


def append_imu_settings(
    settings: Path,
    calibration: dict[str, Any],
    camera: dict[str, Any],
    samples: list[tuple[int, np.ndarray, np.ndarray]],
    camera_rotation: np.ndarray | None,
    fast_init: int,
) -> dict[str, float]:
    intervals = np.diff(np.asarray([timestamp for timestamp, _accel, _gyro in samples], dtype=np.int64))
    intervals = intervals[intervals > 0]
    if len(intervals) == 0:
        raise ValueError("IMU timestamps are not strictly increasing")
    rate_hz = 1e9 / float(np.median(intervals))
    t_imu_cam = factory_t_imu_camera(camera, "imu-to-camera", "wxyz")
    if camera_rotation is not None:
        t_imu_cam[:3, :3] = camera_rotation.T
    t_body_cam = inverse(t_imu_cam)
    noise = calibration["noise"]
    settings.open("a", encoding="utf-8").write(
        "\n# Tracking IMU calibration. T_b_c1 is left camera -> IMU/body.\n"
        + matrix_yaml("IMU.T_b_c1", t_body_cam)
        + f"IMU.Frequency: {rate_hz:.12g}\n"
        + f"IMU.NoiseGyro: {float(noise['gyro_noise_std_rads'][0]) / np.sqrt(rate_hz):.12g}\n"
        + f"IMU.NoiseAcc: {float(noise['accel_noise_std_mps2'][0]) / np.sqrt(rate_hz):.12g}\n"
        + f"IMU.GyroWalk: {float(noise['gyro_bias_std_rads'][0]):.12g}\n"
        + f"IMU.AccWalk: {float(noise['accel_bias_std_mps2'][0]):.12g}\n"
        + f"IMU.fastInit: {int(fast_init)}\n",
    )
    return {"imu_rate_hz": rate_hz}


def export_euroc(
    destination: Path,
    episode: Path,
    frames: list[tuple[int, int]],
    samples: list[tuple[int, np.ndarray, np.ndarray]],
    width: int,
    height: int,
    scale: float,
    jpeg_quality: int,
) -> None:
    left, right = destination / "mav0/cam0/data", destination / "mav0/cam1/data"
    imu = destination / "mav0/imu0"
    left.mkdir(parents=True, exist_ok=True)
    right.mkdir(parents=True, exist_ok=True)
    imu.mkdir(parents=True, exist_ok=True)
    wanted = dict(frames)
    written: list[int] = []
    container = av.open(str(episode / "tracking.mp4"))
    try:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            timestamp = wanted.get(index)
            if timestamp is None:
                continue
            image = frame.to_ndarray(format="bgr24")
            if image.shape[:2] != (height, width * 2):
                raise ValueError(f"unexpected tracking frame shape {image.shape[1]}x{image.shape[0]}")
            for target, half in ((left, image[:, :width]), (right, image[:, width:])):
                if scale != 1.0:
                    half = cv2.resize(half, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", half, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if not ok:
                    raise RuntimeError(f"failed to encode tracking frame {index}")
                encoded.tofile(str(target / f"{timestamp}.png"))
            written.append(timestamp)
    finally:
        container.close()
    if written != [timestamp for _index, timestamp in frames]:
        raise RuntimeError(f"exported {len(written)}/{len(frames)} tracking stereo pairs")
    (destination / "times.txt").write_text("\n".join(str(timestamp) for timestamp in written) + "\n", encoding="utf-8")
    with (imu / "data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["#timestamp [ns]", "gx", "gy", "gz", "ax", "ay", "az"])
        for timestamp, acceleration, gyro in samples:
            writer.writerow([timestamp, *gyro.tolist(), *acceleration.tolist()])


def loop_evidence(log: Path) -> list[str]:
    text = log.read_text(encoding="utf-8", errors="replace")
    return re.findall(r"(?:\*Loop detected|PR: Loop detected[^\n]*|Correct Loop[^\n]*)", text)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.2 <= args.image_scale <= 1.0:
        raise ValueError("--image-scale must be in [0.2, 1.0]")
    episode = args.episode_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    required = (
        "accel.csv", "gyro.csv", "head_pose.csv", "imu_calibration.json",
        "tracking.mp4", "tracking_metainfo.csv", "camera_params_tracking.json",
    )
    missing = [name for name in required if not (episode / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {episode}")
    output.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((episode / "imu_calibration.json").read_text(encoding="utf-8"))
    image = json.loads((episode / "camera_params_tracking.json").read_text(encoding="utf-8"))
    samples = corrected_imu(episode, calibration)
    offset_ns = round(camera_time_offset_sec(calibration, "tracking", None) * 1e9)
    if args.max_pairs < 0:
        raise ValueError("--max-pairs must be non-negative")
    frames = load_frames(episode, "tracking", samples, 1, args.max_pairs, offset_ns)
    camera_rotation = load_camera_from_imu_rotation(args.camera_from_imu_rotation_json)
    stereo_camera1_from_camera0 = load_stereo_camera1_from_camera0(args.stereo_camera1_from_camera0_json)
    settings = output / "orbslam3_tracking_stereo_inertial_loop.yaml"
    geometry = write_settings(
        settings,
        image,
        args.image_scale,
        args.nfeatures,
        args.ini_th_fast,
        args.min_th_fast,
        camera1_from_camera0=stereo_camera1_from_camera0,
    )
    imu_info = append_imu_settings(settings, calibration, image["cameras"][0], samples, camera_rotation, args.imu_fast_init)
    reference = output / "head_pose_reference.tum"
    write_reference(episode / "head_pose.csv", reference)
    orb_root = args.orb_root.expanduser().resolve()
    binary = orb_root / "Examples/Stereo-Inertial/stereo_inertial_euroc"
    vocabulary = orb_root / "Vocabulary/ORBvoc.txt"
    if not binary.is_file() or not vocabulary.is_file():
        raise FileNotFoundError(f"ORB-SLAM3 binary or vocabulary missing under {orb_root}")
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_intermediate:
        work = output / "orb_export"
        if work.exists():
            raise FileExistsError(f"refusing to overwrite existing intermediate directory: {work}")
        work.mkdir()
    else:
        temporary = tempfile.TemporaryDirectory(prefix=f"sxr_tracking_orb_si_{episode.name}_", dir="/tmp")
        work = Path(temporary.name)
    name = "sxr_tracking_orb_si_loop"
    try:
        camera = image["cameras"][0]
        export_euroc(work, episode, frames, samples, int(camera["width"]), int(camera["height"]), args.image_scale, args.jpeg_quality)
        command = [str(binary), str(vocabulary), str(settings), str(work), str(work / "times.txt"), name]
        environment = os.environ.copy()
        environment["ORB_SLAM3_ENABLE_VIEWER"] = "0"
        with (output / "orbslam3.log").open("w", encoding="utf-8") as handle:
            handle.write("COMMAND: " + " ".join(command) + "\n")
            handle.write(f"TRACKING_CAMERA_TO_IMU_OFFSET_NS: {offset_ns}\n\n")
            status = subprocess.run(
                command, cwd=output, env=environment, stdout=handle, stderr=subprocess.STDOUT,
                timeout=args.timeout_sec, check=False,
            ).returncode
        if status:
            raise RuntimeError(f"ORB-SLAM3 exited {status}; see {output / 'orbslam3.log'}")
    finally:
        if temporary is not None:
            temporary.cleanup()
    source = output / f"f_{name}.txt"
    if not source.is_file():
        raise RuntimeError(f"ORB-SLAM3 did not create {source.name}; see {output / 'orbslam3.log'}")
    estimate = output / "orb_stereo_inertial_loop_trajectory.tum"
    pose_count = convert_euroc(source, estimate)
    if pose_count < 3:
        raise RuntimeError(f"ORB-SLAM3 produced only {pose_count} usable poses")
    evaluate(reference, estimate, output / "evaluation")
    build_viewer(reference, estimate, output / "evaluation/trajectory_viewer")
    evidence = loop_evidence(output / "orbslam3.log")
    summary = {
        "algorithm": "ORB-SLAM3 stereo-inertial with native loop closing",
        "episode": str(episode),
        "stream": "tracking",
        "image_pairs": len(frames),
        "trajectory_poses": pose_count,
        "trajectory_coverage": pose_count / len(frames),
        "image_scale": args.image_scale,
        "factory_camera_time_offset_ns": offset_ns,
        "camera_from_imu_rotation_override": str(args.camera_from_imu_rotation_json),
        "stereo_camera1_from_camera0_source": str(args.stereo_camera1_from_camera0_json),
        "imu_fast_init": args.imu_fast_init,
        "loop_detection_messages": evidence,
        "loop_detected": bool(evidence),
        "head_pose_note": "Used only after SLAM for scale-gated evaluation; not an independent ground truth.",
        **geometry,
        **imu_info,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {episode.name}: {pose_count}/{len(frames)} poses; loop_detected={bool(evidence)} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
