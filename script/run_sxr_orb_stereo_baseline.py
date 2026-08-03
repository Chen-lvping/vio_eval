#!/usr/bin/env python3
"""Run the full-sequence ORB-SLAM3 stereo baseline on one SXR Ego episode.

The recorder stores side-by-side RGB in ``rgb.mp4`` and timing/poses in
``sensor.mcap``.  Frames are decoded into a temporary EuRoC layout as JPEG
payloads with a ``.png`` filename (OpenCV identifies the payload by header),
so long episodes do not consume multi-gigabyte persistent PNG exports.  Only
the trajectory, configuration, logs and evaluation artifacts are retained.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import av
import cv2
import numpy as np

from check_episode_data import CheckResult, inspect_mcap, iter_mcap_messages, resolve_data_dir
from trajectory_scale_gate import require_reasonable_scale


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORB_ROOT = ROOT / ".orbslam3_determinism_kfcull"
RGB_META_FORMAT = "<qqqIIII"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--orb-root", type=Path, default=DEFAULT_ORB_ROOT)
    parser.add_argument("--binary-name", default="stereo_euroc", help="Stereo executable under Examples/Stereo")
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--jpeg-quality", type=int, default=92, choices=range(50, 101))
    parser.add_argument("--nfeatures", type=int, default=3000, help="ORB feature budget per image")
    parser.add_argument("--ini-th-fast", type=int, default=12, help="initial FAST threshold")
    parser.add_argument("--min-th-fast", type=int, default=3, help="fallback FAST threshold")
    parser.add_argument("--clahe", action="store_true", help="apply mild grayscale local-contrast enhancement before export")
    parser.add_argument("--timeout-sec", type=int, default=900)
    return parser.parse_args(argv)


def read_records(sensor_path: Path) -> tuple[list[tuple[int, int]], list[int]]:
    frames: list[tuple[int, int]] = []
    imu_times: list[int] = []
    for _schema, channel, message in iter_mcap_messages(sensor_path):
        topic = str(getattr(channel, "topic", "")).strip("/")
        payload = bytes(getattr(message, "data", b""))
        if topic == "rgb_metainfo" and len(payload) >= struct.calcsize(RGB_META_FORMAT):
            _start, mid_utc_ns, _pts_us, frame_index, _id, _duration, _gain = struct.unpack_from(RGB_META_FORMAT, payload)
            frames.append((int(frame_index), int(mid_utc_ns)))
        elif topic in {"imu/accel", "imu/gyro"} and len(payload) >= 8:
            timestamp = int(struct.unpack_from("<q", payload)[0])
            if timestamp > 0:
                imu_times.append(timestamp)
    return sorted(frames), sorted(imu_times)


def head_start_offset_ns(metadata: dict[str, Any]) -> int:
    for item in metadata.get("head_pose_details", []):
        if isinstance(item, dict) and item.get("name") == "head_pose" and item.get("start_offset_us") is not None:
            return int(item["start_offset_us"]) * 1000
    raise KeyError("metadata.head_pose_details missing head_pose.start_offset_us")


def inv_se3(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return out


def matrix_yaml(name: str, matrix: np.ndarray) -> list[str]:
    values = ", ".join(f"{float(value):.12g}" for value in matrix.reshape(-1))
    return [name + ": !!opencv-matrix", "  rows: 4", "  cols: 4", "  dt: f", f"  data: [{values}]"]


def write_settings(path: Path, calibration: dict[str, Any], nfeatures: int, ini_th_fast: int, min_th_fast: int) -> None:
    if nfeatures < 500 or ini_th_fast < 1 or min_th_fast < 1 or min_th_fast > ini_th_fast:
        raise ValueError("invalid ORB feature / FAST thresholds")
    rgb = calibration["observation"]["images"]["rgb"]
    cam0, cam1 = rgb["cam0"], rgb["cam1"]
    intr0, intr1 = cam0["intrinsics"]["2328x1748"], cam1["intrinsics"]["2328x1748"]
    d0, d1 = cam0["distortion_coeffs"][:4], cam1["distortion_coeffs"][:4]
    extrinsics = rgb["extrinsics"]
    t_i_c0 = np.asarray(extrinsics["T_ic_imu0_cam0"], dtype=np.float64)
    t_i_c1 = np.asarray(extrinsics["T_ic_imu0_cam1"], dtype=np.float64)
    t_c0_c1 = t_i_c1 @ inv_se3(t_i_c0)
    lines = ["%YAML:1.0", "", 'File.version: "1.0"', "", 'Camera.type: "KannalaBrandt8"', ""]
    for index, intr, distortion in ((1, intr0, d0), (2, intr1, d1)):
        lines.extend([
            f"Camera{index}.fx: {intr['fx']:.12g}", f"Camera{index}.fy: {intr['fy']:.12g}",
            f"Camera{index}.cx: {intr['ppx']:.12g}", f"Camera{index}.cy: {intr['ppy']:.12g}",
            *[f"Camera{index}.k{i + 1}: {value:.12g}" for i, value in enumerate(distortion)], "",
        ])
    lines.extend(matrix_yaml("Stereo.T_c1_c2", t_c0_c1))
    lines.extend([
        "", "Camera1.overlappingBegin: 0", "Camera1.overlappingEnd: 2327",
        "Camera2.overlappingBegin: 0", "Camera2.overlappingEnd: 2327",
        "Camera.width: 2328", "Camera.height: 1748", "Camera.fps: 60", "Camera.RGB: 0",
        "Stereo.ThDepth: 40.0", "", f"ORBextractor.nFeatures: {nfeatures}", "ORBextractor.scaleFactor: 1.2",
        f"ORBextractor.nLevels: 8", f"ORBextractor.iniThFAST: {ini_th_fast}", f"ORBextractor.minThFAST: {min_th_fast}", "",
        "Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1.0", "Viewer.GraphLineWidth: 0.9",
        "Viewer.PointSize: 2.0", "Viewer.CameraSize: 0.08", "Viewer.CameraLineWidth: 3.0",
        "Viewer.ViewpointX: 0.0", "Viewer.ViewpointY: -0.7", "Viewer.ViewpointZ: -3.5", "Viewer.ViewpointF: 500.0",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reference(path: Path, poses: list[tuple[int, tuple[float, float, float], tuple[float, float, float, float]]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for timestamp, position, quaternion in sorted(poses):
            handle.write(f"{timestamp * 1e-9:.9f} {position[0]:.9f} {position[1]:.9f} {position[2]:.9f} {quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} {quaternion[3]:.9f}\n")


def convert_euroc_to_tum(source: Path, destination: Path) -> int:
    count = 0
    with source.open(encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for line in src:
            fields = line.split()
            if len(fields) == 8:
                dst.write(f"{float(fields[0]) * 1e-9:.9f} {' '.join(fields[1:])}\n")
                count += 1
    return count


def evaluate(reference: Path, estimate: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    require_reasonable_scale(reference, estimate, out_dir / "scale_gate.json")
    commands = {
        "ape_translation.log": ["evo_ape", "tum", str(reference), str(estimate), "--align", "--pose_relation", "trans_part"],
        "rpe_translation.log": ["evo_rpe", "tum", str(reference), str(estimate), "--align", "--pose_relation", "trans_part", "--delta", "1", "--delta_unit", "f"],
    }
    for name, command in commands.items():
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (out_dir / name).write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{' '.join(command[:2])} failed; see {out_dir / name}")


def build_viewer(reference: Path, estimate: Path, out_dir: Path) -> None:
    """Create a self-contained, timestamp-aware local comparison viewer."""
    command = [
        "python3", str(ROOT / "script/visualize/visualize_trajectory_pair.py"),
        "--ref", str(reference), "--est", str(estimate), "--output-dir", str(out_dir),
        "--ref-name", "SXR device reference (head_pose)",
        "--est-name", "ORB-SLAM3 stereo (cam0)",
        "--title", "SXR ORB-SLAM3 Stereo vs device reference",
    ]
    subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _root, episode = resolve_data_dir(args.episode_dir)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((episode / "calibration.json").read_text(encoding="utf-8"))
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    frames, imu_times = read_records(episode / "sensor.mcap")
    head, _hand, _channels, _schemas = inspect_mcap(episode / "sensor.mcap", CheckResult())
    if not frames or len(head.poses) < 2 or len(imu_times) < 2:
        raise ValueError("episode lacks rgb_metainfo, head_pose or IMU timestamps")
    head_start = min(pose[0] for pose in head.poses)
    offset = head_start_offset_ns(metadata)
    aligned = [(index, head_start + mid - offset) for index, mid in frames]
    aligned = [(index, timestamp) for index, timestamp in aligned if imu_times[0] <= timestamp <= imu_times[-1]]
    if len(aligned) < 3:
        raise ValueError("fewer than three RGB frames overlap the IMU timeline")
    write_settings(output / "orbslam3_sxr_stereo.yaml", calibration, args.nfeatures, args.ini_th_fast, args.min_th_fast)
    reference = output / "head_pose_reference.tum"
    write_reference(reference, head.poses)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"sxr_{episode.name}_", dir=args.temp_root) as temp_name:
        temp = Path(temp_name)
        left = temp / "mav0/cam0/data"
        right = temp / "mav0/cam1/data"
        left.mkdir(parents=True)
        right.mkdir(parents=True)
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
        times = temp / "times.txt"
        times.write_text("\n".join(str(timestamp) for _index, timestamp in aligned) + "\n", encoding="utf-8")
        orb_root = args.orb_root.expanduser().resolve()
        binary = orb_root / "Examples/Stereo" / args.binary_name
        vocabulary = orb_root / "Vocabulary/ORBvoc.txt"
        if not binary.is_file() or not vocabulary.is_file():
            raise FileNotFoundError(f"ORB binary/vocabulary missing: {binary}, {vocabulary}")
        command = [str(binary), str(vocabulary), str(output / "orbslam3_sxr_stereo.yaml"), str(temp), str(times), "sxr_orb_stereo"]
        with (output / "orbslam3.log").open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + " ".join(command) + "\n\n")
            status = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_sec, check=False).returncode
        if status:
            raise RuntimeError(f"ORB-SLAM3 exited {status}; see {output / 'orbslam3.log'}")
    raw = output / "f_sxr_orb_stereo.txt"
    estimate = output / "orb_stereo_trajectory.tum"
    count = convert_euroc_to_tum(raw, estimate)
    if count < 3:
        raise RuntimeError(f"ORB output only has {count} poses")
    evaluate(reference, estimate, output / "evaluation")
    build_viewer(reference, estimate, output / "evaluation/trajectory_viewer")
    (output / "run_summary.txt").write_text(
        f"episode={episode}\nmode=ORB-SLAM3 Stereo (no IMU)\nimage_pairs={len(aligned)}\ntrajectory_poses={count}\njpeg_quality={args.jpeg_quality}\norb_features={args.nfeatures}\nfast_thresholds={args.ini_th_fast}/{args.min_th_fast}\nclahe={args.clahe}\ntemporary_euroc_export=cleaned\n",
        encoding="utf-8",
    )
    print(f"[OK] {episode.name}: {count}/{len(aligned)} poses -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
