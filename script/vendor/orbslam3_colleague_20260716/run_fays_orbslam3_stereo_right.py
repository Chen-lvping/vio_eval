#!/usr/bin/env python3
"""Offline ORB-SLAM3 runner for Fays stereo_right MKV + MCAP IMU data."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_colleague_sync")
DEFAULT_EPISODE_DIR = Path("/home/junquan/dm_test/robot_traj_test/0618/episode_20260618_0002")
DEFAULT_ROBOT_JSON = Path("/home/junquan/dm_test/robot_traj_test/0618/rm75_pose_traj_2.json")
DEFAULT_T_TCP_CAM = np.array(
    [
        [0.854672738, -0.422689316, 0.301443615, 0.020087823],
        [0.519166541, 0.694970001, -0.497476432, -0.097333617],
        [0.000783703, 0.581678983, 0.813418064, 0.052300458],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
POSE_CSV_COLUMNS = ["Timestamp_us", "Vx", "Vy", "Vz", "X", "Y", "Z", "Quat_X", "Quat_Y", "Quat_Z", "Quat_W"]


@dataclass(frozen=True)
class PipelinePaths:
    episode_dir: Path
    mkv_file: Path
    mcap_file: Path
    calibration_json: Path
    work_dir: Path
    sequence_dir: Path
    left_dir: Path
    right_dir: Path
    timestamps_file: Path
    imu_csv: Path
    settings_yaml: Path
    raw_trajectory: Path
    keyframe_trajectory: Path
    output_csv: Path
    eval_dir: Path


@dataclass(frozen=True)
class ImuRecord:
    timestamp_ns: int
    gyro: Tuple[float, float, float]
    accel: Tuple[float, float, float]


@dataclass(frozen=True)
class PoseRow:
    timestamp_s: float
    position: np.ndarray
    quat_xyzw: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--mkv-file", type=Path)
    parser.add_argument("--mcap-file", type=Path)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--camera-key", default="stereo_right")
    parser.add_argument("--imu-key")
    parser.add_argument("--mcap-camera-topic", default="c")
    parser.add_argument("--mcap-imu-topic", default="i")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--tracking-mode", choices=["stereo", "stereo-inertial"], default="stereo-inertial")
    parser.add_argument("--eval-pose-frame", choices=["cam0", "imu"], help="Pose frame stored in output CSV; default follows tracking-mode.")
    parser.add_argument("--orbslam-binary", type=Path)
    parser.add_argument("--vocabulary", type=Path, default=REPO_ROOT / "Vocabulary/ORBvoc.txt")
    parser.add_argument("--trajectory-name", default="orbslam3_stereo_right")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug limit; 0 means all frames.")
    parser.add_argument("--start-time-offset-sec", type=float, default=0.0, help="Skip camera frames before this offset from the first selected frame.")
    parser.add_argument("--cam-time-offset-sec", type=float, default=0.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-orbslam", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--robot-json", type=Path, default=DEFAULT_ROBOT_JSON)
    parser.add_argument("--hand-eye-json", type=Path)
    parser.add_argument("--t-max-diff", type=float, default=0.01)
    parser.add_argument("--run-evo", action="store_true")
    parser.add_argument("--imu-noise-gyro", type=float)
    parser.add_argument("--imu-noise-acc", type=float)
    parser.add_argument("--imu-gyro-walk", type=float)
    parser.add_argument("--imu-acc-walk", type=float)
    parser.add_argument("--imu-frequency", type=float)
    parser.add_argument("--imu-convert-discrete-noise", action="store_true", help="Treat calibration noise_density_discrete as per-sample noise and convert it to continuous density before writing ORB-SLAM3 YAML.")
    parser.add_argument("--imu-fast-init", type=int, choices=(0, 1), default=0, help="Set ORB-SLAM3 IMU.fastInit; enable only for low-excitation trajectories.")
    parser.add_argument("--orb-features", type=int, default=1200)
    parser.add_argument("--orb-scale-factor", type=float, default=1.2)
    parser.add_argument("--orb-levels", type=int, default=8)
    parser.add_argument("--orb-init-fast", type=int, default=20)
    parser.add_argument("--orb-min-fast", type=int, default=7)
    parser.add_argument("--offline-accurate", action="store_true", help="Use the offline stereo runner with local mapping waits and final GBA.")
    parser.add_argument("--gba-iterations", type=int, default=30)
    parser.add_argument("--offline-max-kf-frames", type=int, default=0)
    parser.add_argument("--offline-close-point-limit", type=int, default=0)
    parser.add_argument("--offline-disable-kf-culling", action="store_true")
    parser.add_argument("--offline-wait-timeout", type=float, default=10.0)
    parser.add_argument("--full-frame-ba-iterations", type=int, default=0, help="Run ORB-SLAM3 internal full-frame visual BA after GBA in offline accurate mode.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> PipelinePaths:
    episode_dir = args.episode_dir.expanduser()
    work_dir = (args.work_dir or episode_dir / "pose_data_orbslam3_right").expanduser()
    sequence_dir = work_dir / "euroc_sequence"
    return PipelinePaths(
        episode_dir=episode_dir,
        mkv_file=(args.mkv_file or episode_dir / "stereo_right.mkv").expanduser(),
        mcap_file=(args.mcap_file or episode_dir / "fays_data_right.mcap").expanduser(),
        calibration_json=(args.calibration_json or episode_dir / "calibration.json").expanduser(),
        work_dir=work_dir,
        sequence_dir=sequence_dir,
        left_dir=sequence_dir / "mav0/cam0/data",
        right_dir=sequence_dir / "mav0/cam1/data",
        timestamps_file=work_dir / "timestamps.txt",
        imu_csv=sequence_dir / "mav0/imu0/data.csv",
        settings_yaml=work_dir / "orbslam3_stereo_right.yaml",
        raw_trajectory=work_dir / f"f_{args.trajectory_name}.txt",
        keyframe_trajectory=work_dir / f"kf_{args.trajectory_name}.txt",
        output_csv=(args.output_csv or work_dir / "pose_data_right.csv").expanduser(),
        eval_dir=(args.eval_dir or work_dir / "evo_tcp_eval_all").expanduser(),
    )


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def assert_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_mcap_messages(mcap_file: Path) -> Iterable[Tuple[Any, Any, Any]]:
    try:
        from mcap.reader import make_reader
        from mcap.stream_reader import StreamReader
    except ImportError as exc:
        raise RuntimeError("Python package 'mcap' is required") from exc
    with mcap_file.open("rb") as handle:
        reader = make_reader(handle)
        yielded = 0
        for item in reader.iter_messages():
            yielded += 1
            yield item
        if yielded:
            return
    with mcap_file.open("rb") as handle:
        schemas: Dict[int, Any] = {}
        channels: Dict[int, Any] = {}
        for record in StreamReader(handle).records:
            name = type(record).__name__
            if name == "Schema":
                schemas[record.id] = record
            elif name == "Channel":
                channels[record.id] = record
            elif name == "Message":
                channel = channels.get(record.channel_id)
                if channel is not None:
                    yield schemas.get(channel.schema_id), channel, record


def require_publish_time_ns(message: Any, topic_name: str) -> int:
    if not hasattr(message, "publish_time"):
        raise RuntimeError(f"MCAP message on topic {topic_name!r} has no publish_time")
    return int(message.publish_time)


def decode_imu_payload(payload: bytes) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if len(payload) >= 48:
        gx, gy, gz, ax, ay, az = struct.unpack("<6d", payload[:48])
    elif len(payload) >= 24:
        gx, gy, gz, ax, ay, az = struct.unpack("<6f", payload[:24])
    else:
        raise ValueError(f"Unexpected IMU payload size: {len(payload)} bytes")
    return (gx, gy, gz), (ax, ay, az)


def load_mcap_data(mcap_file: Path, camera_topic: str, imu_topic: str, cam_offset_s: float) -> Tuple[Dict[int, int], List[ImuRecord]]:
    camera_timestamps: Dict[int, int] = {}
    imu_records: List[ImuRecord] = []
    cam_first_log = cam_base_pub = imu_first_log = imu_base_pub = None
    cam_offset_ns = int(round(cam_offset_s * 1_000_000_000))
    for _schema, channel, message in iter_mcap_messages(mcap_file):
        if channel.topic == camera_topic:
            if len(message.data) < 4:
                continue
            log_ns = int(message.log_time)
            pub_ns = require_publish_time_ns(message, channel.topic)
            if cam_first_log is None:
                cam_first_log, cam_base_pub = log_ns, pub_ns
            frame_index = struct.unpack("<I", message.data[:4])[0]
            camera_timestamps[frame_index] = int(cam_base_pub + (log_ns - cam_first_log) + cam_offset_ns)
        elif channel.topic == imu_topic:
            log_ns = int(message.log_time)
            pub_ns = require_publish_time_ns(message, channel.topic)
            if imu_first_log is None:
                imu_first_log, imu_base_pub = log_ns, pub_ns
            try:
                gyro, accel = decode_imu_payload(bytes(message.data))
            except ValueError as exc:
                print(f"[WARN] {exc}; skipping IMU sample", file=sys.stderr)
                continue
            imu_records.append(ImuRecord(int(imu_base_pub + (log_ns - imu_first_log)), gyro, accel))
    imu_records.sort(key=lambda item: item.timestamp_ns)
    if not camera_timestamps or not imu_records:
        raise RuntimeError("MCAP did not contain usable camera timestamps and IMU records")
    print(f"[MCAP] camera={len(camera_timestamps)}, imu={len(imu_records)}")
    return camera_timestamps, imu_records


def video_frame_count(mkv_file: Path) -> int:
    cap = cv2.VideoCapture(str(mkv_file))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mkv_file}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if count <= 0:
        raise RuntimeError(f"Cannot determine frame count: {mkv_file}")
    return count


def select_video_frames(mkv_file: Path, camera_timestamps: Dict[int, int], imu_records: Sequence[ImuRecord], max_frames: int, start_time_offset_s: float = 0.0) -> List[Tuple[int, int]]:
    total_frames = video_frame_count(mkv_file)
    imu_start, imu_end = imu_records[0].timestamp_ns, imu_records[-1].timestamp_ns
    selected = [(idx, ts) for idx, ts in sorted(camera_timestamps.items()) if 0 <= idx < total_frames and imu_start <= ts <= imu_end]
    if selected and start_time_offset_s > 0.0:
        cutoff_ns = selected[0][1] + int(round(start_time_offset_s * 1_000_000_000))
        selected = [(idx, ts) for idx, ts in selected if ts >= cutoff_ns]
    if max_frames > 0:
        selected = selected[:max_frames]
    if not selected:
        raise RuntimeError("No video frames overlap the IMU range")
    print(f"[Select] frames={len(selected)} / {total_frames}, time={selected[0][1]*1e-9:.6f}..{selected[-1][1]*1e-9:.6f}s")
    return selected


def to_gray(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def extract_stereo_images(mkv_file: Path, selected_frames: Sequence[Tuple[int, int]], left_dir: Path, right_dir: Path, overwrite: bool) -> None:
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    ts_by_idx = dict(selected_frames)
    pending = set(ts_by_idx)
    cap = cv2.VideoCapture(str(mkv_file))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mkv_file}")
    idx = saved = 0
    while pending:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in pending:
            ts = ts_by_idx[idx]
            half = frame.shape[0] // 2
            left_path = left_dir / f"{ts}.png"
            right_path = right_dir / f"{ts}.png"
            if overwrite or not left_path.exists():
                cv2.imwrite(str(left_path), to_gray(frame[:half, :]))
            if overwrite or not right_path.exists():
                cv2.imwrite(str(right_path), to_gray(frame[half:, :]))
            pending.remove(idx)
            saved += 1
            if saved % 100 == 0:
                print(f"[Images] saved {saved} stereo pairs")
        idx += 1
    cap.release()
    if pending:
        raise RuntimeError(f"Video ended before extracting {len(pending)} frames")
    print(f"[Images] saved/verified {saved} stereo pairs")


def write_timestamps_file(path: Path, selected_frames: Sequence[Tuple[int, int]], overwrite: bool) -> None:
    assert_can_write(path, overwrite)
    with path.open("w", encoding="utf-8") as handle:
        for _idx, ts in selected_frames:
            handle.write(f"{ts}\n")


def write_imu_csv(path: Path, imu_records: Sequence[ImuRecord], last_cam_ts: int, overwrite: bool) -> List[ImuRecord]:
    assert_can_write(path, overwrite)
    filtered = []
    kept_first_after_end = False
    for record in imu_records:
        if record.timestamp_ns <= last_cam_ts:
            filtered.append(record)
        elif not kept_first_after_end:
            filtered.append(record)
            kept_first_after_end = True
            break
    if not filtered:
        raise RuntimeError("No IMU samples before selected video end")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for record in filtered:
            handle.write("{},{:.12g},{:.12g},{:.12g},{:.12g},{:.12g},{:.12g}\n".format(record.timestamp_ns, *record.gyro, *record.accel))
    print(f"[IMU] wrote {len(filtered)} samples")
    return filtered


def matrix_from_json(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{label} must be a 4x4 matrix")
    return matrix


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = transform[:3, :3].T
    inv[:3, 3] = -transform[:3, :3].T.dot(transform[:3, 3])
    return inv


def infer_imu_key(camera_key: str) -> str:
    return "imu_right" if camera_key.endswith("_right") else "imu_left"


def load_stereo_calibration(calibration_json: Path, camera_key: str, imu_key: Optional[str]) -> Dict[str, Any]:
    calibration = json.loads(calibration_json.read_text(encoding="utf-8"))
    stereo = calibration.get("observation", {}).get("images", {}).get(camera_key)
    if not isinstance(stereo, dict):
        raise KeyError(f"observation.images.{camera_key} not found")
    resolved_imu_key = imu_key or infer_imu_key(camera_key)
    imu = calibration.get("observation", {}).get("imu", {}).get(resolved_imu_key, {})
    return {"stereo": stereo, "imu": imu, "imu_key": resolved_imu_key}


def select_intrinsics(camera: Dict[str, Any]) -> Tuple[int, int, Dict[str, float]]:
    intrinsics = camera.get("intrinsics")
    if not isinstance(intrinsics, dict) or not intrinsics:
        raise ValueError("camera intrinsics are missing")
    key = sorted(intrinsics.keys())[0]
    width, height = key.lower().split("x", 1)
    return int(width), int(height), intrinsics[key]


def yaml_matrix(name: str, matrix: np.ndarray) -> str:
    values = ", ".join(f"{float(v):.12g}" for v in matrix.reshape(-1))
    return f"{name}: !!opencv-matrix\n  rows: 4\n  cols: 4\n  dt: f\n  data: [{values}]\n"


def median_imu_frequency(imu_records: Sequence[ImuRecord]) -> float:
    if len(imu_records) < 2:
        return 200.0
    stamps = np.asarray([record.timestamp_ns for record in imu_records], dtype=np.float64) * 1e-9
    deltas = np.diff(stamps)
    deltas = deltas[deltas > 0.0]
    return float(round(1.0 / float(np.median(deltas)), 3)) if deltas.size else 200.0


def imu_noise(cli_value: Optional[float], imu: Dict[str, Any], sensor: str, key: str, fallback: float, sample_rate_hz: Optional[float] = None, convert_discrete: bool = False) -> float:
    if cli_value is not None:
        return float(cli_value)
    sensor_config = imu.get(sensor, {})
    if key not in sensor_config:
        return float(fallback)
    value = float(sensor_config[key])
    if convert_discrete and key.endswith("_discrete") and sample_rate_hz and sample_rate_hz > 0.0:
        return value / math.sqrt(sample_rate_hz)
    return value


def generate_settings_yaml(path: Path, calibration_json: Path, camera_key: str, imu_key: Optional[str], imu_records: Sequence[ImuRecord], args: argparse.Namespace, overwrite: bool) -> None:
    assert_can_write(path, overwrite)
    calib = load_stereo_calibration(calibration_json, camera_key, imu_key)
    stereo, imu = calib["stereo"], calib["imu"]
    cam0, cam1 = stereo["cam0"], stereo["cam1"]
    width, height, intr0 = select_intrinsics(cam0)
    width1, height1, intr1 = select_intrinsics(cam1)
    if (width, height) != (width1, height1):
        raise ValueError("cam0/cam1 resolutions differ")
    dist0 = [float(v) for v in cam0["distortion_coeffs"]]
    dist1 = [float(v) for v in cam1["distortion_coeffs"]]
    ext = stereo["extrinsics"]
    t_cam0_imu = matrix_from_json(ext["T_ic_cam0_to_imu0"], "T_ic_cam0_to_imu0")
    t_cam1_imu = matrix_from_json(ext["T_ic_cam1_to_imu0"], "T_ic_cam1_to_imu0")
    t_cam0_cam1 = t_cam0_imu.dot(invert_transform(t_cam1_imu))
    imu_freq = args.imu_frequency or median_imu_frequency(imu_records)
    system_lines = []
    if args.offline_max_kf_frames > 0:
        system_lines.append(f"System.MaxKeyFrameIntervalFrames: {args.offline_max_kf_frames}")
    if args.offline_close_point_limit > 0:
        system_lines.append(f"System.ClosePointCreationLimit: {args.offline_close_point_limit}")
    if args.offline_disable_kf_culling:
        system_lines.append("System.DisableKeyFrameCulling: 1")
    system_options = "\n".join(system_lines)
    if system_options:
        system_options += "\n"
    text = f"""%YAML:1.0
File.version: "1.0"
Camera.type: "KannalaBrandt8"

Camera1.fx: {float(intr0['fx']):.12g}
Camera1.fy: {float(intr0['fy']):.12g}
Camera1.cx: {float(intr0['ppx']):.12g}
Camera1.cy: {float(intr0['ppy']):.12g}
Camera1.k1: {dist0[0]:.12g}
Camera1.k2: {dist0[1]:.12g}
Camera1.k3: {dist0[2]:.12g}
Camera1.k4: {dist0[3]:.12g}

Camera2.fx: {float(intr1['fx']):.12g}
Camera2.fy: {float(intr1['fy']):.12g}
Camera2.cx: {float(intr1['ppx']):.12g}
Camera2.cy: {float(intr1['ppy']):.12g}
Camera2.k1: {dist1[0]:.12g}
Camera2.k2: {dist1[1]:.12g}
Camera2.k3: {dist1[2]:.12g}
Camera2.k4: {dist1[3]:.12g}
{yaml_matrix('Stereo.T_c1_c2', t_cam0_cam1)}
Camera1.overlappingBegin: 0
Camera1.overlappingEnd: {width - 1}
Camera2.overlappingBegin: 0
Camera2.overlappingEnd: {width - 1}
Camera.width: {width}
Camera.height: {height}
Camera.fps: {int(round(float(stereo.get('fps', 25.0))))}
Camera.RGB: 1
Stereo.ThDepth: 40.0
{yaml_matrix('IMU.T_b_c1', t_cam0_imu)}
IMU.NoiseGyro: {imu_noise(args.imu_noise_gyro, imu, 'gyroscope', 'noise_density_discrete', 0.00016, imu_freq, args.imu_convert_discrete_noise):.12g}
IMU.NoiseAcc: {imu_noise(args.imu_noise_acc, imu, 'accelerometer', 'noise_density_discrete', 0.0028, imu_freq, args.imu_convert_discrete_noise):.12g}
IMU.GyroWalk: {imu_noise(args.imu_gyro_walk, imu, 'gyroscope', 'random_walk', 0.000022):.12g}
IMU.AccWalk: {imu_noise(args.imu_acc_walk, imu, 'accelerometer', 'random_walk', 0.00086):.12g}
IMU.Frequency: {float(imu_freq):.12g}
IMU.fastInit: {int(args.imu_fast_init)}
IMU.InsertKFsWhenLost: 1
ORBextractor.nFeatures: {args.orb_features}
ORBextractor.scaleFactor: {args.orb_scale_factor:.12g}
ORBextractor.nLevels: {args.orb_levels}
ORBextractor.iniThFAST: {args.orb_init_fast}
ORBextractor.minThFAST: {args.orb_min_fast}
{system_options}Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
"""
    path.write_text(text, encoding="utf-8")
    print(f"[Settings] wrote {path}")


def prepare_dataset(paths: PipelinePaths, args: argparse.Namespace) -> None:
    for path, label in [(paths.mkv_file, "MKV"), (paths.mcap_file, "MCAP"), (paths.calibration_json, "calibration")]:
        require_file(path, label)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    camera_timestamps, imu_records = load_mcap_data(paths.mcap_file, args.mcap_camera_topic, args.mcap_imu_topic, args.cam_time_offset_sec)
    selected = select_video_frames(paths.mkv_file, camera_timestamps, imu_records, args.max_frames, args.start_time_offset_sec)
    extract_stereo_images(paths.mkv_file, selected, paths.left_dir, paths.right_dir, args.overwrite)
    write_timestamps_file(paths.timestamps_file, selected, args.overwrite)
    filtered_imu = write_imu_csv(paths.imu_csv, imu_records, selected[-1][1], args.overwrite)
    generate_settings_yaml(paths.settings_yaml, paths.calibration_json, args.camera_key, args.imu_key, filtered_imu, args, args.overwrite)


def run_orbslam(paths: PipelinePaths, args: argparse.Namespace) -> None:
    if args.offline_accurate and args.tracking_mode != "stereo":
        raise ValueError("--offline-accurate currently supports --tracking-mode stereo only")
    if args.orbslam_binary:
        binary = args.orbslam_binary
    elif args.offline_accurate:
        binary = REPO_ROOT / "Examples/Stereo/stereo_euroc_offline"
    else:
        binary = REPO_ROOT / "Examples/Stereo/stereo_euroc" if args.tracking_mode == "stereo" else REPO_ROOT / "Examples/Stereo-Inertial/stereo_inertial_euroc"
    require_file(binary, "ORB-SLAM3 binary")
    require_file(args.vocabulary, "ORB vocabulary")
    if paths.raw_trajectory.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {paths.raw_trajectory}; pass --overwrite")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(REPO_ROOT / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    cmd = [str(binary), str(args.vocabulary), str(paths.settings_yaml), str(paths.sequence_dir), str(paths.timestamps_file), args.trajectory_name]
    if args.offline_accurate:
        cmd.extend([str(args.gba_iterations), "0", str(args.offline_wait_timeout), str(args.full_frame_ba_iterations)])
    print("[ORB-SLAM3]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(paths.work_dir), env=env, check=True)
    require_file(paths.raw_trajectory, "raw ORB-SLAM3 trajectory")
    if args.offline_accurate:
        require_file(paths.keyframe_trajectory, "raw ORB-SLAM3 keyframe trajectory")


def normalize_quat_xyzw(values: Iterable[float]) -> np.ndarray:
    quat = np.asarray(list(values), dtype=np.float64)
    norm = np.linalg.norm(quat)
    if quat.shape != (4,) or norm <= 0.0:
        raise ValueError("invalid quaternion")
    quat = quat / norm
    return -quat if quat[3] < 0 else quat


def quat_to_rotation_matrix(quat_values: Iterable[float]) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat_values)
    return np.array(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
            [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
            [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        quat = [(rotation[2, 1] - rotation[1, 2]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale, 0.25 * scale]
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        quat = [0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale]
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        quat = [(rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale]
    else:
        scale = math.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
        quat = [(rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale, (rotation[1, 0] - rotation[0, 1]) / scale]
    return normalize_quat_xyzw(quat)


def transform_from_pose(position: Iterable[float], quat_xyzw: Iterable[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_to_rotation_matrix(quat_xyzw)
    transform[:3, 3] = np.asarray(list(position), dtype=np.float64)
    return transform


def pose_from_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return transform[:3, 3].copy(), rotation_matrix_to_quat_xyzw(transform[:3, :3])


def load_orb_trajectory(path: Path) -> List[PoseRow]:
    rows: List[PoseRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().replace(",", " ").split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) != 8:
                raise ValueError(f"bad ORB-SLAM3 trajectory line: {line.strip()}")
            rows.append(PoseRow(float(parts[0]) * 1e-9, np.asarray([float(parts[1]), float(parts[2]), float(parts[3])]), normalize_quat_xyzw([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])))
    if not rows:
        raise ValueError(f"{path} contains no poses")
    return sorted(rows, key=lambda row: row.timestamp_s)


def compute_velocities(rows: Sequence[PoseRow]) -> List[np.ndarray]:
    if len(rows) == 1:
        return [np.zeros(3)]
    velocities = []
    for i, row in enumerate(rows):
        prev_row = rows[max(0, i - 1)]
        next_row = rows[min(len(rows) - 1, i + 1)]
        dt = next_row.timestamp_s - prev_row.timestamp_s
        velocities.append(np.zeros(3) if dt <= 0 else (next_row.position - prev_row.position) / dt)
    return velocities


def convert_orb_trajectory_to_pose_csv(raw_path: Path, output_csv: Path, overwrite: bool) -> None:
    assert_can_write(output_csv, overwrite)
    rows = load_orb_trajectory(raw_path)
    velocities = compute_velocities(rows)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(POSE_CSV_COLUMNS)
        for row, velocity in zip(rows, velocities):
            writer.writerow([int(round(row.timestamp_s * 1_000_000)), f"{velocity[0]:.9f}", f"{velocity[1]:.9f}", f"{velocity[2]:.9f}", f"{row.position[0]:.9f}", f"{row.position[1]:.9f}", f"{row.position[2]:.9f}", f"{row.quat_xyzw[0]:.9f}", f"{row.quat_xyzw[1]:.9f}", f"{row.quat_xyzw[2]:.9f}", f"{row.quat_xyzw[3]:.9f}"])
    print(f"[CSV] wrote {len(rows)} poses: {output_csv}")


def read_vector(value: Any, names: Tuple[str, ...], label: str) -> List[float]:
    if isinstance(value, dict):
        return [float(value[name]) for name in names]
    if isinstance(value, (list, tuple)) and len(value) == len(names):
        return [float(v) for v in value]
    raise ValueError(f"{label} has unsupported format")


def load_robot_tcp_rows(path: Path) -> List[PoseRow]:
    samples = json.loads(path.read_text(encoding="utf-8")).get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path} has no samples")
    rows = []
    for sample in samples:
        timestamp_s = float(sample.get("timestamp_s", sample.get("timestamp")))
        position = np.asarray(read_vector(sample["position_m"], ("x", "y", "z"), "position_m"))
        if "quaternion_xyzw" in sample:
            quat = read_vector(sample["quaternion_xyzw"], ("x", "y", "z", "w"), "quaternion_xyzw")
        else:
            qw, qx, qy, qz = read_vector(sample["quaternion_wxyz"], ("w", "x", "y", "z"), "quaternion_wxyz")
            quat = [qx, qy, qz, qw]
        rows.append(PoseRow(timestamp_s, position, normalize_quat_xyzw(quat)))
    return sorted(rows, key=lambda row: row.timestamp_s)


def load_pose_csv_rows(path: Path) -> List[PoseRow]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(PoseRow(float(row["Timestamp_us"]) * 1e-6, np.asarray([float(row["X"]), float(row["Y"]), float(row["Z"])]), normalize_quat_xyzw([float(row["Quat_X"]), float(row["Quat_Y"]), float(row["Quat_Z"]), float(row["Quat_W"])])))
    if not rows:
        raise ValueError(f"{path} contains no pose rows")
    return sorted(rows, key=lambda row: row.timestamp_s)


def load_hand_eye(path: Optional[Path]) -> np.ndarray:
    if path is None:
        return DEFAULT_T_TCP_CAM.copy()
    return matrix_from_json(json.loads(path.read_text(encoding="utf-8"))["T_tcp_cam"], "T_tcp_cam")


def load_t_cam_imu(calibration_json: Path, camera_key: str) -> np.ndarray:
    stereo = load_stereo_calibration(calibration_json, camera_key, None)["stereo"]
    return matrix_from_json(stereo["extrinsics"]["T_ic_cam0_to_imu0"], "T_ic_cam0_to_imu0")


def vio_imu_rows_to_tcp_rows(vio_rows: Sequence[PoseRow], t_imu_tcp: np.ndarray) -> List[PoseRow]:
    result = []
    for row in vio_rows:
        position, quat = pose_from_transform(transform_from_pose(row.position, row.quat_xyzw).dot(t_imu_tcp))
        result.append(PoseRow(row.timestamp_s, position, quat))
    return result


def write_tum(path: Path, rows: Sequence[PoseRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write("{:.9f} {:.10f} {:.10f} {:.10f} {:.10f} {:.10f} {:.10f} {:.10f}\n".format(row.timestamp_s, row.position[0], row.position[1], row.position[2], row.quat_xyzw[0], row.quat_xyzw[1], row.quat_xyzw[2], row.quat_xyzw[3]))


def associate_rows(ref_rows: Sequence[PoseRow], est_rows: Sequence[PoseRow], max_diff_s: float) -> List[Tuple[PoseRow, PoseRow]]:
    est_times = np.asarray([row.timestamp_s for row in est_rows])
    pairs = []
    for ref in ref_rows:
        idx = int(np.searchsorted(est_times, ref.timestamp_s))
        choices = [i for i in (idx - 1, idx) if 0 <= i < len(est_rows)]
        if choices:
            best = min(choices, key=lambda i: abs(est_rows[i].timestamp_s - ref.timestamp_s))
            if abs(est_rows[best].timestamp_s - ref.timestamp_s) <= max_diff_s:
                pairs.append((ref, est_rows[best]))
    return pairs


def umeyama_align(source: np.ndarray, target: np.ndarray, with_scale: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    src_mean, tgt_mean = np.mean(source, axis=0), np.mean(target, axis=0)
    src_centered, tgt_centered = source - src_mean, target - tgt_mean
    cov = tgt_centered.T.dot(src_centered) / source.shape[0]
    u, s, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u.dot(vt)) < 0:
        sign[2, 2] = -1
    rotation = u.dot(sign).dot(vt)
    scale = float(np.sum(s * np.diag(sign)) / np.mean(np.sum(src_centered * src_centered, axis=1))) if with_scale else 1.0
    translation = tgt_mean - scale * rotation.dot(src_mean)
    return scale, rotation, translation


def error_stats(values: np.ndarray) -> Dict[str, float]:
    return {"rmse": float(math.sqrt(np.mean(values * values))), "mean": float(np.mean(values)), "median": float(np.median(values)), "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values)), "sse": float(np.sum(values * values))}


def evaluate_pairs(pairs: Sequence[Tuple[PoseRow, PoseRow]], with_scale: bool) -> Dict[str, Any]:
    if len(pairs) < 3:
        raise ValueError(f"Need at least 3 associated pairs, got {len(pairs)}")
    ref_pos = np.asarray([p[0].position for p in pairs])
    est_pos = np.asarray([p[1].position for p in pairs])
    scale, rot, trans = umeyama_align(est_pos, ref_pos, with_scale)
    trans_errors, angle_errors, rot_part_errors = [], [], []
    for ref, est in pairs:
        trans_errors.append(float(np.linalg.norm(ref.position - (scale * rot.dot(est.position) + trans))))
        r_ref = quat_to_rotation_matrix(ref.quat_xyzw)
        r_est = rot.dot(quat_to_rotation_matrix(est.quat_xyzw))
        delta = r_ref.T.dot(r_est)
        angle_errors.append(float(math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) * 0.5))))))
        rot_part_errors.append(float(np.linalg.norm(r_ref - r_est, ord="fro")))
    trans_stats = error_stats(np.asarray(trans_errors))
    return {"trans_part": trans_stats, "angle_deg": error_stats(np.asarray(angle_errors)), "rot_part": error_stats(np.asarray(rot_part_errors)), "pairs": len(pairs), "scale": scale, "start": pairs[0][0].timestamp_s, "end": pairs[-1][0].timestamp_s, "duration": pairs[-1][0].timestamp_s - pairs[0][0].timestamp_s, "pass_10mm_rmse": trans_stats["rmse"] <= 0.01}


def alignment_error_records(pairs: Sequence[Tuple[PoseRow, PoseRow]], with_scale: bool) -> List[Dict[str, float]]:
    ref_pos = np.asarray([p[0].position for p in pairs])
    est_pos = np.asarray([p[1].position for p in pairs])
    scale, rot, trans = umeyama_align(est_pos, ref_pos, with_scale)
    start_s = pairs[0][0].timestamp_s
    records = []
    for ref, est in pairs:
        error_m = float(np.linalg.norm(ref.position - (scale * rot.dot(est.position) + trans)))
        records.append({"timestamp_s": ref.timestamp_s, "relative_s": ref.timestamp_s - start_s, "error_m": error_m, "error_mm": error_m * 1000.0})
    return records


def trajectory_error_diagnostics(pairs: Sequence[Tuple[PoseRow, PoseRow]], with_scale: bool, bin_seconds: float = 5.0, top_n: int = 20) -> Dict[str, Any]:
    records = alignment_error_records(pairs, with_scale)
    top_errors = sorted(records, key=lambda item: item["error_m"], reverse=True)[:top_n]
    bins = []
    if records:
        duration = records[-1]["relative_s"]
        start = 0.0
        while start <= duration + 1e-9:
            end = start + bin_seconds
            values = np.asarray([item["error_m"] for item in records if start <= item["relative_s"] < end], dtype=np.float64)
            if values.size:
                stats = error_stats(values)
                bins.append({
                    "start_s": start,
                    "end_s": end,
                    "count": int(values.size),
                    "rmse_m": stats["rmse"],
                    "mean_m": stats["mean"],
                    "max_m": stats["max"],
                    "rmse_mm": stats["rmse"] * 1000.0,
                    "mean_mm": stats["mean"] * 1000.0,
                    "max_mm": stats["max"] * 1000.0,
                })
            start = end
    return {"top_errors": top_errors, "bins_5s": bins}


def load_orb_timestamps_s(path: Path) -> List[float]:
    if not path.is_file():
        return []
    timestamps = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().replace(",", " ").split()
            if parts and not parts[0].startswith("#"):
                timestamps.append(float(parts[0]) * 1e-9)
    return timestamps


def keyframe_diagnostics(raw_trajectory: Path, keyframe_trajectory: Path) -> Dict[str, Any]:
    frame_times = load_orb_timestamps_s(raw_trajectory)
    keyframe_times = load_orb_timestamps_s(keyframe_trajectory)
    if not keyframe_times:
        return {"count": 0, "path": str(keyframe_trajectory), "available": False}
    start_s = frame_times[0] if frame_times else keyframe_times[0]
    rel_times = [t - start_s for t in keyframe_times]
    gaps = [b - a for a, b in zip(rel_times, rel_times[1:])]
    return {
        "count": len(keyframe_times),
        "path": str(keyframe_trajectory),
        "available": True,
        "relative_times_s": rel_times,
        "gaps_s": gaps,
        "max_gap_s": max(gaps) if gaps else 0.0,
    }


def run_evo_commands(eval_dir: Path, ref_tum: Path, est_tum: Path, start_s: float, end_s: float, max_diff_s: float) -> None:
    evo_ape = shutil.which("evo_ape")
    if evo_ape is None:
        print("[Eval] evo_ape not found; skipped")
        return
    for mode, scale_flag in [("se3", False), ("sim3", True)]:
        for metric in ["trans_part", "angle_deg", "rot_part"]:
            cmd = [evo_ape, "tum", str(ref_tum), str(est_tum), "-a", "-r", metric, "--t_max_diff", str(max_diff_s), "--t_start", f"{start_s:.9f}", "--t_end", f"{end_s:.9f}", "--save_results", str(eval_dir / f"all_{metric}_{mode}.zip"), "--no_warnings", "-v"]
            if scale_flag:
                cmd.insert(5, "-s")
            print("[Eval/evo]", " ".join(cmd))
            subprocess.run(cmd, check=False)


def evaluate_trajectory(paths: PipelinePaths, args: argparse.Namespace) -> None:
    require_file(paths.output_csv, "estimated CSV")
    require_file(args.robot_json, "robot JSON")
    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    pose_frame = args.eval_pose_frame or ("cam0" if args.tracking_mode == "stereo" else "imu")
    t_tcp_cam = load_hand_eye(args.hand_eye_json)
    robot_rows = load_robot_tcp_rows(args.robot_json)
    pose_rows = load_pose_csv_rows(paths.output_csv)
    if pose_frame == "cam0":
        t_cam_tcp = invert_transform(t_tcp_cam)
        est_tcp_rows = []
        for row in pose_rows:
            position, quat = pose_from_transform(transform_from_pose(row.position, row.quat_xyzw).dot(t_cam_tcp))
            est_tcp_rows.append(PoseRow(row.timestamp_s, position, quat))
        formula = "T_world_tcp = T_world_cam0 * inv(T_tcp_cam)"
    else:
        t_tcp_imu = t_tcp_cam.dot(load_t_cam_imu(paths.calibration_json, args.camera_key))
        est_tcp_rows = vio_imu_rows_to_tcp_rows(pose_rows, invert_transform(t_tcp_imu))
        formula = "T_world_tcp = T_world_imu * inv(T_tcp_cam * T_cam_imu)"
    common_start = max(robot_rows[0].timestamp_s, est_tcp_rows[0].timestamp_s)
    common_end = min(robot_rows[-1].timestamp_s, est_tcp_rows[-1].timestamp_s)
    pairs = associate_rows([r for r in robot_rows if common_start <= r.timestamp_s <= common_end], [r for r in est_tcp_rows if common_start <= r.timestamp_s <= common_end], args.t_max_diff)
    ref_tum, est_tum = paths.eval_dir / "ref_tcp.tum", paths.eval_dir / "est_tcp.tum"
    write_tum(ref_tum, robot_rows)
    write_tum(est_tum, est_tcp_rows)
    run_config = {
        "tracking_mode": args.tracking_mode,
        "offline_accurate": bool(args.offline_accurate),
        "gba_iterations": args.gba_iterations,
        "orb_features": args.orb_features,
        "orb_scale_factor": args.orb_scale_factor,
        "orb_levels": args.orb_levels,
        "orb_init_fast": args.orb_init_fast,
        "orb_min_fast": args.orb_min_fast,
        "cam_time_offset_sec": args.cam_time_offset_sec,
        "offline_max_kf_frames": args.offline_max_kf_frames,
        "offline_close_point_limit": args.offline_close_point_limit,
        "offline_disable_kf_culling": bool(args.offline_disable_kf_culling),
        "full_frame_ba_iterations": args.full_frame_ba_iterations,
    }
    summary = {"inputs": {"estimated_csv": str(paths.output_csv), "robot_json": str(args.robot_json), "calibration_json": str(paths.calibration_json), "formula": formula, "pose_frame": pose_frame, "raw_trajectory": str(paths.raw_trajectory), "keyframe_trajectory": str(paths.keyframe_trajectory), "run_config": run_config}, "time_ranges": {"ref_start": robot_rows[0].timestamp_s, "ref_end": robot_rows[-1].timestamp_s, "est_start": est_tcp_rows[0].timestamp_s, "est_end": est_tcp_rows[-1].timestamp_s, "common_start": common_start, "common_end": common_end, "common_duration": max(0.0, common_end - common_start)}, "association": {"t_max_diff": args.t_max_diff, "pairs": len(pairs)}}
    if len(pairs) < 3:
        summary["error"] = f"Need at least 3 associated pairs, got {len(pairs)}"
        summary["results"] = {}
        (paths.eval_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[Eval] insufficient associated pairs: {len(pairs)}; wrote {paths.eval_dir / 'summary.json'}")
        return
    se3, sim3 = evaluate_pairs(pairs, False), evaluate_pairs(pairs, True)
    summary["results"] = {"se3": {"all": se3}, "sim3": {"all": sim3}}
    summary["diagnostics"] = {
        "se3": trajectory_error_diagnostics(pairs, False),
        "sim3": trajectory_error_diagnostics(pairs, True),
        "keyframes": keyframe_diagnostics(paths.raw_trajectory, paths.keyframe_trajectory),
    }
    (paths.eval_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Eval] SE3 RMSE {se3['trans_part']['rmse'] * 1000:.4f} mm, mean {se3['trans_part']['mean'] * 1000:.4f} mm, pairs {se3['pairs']}, pass<10mm={se3['pass_10mm_rmse']}")
    print(f"[Eval] Sim3 RMSE {sim3['trans_part']['rmse'] * 1000:.4f} mm, mean {sim3['trans_part']['mean'] * 1000:.4f} mm, scale {sim3['scale']:.6f}, pass<10mm={sim3['pass_10mm_rmse']}")
    if args.run_evo:
        run_evo_commands(paths.eval_dir, ref_tum, est_tum, common_start, common_end, args.t_max_diff)


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    if args.evaluate_only:
        evaluate_trajectory(paths, args)
        return 0
    if not args.skip_prepare:
        prepare_dataset(paths, args)
    if args.prepare_only:
        print(f"[Done] prepared dataset in {paths.work_dir}")
        return 0
    if not args.skip_orbslam:
        run_orbslam(paths, args)
    convert_orb_trajectory_to_pose_csv(paths.raw_trajectory, paths.output_csv, args.overwrite)
    if args.eval:
        evaluate_trajectory(paths, args)
    print(f"[Done] output CSV: {paths.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
