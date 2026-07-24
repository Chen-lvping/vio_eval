#!/usr/bin/env python3
"""Run ORB-SLAM3 directly on a ugripper single-MCAP episode without GT."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODE = REPO_ROOT / "data/0704/episode_20260704_0009"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/evaluation/workbench"
DEFAULT_ORB_ROOT = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_clean")
DEFAULT_PANGOLIN_LIB = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/Pangolin_v06/install/lib")
DEFAULT_CONFIG_GENERATOR = Path("/home/chenlvping/5_skill/lwm/vinsfusion_ws/scripts/gen_orbslam3_config_from_calib.py")

TOPIC_MAP = {
    "stereo_left": {
        "image": "/stereo_left/image_compressed",
        "imu": "/imu_left",
        "calibration_key": "stereo_left",
        "vins_side": "left",
    },
    "stereo_right": {
        "image": "/stereo_right/image_compressed",
        "imu": "/imu_right",
        "calibration_key": "stereo_right",
        "vins_side": "right",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--mcap-path", type=Path, default=None, help="Override the single source .mcap file")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--camera-rig", choices=tuple(TOPIC_MAP), default="stereo_right")
    parser.add_argument("--mode", choices=("stereo", "stereo-inertial"), default="stereo-inertial")
    parser.add_argument("--feature-preset", choices=("baseline", "low-texture", "aggressive"), default="low-texture")
    parser.add_argument("--orb-root", type=Path, default=DEFAULT_ORB_ROOT)
    parser.add_argument("--config-generator", type=Path, default=DEFAULT_CONFIG_GENERATOR)
    parser.add_argument("--vins-config", type=Path, default=None)
    parser.add_argument(
        "--vins-noise-only",
        action="store_true",
        help="Use only IMU noise from --vins-config; preserve MCAP calibration extrinsics.",
    )
    parser.add_argument("--match-vins-config", action="store_true")
    parser.add_argument("--vins-noise-mode", choices=("copy", "orb_from_vins"), default="orb_from_vins")
    parser.add_argument("--camera-time-shift-sec", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=360)
    parser.add_argument("--trajectory-name", default="")
    parser.add_argument("--extra-ld-path", action="append", default=[])
    parser.add_argument("--no-ldd-check", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8)
    parser.add_argument("--exposure-gain", type=float, default=1.0)
    parser.add_argument("--nfeatures", type=int, default=None)
    parser.add_argument("--scale-factor", type=float, default=None)
    parser.add_argument("--nlevels", type=int, default=None)
    parser.add_argument("--ini-fast", type=int, default=None)
    parser.add_argument("--min-fast", type=int, default=None)
    parser.add_argument("--th-depth", type=float, default=None)
    parser.add_argument("--gyro-noise", type=float, default=None)
    parser.add_argument("--acc-noise", type=float, default=None)
    parser.add_argument("--gyro-walk", type=float, default=None)
    parser.add_argument("--acc-walk", type=float, default=None)
    parser.add_argument("--imu-fast-init", type=int, choices=(0, 1), default=0)
    return parser.parse_args()


def default_output_dir(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    suffix = f"{args.episode_dir.name}_{args.camera_rig}_{args.mode}_{args.feature_preset}"
    if args.vins_config is not None or args.match_vins_config:
        suffix += "_vins_match"
    suffix += "_single_mcap"
    return args.output_root / f"orbslam3_ugripper_runs_{stamp}" / suffix


def resolve_mcap_path(args: argparse.Namespace) -> Path:
    if args.mcap_path is not None:
        return args.mcap_path.expanduser().resolve()
    candidates = sorted(args.episode_dir.glob("*.mcap"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one .mcap under {args.episode_dir}, found {len(candidates)}")
    return candidates[0].resolve()


def resolve_vins_config(args: argparse.Namespace) -> Path | None:
    if args.vins_config is not None:
        return args.vins_config.expanduser().resolve()
    if not args.match_vins_config:
        return None
    vins_side = TOPIC_MAP[args.camera_rig]["vins_side"]
    candidate = args.episode_dir / "pose_data_vins" / "vio_log" / vins_side / "generated_config" / "StereoIMU-vinsfusion.yaml"
    return candidate.resolve()


def read_existing_times(times_file: Path) -> list[int]:
    if not times_file.is_file():
        return []
    values: list[int] = []
    for line in times_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            values.append(int(line.split()[0]))
    return values


def conversion_complete(output_dir: Path, max_frames: int) -> bool:
    manifest_file = output_dir / "export_manifest.json"
    if manifest_file.is_file():
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if int(manifest.get("max_frames", 0)) != int(max_frames):
                return False
        except Exception:
            return False
    elif max_frames == 0:
        return False
    times = read_existing_times(output_dir / "times.txt")
    if not times:
        return False
    if max_frames > 0 and len(times) != max_frames:
        return False
    cam0_dir = output_dir / "mav0/cam0/data"
    cam1_dir = output_dir / "mav0/cam1/data"
    return all((cam0_dir / f"{ts}.png").is_file() and (cam1_dir / f"{ts}.png").is_file() for ts in times)


def reset_conversion_outputs(output_dir: Path) -> None:
    for rel in ("mav0/cam0/data", "mav0/cam1/data"):
        path = output_dir / rel
        if path.exists():
            shutil.rmtree(path)
    for rel in ("times.txt", "export_manifest.json", "mav0/cam0/data.csv", "mav0/cam1/data.csv", "mav0/imu0/data.csv"):
        path = output_dir / rel
        if path.exists():
            path.unlink()


def save_camera_csv(csv_path: Path, timestamps: list[int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["#timestamp [ns]", "filename"])
        for ts in timestamps:
            writer.writerow([ts, f"{ts}.png"])


def save_imu_csv(csv_path: Path, imu_records: list[tuple[int, float, float, float, float, float, float]], end_timestamp_ns: int | None) -> int:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["#timestamp [ns]", "gx", "gy", "gz", "ax", "ay", "az"])
        for row in imu_records:
            if end_timestamp_ns is not None and row[0] > end_timestamp_ns:
                break
            writer.writerow(row)
            count += 1
    return count


def make_clahe(args: argparse.Namespace):
    if not args.clahe:
        return None
    return cv2.createCLAHE(clipLimit=float(args.clahe_clip_limit), tileGridSize=(int(args.clahe_tile_grid_size), int(args.clahe_tile_grid_size)))


def apply_clahe(image: np.ndarray, clahe) -> np.ndarray:
    if clahe is None:
        return image
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_exposure_gain(image: np.ndarray, exposure_gain: float) -> np.ndarray:
    gain = float(exposure_gain)
    if abs(gain - 1.0) < 1e-6:
        return image
    if gain <= 0.0:
        raise ValueError("--exposure-gain must be positive")
    return cv2.convertScaleAbs(image, alpha=gain, beta=0.0)


def extract_calibration_json(mcap_path: Path, output_path: Path) -> dict:
    with mcap_path.open("rb") as handle:
        reader = make_reader(handle)
        for _schema, channel, message in reader.iter_messages():
            if channel.topic == "/calibration":
                text = bytes(message.data).decode("utf-8")
                output_path.write_text(text, encoding="utf-8")
                return json.loads(text)
    raise RuntimeError(f"/calibration topic not found in {mcap_path}")


def export_single_mcap(args: argparse.Namespace, mcap_path: Path, calibration_json_path: Path) -> dict[str, int | float | str]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.force_export:
        reset_conversion_outputs(output_dir)
    elif conversion_complete(output_dir, args.max_frames):
        times = read_existing_times(output_dir / "times.txt")
        return {
            "status": "reused",
            "frames": len(times),
            "first_timestamp_ns": times[0],
            "last_timestamp_ns": times[-1],
        }

    topic_cfg = TOPIC_MAP[args.camera_rig]
    image_topic = topic_cfg["image"]
    imu_topic = topic_cfg["imu"]
    cam0_dir = output_dir / "mav0/cam0/data"
    cam1_dir = output_dir / "mav0/cam1/data"
    cam0_dir.mkdir(parents=True, exist_ok=True)
    cam1_dir.mkdir(parents=True, exist_ok=True)
    clahe = make_clahe(args)
    codec = av.CodecContext.create("hevc", "r")
    compressed_cls = get_message("sensor_msgs/msg/CompressedImage")
    imu_cls = get_message("sensor_msgs/msg/Imu")
    saved_timestamps: list[int] = []
    imu_records: list[tuple[int, float, float, float, float, float, float]] = []
    decoded_shape: tuple[int, int, int] | None = None

    with mcap_path.open("rb") as handle:
        reader = make_reader(handle)
        for _schema, channel, message in reader.iter_messages():
            if channel.topic == imu_topic:
                imu_msg = deserialize_message(message.data, imu_cls)
                imu_records.append(
                    (
                        int(message.log_time),
                        float(imu_msg.angular_velocity.x),
                        float(imu_msg.angular_velocity.y),
                        float(imu_msg.angular_velocity.z),
                        float(imu_msg.linear_acceleration.x),
                        float(imu_msg.linear_acceleration.y),
                        float(imu_msg.linear_acceleration.z),
                    )
                )
                continue
            if channel.topic != image_topic:
                continue
            image_msg = deserialize_message(message.data, compressed_cls)
            timestamp_ns = int(message.log_time) + int(round(float(args.camera_time_shift_sec) * 1e9))
            frames = codec.decode(av.Packet(bytes(image_msg.data)))
            if not frames:
                continue
            frame = frames[0].to_ndarray(format="bgr24")
            decoded_shape = tuple(int(v) for v in frame.shape)
            half_height = frame.shape[0] // 2
            cam0 = frame[:half_height, :]
            cam1 = frame[half_height:, :]
            if not args.color:
                cam0 = cv2.cvtColor(cam0, cv2.COLOR_BGR2GRAY)
                cam1 = cv2.cvtColor(cam1, cv2.COLOR_BGR2GRAY)
            cam0 = apply_exposure_gain(cam0, args.exposure_gain)
            cam1 = apply_exposure_gain(cam1, args.exposure_gain)
            cam0 = apply_clahe(cam0, clahe)
            cam1 = apply_clahe(cam1, clahe)
            name = f"{timestamp_ns}.png"
            if not cv2.imwrite(str(cam0_dir / name), cam0):
                raise RuntimeError(f"failed to write {cam0_dir / name}")
            if not cv2.imwrite(str(cam1_dir / name), cam1):
                raise RuntimeError(f"failed to write {cam1_dir / name}")
            saved_timestamps.append(timestamp_ns)
            if args.max_frames > 0 and len(saved_timestamps) >= args.max_frames:
                break

    if not saved_timestamps:
        raise RuntimeError(f"no frames exported from {image_topic} in {mcap_path}")
    imu_records.sort(key=lambda row: row[0])
    (output_dir / "times.txt").write_text("".join(f"{ts}\n" for ts in saved_timestamps), encoding="utf-8")
    save_camera_csv(output_dir / "mav0/cam0/data.csv", saved_timestamps)
    save_camera_csv(output_dir / "mav0/cam1/data.csv", saved_timestamps)
    imu_count = save_imu_csv(output_dir / "mav0/imu0/data.csv", imu_records, saved_timestamps[-1] if args.mode == "stereo-inertial" else None)
    manifest = {
        "episode_dir": str(args.episode_dir),
        "mcap_path": str(mcap_path),
        "camera_rig": args.camera_rig,
        "mode": args.mode,
        "image_topic": image_topic,
        "imu_topic": imu_topic,
        "calibration_json": str(calibration_json_path),
        "max_frames": int(args.max_frames),
        "frames": len(saved_timestamps),
        "imu_samples": imu_count,
        "first_timestamp_ns": saved_timestamps[0],
        "last_timestamp_ns": saved_timestamps[-1],
        "camera_time_shift_sec": float(args.camera_time_shift_sec),
        "decoded_shape": list(decoded_shape) if decoded_shape is not None else [],
        "color": bool(args.color),
        "clahe": bool(args.clahe),
        "exposure_gain": float(args.exposure_gain),
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "exported",
        "frames": len(saved_timestamps),
        "imu_samples": imu_count,
        "first_timestamp_ns": saved_timestamps[0],
        "last_timestamp_ns": saved_timestamps[-1],
        "decoded_shape": decoded_shape or (),
    }


def generate_orb_settings(args: argparse.Namespace, shim_episode_dir: Path, vins_config: Path | None) -> Path:
    settings = args.output_dir / f"orbslam3_{args.camera_rig}_{args.mode}.yaml"
    cmd = [
        sys.executable,
        str(args.config_generator),
        str(shim_episode_dir),
        str(settings),
        "--camera-rig",
        args.camera_rig,
        "--mode",
        args.mode,
        "--feature-preset",
        args.feature_preset,
    ]
    if vins_config is not None:
        cmd.extend(["--vins-config", str(vins_config), "--vins-noise-mode", args.vins_noise_mode])
        if args.vins_noise_only:
            cmd.append("--vins-noise-only")
    for attr, option in (
        ("nfeatures", "--nfeatures"),
        ("scale_factor", "--scale-factor"),
        ("nlevels", "--nlevels"),
        ("ini_fast", "--ini-fast"),
        ("min_fast", "--min-fast"),
        ("th_depth", "--th-depth"),
        ("gyro_noise", "--gyro-noise"),
        ("acc_noise", "--acc-noise"),
        ("gyro_walk", "--gyro-walk"),
        ("acc_walk", "--acc-walk"),
        ("imu_fast_init", "--imu-fast-init"),
    ):
        value = getattr(args, attr)
        if value is not None:
            cmd.extend([option, str(value)])
    subprocess.run(cmd, check=True)
    return settings


def common_ld_paths(orb_root: Path, extra_paths: list[str]) -> list[str]:
    candidates = [
        DEFAULT_PANGOLIN_LIB,
        orb_root / "lib",
        orb_root / "Thirdparty/DBoW2/lib",
        orb_root / "Thirdparty/g2o/lib",
    ]
    result: list[str] = []
    for raw in extra_paths:
        for part in raw.split(":"):
            if part:
                result.append(part)
    for path in candidates:
        if Path(path).is_dir():
            result.append(str(path))
    seen: set[str] = set()
    unique: list[str] = []
    for item in result:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def orb_executable(orb_root: Path, mode: str) -> Path:
    if mode == "stereo":
        return orb_root / "Examples/Stereo/stereo_euroc"
    return orb_root / "Examples/Stereo-Inertial/stereo_inertial_euroc"


def ldd_missing(executable: Path, env: dict[str, str], log_path: Path) -> list[str]:
    proc = subprocess.run(["ldd", str(executable)], env=env, text=True, capture_output=True, check=False)
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return [line.strip() for line in proc.stdout.splitlines() if "not found" in line]


def trajectory_has_rows(output_dir: Path, trajectory_name: str) -> bool:
    candidates = [
        output_dir / f"f_{trajectory_name}.txt",
        output_dir / f"kf_{trajectory_name}.txt",
        output_dir / "CameraTrajectory.txt",
        output_dir / "KeyFrameTrajectory.txt",
    ]
    for path in candidates:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip() and not line.startswith("#"):
                return True
    return False


def run_orbslam(args: argparse.Namespace, settings_file: Path) -> int:
    orb_root = args.orb_root.resolve()
    executable = orb_executable(orb_root, args.mode)
    vocabulary = orb_root / "Vocabulary/ORBvoc.txt"
    for path in (executable, vocabulary, settings_file, args.output_dir / "times.txt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    trajectory_name = args.trajectory_name or f"{args.episode_dir.name}_{args.camera_rig}_{args.mode}"
    cmd = [
        str(executable),
        str(vocabulary),
        str(settings_file),
        str(args.output_dir),
        str((args.output_dir / "times.txt").resolve()),
        trajectory_name,
    ]
    env = os.environ.copy()
    env["ORB_SLAM3_ENABLE_VIEWER"] = "1" if args.viewer else "0"
    ld_paths = common_ld_paths(orb_root, list(args.extra_ld_path))
    if env.get("LD_LIBRARY_PATH"):
        ld_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)
    ldd_log = args.output_dir / "ldd.log"
    missing = ldd_missing(executable, env, ldd_log)
    if missing and not args.no_ldd_check:
        print("[RUN] Missing runtime libraries; see", ldd_log)
        for line in missing:
            print("  " + line)
        return 30
    run_log = args.output_dir / "orbslam3_native.log"
    with run_log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.write("ORB_SLAM3_ENABLE_VIEWER=" + env["ORB_SLAM3_ENABLE_VIEWER"] + "\n")
        handle.write("LD_LIBRARY_PATH=" + env.get("LD_LIBRARY_PATH", "") + "\n\n")
        handle.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(args.output_dir),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_sec if args.timeout_sec > 0 else None,
                check=False,
            )
        except subprocess.TimeoutExpired:
            handle.write(f"\n[TIMEOUT] ORB-SLAM3 exceeded {args.timeout_sec} seconds\n")
            return 0 if trajectory_has_rows(args.output_dir, trajectory_name) else 124
    if proc.returncode != 0 and not trajectory_has_rows(args.output_dir, trajectory_name):
        return int(proc.returncode)
    return 0 if trajectory_has_rows(args.output_dir, trajectory_name) else 40


def main() -> int:
    args = parse_args()
    args.episode_dir = args.episode_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else default_output_dir(args).resolve())
    args.orb_root = args.orb_root.expanduser().resolve()
    args.config_generator = args.config_generator.expanduser().resolve()
    mcap_path = resolve_mcap_path(args)
    vins_config = resolve_vins_config(args)
    if vins_config is not None and not vins_config.is_file():
        raise FileNotFoundError(vins_config)
    if args.vins_noise_only and vins_config is None:
        raise ValueError("--vins-noise-only requires --vins-config")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shim_episode_dir = args.output_dir / "episode_shim"
    shim_episode_dir.mkdir(parents=True, exist_ok=True)
    calibration_json_path = shim_episode_dir / "calibration.json"
    _calib = extract_calibration_json(mcap_path, calibration_json_path)
    export_stats = export_single_mcap(args, mcap_path, calibration_json_path)
    print(f"[EXPORT] {export_stats}")
    settings_file = generate_orb_settings(args, shim_episode_dir, vins_config)
    print(f"[CONFIG] {settings_file}")
    if args.prepare_only:
        print(f"[DONE] prepared ORB inputs under {args.output_dir}")
        return 0
    rc = run_orbslam(args, settings_file)
    if rc == 0:
        print(f"[DONE] ORB-SLAM3 trajectory written under {args.output_dir}")
    else:
        print(f"[ERROR] ORB-SLAM3 failed with code {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
