#!/usr/bin/env python3
"""Capture a short Hanpu/YCTC stereo+IMU clip and run ORB-SLAM3 stereo on it."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import struct
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/hanpu/runs"
DEFAULT_ORB_ROOT = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_clean")

MAGIC = b"SY"
PROTOCOL_VERSION = 3
BAUD = 921600
MSG_START = 0x01
MSG_STOP = 0x02
MSG_IMU_DATA = 0x04
MSG_FW_VERSION = 0x08


@dataclass
class BridgeFrame:
    msg_type: int
    flags: int
    seq: int
    payload: bytes


@dataclass
class JpegFrame:
    frame_index: int
    jpeg_bytes: bytes
    left_pts_us: int
    right_pts_us: int
    left_exp_us: int
    right_exp_us: int


def crc16_ibm(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 1) else (crc >> 1)
    return crc & 0xFFFF


def build_frame(msg_type: int, seq: int, payload: bytes = b"", flags: int = 0) -> bytes:
    header = (
        MAGIC
        + bytes([PROTOCOL_VERSION, msg_type, flags])
        + struct.pack("<H", seq)
        + struct.pack("<H", len(payload))
        + b"\x00"
    )
    return header + payload + struct.pack("<H", crc16_ibm(header + payload))


def read_exact(port: serial.Serial, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = port.read(size - len(data))
        if not chunk:
            break
        data += chunk
    return data


def read_bridge_frame(port: serial.Serial, timeout_sec: float) -> BridgeFrame | None:
    deadline = time.time() + timeout_sec
    sync = bytearray()
    while time.time() < deadline:
        byte = port.read(1)
        if not byte:
            continue
        sync += byte
        if len(sync) > 2:
            sync = sync[-2:]
        if bytes(sync) != MAGIC:
            continue
        header_tail = read_exact(port, 8)
        if len(header_tail) != 8:
            continue
        header = MAGIC + header_tail
        if header[2] != PROTOCOL_VERSION:
            continue
        payload_len = struct.unpack_from("<H", header, 7)[0]
        payload = read_exact(port, payload_len)
        crc = read_exact(port, 2)
        if len(payload) != payload_len or len(crc) != 2:
            continue
        if struct.unpack("<H", crc)[0] != crc16_ibm(header + payload):
            continue
        return BridgeFrame(
            msg_type=header[3],
            flags=header[4],
            seq=struct.unpack_from("<H", header, 5)[0],
            payload=payload,
        )
    return None


def common_ld_paths(orb_root: Path) -> list[str]:
    candidates = [
        orb_root / "lib",
        orb_root / "Thirdparty/DBoW2/lib",
        orb_root / "Thirdparty/g2o/lib",
        Path("/home/chenlvping/1_DM_work/orbslam3_fresh/Pangolin_v06/install/lib"),
        Path("/home/chenlvping/orb3_build/pangolin_v0.6_native/install/lib"),
    ]
    values: list[str] = []
    for path in candidates:
        if path.is_dir():
            values.append(str(path))
    current = os.environ.get("LD_LIBRARY_PATH")
    if current:
        values.append(current)
    seen = set()
    ordered: list[str] = []
    for item in values:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def parse_fw_payload(payload: bytes) -> dict[str, str]:
    code_len, name_len, build_len = struct.unpack_from("<HHH", payload, 0)
    offset = 6
    code = payload[offset : offset + code_len].decode("ascii", errors="replace")
    offset += code_len
    name = payload[offset : offset + name_len].decode("ascii", errors="replace")
    offset += name_len
    build = payload[offset : offset + build_len].decode("ascii", errors="replace")
    return {"code": code, "name": name, "build": build}


def capture_imu_samples(port_name: str, duration_sec: float, raw_csv_path: Path) -> dict[str, object]:
    rows: list[tuple[int, float, float, float, float, float, float]] = []
    fw_info: dict[str, str] = {}
    with serial.Serial(port_name, BAUD, timeout=0.1) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()

        port.write(build_frame(MSG_FW_VERSION, 1))
        fw = read_bridge_frame(port, 2.0)
        if fw is None or fw.msg_type != MSG_FW_VERSION:
            raise RuntimeError("failed to read FW_VERSION")
        fw_info = parse_fw_payload(fw.payload)

        port.write(build_frame(MSG_START, 2, b"\x01\x00\x00\x00"))
        ack = read_bridge_frame(port, 2.0)
        if ack is None or ack.msg_type != MSG_START or ack.flags != 0x01:
            raise RuntimeError("failed to read START ack for IMU")

        deadline = time.time() + duration_sec
        while time.time() < deadline:
            frame = read_bridge_frame(port, 0.3)
            if frame is None or frame.msg_type != MSG_IMU_DATA:
                continue
            _, _, _, gyro_count, acc_count = struct.unpack_from("<IQQHH", frame.payload, 0)
            pos = 24
            gyros: list[tuple[int, float, float, float]] = []
            for _ in range(gyro_count):
                x, y, z, _temp, pts_us = struct.unpack_from("<hhhIQ", frame.payload, pos)
                pos += 18
                gyros.append((pts_us, x * (1000.0 / 32768.0) * np.pi / 180.0, y * (1000.0 / 32768.0) * np.pi / 180.0, z * (1000.0 / 32768.0) * np.pi / 180.0))
            accs: list[tuple[int, float, float, float]] = []
            for _ in range(acc_count):
                x, y, z, _temp, pts_us = struct.unpack_from("<hhhIQ", frame.payload, pos)
                pos += 18
                scale = 4.0 / 32768.0 * 9.80665
                accs.append((pts_us, x * scale, y * scale, z * scale))

            by_ts: dict[int, list[float]] = {}
            for ts, gx, gy, gz in gyros:
                by_ts.setdefault(ts, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                by_ts[ts][0:3] = [gx, gy, gz]
            for ts, ax, ay, az in accs:
                by_ts.setdefault(ts, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                by_ts[ts][3:6] = [ax, ay, az]
            for ts in sorted(by_ts):
                values = by_ts[ts]
                if any(np.isnan(v) for v in values):
                    continue
                rows.append((int(ts * 1000), float(values[0]), float(values[1]), float(values[2]), float(values[3]), float(values[4]), float(values[5])))

        port.write(build_frame(MSG_STOP, 3))
        stop_deadline = time.time() + 2.0
        while time.time() < stop_deadline:
            frame = read_bridge_frame(port, 0.3)
            if frame is not None and frame.msg_type == MSG_STOP and frame.flags == 0x01:
                break

    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_ns", "gx_rad_s", "gy_rad_s", "gz_rad_s", "ax_m_s2", "ay_m_s2", "az_m_s2"])
        writer.writerows(rows)

    deltas = [curr[0] - prev[0] for prev, curr in zip(rows, rows[1:]) if curr[0] > prev[0]]
    return {
        "fw_info": fw_info,
        "imu_rows": len(rows),
        "imu_rate_hz": (1e9 / statistics.mean(deltas)) if deltas else 0.0,
        "first_timestamp_ns": rows[0][0] if rows else None,
        "last_timestamp_ns": rows[-1][0] if rows else None,
    }


def capture_mjpeg(video_device: str, duration_sec: float, video_size: str, raw_stream_path: Path) -> None:
    raw_stream_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "v4l2",
        "-input_format",
        "mjpeg",
        "-video_size",
        video_size,
        "-framerate",
        "30",
        "-i",
        video_device,
        "-t",
        str(duration_sec),
        "-c",
        "copy",
        str(raw_stream_path),
    ]
    subprocess.run(cmd, check=True)


def find_jpeg_eoi(data: bytes, start: int) -> int:
    pos = start
    length = len(data)
    while pos + 1 < length:
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker == 0x00 or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        if marker == 0xD9:
            return pos
        pos += 1
    return -1


def parse_yctc_block(payload: bytes) -> tuple[int, int, int, int] | None:
    magic = b"YCTC"
    pos = payload.find(magic)
    while pos != -1:
        if pos + 32 <= len(payload):
            version = struct.unpack_from("<H", payload, pos + 4)[0]
            block_size = struct.unpack_from("<H", payload, pos + 6)[0]
            if version == 1 and block_size == 32:
                left_pts = struct.unpack_from("<Q", payload, pos + 8)[0]
                right_pts = struct.unpack_from("<Q", payload, pos + 16)[0]
                left_exp = struct.unpack_from("<I", payload, pos + 24)[0]
                right_exp = struct.unpack_from("<I", payload, pos + 28)[0]
                return left_pts, right_pts, left_exp, right_exp
        pos = payload.find(magic, pos + 1)
    return None


def parse_mjpeg_stream(raw_stream_path: Path) -> list[JpegFrame]:
    data = raw_stream_path.read_bytes()
    frames: list[JpegFrame] = []
    pos = 0
    frame_index = 0
    length = len(data)
    while pos + 1 < length:
        if data[pos] != 0xFF or data[pos + 1] != 0xD8:
            pos += 1
            continue
        frame_start = pos
        pos += 2
        yctc_info: tuple[int, int, int, int] | None = None
        while pos + 1 < length:
            if data[pos] != 0xFF:
                pos += 1
                continue
            marker_pos = pos
            while pos < length and data[pos] == 0xFF:
                pos += 1
            if pos >= length:
                break
            marker = data[pos]
            if marker == 0x00:
                pos += 1
                continue
            if marker == 0xD9:
                frame_end = pos + 1
                if yctc_info is not None:
                    frames.append(
                        JpegFrame(
                            frame_index=frame_index,
                            jpeg_bytes=data[frame_start : frame_end + 1],
                            left_pts_us=yctc_info[0],
                            right_pts_us=yctc_info[1],
                            left_exp_us=yctc_info[2],
                            right_exp_us=yctc_info[3],
                        )
                    )
                frame_index += 1
                pos = frame_end + 1
                break
            if (0xD0 <= marker <= 0xD7) or marker == 0x01:
                pos += 1
                continue
            if pos + 2 >= length:
                return frames
            seg_len = (data[pos + 1] << 8) | data[pos + 2]
            if seg_len < 2 or pos + 1 + seg_len > length:
                return frames
            if marker == 0xEF:
                payload = data[pos + 3 : pos + 1 + seg_len]
                yctc_info = parse_yctc_block(payload)
            if marker == 0xDA:
                eoi = find_jpeg_eoi(data, pos + 1 + seg_len)
                if eoi == -1:
                    return frames
                if yctc_info is not None:
                    frames.append(
                        JpegFrame(
                            frame_index=frame_index,
                            jpeg_bytes=data[frame_start : eoi + 2],
                            left_pts_us=yctc_info[0],
                            right_pts_us=yctc_info[1],
                            left_exp_us=yctc_info[2],
                            right_exp_us=yctc_info[3],
                        )
                    )
                frame_index += 1
                pos = eoi + 2
                break
            pos += 1 + seg_len
        else:
            break
    return frames


def export_euroc(frames: list[JpegFrame], imu_csv_path: Path, export_dir: Path, crop_top_only: bool) -> dict[str, object]:
    if not frames:
        raise RuntimeError("no MJPEG frames with YCTC user data were found")
    if export_dir.exists():
        shutil.rmtree(export_dir)
    cam0_dir = export_dir / "mav0/cam0/data"
    cam1_dir = export_dir / "mav0/cam1/data"
    imu0_path = export_dir / "mav0/imu0/data.csv"
    cam0_dir.mkdir(parents=True, exist_ok=True)
    cam1_dir.mkdir(parents=True, exist_ok=True)
    imu0_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps_ns: list[int] = []
    width = None
    height = None
    for item in frames:
        arr = np.frombuffer(item.jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        full_h, full_w = frame.shape[:2]
        if crop_top_only:
            frame = frame[: full_h // 2, :]
        left = frame[:, : frame.shape[1] // 2]
        right = frame[:, frame.shape[1] // 2 :]
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        timestamp_ns = int(item.left_pts_us) * 1000
        name = f"{timestamp_ns}.png"
        cv2.imwrite(str(cam0_dir / name), left)
        cv2.imwrite(str(cam1_dir / name), right)
        timestamps_ns.append(timestamp_ns)
        width = int(left.shape[1])
        height = int(left.shape[0])

    if not timestamps_ns or width is None or height is None:
        raise RuntimeError("failed to decode any MJPEG frame")

    (export_dir / "times.txt").write_text("".join(f"{ts}\n" for ts in timestamps_ns), encoding="utf-8")
    for cam_csv in (export_dir / "mav0/cam0/data.csv", export_dir / "mav0/cam1/data.csv"):
        with cam_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["#timestamp [ns]", "filename"])
            for ts in timestamps_ns:
                writer.writerow([ts, f"{ts}.png"])
    shutil.copyfile(imu_csv_path, imu0_path)

    return {
        "frames": len(timestamps_ns),
        "first_timestamp_ns": timestamps_ns[0],
        "last_timestamp_ns": timestamps_ns[-1],
        "camera_width": width,
        "camera_height": height,
    }


def write_placeholder_orb_settings(settings_path: Path, width: int, height: int, fps: float, baseline_m: float) -> None:
    fx = width * 0.52
    fy = height * 0.92
    cx = width / 2.0
    cy = height / 2.0
    settings_path.write_text(
        f"""%YAML:1.0

File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {fx:.6f}
Camera1.fy: {fy:.6f}
Camera1.cx: {cx:.6f}
Camera1.cy: {cy:.6f}
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera2.fx: {fx:.6f}
Camera2.fy: {fy:.6f}
Camera2.cx: {cx:.6f}
Camera2.cy: {cy:.6f}
Camera2.k1: 0.0
Camera2.k2: 0.0
Camera2.p1: 0.0
Camera2.p2: 0.0
Camera.width: {width}
Camera.height: {height}
Camera.fps: {int(round(fps))}
Camera.RGB: 1
Stereo.ThDepth: 35.0
Stereo.T_c1_c2: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [1.0, 0.0, 0.0, {baseline_m:.6f},
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]
ORBextractor.nFeatures: 4200
ORBextractor.scaleFactor: 1.15
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 7
ORBextractor.minThFAST: 2
Viewer.KeyFrameSize: 0.05
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
""",
        encoding="utf-8",
    )


def trajectory_has_rows(run_dir: Path) -> bool:
    for name in ("CameraTrajectory.txt", "KeyFrameTrajectory.txt", "f_hanpu_smoketest.txt", "kf_hanpu_smoketest.txt"):
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return True
    return False


def run_orbslam3_stereo(orb_root: Path, export_dir: Path, settings_path: Path, run_dir: Path, timeout_sec: int, viewer: bool) -> int:
    executable = orb_root / "Examples/Stereo/stereo_euroc"
    vocabulary = orb_root / "Vocabulary/ORBvoc.txt"
    for path in (executable, vocabulary, settings_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(common_ld_paths(orb_root))
    env["ORB_SLAM3_ENABLE_VIEWER"] = "1" if viewer else "0"
    cmd = [
        str(executable),
        str(vocabulary),
        str(settings_path),
        str(export_dir),
        str(export_dir / "times.txt"),
        "hanpu_smoketest",
    ]
    log_path = run_dir / "orbslam3_native.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.write("LD_LIBRARY_PATH=" + env["LD_LIBRARY_PATH"] + "\n")
        handle.write("ORB_SLAM3_ENABLE_VIEWER=" + env["ORB_SLAM3_ENABLE_VIEWER"] + "\n\n")
        handle.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=run_dir,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 0 if trajectory_has_rows(run_dir) else 124
    if proc.returncode != 0 and trajectory_has_rows(run_dir):
        return 0
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-device", default="/dev/video0")
    parser.add_argument("--imu-port", default="/dev/ttyACM0")
    parser.add_argument("--video-size", default="3840x1080")
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--orb-root", type=Path, default=DEFAULT_ORB_ROOT)
    parser.add_argument("--baseline-m", type=float, default=0.06)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-orb", action="store_true")
    parser.add_argument("--no-top-crop", action="store_true", help="Use the full decoded frame instead of the top half only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root.expanduser().resolve() / f"hanpu_orb_smoke_{stamp}"
    raw_dir = run_root / "raw"
    export_dir = run_root / "euroc_export"
    orb_dir = run_root / "orbslam3_run"
    raw_video = raw_dir / "stereo_raw.mjpg"
    raw_imu = raw_dir / "imu_raw.csv"
    settings_path = run_root / "orbslam3_stereo_placeholder.yaml"

    run_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    capture_started = time.time()
    video_proc = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "v4l2",
            "-input_format",
            "mjpeg",
            "-video_size",
            args.video_size,
            "-framerate",
            "30",
            "-i",
            args.video_device,
            "-t",
            str(args.duration_sec),
            "-c",
            "copy",
            str(raw_video),
        ]
    )
    try:
        imu_summary = capture_imu_samples(args.imu_port, args.duration_sec + 0.4, raw_imu)
    finally:
        video_proc.wait(timeout=max(int(args.duration_sec) + 10, 15))
    capture_finished = time.time()

    frames = parse_mjpeg_stream(raw_video)
    export_summary = export_euroc(frames, raw_imu, export_dir, crop_top_only=not args.no_top_crop)
    write_placeholder_orb_settings(
        settings_path,
        width=int(export_summary["camera_width"]),
        height=int(export_summary["camera_height"]),
        fps=float(args.fps),
        baseline_m=float(args.baseline_m),
    )

    orb_returncode = None
    if not args.no_orb:
        orb_returncode = run_orbslam3_stereo(
            orb_root=args.orb_root.expanduser().resolve(),
            export_dir=export_dir,
            settings_path=settings_path,
            run_dir=orb_dir,
            timeout_sec=args.timeout_sec,
            viewer=args.viewer,
        )

    manifest = {
        "run_root": str(run_root),
        "capture_started_unix": capture_started,
        "capture_finished_unix": capture_finished,
        "video_device": args.video_device,
        "imu_port": args.imu_port,
        "video_size": args.video_size,
        "duration_sec": args.duration_sec,
        "crop_top_only": not args.no_top_crop,
        "baseline_m": args.baseline_m,
        "imu_summary": imu_summary,
        "export_summary": export_summary,
        "orb_returncode": orb_returncode,
        "trajectory_written": trajectory_has_rows(orb_dir) if orb_returncode is not None else False,
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"run_root={run_root}")
    print(f"raw_video={raw_video}")
    print(f"euroc_export={export_dir}")
    print(f"frames={export_summary['frames']}")
    print(f"camera_size={export_summary['camera_width']}x{export_summary['camera_height']}")
    print(f"imu_rows={imu_summary['imu_rows']}")
    if orb_returncode is not None:
        print(f"orb_returncode={orb_returncode}")
        print(f"trajectory_written={trajectory_has_rows(orb_dir)}")
        print(f"orb_log={orb_dir / 'orbslam3_native.log'}")

    if not args.keep_raw:
        raw_video.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
