#!/usr/bin/env python3
"""Manually record Hanpu stereo+IMU, then run ORB-SLAM3 and plot the trajectory."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_VIDEO = "/dev/v4l/by-id/usb-YCTC_YCTC_SC233HGS_0152312181647-video-index0"
DEFAULT_IMU = "/dev/ttyACM0"
DEFAULT_ORB_ROOT = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_clean")
BAUD = 921600
MSG_IMU_DATA = 0x04
MSG_STOP = 0x02
CONTROL_ACK_FLAG = 0x01


def read_bridge_frame(port: serial.Serial, timeout_sec: float):
    from script.capture.run_hanpu_orbslam3_smoketest import read_bridge_frame as read_frame

    return read_frame(port, timeout_sec)


def send_stop(port: serial.Serial) -> None:
    from script.capture.run_hanpu_orbslam3_smoketest import build_frame

    port.write(build_frame(MSG_STOP, 20))
    deadline = time.time() + 1.5
    while time.time() < deadline:
        frame = read_bridge_frame(port, 0.2)
        if frame is not None and frame.msg_type == MSG_STOP and frame.flags == CONTROL_ACK_FLAG:
            return


def capture_imu(stop_event: threading.Event, port_name: str, output_path: Path, error_box: list[BaseException]) -> None:
    rows: list[tuple[int, float, float, float, float, float, float]] = []
    try:
        from script.capture.run_hanpu_orbslam3_smoketest import (
            request_fw_info,
            start_imu_stream,
        )

        with serial.Serial(port_name, BAUD, timeout=0.1) as port:
            fw_info = request_fw_info(port)
            print(f"imu_fw={fw_info['name']}", flush=True)
            start_imu_stream(port)
            print("imu_stream=started", flush=True)

            while not stop_event.is_set():
                frame = read_bridge_frame(port, 0.3)
                if frame is None or frame.msg_type != MSG_IMU_DATA:
                    continue
                _, _, _, gyro_count, acc_count = struct.unpack_from("<IQQHH", frame.payload, 0)
                pos = 24
                gyros = []
                for _ in range(gyro_count):
                    x, y, z, _temp, pts_us = struct.unpack_from("<hhhIQ", frame.payload, pos)
                    pos += 18
                    scale = 1000.0 / 32768.0 * np.pi / 180.0
                    gyros.append((pts_us, x * scale, y * scale, z * scale))
                accs = []
                for _ in range(acc_count):
                    x, y, z, _temp, pts_us = struct.unpack_from("<hhhIQ", frame.payload, pos)
                    pos += 18
                    scale = 4.0 / 32768.0 * 9.80665
                    accs.append((pts_us, x * scale, y * scale, z * scale))

                by_ts: dict[int, list[float]] = {}
                for ts, gx, gy, gz in gyros:
                    by_ts.setdefault(ts, [np.nan] * 6)[0:3] = [gx, gy, gz]
                for ts, ax, ay, az in accs:
                    by_ts.setdefault(ts, [np.nan] * 6)[3:6] = [ax, ay, az]
                for ts, values in sorted(by_ts.items()):
                    if not any(np.isnan(value) for value in values):
                        rows.append((ts * 1000, *values))
            send_stop(port)
    except BaseException as exc:
        error_box.append(exc)
        stop_event.set()
    finally:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp_ns", "gx_rad_s", "gy_rad_s", "gz_rad_s", "ax_m_s2", "ay_m_s2", "az_m_s2"])
            writer.writerows(rows)
        print(f"imu_rows={len(rows)}", flush=True)


def read_pipe_frame(pipe, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = pipe.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def stop_video(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-device", default=DEFAULT_VIDEO)
    parser.add_argument("--imu-port", default=DEFAULT_IMU)
    parser.add_argument("--video-size", default="3840x1080")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data/hanpu/runs")
    parser.add_argument("--calibration-json", type=Path, default=REPO_ROOT / "data/hanpu/calibration/stereo_calibration.json")
    parser.add_argument("--orb-root", type=Path, default=DEFAULT_ORB_ROOT)
    parser.add_argument("--orb-viewer", action="store_true")
    parser.add_argument("--no-orb", action="store_true", help="Stop after recording, EuRoC export, and real calibration YAML generation")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root.expanduser().resolve() / f"hanpu_manual_{stamp}"
    raw_dir = run_root / "raw"
    raw_video = raw_dir / "stereo_raw.mjpg"
    raw_imu = raw_dir / "imu_raw.csv"
    raw_dir.mkdir(parents=True, exist_ok=True)

    preview_width, preview_height = 1920, 540
    frame_bytes = preview_width * preview_height * 3
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "v4l2", "-input_format", "mjpeg", "-video_size", args.video_size,
        "-framerate", str(args.fps), "-i", args.video_device,
        "-map", "0:v", "-c", "copy", str(raw_video),
        "-map", "0:v", "-vf", f"fps=10,scale={preview_width}:{preview_height}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    video = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None)
    stop_event = threading.Event()
    imu_error: list[BaseException] = []
    imu_thread = threading.Thread(target=capture_imu, args=(stop_event, args.imu_port, raw_imu, imu_error), daemon=True)
    imu_thread.start()

    print("Recording started. Press q in the preview window or Ctrl-C to stop.", flush=True)
    cv2.namedWindow("hanpu_record_preview", cv2.WINDOW_NORMAL)
    try:
        while not stop_event.is_set():
            if video.stdout is None:
                break
            data = read_pipe_frame(video.stdout, frame_bytes)
            if data is None:
                break
            frame = np.frombuffer(data, dtype=np.uint8).reshape((preview_height, preview_width, 3)).copy()
            cv2.putText(frame, "Press q to stop", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("hanpu_record_preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                stop_event.set()
                break
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        stop_video(video)
        imu_thread.join(timeout=5)
        cv2.destroyAllWindows()

    if imu_error:
        raise RuntimeError(f"IMU capture failed: {imu_error[0]}")
    if not raw_video.exists() or raw_video.stat().st_size == 0:
        raise RuntimeError(f"raw video was not written: {raw_video}")

    from script.capture.hanpu_calibration_to_orbslam3_yaml import write_yaml
    from script.capture.run_hanpu_orbslam3_smoketest import export_euroc, parse_mjpeg_stream

    frames = parse_mjpeg_stream(raw_video)
    export_dir = run_root / "euroc_export"
    summary = export_euroc(frames, raw_imu, export_dir, crop_top_only=False)
    print(f"export_frames={summary['frames']}", flush=True)

    real_yaml = run_root / "hanpu_stereo_real_1920x1080.yaml"
    write_yaml(args.calibration_json.expanduser().resolve(), real_yaml, 1920, 1080, float(args.fps), 35.0)

    if args.no_orb:
        print(f"run_root={run_root}")
        print(f"euroc_export={export_dir}")
        print(f"calibration_yaml={real_yaml}")
        return 0

    orb_root = args.orb_root.expanduser().resolve()
    orb_dir = run_root / "orbslam3_real_run"
    orb_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join([
        str(orb_root / "lib"),
        str(orb_root / "Thirdparty/DBoW2/lib"),
        str(orb_root / "Thirdparty/g2o/lib"),
        "/home/chenlvping/1_DM_work/orbslam3_fresh/Pangolin_v06/install/lib",
        env.get("LD_LIBRARY_PATH", ""),
    ])
    env["ORB_SLAM3_ENABLE_VIEWER"] = "1" if args.orb_viewer else "0"
    orb_command = [
        str(orb_root / "Examples/Stereo/stereo_euroc"),
        str(orb_root / "Vocabulary/ORBvoc.txt"),
        str(real_yaml),
        str(export_dir),
        str(export_dir / "times.txt"),
        "hanpu_real_stereo",
    ]
    log_path = orb_dir / "orbslam3_real.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            orb_command,
            cwd=orb_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, orb_command)

    trajectory = orb_dir / "f_hanpu_real_stereo.txt"
    print(f"run_root={run_root}")
    print(f"orb_log={log_path}")
    print(f"trajectory={trajectory}")
    if not args.no_plot:
        subprocess.run(["evo_traj", "euroc", str(trajectory), "--plot", "--plot_mode", "xyz"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
