#!/usr/bin/env python3
"""Probe a YCTC SC233HGS stereo-IMU device over its CDC ACM bridge."""

from __future__ import annotations

import argparse
import statistics
import struct
import time
from dataclasses import dataclass
from typing import Iterable

import serial


MAGIC = b"SY"
PROTOCOL_VERSION = 3
DEFAULT_BAUD = 921600

MSG_START = 0x01
MSG_STOP = 0x02
MSG_IMU_DATA = 0x04
MSG_MAG_DATA = 0x05
MSG_FW_VERSION = 0x08
CONTROL_ACK_FLAG = 0x01

STREAM_MASKS = {
    "imu": 0x01,
    "mag": 0x02,
    "imu+mag": 0x03,
}


@dataclass
class Frame:
    msg_type: int
    flags: int
    seq: int
    payload: bytes


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
    crc = crc16_ibm(header + payload)
    return header + payload + struct.pack("<H", crc)


def read_exact(port: serial.Serial, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = port.read(size - len(data))
        if not chunk:
            break
        data += chunk
    return data


def read_frame(port: serial.Serial, timeout_sec: float) -> Frame | None:
    deadline = time.time() + timeout_sec
    sync = bytearray()
    while time.time() < deadline:
        byte = port.read(1)
        if not byte:
            continue
        sync += byte
        if len(sync) > len(MAGIC):
            sync = sync[-len(MAGIC) :]
        if bytes(sync) != MAGIC:
            continue

        header_tail = read_exact(port, 8)
        if len(header_tail) != 8:
            continue
        header = MAGIC + header_tail
        version = header[2]
        payload_len = struct.unpack_from("<H", header, 7)[0]
        payload = read_exact(port, payload_len)
        crc = read_exact(port, 2)
        if len(payload) != payload_len or len(crc) != 2:
            continue
        if version != PROTOCOL_VERSION:
            continue
        if struct.unpack("<H", crc)[0] != crc16_ibm(header + payload):
            continue
        return Frame(
            msg_type=header[3],
            flags=header[4],
            seq=struct.unpack_from("<H", header, 5)[0],
            payload=payload,
        )
    return None


def drain_frames(port: serial.Serial, duration_sec: float) -> list[Frame]:
    frames: list[Frame] = []
    deadline = time.time() + max(duration_sec, 0.0)
    while time.time() < deadline:
        frame = read_frame(port, min(0.05, max(deadline - time.time(), 0.01)))
        if frame is not None:
            frames.append(frame)
    return frames


def send_stop_and_wait(port: serial.Serial, seq: int, timeout_sec: float) -> bool:
    port.write(build_frame(MSG_STOP, seq))
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        frame = read_frame(port, min(0.2, max(deadline - time.time(), 0.01)))
        if frame is None:
            continue
        if frame.msg_type == MSG_STOP and frame.flags == CONTROL_ACK_FLAG and frame.seq == seq:
            return True
    return False


def request_fw_payload(port: serial.Serial, retries: int = 3) -> bytes:
    for attempt in range(retries):
        seq = 1 + attempt
        port.reset_input_buffer()
        port.reset_output_buffer()
        send_stop_and_wait(port, 100 + seq, 0.4)
        drain_frames(port, 0.1)
        time.sleep(0.05)
        port.write(build_frame(MSG_FW_VERSION, seq))
        deadline = time.time() + 2.0
        while time.time() < deadline:
            frame = read_frame(port, min(0.25, max(deadline - time.time(), 0.01)))
            if frame is None:
                continue
            if frame.msg_type == MSG_FW_VERSION and frame.flags == CONTROL_ACK_FLAG and frame.seq == seq:
                return frame.payload
    raise RuntimeError("failed to read FW_VERSION response after retries")


def start_stream(port: serial.Serial, mask: int, retries: int = 3) -> int:
    payload = bytes([mask, 0, 0, 0])
    for attempt in range(retries):
        seq = 10 + attempt
        drain_frames(port, 0.05)
        port.write(build_frame(MSG_START, seq, payload))
        deadline = time.time() + 2.0
        while time.time() < deadline:
            frame = read_frame(port, min(0.25, max(deadline - time.time(), 0.01)))
            if frame is None:
                continue
            if frame.msg_type == MSG_START and frame.flags == CONTROL_ACK_FLAG and frame.seq == seq:
                return frame.payload[0]
    raise RuntimeError("failed to read START ack after retries")


def format_rate(samples: list[int]) -> str:
    deltas = [curr - prev for prev, curr in zip(samples, samples[1:]) if curr > prev]
    if not deltas:
        return "n/a"
    mean_dt = statistics.mean(deltas)
    return (
        f"{1e6 / mean_dt:.3f} Hz"
        f" (mean_dt={mean_dt:.1f} us, min={min(deltas)}, max={max(deltas)})"
    )


def parse_fw_payload(payload: bytes) -> tuple[str, str, str]:
    code_len, name_len, build_len = struct.unpack_from("<HHH", payload, 0)
    offset = 6
    code = payload[offset : offset + code_len].decode("ascii", errors="replace")
    offset += code_len
    name = payload[offset : offset + name_len].decode("ascii", errors="replace")
    offset += name_len
    build = payload[offset : offset + build_len].decode("ascii", errors="replace")
    return code, name, build


def summarize_frames(frames: Iterable[Frame]) -> dict[str, object]:
    imu_frames = 0
    mag_frames = 0
    gyro_pts: list[int] = []
    acc_pts: list[int] = []
    mag_pts: list[int] = []
    imu_windows: list[tuple[int, int]] = []

    for frame in frames:
        if frame.msg_type == MSG_IMU_DATA:
            imu_frames += 1
            generation, window_begin, window_end, gyro_count, acc_count = struct.unpack_from(
                "<IQQHH", frame.payload, 0
            )
            imu_windows.append((window_begin, window_end))
            pos = 24
            for _ in range(gyro_count):
                gyro_pts.append(struct.unpack_from("<hhhIQ", frame.payload, pos)[-1])
                pos += 18
            for _ in range(acc_count):
                acc_pts.append(struct.unpack_from("<hhhIQ", frame.payload, pos)[-1])
                pos += 18
        elif frame.msg_type == MSG_MAG_DATA:
            mag_frames += 1
            _, count, _ = struct.unpack_from("<IHH", frame.payload, 0)
            pos = 8
            for _ in range(count):
                mag_pts.append(struct.unpack_from("<iiiB3xQ", frame.payload, pos)[-1])
                pos += 24

    summary: dict[str, object] = {
        "imu_frames": imu_frames,
        "mag_frames": mag_frames,
        "gyro_samples": len(gyro_pts),
        "acc_samples": len(acc_pts),
        "mag_samples": len(mag_pts),
        "gyro_rate": format_rate(gyro_pts),
        "acc_rate": format_rate(acc_pts),
        "mag_rate": format_rate(mag_pts),
    }
    if imu_windows:
        summary["imu_window_begin_us"] = imu_windows[0][0]
        summary["imu_window_end_us"] = imu_windows[-1][1]
    if gyro_pts:
        summary["gyro_first_pts_us"] = gyro_pts[0]
        summary["gyro_last_pts_us"] = gyro_pts[-1]
    if acc_pts:
        summary["acc_first_pts_us"] = acc_pts[0]
        summary["acc_last_pts_us"] = acc_pts[-1]
    if mag_pts:
        summary["mag_first_pts_us"] = mag_pts[0]
        summary["mag_last_pts_us"] = mag_pts[-1]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--stream", choices=tuple(STREAM_MASKS), default="imu+mag")
    parser.add_argument("--duration-sec", type=float, default=1.5)
    parser.add_argument("--read-timeout-sec", type=float, default=0.2)
    parser.add_argument("--fw-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with serial.Serial(args.port, args.baud, timeout=args.read_timeout_sec) as port:
        code, name, build = parse_fw_payload(request_fw_payload(port))
        print(f"fw_code={code}")
        print(f"fw_name={name}")
        print(f"fw_build={build}")

        if args.fw_only:
            return 0

        mask = STREAM_MASKS[args.stream]
        ack_stream = start_stream(port, mask)
        print(f"start_ack_stream=0x{ack_stream:02x}")

        frames: list[Frame] = []
        end_time = time.time() + max(args.duration_sec, 0.1)
        while time.time() < end_time:
            frame = read_frame(port, timeout_sec=0.3)
            if frame is not None:
                frames.append(frame)

        if not send_stop_and_wait(port, 20, 1.5):
            raise RuntimeError("failed to read STOP ack")

    summary = summarize_frames(frames)
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
