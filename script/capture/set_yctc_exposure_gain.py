#!/usr/bin/env python3
"""Set and verify SC233HGS exposure and ISP system gain over the YCTC bridge."""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import serial

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.capture.run_hanpu_orbslam3_smoketest import (
    BAUD,
    CONTROL_ACK_FLAG,
    build_frame,
    read_bridge_frame,
    send_stop_and_wait,
)


MSG_PWM_EXPOSURE = 0x0B
MSG_GAIN = 0x0C


def request(port: serial.Serial, msg_type: int, seq: int, payload: bytes):
    port.write(build_frame(msg_type, seq, payload))
    deadline = time.time() + 2.0
    while time.time() < deadline:
        frame = read_bridge_frame(port, 0.25)
        if frame is None:
            continue
        if frame.msg_type == msg_type and frame.flags == CONTROL_ACK_FLAG and frame.seq == seq:
            return frame
    raise RuntimeError(f"no ACK for message 0x{msg_type:02x}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--exposure-us", type=int, default=10000, help="Exposure time in microseconds")
    parser.add_argument("--gain", type=lambda value: int(value, 0), default=0x1000, help="22.10 fixed-point gain, default 0x1000 (4x)")
    args = parser.parse_args()
    if args.exposure_us <= 0 or args.exposure_us > 33270:
        raise SystemExit("exposure must be in 1..33270 us")
    if args.gain < 0x400:
        raise SystemExit("gain must be at least 0x400")

    with serial.Serial(args.port, BAUD, timeout=0.1) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()
        send_stop_and_wait(port, 90, 1.0)

        exposure = request(
            port,
            MSG_PWM_EXPOSURE,
            91,
            struct.pack("<II", args.exposure_us * 1000, 0),
        )
        gain = request(port, MSG_GAIN, 92, struct.pack("<I", args.gain))

    applied_exposure_ns, pulse_width_ns = struct.unpack_from("<II", exposure.payload, 0)
    applied_gain = struct.unpack_from("<I", gain.payload, 0)[0]
    print(f"exposure_us={applied_exposure_ns / 1000.0:.1f}")
    print(f"pulse_width_us={pulse_width_ns / 1000.0:.1f}")
    print(f"gain_raw=0x{applied_gain:08x}")
    print(f"gain_multiplier={applied_gain / 1024.0:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
