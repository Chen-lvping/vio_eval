#!/usr/bin/env python3
"""Recover an RM75-compatible pose JSON from a TUM trajectory file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_tum_line(line: str, line_no: int) -> dict[str, Any]:
    text = line.strip()
    if not text or text.startswith("#"):
        raise ValueError("skip")

    parts = text.split()
    if len(parts) != 8:
        raise ValueError(f"line {line_no}: expected 8 columns, got {len(parts)}")

    timestamp_s, x, y, z, qx, qy, qz, qw = (float(part) for part in parts)
    return {
        "timestamp_s": timestamp_s,
        "timestamp_before_s": timestamp_s,
        "timestamp_after_s": timestamp_s,
        "timestamp_midpoint_s": timestamp_s,
        "monotonic_before_s": 0.0,
        "monotonic_after_s": 0.0,
        "read_duration_sec": 0.0,
        "position_m": {"x": x, "y": y, "z": z},
        "quaternion_wxyz": {"w": qw, "x": qx, "y": qy, "z": qz},
    }


def load_tum(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            sample = parse_tum_line(raw, line_no)
        except ValueError as exc:
            if str(exc) == "skip":
                continue
            raise
        sample["sample_index"] = len(samples)
        samples.append(sample)
    if not samples:
        raise RuntimeError(f"no trajectory samples found in {path}")
    return samples


def build_payload(samples: list[dict[str, Any]], source_tum: Path) -> dict[str, Any]:
    start_time_s = float(samples[0]["timestamp_s"])
    end_time_s = float(samples[-1]["timestamp_s"])
    duration_s = max(0.0, end_time_s - start_time_s)
    sample_count = len(samples)
    actual_rate_hz = (sample_count / duration_s) if duration_s > 0 else 0.0

    return {
        "model": "RM-75-B",
        "frame": "current work frame / current tool TCP",
        "source_tum": str(source_tum.resolve()),
        "recovered_from": "gt_tcp.tum",
        "recovery_note": (
            "Recovered from an evaluation TUM export. Original per-sample host timing "
            "metadata was not preserved, so timestamp_before/after and monotonic timing "
            "fields were reconstructed from timestamp_s."
        ),
        "target_rate_hz": actual_rate_hz,
        "actual_rate_hz": actual_rate_hz,
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "duration_s": duration_s,
        "sample_count": sample_count,
        "capture_started_host_s": start_time_s,
        "capture_ended_host_s": end_time_s,
        "capture_started_monotonic_s": 0.0,
        "capture_ended_monotonic_s": 0.0,
        "timestamp_source": "recovered_from_tum_timestamp",
        "timestamp_policy": "tum_timestamp_reused_for_all_sample_time_fields",
        "scheduler_clock": "recovered_offline",
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover an RM75-compatible pose JSON from an evaluation gt_tcp.tum file."
    )
    parser.add_argument("input_tum", type=Path, help="Input TUM trajectory path")
    parser.add_argument("output_json", type=Path, help="Recovered output JSON path")
    args = parser.parse_args()

    input_tum = args.input_tum.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    samples = load_tum(input_tum)
    payload = build_payload(samples, input_tum)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Recovered {payload['sample_count']} samples from {input_tum} "
        f"to {output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
