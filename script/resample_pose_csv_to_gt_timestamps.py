#!/usr/bin/env python3
"""Resample a pose CSV onto GT timestamps after applying a fixed time offset."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


CSV_HEADER = ["Timestamp_us", "X", "Y", "Z", "Quat_X", "Quat_Y", "Quat_Z", "Quat_W"]


@dataclass(frozen=True)
class PoseSeries:
    times_s: np.ndarray
    positions: np.ndarray
    quats_xyzw: np.ndarray


def normalize_timestamp(value: float) -> float:
    value = float(value)
    av = abs(value)
    if av > 1e17:
        return value * 1e-9
    if av > 1e13:
        return value * 1e-6
    if av > 1e10:
        return value * 1e-3
    return value


def read_estimate_csv(path: Path) -> PoseSeries:
    times_s = []
    positions = []
    quats = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            times_s.append(float(row["Timestamp_us"]) * 1e-6)
            positions.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
            quats.append([float(row["Quat_X"]), float(row["Quat_Y"]), float(row["Quat_Z"]), float(row["Quat_W"])])
    if not times_s:
        raise RuntimeError(f"no poses found in {path}")
    return PoseSeries(
        times_s=np.asarray(times_s, dtype=float),
        positions=np.asarray(positions, dtype=float),
        quats_xyzw=np.asarray(quats, dtype=float),
    )


def load_gt_times(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", payload)
    times = []
    for sample in samples:
        raw = sample.get("timestamp", sample.get("timestamp_s", sample.get("timestamp_us")))
        if raw is None:
            raise KeyError("timestamp missing in GT sample")
        times.append(normalize_timestamp(float(raw)))
    if not times:
        raise RuntimeError(f"no timestamps found in {path}")
    return np.asarray(times, dtype=float)


def interpolate_pose(series: PoseSeries, query_times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(series.times_s, dtype=float)
    pos = np.asarray(series.positions, dtype=float)
    rot = Rotation.from_quat(np.asarray(series.quats_xyzw, dtype=float))

    interp_pos = np.column_stack(
        [
            np.interp(query_times_s, times, pos[:, 0]),
            np.interp(query_times_s, times, pos[:, 1]),
            np.interp(query_times_s, times, pos[:, 2]),
        ]
    )
    slerp = Slerp(times, rot)
    interp_quat = slerp(query_times_s).as_quat()
    return interp_pos, interp_quat


def write_pose_csv(path: Path, times_s: np.ndarray, positions: np.ndarray, quats_xyzw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for t, p, q in zip(times_s, positions, quats_xyzw):
            writer.writerow(
                [
                    str(int(round(float(t) * 1_000_000.0))),
                    f"{float(p[0]):.9f}",
                    f"{float(p[1]):.9f}",
                    f"{float(p[2]):.9f}",
                    f"{float(q[0]):.9f}",
                    f"{float(q[1]):.9f}",
                    f"{float(q[2]):.9f}",
                    f"{float(q[3]):.9f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate-csv", type=Path, required=True)
    parser.add_argument("--gt-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--time-offset-sec",
        type=float,
        required=True,
        help="Offset added to estimate timestamps before interpolation onto GT times.",
    )
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=0.10,
        help="Reject GT timestamps whose nearest estimate bracket is farther than this gap.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    estimate_csv = args.estimate_csv.expanduser().resolve()
    gt_json = args.gt_json.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    max_gap_sec = float(args.max_gap_sec)

    series = read_estimate_csv(estimate_csv)
    shifted_series = PoseSeries(
        times_s=np.asarray(series.times_s, dtype=float) + float(args.time_offset_sec),
        positions=series.positions,
        quats_xyzw=series.quats_xyzw,
    )
    gt_times_s = load_gt_times(gt_json)

    valid_mask = (gt_times_s >= shifted_series.times_s[0]) & (gt_times_s <= shifted_series.times_s[-1])
    overlap_times = gt_times_s[valid_mask]
    if overlap_times.size == 0:
        raise RuntimeError("no overlap between shifted estimate and GT timestamps")

    left_indices = np.searchsorted(shifted_series.times_s, overlap_times, side="right") - 1
    right_indices = np.clip(left_indices + 1, 0, shifted_series.times_s.size - 1)
    left_indices = np.clip(left_indices, 0, shifted_series.times_s.size - 1)
    left_dt = np.abs(overlap_times - shifted_series.times_s[left_indices])
    right_dt = np.abs(shifted_series.times_s[right_indices] - overlap_times)
    bracket_gap = np.maximum(left_dt, right_dt)
    keep_mask = bracket_gap <= max_gap_sec
    final_times = overlap_times[keep_mask]
    if final_times.size == 0:
        raise RuntimeError("no GT timestamps survived max-gap filtering")

    interp_pos, interp_quat = interpolate_pose(shifted_series, final_times)
    write_pose_csv(output_csv, final_times, interp_pos, interp_quat)

    manifest = {
        "estimate_csv": str(estimate_csv),
        "gt_json": str(gt_json),
        "output_csv": str(output_csv),
        "time_offset_sec": float(args.time_offset_sec),
        "max_gap_sec": max_gap_sec,
        "input_pose_count": int(series.times_s.size),
        "gt_time_count": int(gt_times_s.size),
        "output_pose_count": int(final_times.size),
        "shifted_estimate_first_s": float(shifted_series.times_s[0]),
        "shifted_estimate_last_s": float(shifted_series.times_s[-1]),
        "output_first_s": float(final_times[0]),
        "output_last_s": float(final_times[-1]),
    }
    output_csv.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {output_csv} ({final_times.size} poses)")
    print(f"[OK] wrote {output_csv.with_suffix('.manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
