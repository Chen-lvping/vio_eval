#!/usr/bin/env python3
"""Audit whether curated RM75 raw episodes can reproduce historical trajectories."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = ROOT / "data/evaluation/curated/rm75_main_clean_ape_le_50mm_20260729/episodes.csv"
DEFAULT_RECOVERED = ROOT / "data/evaluation/recovered/rm75_main_clean_portfolio_trajectories_20260730"
DEFAULT_OUTPUT = ROOT / "data/evaluation/workbench/rm75_historical_input_audit_20260731"
MCAP_READER = ROOT / "script/vendor/orbslam3_colleague_20260716/run_fays_orbslam3_stereo_right.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--recovered-root", type=Path, default=DEFAULT_RECOVERED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("rm75_historical_mcap_reader", MCAP_READER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import MCAP reader: {MCAP_READER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def csv_time_range(path: Path) -> tuple[float, float, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    timestamps = [float(row["Timestamp_us"]) * 1e-6 for row in rows]
    return timestamps[0], timestamps[-1], len(timestamps)


def gt_time_range(path: Path) -> tuple[float, float, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not samples:
        raise ValueError(f"ground truth has no samples: {path}")
    timestamps = [float(sample.get("timestamp_s", sample.get("timestamp"))) for sample in samples]
    return timestamps[0], timestamps[-1], len(timestamps)


def validate_gt(path: Path) -> tuple[float, float, int, bool, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not samples:
        raise ValueError(f"ground truth has no samples: {path}")
    timestamps: list[float] = []
    values_finite = True
    for sample in samples:
        timestamps.append(float(sample.get("timestamp_s", sample.get("timestamp"))))
        position = sample.get("position_m", {})
        quaternion = sample.get("quaternion_xyzw", sample.get("quaternion_wxyz", {}))
        if isinstance(position, dict):
            pose_values = list(position.values())
        else:
            pose_values = list(position)
        if isinstance(quaternion, dict):
            pose_values.extend(quaternion.values())
        else:
            pose_values.extend(quaternion)
        values_finite = values_finite and all(math.isfinite(float(value)) for value in pose_values)
    monotonic = all(current > previous for previous, current in zip(timestamps, timestamps[1:]))
    return timestamps[0], timestamps[-1], len(timestamps), monotonic, values_finite


def overlap_seconds(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    index = args.index.expanduser().resolve()
    recovered_root = args.recovered_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = load_runner()

    with index.open(newline="", encoding="utf-8") as handle:
        sources = list(csv.DictReader(handle))

    rows: list[dict[str, object]] = []
    for source in sources:
        raw = Path(source["raw_dataset"])
        gt = Path(source["ground_truth"])
        tag = raw.name
        historical = recovered_root / tag / "pose_raw.csv"
        row: dict[str, object] = {
            "episode": source["episode"],
            "tag": tag,
            "status": "not_ready",
            "reason": "",
            "raw_exists": raw.is_dir(),
            "gt_exists": gt.is_file(),
            "historical_pose_exists": historical.is_file(),
            "video_exists": (raw / "stereo_right.mkv").is_file(),
            "mcap_exists": (raw / "fays_data_right.mcap").is_file(),
            "video_frame_count": "",
            "mcap_camera_count": "",
            "mcap_imu_count": "",
            "historical_camera_timestamp_match_ratio": "",
            "raw_contains_historical_time_range": "",
            "gt_timestamps_strictly_increasing": "",
            "gt_values_finite": "",
            "raw_camera_start_s": "",
            "raw_camera_end_s": "",
            "historical_pose_start_s": "",
            "historical_pose_end_s": "",
            "gt_start_s": "",
            "gt_end_s": "",
            "raw_historical_overlap_s": "",
            "historical_gt_overlap_s": "",
            "historical_pose_count": "",
            "gt_sample_count": "",
            "raw_dataset": str(raw),
            "ground_truth": str(gt),
            "historical_pose": str(historical),
        }
        missing = []
        if not raw.is_dir():
            missing.append("raw_missing")
        if not gt.is_file():
            missing.append("gt_missing")
        if not historical.is_file():
            missing.append("historical_pose_missing")
        if not (raw / "stereo_right.mkv").is_file():
            missing.append("video_missing")
        if not (raw / "fays_data_right.mcap").is_file():
            missing.append("mcap_missing")
        if missing:
            row["reason"] = ";".join(missing)
            rows.append(row)
            continue

        try:
            hist_start, hist_end, hist_count = csv_time_range(historical)
            gt_start, gt_end, gt_count, gt_monotonic, gt_finite = validate_gt(gt)
            camera, imu = runner.load_mcap_data(raw / "fays_data_right.mcap", "c", "i", 0.0)
            camera_times = [timestamp * 1e-9 for timestamp in camera.values()]
            camera_timestamp_us = {int(round(timestamp * 1e-3)) for timestamp in camera.values()}
            with historical.open(newline="", encoding="utf-8") as handle:
                historical_timestamp_us = [int(round(float(item["Timestamp_us"]))) for item in csv.DictReader(handle)]
            # Historical CSV serialization can move a large epoch timestamp by
            # one microsecond through float round-tripping. Treat only that
            # quantization error as an exact camera-frame match.
            matched_timestamps = sum(
                any(candidate in camera_timestamp_us for candidate in (timestamp - 1, timestamp, timestamp + 1))
                for timestamp in historical_timestamp_us
            )
            timestamp_match_ratio = matched_timestamps / len(historical_timestamp_us)
            camera_range = (min(camera_times), max(camera_times))
            historical_range = (hist_start, hist_end)
            gt_range = (gt_start, gt_end)
            raw_overlap = overlap_seconds(camera_range, historical_range)
            gt_overlap = overlap_seconds(historical_range, gt_range)
            raw_contains_historical = (
                camera_range[0] <= historical_range[0] + 1e-3
                and camera_range[1] >= historical_range[1] - 1e-3
            )
            frame_count = runner.video_frame_count(raw / "stereo_right.mkv")
            video_covers_camera_indices = frame_count > max(camera)
            row.update(
                {
                    "video_frame_count": frame_count,
                    "mcap_camera_count": len(camera),
                    "mcap_imu_count": len(imu),
                    "historical_camera_timestamp_match_ratio": timestamp_match_ratio,
                    "raw_contains_historical_time_range": raw_contains_historical,
                    "gt_timestamps_strictly_increasing": gt_monotonic,
                    "gt_values_finite": gt_finite,
                    "raw_camera_start_s": camera_range[0],
                    "raw_camera_end_s": camera_range[1],
                    "historical_pose_start_s": hist_start,
                    "historical_pose_end_s": hist_end,
                    "gt_start_s": gt_start,
                    "gt_end_s": gt_end,
                    "raw_historical_overlap_s": raw_overlap,
                    "historical_gt_overlap_s": gt_overlap,
                    "historical_pose_count": hist_count,
                    "gt_sample_count": gt_count,
                }
            )
            reasons = []
            if not video_covers_camera_indices:
                reasons.append("video_does_not_cover_mcap_frame_indices")
            if not raw_contains_historical:
                reasons.append("raw_time_domain_mismatch")
            if timestamp_match_ratio < 0.999:
                reasons.append("raw_content_or_timestamp_mismatch")
            if gt_overlap <= 0.0:
                reasons.append("historical_gt_mismatch")
            if not gt_monotonic:
                reasons.append("gt_timestamps_not_strictly_increasing")
            if not gt_finite:
                reasons.append("gt_contains_nonfinite_values")
            row["reason"] = ";".join(reasons)
            row["status"] = "ready" if not reasons else "not_ready"
        except Exception as exc:  # Keep the full audit running after one corrupt episode.
            row["reason"] = f"audit_error:{type(exc).__name__}:{exc}"
        rows.append(row)

    write_csv(output_dir / "input_audit.csv", rows)
    summary = {
        "index": str(index),
        "recovered_root": str(recovered_root),
        "total": len(rows),
        "ready": sum(row["status"] == "ready" for row in rows),
        "not_ready": sum(row["status"] != "ready" for row in rows),
        "rows": rows,
    }
    (output_dir / "input_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("total", "ready", "not_ready")}))
    return 0 if summary["not_ready"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
