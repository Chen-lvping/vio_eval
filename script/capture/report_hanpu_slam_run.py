#!/usr/bin/env python3
"""Generate reproducible camera and SLAM quality metrics for one Hanpu run."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.capture.run_hanpu_orbslam3_smoketest import parse_mjpeg_stream


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip() and not line.lstrip().startswith("#"))


def imu_metrics(path: Path) -> dict[str, object]:
    timestamps: list[int] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    timestamps.append(int(row["timestamp_ns"]))
                except (KeyError, TypeError, ValueError):
                    continue
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    return {
        "rows": len(timestamps),
        "rate_hz": 1e9 / statistics.mean(deltas) if deltas else None,
        "dt_ns": summarize([float(value) for value in deltas]),
    }


def load_calibration(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    calibration = document["calibration"]
    return {
        "serial_number": document.get("header", {}).get("serial_number", ""),
        "stereo_rms_px": document.get("metrics", {}).get("stereo_rms"),
        "baseline_mm": document.get("metrics", {}).get("baseline_mm"),
        "K1": np.array(calibration["K1"], dtype=np.float64),
        "D1": np.array(calibration["D1"], dtype=np.float64).reshape(-1, 1),
        "K2": np.array(calibration["K2"], dtype=np.float64),
        "D2": np.array(calibration["D2"], dtype=np.float64).reshape(-1, 1),
        "R": np.array(calibration["R"], dtype=np.float64),
        "T": np.array(calibration["T"], dtype=np.float64).reshape(3, 1) / 1000.0,
    }


def match_metrics(frames, calibration: dict[str, object], stride: int) -> dict[str, object]:
    sampled = frames[::max(stride, 1)]
    if not sampled:
        return {"sampled_frames": 0}
    first = cv2.imdecode(np.frombuffer(sampled[0].jpeg_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    height, full_width = first.shape
    width = full_width // 2
    size = (width, height)
    K1 = calibration["K1"]
    D1 = calibration["D1"]
    K2 = calibration["K2"]
    D2 = calibration["D2"]
    R = calibration["R"]
    T = calibration["T"]
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    map1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1[:, :3], size, cv2.CV_32F)
    map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2[:, :3], size, cv2.CV_32F)
    orb = cv2.ORB_create(nfeatures=4200, scaleFactor=1.15, nlevels=8, fastThreshold=7)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    left_images = []
    right_images = []
    image_metrics = []
    for item in sampled:
        image = cv2.imdecode(np.frombuffer(item.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        left = image[:, :width]
        right = image[:, width:]
        left_rect = cv2.remap(left, *map1, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, *map2, cv2.INTER_LINEAR)
        left_images.append(left_rect)
        right_images.append(right_rect)
        left_kp = orb.detect(left_rect, None)
        right_kp = orb.detect(right_rect, None)
        image_metrics.append({
            "left_mean": float(left_rect.mean()),
            "right_mean": float(right_rect.mean()),
            "left_std": float(left_rect.std()),
            "right_std": float(right_rect.std()),
            "left_laplacian_var": float(cv2.Laplacian(left_rect, cv2.CV_64F).var()),
            "right_laplacian_var": float(cv2.Laplacian(right_rect, cv2.CV_64F).var()),
            "left_keypoints": len(left_kp),
            "right_keypoints": len(right_kp),
        })

    def pair_matches(first_image, second_image) -> tuple[int, float | None]:
        _, first_desc = orb.detectAndCompute(first_image, None)
        _, second_desc = orb.detectAndCompute(second_image, None)
        if first_desc is None or second_desc is None:
            return 0, None
        matches = list(matcher.match(first_desc, second_desc))
        distances = [float(item.distance) for item in matches]
        return len(matches), statistics.median(distances) if distances else None

    stereo = [pair_matches(left, right) for left, right in zip(left_images, right_images)]
    temporal = [pair_matches(first, second) for first, second in zip(left_images, left_images[1:])]
    return {
        "image_size": {"width": width, "height": height},
        "sampled_frames": len(sampled),
        "image_metrics": {
            key: summarize([float(item[key]) for item in image_metrics])
            for key in image_metrics[0]
        },
        "stereo_matches": summarize([float(item[0]) for item in stereo]),
        "stereo_median_hamming": summarize([float(item[1]) for item in stereo if item[1] is not None]),
        "temporal_matches": summarize([float(item[0]) for item in temporal]),
        "temporal_median_hamming": summarize([float(item[1]) for item in temporal if item[1] is not None]),
    }


def log_metrics(run_root: Path) -> dict[str, int]:
    text = ""
    for path in run_root.rglob("*.log"):
        text += path.read_text(encoding="utf-8", errors="ignore")
    return {
        "fail_to_track_local_map": len(re.findall(r"Fail to track local map", text)),
        "less_than_15_matches": len(re.findall(r"Less than 15 matches", text)),
        "frames_set_to_lost": len(re.findall(r"Frames set to lost", text)),
        "map_created": len(re.findall(r"New Map created", text)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--calibration-json", type=Path, default=Path("data/hanpu/calibration/stereo_calibration.json"))
    parser.add_argument("--sample-stride", type=int, default=10)
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    raw_video = run_root / "raw/stereo_raw.mjpg"
    if not raw_video.is_file():
        raise SystemExit(f"raw video not found: {raw_video}")
    frames = parse_mjpeg_stream(raw_video)
    pts = [item.left_pts_us for item in frames]
    deltas = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    calibration = load_calibration(args.calibration_json.expanduser().resolve())
    report: dict[str, object] = {
        "run_root": str(run_root),
        "frames": len(frames),
        "duration_s": (pts[-1] - pts[0]) / 1e6 if len(pts) >= 2 else None,
        "video_rate_hz": 1e6 / statistics.mean(deltas) if deltas else None,
        "video_dt_us": summarize([float(value) for value in deltas]),
        "positive_dt_count": len(deltas),
        "nonpositive_dt_count": sum(1 for a, b in zip(pts, pts[1:]) if b <= a),
        "stereo_pts_equal": all(item.left_pts_us == item.right_pts_us for item in frames),
        "left_exposure_us": summarize([float(item.left_exp_us) for item in frames]),
        "right_exposure_us": summarize([float(item.right_exp_us) for item in frames]),
        "imu": imu_metrics(run_root / "raw/imu_raw.csv"),
        "calibration": {
            "serial_number": calibration["serial_number"],
            "stereo_rms_px": calibration["stereo_rms_px"],
            "baseline_mm": calibration["baseline_mm"],
        },
        "image_and_matching": match_metrics(frames, calibration, args.sample_stride),
        "orb_log": log_metrics(run_root),
        "trajectory_rows": count_rows(run_root / "orbslam3_real_run/f_hanpu_real_stereo.txt"),
        "keyframe_rows": count_rows(run_root / "orbslam3_real_run/kf_hanpu_real_stereo.txt"),
    }
    json_path = run_root / "camera_slam_report.json"
    md_path = run_root / "camera_slam_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    image = report["image_and_matching"]
    md_path.write_text(
        "# Hanpu Camera and SLAM Report\n\n"
        f"- Run: `{run_root}`\n"
        f"- Video: {report['frames']} frames, {report['video_rate_hz']:.6f} Hz, {report['duration_s']:.3f} s\n"
        f"- Stereo PTS equal: `{report['stereo_pts_equal']}`\n"
        f"- Exposure left/right median: {report['left_exposure_us']['median']} / {report['right_exposure_us']['median']} us\n"
        f"- IMU: {report['imu']['rows']} rows, {report['imu']['rate_hz']} Hz\n"
        f"- Calibration: RMS {report['calibration']['stereo_rms_px']} px, baseline {report['calibration']['baseline_mm']} mm\n"
        f"- ORB keypoints mean: {image['image_metrics']['left_keypoints']['mean']} / {image['image_metrics']['right_keypoints']['mean']}\n"
        f"- Stereo matches mean: {image['stereo_matches']['mean']}\n"
        f"- Temporal matches mean: {image['temporal_matches']['mean']}\n"
        f"- ORB log: {report['orb_log']}\n"
        f"- Trajectory/keyframes: {report['trajectory_rows']} / {report['keyframe_rows']} rows\n",
        encoding="utf-8",
    )
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    print(json.dumps(report["orb_log"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
