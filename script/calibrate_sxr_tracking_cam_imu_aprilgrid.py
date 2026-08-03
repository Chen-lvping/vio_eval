#!/usr/bin/env python3
"""Fit tracking-camera rotation and timing from AprilTag motion and gyro data.

Only tag corners, tracking timestamps, and the calibrated gyro are used.  A
single visible tag is sufficient because its fixed board-frame orientation
cancels when forming consecutive camera rotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import av
import cv2
import numpy as np
from pupil_apriltags import Detector

from calibrate_sxr_tracking_stereo_aprilgrid import tag_pose
from validate_sxr_tracking_aprilgrid import fit_rotation, interpolate, read_gyro, rotation_distance_deg


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--tag-family", default="tag16h5")
    parser.add_argument("--tag-size", type=float, default=0.055)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--maximum-pair-dt-sec", type=float, default=0.25)
    parser.add_argument("--offset-span-ms", type=float, default=30.0)
    parser.add_argument("--offset-step-ms", type=float, default=0.5)
    return parser.parse_args(argv)


def timestamps(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        return np.asarray([int(row["mid_exposure_utc_ns"]) * 1e-9 for row in csv.DictReader(handle)], dtype=np.float64)


def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 30 or np.std(first) < 1e-8 or np.std(second) < 1e-8:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tag_size <= 0.0 or args.frame_stride < 1 or args.maximum_pair_dt_sec <= 0.0:
        raise ValueError("invalid AprilGrid timing-calibration settings")
    episode, output = args.episode_dir.expanduser().resolve(), args.output_json.expanduser().resolve()
    required = ("tracking.mp4", "tracking_metainfo.csv", "camera_params_tracking.json", "imu_calibration.json", "gyro.csv")
    missing = [name for name in required if not (episode / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {episode}")
    image = json.loads((episode / "camera_params_tracking.json").read_text(encoding="utf-8"))
    calibration = json.loads((episode / "imu_calibration.json").read_text(encoding="utf-8"))
    camera = image["cameras"][0]
    frame_stamps = timestamps(episode / "tracking_metainfo.csv")
    gyro_stamps, gyro = read_gyro(episode / "gyro.csv", calibration)
    detector = Detector(families=args.tag_family, nthreads=1, quad_decimate=1.0, refine_edges=1)
    previous: dict[int, tuple[int, np.ndarray]] = {}
    visual_times: list[float] = []
    visual_vectors: list[np.ndarray] = []
    detections = 0
    container = av.open(str(episode / "tracking.mp4"))
    try:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index >= len(frame_stamps):
                break
            if index % args.frame_stride:
                continue
            gray = np.ascontiguousarray(frame.to_ndarray(format="gray")[:, : frame.width // 2])
            for tag in detector.detect(gray):
                pose = tag_pose(np.asarray(tag.corners, dtype=np.float64).copy(), camera, args.tag_size)
                if pose is None:
                    continue
                tag_id = int(tag.tag_id)
                detections += 1
                prior = previous.get(tag_id)
                if prior is not None:
                    prior_index, prior_pose = prior
                    dt = float(frame_stamps[index] - frame_stamps[prior_index])
                    if 1e-6 < dt <= args.maximum_pair_dt_sec:
                        relative = pose[:3, :3] @ prior_pose[:3, :3].T
                        visual_vectors.append(-cv2.Rodrigues(relative)[0].reshape(3) / dt)
                        visual_times.append(0.5 * (frame_stamps[index] + frame_stamps[prior_index]))
                previous[tag_id] = (index, pose)
    finally:
        container.close()
    visual = np.asarray(visual_vectors, dtype=np.float64)
    times = np.asarray(visual_times, dtype=np.float64)
    if len(visual) < 50:
        raise RuntimeError(f"only {len(visual)} AprilGrid/gyro rotation pairs")
    visual_rates, gyro_rates = np.linalg.norm(visual, axis=1), np.linalg.norm(gyro, axis=1)
    offsets = np.arange(-args.offset_span_ms, args.offset_span_ms + 0.5 * args.offset_step_ms, args.offset_step_ms) * 1e-3
    scan = []
    for offset in offsets:
        sampled = np.interp(times + offset, gyro_stamps, gyro_rates, left=np.nan, right=np.nan)
        valid = np.isfinite(sampled)
        scan.append({"camera_to_imu_offset_s": float(offset), "correlation": correlation(visual_rates[valid], sampled[valid]), "samples": int(valid.sum())})
    ranked = [row for row in scan if row["correlation"] is not None]
    if not ranked:
        raise RuntimeError("not enough AprilGrid/gyro pairs for timing scan")
    best = max(ranked, key=lambda row: float(row["correlation"]))
    sampled_gyro = interpolate(times + float(best["camera_to_imu_offset_s"]), gyro_stamps, gyro)
    active = np.isfinite(sampled_gyro).all(axis=1) & (visual_rates > 0.08) & (np.linalg.norm(sampled_gyro, axis=1) > 0.08)
    if int(active.sum()) < 50:
        raise RuntimeError("not enough excited AprilGrid/gyro pairs")
    fitted, cosine, rmse_deg = fit_rotation(sampled_gyro[active], visual[active])
    factory_offset = float(calibration["imu"]["time_alignment_s"]["cameras"]["trackingA"])
    nearest = min(scan, key=lambda row: abs(float(row["camera_to_imu_offset_s"]) - factory_offset))
    payload = {
        "method": "AprilTag per-tag temporal rotations versus calibrated gyro; head_pose not read",
        "episode": str(episode), "tag_family": args.tag_family, "tag_size_m": args.tag_size,
        "tag_detections": detections, "relative_rotation_pairs": len(visual), "excited_pairs": int(active.sum()),
        "factory_camera_to_imu_offset_s": factory_offset,
        "best_camera_to_imu_offset_s": best["camera_to_imu_offset_s"], "best_correlation": best["correlation"],
        "factory_nearest_correlation": nearest["correlation"],
        "extrinsic_rotation_validation": {"fitted_rotation_camera_from_imu": fitted.tolist(), "fitted_to_visual_mean_cosine": cosine, "fitted_angular_rmse_deg": rmse_deg},
        "offset_scan": scan,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "offset_scan"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
