#!/usr/bin/env python3
"""Run fixed-offset, GT-independent ablations for adaptive IMU initialization."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "script" / "run_orbslam3_tcp_eval.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--orb-root", type=Path, required=True)
    parser.add_argument("--episode-pattern", default="episode_*")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--fixed-offset-sec", type=float, default=0.0)
    return parser.parse_args()


def ground_truth_for(episode: Path, ground_truth_root: Path) -> Path:
    suffix = episode.name.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"cannot infer episode ID from {episode.name}")
    episode_id = int(suffix)
    candidates = (
        ground_truth_root / f"rm75_pose_traj{episode_id:02d}.json",
        ground_truth_root / f"rm75_pose_traj_{episode_id}.json",
        ground_truth_root / f"rm75_pose_traj{episode_id}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("missing ground truth; checked:\n" + "\n".join(map(str, candidates)))


def profiles() -> dict[str, dict[str, str]]:
    offline = {"ORB_SLAM3_OFFLINE_WAIT_LOCAL_MAPPING": "1", "ORB_SLAM3_OFFLINE_WAIT_TIMEOUT_SEC": "10"}
    return {
        "baseline": dict(offline),
        "translation": {**offline, "ORB_SLAM3_ADAPTIVE_IMU_INIT": "1", "ORB_SLAM3_ADAPTIVE_IMU_INIT_MIN_TIME_SEC": "1.5", "ORB_SLAM3_ADAPTIVE_IMU_INIT_MIN_TRANSLATION_M": "0.08", "ORB_SLAM3_ADAPTIVE_IMU_INIT_MIN_ROTATION_DEG": "1000000"},
        "rotation": {**offline, "ORB_SLAM3_ADAPTIVE_IMU_INIT": "1", "ORB_SLAM3_ADAPTIVE_IMU_INIT_MIN_TIME_SEC": "1.5", "ORB_SLAM3_ADAPTIVE_IMU_INIT_MIN_TRANSLATION_M": "1000000", "ORB_SLAM3_ADAPTIVE_IMU_INIT_MIN_ROTATION_DEG": "5.0"},
        "combined": {**offline, "ORB_SLAM3_ADAPTIVE_IMU_INIT": "1", "ORB_SLAM3_ADAPTIVE_IMU_INIT_MIN_TIME_SEC": "1.5", "ORB_SLAM3_ADAPTIVE_IMU_INIT_MIN_TRANSLATION_M": "0.08", "ORB_SLAM3_ADAPTIVE_IMU_INIT_MIN_ROTATION_DEG": "5.0"},
    }


def read_summary(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: float(row["rmse"]) for row in csv.DictReader(handle)}


def main() -> int:
    args = parse_args()
    args.episode_root = args.episode_root.expanduser().resolve()
    args.gt_root = args.gt_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.orb_root = args.orb_root.expanduser().resolve()
    episodes = sorted(path for path in args.episode_root.glob(args.episode_pattern) if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"no episodes match {args.episode_pattern!r}")
    if not RUNNER.is_file():
        raise FileNotFoundError(RUNNER)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for profile, environment in profiles().items():
        for episode in episodes:
            root = args.output_root / profile / episode.name
            run_dir, evaluation_dir, log_path = root / "run", root / "evaluation", root / "driver.log"
            root.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(RUNNER), "--episode-dir", str(episode), "--ground-truth", str(ground_truth_for(episode, args.gt_root)), "--output-dir", str(run_dir), "--eval-dir", str(evaluation_dir), "--orb-root", str(args.orb_root), "--camera-rig", "stereo_right", "--mode", "stereo-inertial", "--feature-preset", "low-texture", "--imu-fast-init", "0", "--nfeatures", "3000", "--ini-fast", "12", "--min-fast", "3", "--no-smooth-trajectory", "--strict-sync-offset-sec", str(args.fixed_offset_sec), "--timeout-sec", str(args.timeout_sec), "--skip-viewer"]
            run_environment = os.environ.copy()
            run_environment.update(environment)
            with log_path.open("w", encoding="utf-8") as handle:
                handle.write("$ " + " ".join(command) + "\n" + json.dumps(environment, sort_keys=True) + "\n\n")
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=run_environment, check=False)
            row: dict[str, object] = {"profile": profile, "episode": episode.name, "status": "ok" if result.returncode == 0 else "failed", "returncode": result.returncode, "environment": json.dumps(environment, sort_keys=True), "run_dir": str(run_dir), "evaluation_dir": str(evaluation_dir), "log_path": str(log_path)}
            summary = evaluation_dir / "summary.csv"
            if result.returncode == 0 and summary.is_file():
                row.update(read_summary(summary))
            rows.append(row)
    fields = sorted({field for row in rows for field in row})
    with (args.output_root / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "ablation_manifest.json").write_text(json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"), "fixed_offset_sec": args.fixed_offset_sec, "profiles": profiles(), "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
