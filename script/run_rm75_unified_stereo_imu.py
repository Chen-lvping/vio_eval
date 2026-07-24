#!/usr/bin/env python3
"""Run the RM75 set with one ORB-SLAM3 stereo-IMU algorithm configuration.

Camera intrinsics/extrinsics and strict-sync offsets remain dataset-specific
calibration inputs. Feature, IMU-noise, initialization, smoothing, backend and
fallback parameters are shared by every episode.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "script/run_orbslam3_rm75_batch_eval.py"
EPISODES = ROOT / "data/gripper/gripper_all_seq"
OFFSETS = ROOT / "data/evaluation/config/rm75_gripper_22_strict_sync_offsets.json"
STEREO_FALLBACK_OFFSETS = ROOT / "data/evaluation/config/rm75_colleague_best_strict_sync_offsets.json"
CALIBRATION_PROFILES = ROOT / "data/evaluation/config/rm75_gripper_22_vins_config_profiles.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-pattern", default="episode_gripper_*")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/evaluation/workbench")
    parser.add_argument("--timeout-sec", type=int, default=480)
    parser.add_argument("--final-ba-iters", type=int, default=0)
    parser.add_argument("--direct-imu", action="store_true", help="Disable all automatic stereo fallback for a raw stereo-IMU comparison run.")
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        str(BATCH),
        "--episode-root", str(EPISODES),
        "--gt-root", str(EPISODES),
        "--output-root", str(args.output_root.expanduser().resolve()),
        "--episode-pattern", args.episode_pattern,
        "--camera-rig", "stereo_right",
        "--mode", "stereo-inertial",
        "--feature-preset", "low-texture",
        "--nfeatures", "3000",
        "--ini-fast", "12",
        "--min-fast", "3",
        "--imu-fast-init", "0",
        "--vins-noise-mode", "orb_from_vins",
        "--vins-config-json", str(CALIBRATION_PROFILES),
        "--strict-sync-offset-json", str(OFFSETS),
        "--min-inertial-coverage", "0.0" if args.direct_imu else "0.25",
        "--final-ba-iters", str(args.final_ba_iters),
        "--timeout-sec", str(args.timeout_sec),
        "--no-offline-deterministic",
    ]
    if args.direct_imu:
        cmd.append("--no-stereo-fallback")
    else:
        cmd.extend(
            [
                "--stereo-shadow-gate",
                "--stereo-shadow-max-disagreement-mm", "15.0",
                "--stereo-fallback-offset-json", str(STEREO_FALLBACK_OFFSETS),
            ]
        )
    if args.skip_viewer:
        cmd.append("--skip-viewer")
    print("[UNIFIED]", " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
