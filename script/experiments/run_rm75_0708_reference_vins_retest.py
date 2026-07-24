#!/usr/bin/env python3
"""Retest the 0708 IMU false-positive episodes with the proven VINS profile.

The reference VINS file has been checked against 0708's calibration: its
camera-to-IMU extrinsics match to numerical precision.  ``--vins-noise-only``
still keeps the episode calibration authoritative and only imports the
historically validated noise model.  The reference td and w7/p2 smoothing are
part of the historical successful stereo-inertial profile.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_SINGLE = ROOT / "script/run_orbslam3_tcp_eval.py"
REFERENCE_VINS = (
    ROOT
    / "data/gripper/gripper_data2/episode_20260618_0004/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml"
)
DEFAULT_OUTPUT = ROOT / "data/evaluation/workbench/rm75_0708_reference_vins_retest_20260722"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", default="0001,0004")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_summary(eval_dir: Path) -> dict[str, float]:
    with (eval_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: float(row["rmse"]) for row in csv.DictReader(handle)}


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not REFERENCE_VINS.is_file():
        raise FileNotFoundError(REFERENCE_VINS)

    rows: list[dict[str, Any]] = []
    for suffix in [value.strip() for value in args.episodes.split(",") if value.strip()]:
        episode = ROOT / f"data/0708/episode_20260708_{suffix}"
        gt = ROOT / f"data/0708/rm75_pose_traj{int(suffix):02d}.json"
        run_dir = output_root / f"episode_20260708_{suffix}_reference_vins"
        eval_dir = output_root / f"eval_episode_20260708_{suffix}"
        log_path = output_root / "logs" / f"episode_20260708_{suffix}.log"
        row: dict[str, Any] = {
            "episode": episode.name,
            "status": "running",
            "reference_vins_config": str(REFERENCE_VINS),
            "profile": "vins_noise_only + apply_vins_td + smooth_w7_p2",
            "run_dir": str(run_dir),
            "eval_dir": str(eval_dir),
        }
        try:
            if not (args.resume and (eval_dir / "summary.csv").is_file()):
                command = [
                    sys.executable, str(RUN_SINGLE),
                    "--episode-dir", str(episode), "--ground-truth", str(gt),
                    "--output-dir", str(run_dir), "--eval-dir", str(eval_dir),
                    "--camera-rig", "stereo_right", "--mode", "stereo-inertial",
                    "--feature-preset", "low-texture", "--nfeatures", "3000",
                    "--ini-fast", "12", "--min-fast", "3", "--imu-fast-init", "0",
                    "--vins-config", str(REFERENCE_VINS), "--vins-noise-only",
                    "--vins-noise-mode", "orb_from_vins", "--apply-vins-td",
                    "--smooth-window", "7", "--smooth-passes", "2",
                    "--strict-sync-offset-sec", "0", "--strict-sync-offset-scan-span-ms", "180",
                    "--strict-sync-offset-scan-step-ms", "10", "--strict-sync-offset-scan-score", "composite",
                    "--timeout-sec", str(args.timeout_sec), "--skip-viewer",
                ]
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as handle:
                    handle.write("$ " + " ".join(command) + "\n\n")
                    subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            manifest = json.loads((eval_dir / "orbslam3_tcp_eval_manifest.json").read_text(encoding="utf-8"))
            summary = read_summary(eval_dir)
            row.update(
                {
                    "status": "ok",
                    "selected_mode": manifest.get("mode"),
                    "fallback_to_stereo": manifest.get("fallback_to_stereo"),
                    "inertial_coverage": manifest.get("inertial_coverage"),
                    "strict_sync_offset_ms": float(manifest["strict_sync_offset_sec"]) * 1000.0,
                    "ape_translation_se3_mm": summary["ape_translation_se3"],
                    "rpe_translation_5cm_mm": summary["rpe_translation_5cm"],
                    "ape_rotation_se3_deg": summary["ape_rotation_se3"],
                }
            )
        except Exception as exc:
            row.update({"status": "failed", "error": str(exc), "log_path": str(log_path)})
        rows.append(row)
        with (output_root / "progress.json").open("w", encoding="utf-8") as handle:
            json.dump({"updated_at": datetime.now().isoformat(timespec="seconds"), "rows": rows}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    fields = sorted({key for row in rows for key in row})
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
