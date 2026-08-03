#!/usr/bin/env python3
"""Serially evaluate the curated RM75 clean set with the current IMU-period fix.

All episodes are recorded with the same device, so this runner deliberately
uses one shared, historically validated VINS IMU-parameter profile.  Per-
episode camera/IMU geometry is still read from each episode's calibration.json
by the exporter/config generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SINGLE = REPO_ROOT / "script" / "run_orbslam3_tcp_eval.py"
DEFAULT_BUNDLE = REPO_ROOT / "data/evaluation/curated/rm75_main_clean_ape_le_50mm_20260729"
DEFAULT_SHARED_VINS_CONFIG = (
    REPO_ROOT
    / "data/gripper/gripper_data2/episode_20260618_0004/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--timeout-sec", type=int, default=720)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--episode-pattern", default="main_*", help="Glob under bundle/episodes.")
    p.add_argument(
        "--shared-vins-config",
        type=Path,
        default=DEFAULT_SHARED_VINS_CONFIG,
        help="One historical VINS configuration supplying shared IMU noise/initialization parameters.",
    )
    p.add_argument("--allow-stereo-fallback", action="store_true", help="Permit stereo fallback; disabled by default for a native IMU comparison.")
    p.add_argument("--stereo-shadow-gate", action="store_true", help="Validate against the stereo branch and safely fall back on disagreement.")
    p.add_argument("--stereo-shadow-max-disagreement-mm", type=float, default=15.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = args.bundle.resolve()
    out = args.output_root.resolve()
    args.shared_vins_config = args.shared_vins_config.resolve()
    if not args.shared_vins_config.is_file():
        raise FileNotFoundError(f"missing shared VINS config: {args.shared_vins_config}")
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for episode_dir in sorted((bundle / "episodes").glob(args.episode_pattern)):
        provenance = json.loads((episode_dir / "provenance.json").read_text(encoding="utf-8"))
        episode = str(provenance["episode"])
        run_dir = out / episode_dir.name / "run"
        eval_dir = out / episode_dir.name / "evaluation"
        result_manifest = eval_dir / "orbslam3_tcp_eval_manifest.json"
        if args.resume and result_manifest.is_file():
            result = json.loads(result_manifest.read_text(encoding="utf-8"))
            rows.append({"episode": episode, "status": "reused", **result["summary_rmse"], "coverage": result.get("inertial_coverage", "")})
            continue
        best = provenance["configs"]["imu_ba10"]
        cmd = [
            sys.executable, str(RUN_SINGLE),
            "--episode-dir", str(provenance["raw_dataset"]),
            "--ground-truth", str(provenance["ground_truth"]),
            "--output-dir", str(run_dir), "--eval-dir", str(eval_dir),
            "--camera-rig", "stereo_right", "--mode", "stereo-inertial",
            "--feature-preset", "low-texture", "--vins-config", str(args.shared_vins_config),
            "--vins-noise-mode", "orb_from_vins", "--imu-fast-init", "0",
            "--nfeatures", "3000", "--ini-fast", "12", "--min-fast", "3",
            "--final-ba-iters", "10", "--no-offline-deterministic", "--offline-wait-timeout-sec", "10",
            "--strict-sync-offset-sec", str(float(best["best_offset_ms"]) / 1000.0),
            "--skip-viewer", "--timeout-sec", str(args.timeout_sec),
        ]
        if not args.allow_stereo_fallback:
            cmd.append("--no-stereo-fallback")
        if args.stereo_shadow_gate:
            cmd.extend(
                [
                    "--stereo-shadow-gate",
                    "--stereo-shadow-max-disagreement-mm",
                    str(args.stereo_shadow_max_disagreement_mm),
                ]
            )
        log = out / "logs" / f"{episode_dir.name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(cmd) + "\n\n")
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
        if proc.returncode == 0 and result_manifest.is_file():
            result = json.loads(result_manifest.read_text(encoding="utf-8"))
            rows.append({"episode": episode, "status": "ok", **result["summary_rmse"], "coverage": result.get("inertial_coverage", "")})
        else:
            rows.append({"episode": episode, "status": f"failed:{proc.returncode}", "ape_translation_se3_rmse": "", "rpe_translation_5cm_rmse": "", "coverage": ""})
        with (out / "progress.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["episode", "status", "ape_translation_se3_rmse", "rpe_translation_5cm_rmse", "coverage"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
    (out / "batch_manifest.json").write_text(json.dumps({"created_at": datetime.now().isoformat(), "bundle": str(bundle), "algorithm": "stereo-inertial: timestamp-guarded early-reset + frequency-aware mImuPer", "episode_pattern": args.episode_pattern, "shared_vins_config": str(args.shared_vins_config), "offline_deterministic": False, "allow_stereo_fallback": args.allow_stereo_fallback, "stereo_shadow_gate": args.stereo_shadow_gate, "episodes": len(rows)}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
