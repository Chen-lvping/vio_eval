#!/usr/bin/env python3
"""Compare a few *global* IMU-noise profiles on representative RM75 episodes.

This is a guardrail experiment, not a per-episode tuning loop.  Every
candidate uses the same shared VINS profile and the same source-level ORB
initialization fixes.  Stereo fallback is explicitly disabled so the result
measures the stereo-inertial configuration itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SINGLE = REPO_ROOT / "script/run_orbslam3_tcp_eval.py"
DEFAULT_BUNDLE = REPO_ROOT / "data/evaluation/curated/rm75_main_clean_ape_le_50mm_20260729"
DEFAULT_SHARED_VINS_CONFIG = (
    REPO_ROOT
    / "data/gripper/gripper_data2/episode_20260618_0004/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--shared-vins-config", type=Path, default=DEFAULT_SHARED_VINS_CONFIG)
    parser.add_argument("--timeout-sec", type=int, default=720)
    return parser.parse_args()


def read_base_noise(config: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    key_map = {"gyr_n": "gyro_noise", "acc_n": "acc_noise", "gyr_w": "gyro_walk", "acc_w": "acc_walk"}
    for raw_line in config.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in key_map:
            values[key_map[key]] = float(value)
    missing = {"gyro_noise", "acc_noise", "gyro_walk", "acc_walk"} - values.keys()
    if missing:
        raise ValueError(f"shared VINS config misses values: {sorted(missing)}")
    # The evaluation runner uses orb_from_vins, so discrete measurement noise
    # is divided by sqrt(IMU frequency=1024); walks are used directly.
    values["gyro_noise"] /= 32.0
    values["acc_noise"] /= 32.0
    return values


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    bundle = args.bundle.resolve()
    shared_config = args.shared_vins_config.resolve()
    if not shared_config.is_file():
        raise FileNotFoundError(shared_config)
    base_noise = read_base_noise(shared_config)
    candidates = (("baseline", 1.0), ("vision_priority_x2", 2.0), ("imu_priority_x0p5", 0.5))
    representatives = ("main_0004", "main_0010", "main_0018")
    rows: list[dict[str, object]] = []

    for candidate_name, scale in candidates:
        for episode_dir_name in representatives:
            provenance = json.loads((bundle / "episodes" / episode_dir_name / "provenance.json").read_text(encoding="utf-8"))
            best = provenance["configs"]["imu_ba10"]
            run_dir = output_root / candidate_name / episode_dir_name / "run"
            eval_dir = output_root / candidate_name / episode_dir_name / "evaluation"
            cmd = [
                sys.executable,
                str(RUN_SINGLE),
                "--episode-dir",
                str(provenance["raw_dataset"]),
                "--ground-truth",
                str(provenance["ground_truth"]),
                "--output-dir",
                str(run_dir),
                "--eval-dir",
                str(eval_dir),
                "--camera-rig",
                "stereo_right",
                "--mode",
                "stereo-inertial",
                "--feature-preset",
                "low-texture",
                "--vins-config",
                str(shared_config),
                "--vins-noise-mode",
                "orb_from_vins",
                "--imu-fast-init",
                "0",
                "--nfeatures",
                "3000",
                "--ini-fast",
                "12",
                "--min-fast",
                "3",
                "--gyro-noise",
                str(base_noise["gyro_noise"] * scale),
                "--acc-noise",
                str(base_noise["acc_noise"] * scale),
                "--gyro-walk",
                str(base_noise["gyro_walk"] * scale),
                "--acc-walk",
                str(base_noise["acc_walk"] * scale),
                "--final-ba-iters",
                "10",
                "--offline-deterministic",
                "--offline-wait-timeout-sec",
                "10",
                "--no-stereo-fallback",
                "--strict-sync-offset-sec",
                str(float(best["best_offset_ms"]) / 1000.0),
                "--skip-viewer",
                "--timeout-sec",
                str(args.timeout_sec),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            log = output_root / candidate_name / "logs" / f"{episode_dir_name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("$ " + " ".join(cmd) + "\n\n" + proc.stdout + "\n" + proc.stderr, encoding="utf-8")
            row: dict[str, object] = {"candidate": candidate_name, "noise_scale": scale, "episode": provenance["episode"], "status": "failed"}
            manifest_path = eval_dir / "orbslam3_tcp_eval_manifest.json"
            if proc.returncode == 0 and manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                row.update({"status": "ok", **manifest["summary_rmse"], "coverage": manifest.get("inertial_coverage", "")})
            rows.append(row)
            output_root.mkdir(parents=True, exist_ok=True)
            with (output_root / "progress.csv").open("w", newline="", encoding="utf-8") as handle:
                fields = ["candidate", "noise_scale", "episode", "status", "ape_translation_se3_rmse", "rpe_translation_5cm_rmse", "coverage"]
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
