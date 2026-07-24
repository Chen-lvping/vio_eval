#!/usr/bin/env python3
"""Evaluate MCAP RM75 episodes with the validated Stereo-IMU BA10 profile.

This intentionally runs the inertial candidate directly.  It does not depend on
the colleague stereo-shadow adapter, whose MCAP input layout is incompatible.
GT is used only for offline strict-sync scoring after trajectory generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from run_rm75_0708_xht_shadow_gate import (
    CONVERT_TUM,
    adapter_command,
    append_log,
    find_mcap,
    gt_path,
    has_rows,
    scan_strict_sync,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vins-config", type=Path, required=True)
    parser.add_argument("--episode-pattern", default="episode_20260708_*")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--final-ba-iters", type=int, choices=(0, 10), default=10)
    parser.add_argument("--offset-span-ms", type=float, default=180.0)
    parser.add_argument("--offset-step-ms", type=float, default=10.0)
    parser.add_argument(
        "--use-full-vins-calibration",
        action="store_true",
        help="Use historical VINS body_T_cam and IMU noise instead of episode calibration extrinsics.",
    )
    parser.add_argument("--camera-time-shift-sec", type=float, default=0.0)
    return parser.parse_args()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_progress(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    fields = sorted({field for row in rows for field in row})
    with (output_root / "ba10_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "updated_at": now(),
        "algorithm": "ORB-SLAM3 Stereo-IMU, fastInit=0",
        "final_ba_iters": args.final_ba_iters,
        "vins_calibration_mode": "full historical" if args.use_full_vins_calibration else "noise only",
        "camera_time_shift_sec": args.camera_time_shift_sec,
        "strict_sync_scan": {"span_ms": args.offset_span_ms, "step_ms": args.offset_step_ms},
        "episodes": rows,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def evaluate_episode(args: argparse.Namespace, episode: Path) -> dict[str, Any]:
    root = args.output_root / episode.name
    work = root / "run"
    log = root / "episode.log"
    result: dict[str, Any] = {
        "episode": episode.name,
        "source_mcap": str(find_mcap(episode)),
        "method": (
            f"stereo_imu_ba{args.final_ba_iters}_vins_full"
            if args.use_full_vins_calibration
            else f"stereo_imu_ba{args.final_ba_iters}_vins_noise"
        ),
        "final_ba_iters": args.final_ba_iters,
        "status": "running",
    }
    try:
        command = [
            "env",
            f"ORB_SLAM3_FINAL_BA_ITERS={args.final_ba_iters}",
            *adapter_command(
                episode,
                find_mcap(episode),
                work,
                "stereo-inertial",
                f"inertial_ba{args.final_ba_iters}",
                args.timeout_sec,
            ),
            "--camera-time-shift-sec",
            str(args.camera_time_shift_sec),
            "--vins-config",
            str(args.vins_config),
        ]
        if not args.use_full_vins_calibration:
            command.append("--vins-noise-only")
        rc = append_log(log, command)
        tum = work / f"f_inertial_ba{args.final_ba_iters}.txt"
        if rc != 0 or not has_rows(tum):
            raise RuntimeError(f"stereo-inertial BA{args.final_ba_iters} failed (exit={rc})")

        pose_csv = root / "pose.csv"
        rc = append_log(log, [sys.executable, str(CONVERT_TUM), str(tum), str(pose_csv)])
        if rc != 0 or not has_rows(pose_csv):
            raise RuntimeError(f"BA{args.final_ba_iters} trajectory conversion failed (exit={rc})")
        best = scan_strict_sync(
            pose_csv,
            "imu",
            work / "episode_shim/calibration.json",
            gt_path(episode, args.gt_root),
            root / "evaluation",
            args.offset_span_ms,
            args.offset_step_ms,
            log,
        )
        result.update(
            {
                "status": "ok",
                "strict_sync_offset_ms": best["offset_ms"],
                "ape_translation_se3_mm": best["ape_translation_se3"],
                "rpe_translation_5cm_mm": best["rpe_translation_5cm"],
                "ape_rotation_se3_deg": best["ape_rotation_se3"],
                "rpe_rotation_5cm_deg": best["rpe_rotation_5cm"],
            }
        )
    except Exception as exc:
        result.update({"status": "error", "error": str(exc)})
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[ERROR] {exc}\n")
    return result


def main() -> int:
    args = parse_args()
    args.episode_root = args.episode_root.expanduser().resolve()
    args.gt_root = args.gt_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.vins_config = args.vins_config.expanduser().resolve()
    if not args.vins_config.is_file():
        raise FileNotFoundError(args.vins_config)
    episodes = sorted(path for path in args.episode_root.glob(args.episode_pattern) if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"no episodes match {args.episode_pattern}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        rows.append(evaluate_episode(args, episode))
        write_progress(args.output_root, rows, args)
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
