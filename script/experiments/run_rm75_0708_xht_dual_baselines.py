#!/usr/bin/env python3
"""Evaluate each single-MCAP RM75 episode with separate stereo BA10 and IMU BA0 runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from run_rm75_0708_xht_shadow_gate import (
    ADAPTER,
    CONVERT_TUM,
    append_log,
    adapter_command,
    find_mcap,
    gt_path,
    has_rows,
    run_ba10_shadow,
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
    parser.add_argument("--offset-span-ms", type=float, default=180.0)
    parser.add_argument("--offset-step-ms", type=float, default=10.0)
    return parser.parse_args()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def evaluate_episode(args: argparse.Namespace, episode: Path) -> dict[str, Any]:
    root = args.output_root / episode.name
    work = root / "stereo_imu_ba0_vins_noise" / "run"
    log = root / "episode.log"
    mcap = find_mcap(episode)
    result: dict[str, Any] = {"episode": episode.name, "status": "running", "source_mcap": str(mcap)}
    try:
        imu_cmd = adapter_command(episode, mcap, work, "stereo-inertial", "inertial", args.timeout_sec)
        imu_cmd.extend(["--vins-config", str(args.vins_config), "--vins-noise-only"])
        rc = append_log(log, imu_cmd)
        imu_tum = work / "f_inertial.txt"
        if rc != 0 or not has_rows(imu_tum):
            raise RuntimeError(f"stereo-inertial BA0 failed (exit={rc})")

        calibration = work / "episode_shim/calibration.json"
        imu_csv = root / "stereo_imu_ba0_vins_noise" / "pose.csv"
        rc = append_log(log, [sys.executable, str(CONVERT_TUM), str(imu_tum), str(imu_csv)])
        if rc != 0 or not has_rows(imu_csv):
            raise RuntimeError(f"inertial trajectory conversion failed (exit={rc})")
        imu_best = scan_strict_sync(
            imu_csv, "imu", calibration, gt_path(episode, args.gt_root),
            root / "stereo_imu_ba0_vins_noise/evaluation", args.offset_span_ms, args.offset_step_ms, log,
        )

        _stereo_tum, stereo_csv, _consistency = run_ba10_shadow(episode, work, log)
        stereo_best = scan_strict_sync(
            stereo_csv, "camera", calibration, gt_path(episode, args.gt_root),
            root / "stereo_ba10/evaluation", args.offset_span_ms, args.offset_step_ms, log,
        )
        result.update({
            "status": "ok",
            "stereo_ba10_ape_mm": stereo_best["ape_translation_se3"],
            "stereo_ba10_rpe_mm": stereo_best["rpe_translation_5cm"],
            "stereo_ba10_offset_ms": stereo_best["offset_ms"],
            "stereo_imu_ba0_ape_mm": imu_best["ape_translation_se3"],
            "stereo_imu_ba0_rpe_mm": imu_best["rpe_translation_5cm"],
            "stereo_imu_ba0_offset_ms": imu_best["offset_ms"],
        })
    except Exception as exc:
        result.update({"status": "error", "error": str(exc)})
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[ERROR] {exc}\n")
    return result


def write_progress(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with (args.output_root / "dual_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "run_manifest.json").write_text(json.dumps({
        "updated_at": now(),
        "algorithms": {
            "stereo_ba10": "offline stereo, GBA=100, final BA=10, smoothing 21/2/9",
            "stereo_imu_ba0_vins_noise": "ORB stereo-inertial BA0, fastInit=0, VINS noise only",
        },
        "vins_config": str(args.vins_config),
        "strict_sync_scan": {"span_ms": args.offset_span_ms, "step_ms": args.offset_step_ms},
        "episodes": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.episode_root = args.episode_root.expanduser().resolve()
    args.gt_root = args.gt_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.vins_config = args.vins_config.expanduser().resolve()
    if not ADAPTER.is_file() or not args.vins_config.is_file():
        raise FileNotFoundError("missing MCAP adapter or VINS noise configuration")
    episodes = sorted(path for path in args.episode_root.glob(args.episode_pattern) if path.is_dir())
    if not episodes:
        raise FileNotFoundError("no episodes matched")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        rows.append(evaluate_episode(args, episode))
        write_progress(args, rows)
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
