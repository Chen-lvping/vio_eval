#!/usr/bin/env python3
"""Evaluate native ORB-SLAM3 stereo trajectories for single-MCAP RM75 episodes.

GT is used only after the stereo trajectory has been generated, to select the
offline strict-sync evaluation offset.  This runner deliberately has no IMU
input or VINS calibration dependency, so it is the shadow-gate fallback
baseline for the XHT MCAP recordings.
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
    parser.add_argument("--episode-pattern", default="episode_20260708_*")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--offset-span-ms", type=float, default=180.0)
    parser.add_argument("--offset-step-ms", type=float, default=10.0)
    parser.add_argument("--camera-time-shift-sec", type=float, default=0.0)
    return parser.parse_args()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def evaluate_episode(args: argparse.Namespace, episode: Path) -> dict[str, Any]:
    root = args.output_root / episode.name
    work = root / "run"
    log = root / "episode.log"
    result: dict[str, Any] = {
        "episode": episode.name,
        "source_mcap": str(find_mcap(episode)),
        "method": "orbslam3_native_stereo_low_texture",
        "status": "running",
    }
    try:
        command = [
            "env",
            "ORB_SLAM3_FINAL_BA_ITERS=10",
            *adapter_command(episode, find_mcap(episode), work, "stereo", "native_stereo", args.timeout_sec),
            "--camera-time-shift-sec",
            str(args.camera_time_shift_sec),
        ]
        rc = append_log(log, command)
        tum = work / "f_native_stereo.txt"
        if rc != 0 or not has_rows(tum):
            raise RuntimeError(f"native stereo failed (exit={rc})")

        pose_csv = root / "pose.csv"
        rc = append_log(log, [sys.executable, str(CONVERT_TUM), str(tum), str(pose_csv)])
        if rc != 0 or not has_rows(pose_csv):
            raise RuntimeError(f"native stereo trajectory conversion failed (exit={rc})")
        best = scan_strict_sync(
            pose_csv,
            "camera",
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


def write_progress(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with (args.output_root / "native_stereo_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "updated_at": now(),
        "algorithm": "ORB-SLAM3 native stereo, low-texture feature preset",
        "camera_time_shift_sec": args.camera_time_shift_sec,
        "strict_sync_scan": {"span_ms": args.offset_span_ms, "step_ms": args.offset_step_ms},
        "episodes": rows,
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    args.episode_root = args.episode_root.expanduser().resolve()
    args.gt_root = args.gt_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    episodes = sorted(path for path in args.episode_root.glob(args.episode_pattern) if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"no episodes match {args.episode_pattern}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        rows.append(evaluate_episode(args, episode))
        write_progress(args, rows)
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
