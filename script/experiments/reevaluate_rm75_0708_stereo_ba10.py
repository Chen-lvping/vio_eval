#!/usr/bin/env python3
"""Formally score saved colleague BA10 stereo shadows with independent offset scans."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from run_rm75_0708_xht_shadow_gate import gt_path, scan_strict_sync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--source-batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--offset-span-ms", type=float, default=180.0)
    parser.add_argument("--offset-step-ms", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.episode_root = args.episode_root.expanduser().resolve()
    args.gt_root = args.gt_root.expanduser().resolve()
    args.source_batch_root = args.source_batch_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for episode in sorted(path for path in args.episode_root.glob("episode_20260708_*") if path.is_dir()):
        source = args.source_batch_root / f"{episode.name}_stereo_right_stereo-inertial_low-texture_fastinit0_smooth_strictsync/stereo_shadow/pose_smooth.csv"
        out = args.output_root / episode.name
        log = out / "reevaluate.log"
        row: dict[str, Any] = {"episode": episode.name, "source_pose_csv": str(source), "status": "running"}
        try:
            if not source.is_file():
                raise FileNotFoundError(source)
            best = scan_strict_sync(
                source, "camera", episode / "calibration.json", gt_path(episode, args.gt_root),
                out / "evaluation", args.offset_span_ms, args.offset_step_ms, log,
            )
            row.update({
                "status": "ok",
                "ape_translation_se3_rmse_mm": best["ape_translation_se3"],
                "rpe_translation_5cm_rmse_mm": best["rpe_translation_5cm"],
                "strict_sync_offset_ms": best["offset_ms"],
            })
        except Exception as exc:
            row.update({"status": "error", "error": str(exc)})
        rows.append(row)
        fields = sorted({key for item in rows for key in item})
        with (args.output_root / "pure_stereo_ba10_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (args.output_root / "provenance.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "colleague ORB-SLAM3 offline stereo GBA=100 + full-frame BA10 + smoothing 21/2/9",
        "source_batch_root": str(args.source_batch_root),
        "strict_sync_scan": {"span_ms": args.offset_span_ms, "step_ms": args.offset_step_ms},
    }, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
