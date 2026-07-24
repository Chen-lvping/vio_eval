#!/usr/bin/env python3
"""Aggregate independently-run RM75 batch shards into one auditable summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    rows = []
    for summary in sorted(root.glob("episode_*/orbslam3_rm75_batch_eval_*/batch_summary.csv")):
        with summary.open("r", newline="", encoding="utf-8") as handle:
            shard_rows = list(csv.DictReader(handle))
        if len(shard_rows) != 1:
            raise ValueError(f"{summary}: expected one row, got {len(shard_rows)}")
        row = shard_rows[0]
        manifest_path = Path(row["single_run_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        row.update(
            {
                "requested_mode": manifest.get("requested_mode", ""),
                "resolved_mode": manifest.get("mode", ""),
                "fallback_to_stereo": manifest.get("fallback_to_stereo", False),
                "fallback_reason": manifest.get("fallback_reason", ""),
                "input_frame_rows": manifest.get("input_frame_rows", ""),
                "inertial_trajectory_rows": manifest.get("inertial_trajectory_rows", ""),
                "inertial_coverage": manifest.get("inertial_coverage", ""),
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: row["episode"])
    if not rows:
        raise FileNotFoundError(f"no batch shard summaries under {root}")
    fields = list(rows[0])
    with (root / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok = [row for row in rows if row["status"] == "ok" and row["ape_translation_se3_rmse_mm"]]
    values = [float(row["ape_translation_se3_rmse_mm"]) for row in ok]
    payload = {
        "total": len(rows),
        "ok": len(ok),
        "within_10mm": sum(value <= 10.0 for value in values),
        "within_20mm": sum(value <= 20.0 for value in values),
        "fallback_to_stereo": sum(str(row["fallback_to_stereo"]).lower() == "true" for row in rows),
        "missing_episodes": [f"episode_gripper_{index:04d}" for index in range(1, 23) if f"episode_gripper_{index:04d}" not in {row["episode"] for row in rows}],
    }
    (root / "aggregate_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
