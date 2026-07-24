#!/usr/bin/env python3
"""Test old per-episode sync offsets on current colleague trajectories."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_colleague_sync")
BASE = ROOT / "data/evaluation/workbench/colleague_all_gt_20260716"
SMOOTH = REPO / "scripts/smooth_pose_csv.py"
RUNNER = REPO / "scripts/run_fays_orbslam3_stereo_right.py"
OUT = BASE / "strictsync_minimal_test"

OLD_BATCHES = {
    "0623": ROOT / "data/evaluation/workbench/orbslam3_rm75_batch_eval_20260707_112227/batch_summary.csv",
    "0624": ROOT / "data/evaluation/workbench/orbslam3_rm75_batch_eval_20260707_110304/batch_summary.csv",
    "0625": ROOT / "data/evaluation/workbench/orbslam3_rm75_batch_eval_20260707_112125/batch_summary.csv",
}
TARGETS = ["0623_0002", "0623_0004", "0624_0001", "0624_0002", "0624_0003", "0624_0004", "0624_0005", "0624_0006", "0625_0002", "0625_0003"]


def old_offsets() -> dict[str, float]:
    result = {}
    for date, path in OLD_BATCHES.items():
        for row in csv.DictReader(path.open()):
            parts = row["episode"].split("_")
            tag = parts[1][4:] + "_" + parts[2]
            if tag.startswith(date):
                result[tag] = float(row["strict_sync_offset_sec"])
    return result


def source_work_dirs() -> dict[str, Path]:
    result = {}
    for batch in [ROOT / "data/evaluation/workbench/colleague_sync_multi_20260716", ROOT / "data/evaluation/workbench/colleague_sync_multi_20260716_more"]:
        for path in sorted(batch.glob("*/evo_tcp_eval_ffba10/summary.json")):
            data = json.loads(path.read_text())
            trajectory = Path(data["inputs"]["raw_trajectory"])
            result[path.parent.parent.name] = trajectory.parent
    return result


def main() -> None:
    offsets = old_offsets()
    source_dirs = source_work_dirs()
    rows = []
    for tag in TARGETS:
        base = BASE / tag
        episode_date, number = tag.split("_")
        episode = ROOT / "data/gripper" / ({"0623": "gripper_data_6_23", "0624": "gripper_data_6_24", "0625": "gripper_data_6_25"}[episode_date]) / f"episode_2026{episode_date}_{number}"
        gt = ROOT / "data/ground_truth" / ({"0623": "rm75_6_23", "0624": "rm75_6_24", "0625": "rm75_6_25"}[episode_date]) / f"rm75_pose_traj_{int(number)}.json"
        work_source = source_dirs.get(tag, base)
        candidates = {"zero": 0.0, "old_strict_direct": offsets[tag]}
        for name, offset in candidates.items():
            work = OUT / tag / name
            work.mkdir(parents=True, exist_ok=True)
            smooth_csv = work / "pose_smooth.csv"
            eval_dir = work / "eval"
            subprocess.run(["python3", str(SMOOTH), "--input-csv", str(base / "pose_raw.csv"), "--output-csv", str(smooth_csv), "--position-window", "21", "--position-poly", "2", "--rotation-window", "9", "--timestamp-offset-sec", str(offset)], check=True)
            subprocess.run(["python3", str(RUNNER), "--evaluate-only", "--tracking-mode", "stereo", "--episode-dir", str(episode), "--work-dir", str(work_source), "--output-csv", str(smooth_csv), "--eval-dir", str(eval_dir), "--robot-json", str(gt), "--trajectory-name", f"colleague_{tag}_ffba10", "--t-max-diff", "0.01"], cwd=REPO, check=True)
            data = json.loads((eval_dir / "summary.json").read_text())
            se3 = data["results"]["se3"]["all"]
            rows.append({"episode": tag, "candidate": name, "offset_sec": offset, "ape_se3_mm": se3["trans_part"]["rmse"] * 1000, "mean_mm": se3["trans_part"]["mean"] * 1000, "pass_10mm": se3["pass_10mm_rmse"]})
            print(f"[DONE] {tag} {name} offset={offset:+.6f} APE={rows[-1]['ape_se3_mm']:.3f} mm", flush=True)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"[DONE] summary={OUT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
