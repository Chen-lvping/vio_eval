#!/usr/bin/env python3
"""Complete current colleague ORB + smoothing evaluation for GT-backed episodes."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_colleague_sync")
OUT = ROOT / "data/evaluation/workbench/colleague_all_gt_20260716"
SMOOTH = REPO / "scripts/smooth_pose_csv.py"
RUNNER = REPO / "scripts/run_fays_orbslam3_stereo_right.py"
VIEWER = ROOT / "script/visualize/visualize_trajectory_pair.py"


def existing_jobs() -> list[dict]:
    jobs = []
    for batch in [
        ROOT / "data/evaluation/workbench/colleague_sync_multi_20260716",
        ROOT / "data/evaluation/workbench/colleague_sync_multi_20260716_more",
    ]:
        for summary in sorted(batch.glob("*/evo_tcp_eval_ffba10/summary.json")):
            data = json.loads(summary.read_text())
            inputs = data["inputs"]
            raw = Path(inputs["estimated_csv"])
            trajectory = Path(inputs["raw_trajectory"])
            jobs.append(
                {
                    "tag": summary.parent.parent.name,
                    "episode": Path(inputs["calibration_json"]).parent.parent,
                    "gt": Path(inputs["robot_json"]),
                    "raw_source": raw,
                    "work_source": trajectory.parent,
                    "trajectory_name": trajectory.stem[2:],
                }
            )
    return jobs


def all_jobs() -> list[dict]:
    jobs = {job["tag"]: job for job in existing_jobs()}
    base = ROOT / "data/gripper/gripper_data_6_24"
    gt = ROOT / "data/ground_truth/rm75_6_24"
    for number in ["0002", "0003", "0005", "0007"]:
        tag = f"0624_{number}"
        jobs[tag] = {
            "tag": tag,
            "episode": base / f"episode_20260624_{number}",
            "gt": gt / f"rm75_pose_traj_{int(number)}.json",
            "raw_source": None,
            "work_source": None,
            "trajectory_name": f"colleague_{tag}_ffba10",
        }
    return [jobs[tag] for tag in sorted(jobs)]


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in all_jobs():
        tag = job["tag"]
        work = OUT / tag
        work.mkdir(parents=True, exist_ok=True)
        raw = work / "pose_raw.csv"
        smooth = work / "pose_smooth.csv"
        eval_dir = work / "eval"
        if job["raw_source"] is None:
            subprocess.run(
                [
                    "python3", str(RUNNER), "--offline-accurate", "--tracking-mode", "stereo",
                    "--episode-dir", str(job["episode"]), "--work-dir", str(work),
                    "--output-csv", str(raw), "--eval-dir", str(work / "eval_raw"),
                    "--trajectory-name", job["trajectory_name"], "--gba-iterations", "100",
                    "--full-frame-ba-iterations", "10", "--orb-features", "1200",
                    "--orb-init-fast", "20", "--orb-min-fast", "7", "--overwrite",
                ], cwd=REPO, check=True
            )
            job["work_source"] = work
        else:
            shutil.copy2(job["raw_source"], raw)
        subprocess.run(
            [
                "python3", str(SMOOTH), "--input-csv", str(raw), "--output-csv", str(smooth),
                "--position-window", "21", "--position-poly", "2", "--rotation-window", "9",
            ], check=True
        )
        subprocess.run(
            [
                "python3", str(RUNNER), "--evaluate-only", "--tracking-mode", "stereo",
                "--episode-dir", str(job["episode"]), "--work-dir", str(job["work_source"]),
                "--output-csv", str(smooth), "--eval-dir", str(eval_dir),
                "--robot-json", str(job["gt"]), "--trajectory-name", job["trajectory_name"],
                "--t-max-diff", "0.01",
            ], cwd=REPO, check=True
        )
        subprocess.run(
            [
                "python3", str(VIEWER), "--ref", str(eval_dir / "ref_tcp.tum"),
                "--est", str(eval_dir / "est_tcp.tum"), "--output-dir", str(eval_dir / "viewer3d"),
                "--ref-name", "TCP ground truth", "--est-name", "colleague ORB + smooth",
                "--title", f"{tag}: ground truth vs current colleague ORB + smooth",
                "--max-points", "3000",
            ], check=True
        )
        data = json.loads((eval_dir / "summary.json").read_text())
        se3 = data["results"]["se3"]["all"]
        sim3 = data["results"]["sim3"]["all"]
        rows.append(
            {
                "episode": tag,
                "status": "ok",
                "ape_translation_se3_rmse_mm": se3["trans_part"]["rmse"] * 1000,
                "ape_translation_sim3_rmse_mm": sim3["trans_part"]["rmse"] * 1000,
                "ape_rotation_se3_rmse_deg": se3["angle_deg"]["rmse"],
                "pairs": se3["pairs"],
                "pass_10mm": se3["pass_10mm_rmse"],
                "viewer_html": f"{tag}/eval/viewer3d/index.html",
            }
        )
        print(f"[DONE] {tag} APE_SE3={rows[-1]['ape_translation_se3_rmse_mm']:.3f} mm", flush=True)
    with (OUT / "batch_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "batch_provenance.json").write_text(
        json.dumps({"algorithm": "colleague ORB GBA100 + full-frame BA10 + smooth(21,2)/rot(9)", "episodes": len(rows)}, indent=2)
    )
    print(f"[DONE] summary={OUT / 'batch_summary.csv'}", flush=True)


if __name__ == "__main__":
    run()
