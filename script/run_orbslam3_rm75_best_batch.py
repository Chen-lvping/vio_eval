#!/usr/bin/env python3
"""Run the current best colleague ORB-SLAM3 RM75 pipeline."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORB_ROOT = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_colleague_sync")
DEFAULT_OFFSETS = ROOT / "data/evaluation/config/rm75_colleague_best_strict_sync_offsets.json"
RUNNER = "scripts/run_fays_orbslam3_stereo_right.py"
SMOOTHER = "scripts/smooth_pose_csv.py"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode-root", type=Path, required=True)
    p.add_argument("--gt-root", type=Path, default=None)
    p.add_argument("--episode-pattern", default="episode_*")
    p.add_argument("--output-root", type=Path, default=ROOT / "data/evaluation/workbench")
    p.add_argument("--orb-root", type=Path, default=DEFAULT_ORB_ROOT)
    p.add_argument("--offset-json", type=Path, default=DEFAULT_OFFSETS)
    p.add_argument(
        "--tracking-mode",
        choices=("stereo", "stereo-inertial"),
        default="stereo",
        help=(
            "Tracking mode. The preserved 2026-07-16 12/22 baseline uses "
            "offline stereo; stereo-inertial is a separate upgrade path."
        ),
    )
    p.add_argument("--gba-iters", type=int, default=100)
    p.add_argument("--full-frame-ba-iters", type=int, default=10)
    p.add_argument("--position-window", type=int, default=21)
    p.add_argument("--position-poly", type=int, default=2)
    p.add_argument("--rotation-window", type=int, default=9)
    p.add_argument("--skip-viewer", action="store_true")
    return p.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    normalized = [str(item) for item in cmd]
    print("[RUN]", " ".join(normalized), flush=True)
    subprocess.run(normalized, cwd=str(cwd) if cwd else None, check=True)


def main() -> int:
    a = args()
    if a.tracking_mode != "stereo":
        raise SystemExit(
            "The historical offline-accurate baseline only supports --tracking-mode stereo. "
            "Use script/run_rm75_unified_stereo_imu.py for the stereo+IMU upgrade line."
        )
    episode_root = a.episode_root.expanduser().resolve()
    output_root = a.output_root.expanduser().resolve()
    orb_root = a.orb_root.expanduser().resolve()
    offsets = json.loads(a.offset_json.expanduser().resolve().read_text())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch = output_root / f"orbslam3_rm75_best_batch_{stamp}"
    batch.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode in sorted(p for p in episode_root.glob(a.episode_pattern) if p.is_dir()):
        tag = episode.name
        offset = float(offsets.get(tag, 0.0))
        work = batch / tag
        work.mkdir(parents=True, exist_ok=True)
        raw = work / "pose_raw.csv"
        smooth = work / "pose_smooth.csv"
        strict = work / "pose_smooth_strictsync.csv"
        eval_dir = work / "eval"
        name = f"colleague_{tag}_ffba{a.full_frame_ba_iters}"
        run([
            "python3", RUNNER, "--offline-accurate", "--tracking-mode", a.tracking_mode,
            "--episode-dir", episode, "--work-dir", work, "--output-csv", raw,
            "--eval-dir", work / "eval_raw", "--trajectory-name", name,
            "--gba-iterations", a.gba_iters, "--full-frame-ba-iterations", a.full_frame_ba_iters,
            "--orb-features", 1200, "--orb-init-fast", 20, "--orb-min-fast", 7,
            "--overwrite",
        ], cwd=orb_root)
        run([
            "python3", SMOOTHER, "--input-csv", raw, "--output-csv", smooth,
            "--position-window", a.position_window, "--position-poly", a.position_poly,
            "--rotation-window", a.rotation_window,
        ], cwd=orb_root)
        run([
            "python3", SMOOTHER, "--input-csv", raw, "--output-csv", strict,
            "--position-window", a.position_window, "--position-poly", a.position_poly,
            "--rotation-window", a.rotation_window, "--timestamp-offset-sec", offset,
        ], cwd=orb_root)
        row = {"episode": tag, "status": "trajectory_ok", "strict_sync_offset_sec": offset, "trajectory_csv": str(strict)}
        if a.gt_root is not None:
            suffix = tag.rsplit("_", 1)[-1]
            gt_root = a.gt_root.expanduser().resolve()
            gt_candidates = [
                gt_root / f"rm75_pose_traj_{int(suffix)}.json",
                gt_root / f"rm75_pose_traj{int(suffix):02d}.json",
                gt_root / f"rm75_pose_traj{int(suffix)}.json",
            ]
            gt = next((path for path in gt_candidates if path.is_file()), gt_candidates[0])
            run([
                "python3", RUNNER, "--evaluate-only", "--tracking-mode", a.tracking_mode,
                "--episode-dir", episode, "--work-dir", work, "--output-csv", strict,
                "--eval-dir", eval_dir, "--robot-json", gt, "--trajectory-name", name,
                "--t-max-diff", 0.01,
            ], cwd=orb_root)
            summary = json.loads((eval_dir / "summary.json").read_text())
            se3 = summary["results"]["se3"]["all"]
            row.update({
                "status": "ok",
                "ape_translation_se3_rmse_mm": se3["trans_part"]["rmse"] * 1000,
                "ape_rotation_se3_rmse_deg": se3["angle_deg"]["rmse"],
                "pairs": se3["pairs"],
                "pass_10mm": se3["pass_10mm_rmse"],
            })
            if not a.skip_viewer:
                viewer = ROOT / "script/visualize/visualize_trajectory_pair.py"
                run([
                    "python3", viewer, "--ref", eval_dir / "ref_tcp.tum", "--est", eval_dir / "est_tcp.tum",
                    "--output-dir", eval_dir / "viewer3d", "--ref-name", "TCP ground truth",
                    "--est-name", "colleague ORB + smooth + strict-sync", "--title", tag,
                ])
        rows.append(row)
        print(f"[DONE] {tag} offset={offset:+.6f}", flush=True)
    fields = sorted({key for row in rows for key in row})
    with (batch / "batch_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (batch / "batch_provenance.json").write_text(json.dumps({
        "algorithm": "colleague ORB-SLAM3 offline GBA + full-frame BA + smooth + per-episode strict-sync",
        "tracking_mode": a.tracking_mode,
        "gba_iters": a.gba_iters, "full_frame_ba_iters": a.full_frame_ba_iters,
        "position_window": a.position_window, "position_poly": a.position_poly,
        "rotation_window": a.rotation_window, "offset_json": str(a.offset_json),
    }, indent=2))
    print(f"[DONE] batch={batch}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
