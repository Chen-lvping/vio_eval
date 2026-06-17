#!/usr/bin/env python3
"""
Run evo-based accuracy evaluation for the corrected stereo_right/cam0 tracks.

This script reuses the existing matched trajectory logic from
evaluate_vins_accuracy.py, exports temporary TUM files with full poses, and
then runs evo_ape / evo_rpe for each algorithm.
"""

from __future__ import annotations

import argparse
import runpy
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "script/evaluate_vins_accuracy.py"
DEFAULT_HAND_EYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data/ground_truth/trajectory_samples0614/trajectory_001.json"
DEFAULT_VINS_ESTIMATE = Path("/home/chenlvping/0614 _test/episode_20260614_0239/right/pose_data.csv")
DEFAULT_VINS_CONFIG = Path("/home/chenlvping/0614 _test/episode_20260614_0239/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml")
DEFAULT_DYNAVINS_ESTIMATE = Path(
    "/home/chenlvping/0VSLAM_ws/results/0614_0239_dynavins_raw/episode_20260614_0239/right/pose_data.csv"
)
DEFAULT_DYNAVINS_CONFIG = Path(
    "/home/chenlvping/0VSLAM_ws/results/0614_0239_dynavins_raw/episode_20260614_0239/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/camera_benchmark/evo_right_cam0_0615"


def load_eval_module() -> Dict[str, object]:
    return runpy.run_path(str(EVAL_SCRIPT), run_name="__evo_eval_loader__")


def write_tum(path: Path, times, positions, rotations, rot_to_quat_xyzw) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for t, pos, rot in zip(times, positions, rotations):
            qx, qy, qz, qw = rot_to_quat_xyzw(rot)
            handle.write(
                f"{t:.9f} {pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )


def make_traj_pair(eval_api: Dict[str, object], gt_path: Path, handeye_path: Path, estimate_path: Path, estimate_frame: str, estimate_config: Path):
    args = Namespace(
        matching="timestamp",
        time_offset_sec=0.0,
        max_time_gap_ms=80.0,
        default_estimate_dt=1.0 / 30.0,
        ground_truth_frame="camera",
        robot_pose_direction="base_to_gripper",
        estimate_frame=estimate_frame,
        estimate_config=estimate_config,
    )
    gt = eval_api["load_robot_ground_truth"](gt_path, handeye_path, args.ground_truth_frame, args.robot_pose_direction)
    est = eval_api["load_estimate"](estimate_path, args.default_estimate_dt)
    if estimate_frame != "camera":
        t_est_cam0, _ = eval_api["estimate_to_camera_transform"](estimate_config, estimate_frame)
        est = eval_api["transform_trajectory"](est, t_est_cam0, "stereo_cam0", "evo export")
    gt_pos, gt_rot, est_pos, est_rot, matched_times, _ = eval_api["prepare_matches"](args, gt, est)
    if gt_rot is None or est_rot is None:
        raise RuntimeError("trajectory export needs rotations in both gt and estimate")
    return matched_times, gt_pos, gt_rot, est_pos, est_rot


def run_cmd(cmd, log_path: Path) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nsee {log_path}")
    return proc.stdout.strip()


def parse_evo_stdout(text: str) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for raw in text.splitlines():
        parts = raw.strip().split()
        if len(parts) == 2:
            key, value = parts
            try:
                stats[key] = float(value)
            except ValueError:
                pass
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--handeye", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--vins-estimate", type=Path, default=DEFAULT_VINS_ESTIMATE)
    parser.add_argument("--vins-config", type=Path, default=DEFAULT_VINS_CONFIG)
    parser.add_argument("--dynavins-estimate", type=Path, default=DEFAULT_DYNAVINS_ESTIMATE)
    parser.add_argument("--dynavins-config", type=Path, default=DEFAULT_DYNAVINS_CONFIG)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    api = load_eval_module()
    rot_to_quat_xyzw = api["rot_to_quat_xyzw"]

    algos = [
        ("vins", args.vins_estimate.expanduser().resolve(), "imu", args.vins_config.expanduser().resolve()),
        ("dynavins", args.dynavins_estimate.expanduser().resolve(), "imu", args.dynavins_config.expanduser().resolve()),
    ]

    summary_rows = []
    for name, estimate_path, estimate_frame, config_path in algos:
        matched_times, gt_pos, gt_rot, est_pos, est_rot = make_traj_pair(
            api,
            args.ground_truth.expanduser().resolve(),
            args.handeye.expanduser().resolve(),
            estimate_path,
            estimate_frame,
            config_path,
        )
        gt_tum = output_dir / f"{name}_gt.tum"
        est_tum = output_dir / f"{name}_est.tum"
        write_tum(gt_tum, matched_times, gt_pos, gt_rot, rot_to_quat_xyzw)
        write_tum(est_tum, matched_times, est_pos, est_rot, rot_to_quat_xyzw)

        evo_ape = str(Path.home() / ".local/bin/evo_ape")
        evo_rpe = str(Path.home() / ".local/bin/evo_rpe")
        common = [gt_tum.as_posix(), est_tum.as_posix(), "--no_warnings"]

        log_dir = output_dir / f"{name}_logs"
        log_dir.mkdir(exist_ok=True)

        s1 = run_cmd(
            [evo_ape, "tum", *common, "-a", "-r", "trans_part", "--change_unit", "mm", "--save_results", str(output_dir / f"{name}_ape_se3.zip")],
            log_dir / "ape_se3.log",
        )
        s2 = run_cmd(
            [evo_ape, "tum", *common, "-a", "-s", "-r", "trans_part", "--change_unit", "mm", "--save_results", str(output_dir / f"{name}_ape_sim3.zip")],
            log_dir / "ape_sim3.log",
        )
        s3 = run_cmd(
            [evo_rpe, "tum", *common, "-a", "-r", "trans_part", "-d", "0.05", "-u", "m", "--pairs_from_reference", "--change_unit", "mm", "--save_results", str(output_dir / f"{name}_rpe_5cm.zip")],
            log_dir / "rpe_5cm.log",
        )

        summary_rows.append((name, gt_tum, est_tum, s1, s2, s3))

    summary = output_dir / "evo_summary.txt"
    summary_csv = output_dir / "evo_summary.csv"
    with summary.open("w", encoding="utf-8") as handle:
        for name, gt_tum, est_tum, s1, s2, s3 in summary_rows:
            handle.write(f"[{name}]\n")
            handle.write(f"gt={gt_tum}\n")
            handle.write(f"est={est_tum}\n\n")
            handle.write("APE SE3\n")
            handle.write(s1 + "\n\n")
            handle.write("APE Sim3\n")
            handle.write(s2 + "\n\n")
            handle.write("RPE 5cm\n")
            handle.write(s3 + "\n\n")
    with summary_csv.open("w", encoding="utf-8") as handle:
        handle.write("algorithm,metric,rmse_mm,mean_mm,median_mm,min_mm,max_mm,std_mm\n")
        for name, _, _, s1, s2, s3 in summary_rows:
            for metric, stdout in [("ape_se3", s1), ("ape_sim3", s2), ("rpe_5cm", s3)]:
                stats = parse_evo_stdout(stdout)
                handle.write(
                    f"{name},{metric},{stats.get('rmse', float('nan')):.6f},"
                    f"{stats.get('mean', float('nan')):.6f},{stats.get('median', float('nan')):.6f},"
                    f"{stats.get('min', float('nan')):.6f},{stats.get('max', float('nan')):.6f},"
                    f"{stats.get('std', float('nan')):.6f}\n"
                )
    print(f"[OK] wrote {summary}")
    print(f"[OK] wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
