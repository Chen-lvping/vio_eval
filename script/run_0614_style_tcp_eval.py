#!/usr/bin/env python3
"""Run the raw-pose TCP evaluation pipeline on newer episode data."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
TCP_EVAL_SCRIPT = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
VIEWER_SCRIPT = REPO_ROOT / "script/visualize_single_tcp_trajectory_3d.py"
DEFAULT_HAND_EYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"
DEFAULT_ESTIMATE_ROOTS = [
    REPO_ROOT / "data/gripper_data_1",
    REPO_ROOT / "data/gripper_data",
]
DEFAULT_GT_ROOT = REPO_ROOT / "data/ground_truth/trajectory_samples0617"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/evaluation/workbench"
AUTO_ESTIMATE_FRAME_BY_ROOT = {
    (REPO_ROOT / "data/gripper_data").resolve(): "vins_base_link",
    (REPO_ROOT / "data/gripper_data_1").resolve(): "imu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        help="Episode id like 20260617_0005 or full name like episode_20260617_0005. "
        "Used to auto-resolve estimate/ground-truth/output paths for the main 0617 raw-pose workflow.",
    )
    parser.add_argument("--estimate", type=Path, help="pose_data.csv path")
    parser.add_argument("--ground-truth", type=Path, help="trajectory_sync_rawpose_XXX.json path")
    parser.add_argument("--output-dir", type=Path, help="Output evaluation directory")
    parser.add_argument("--handeye-yaml", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument(
        "--estimate-frame",
        choices=["auto", "vins_base_link", "imu", "camera"],
        default="auto",
        help="Pose frame stored in pose_data.csv. auto selects a known 0617 convention by source directory.",
    )
    parser.add_argument(
        "--time-association",
        choices=["interpolate", "evo"],
        default="interpolate",
        help="interpolate keeps the current GT interpolation flow; evo exports raw timestamps and uses evo --t_offset/--t_max_diff.",
    )
    parser.add_argument("--max-time-gap-ms", type=float, default=80.0)
    parser.add_argument("--time-offset-sec", type=float, default=0.0)
    parser.add_argument("--t-max-diff-sec", type=float, default=0.01)
    parser.add_argument("--time-offset-auto", action="store_true")
    parser.add_argument("--time-offset-auto-span-sec", type=float, default=2.0)
    parser.add_argument("--time-offset-auto-step-sec", type=float, default=0.01)
    parser.add_argument("--time-offset-auto-score", choices=["translation", "rotation"], default="translation")
    parser.add_argument("--time-offset-auto-min-samples", type=int, default=50)
    parser.add_argument("--rpe-distance-m", type=float, default=0.05)
    parser.add_argument("--viewer-max-points", type=int, default=3000)
    parser.add_argument("--skip-viewer", action="store_true")
    return parser.parse_args()


def normalize_episode_token(value: str) -> Tuple[str, str]:
    match = re.search(r"(?:episode_)?(\d{8})_(\d{4})", value)
    if not match:
        raise ValueError(f"cannot parse episode from {value!r}")
    return match.group(1), match.group(2)


def infer_episode(args: argparse.Namespace) -> Tuple[str, str]:
    candidates = []
    if args.episode:
        candidates.append(args.episode)
    if args.estimate:
        candidates.append(str(args.estimate))
    if args.ground_truth:
        candidates.append(str(args.ground_truth))
    for candidate in candidates:
        try:
            return normalize_episode_token(candidate)
        except ValueError:
            continue
    raise ValueError("provide --episode or a path containing episode_YYYYMMDD_NNNN")


def estimate_candidates(date_str: str, seq: str) -> Iterable[Path]:
    episode_dir = f"episode_{date_str}_{seq}"
    for root in DEFAULT_ESTIMATE_ROOTS:
        yield root / episode_dir / "right" / "pose_data.csv"
        yield root / episode_dir / "pose_data" / "pose_data_right.csv"


def default_estimate_path(date_str: str, seq: str) -> Path:
    for candidate in estimate_candidates(date_str, seq):
        if candidate.is_file():
            return candidate
    tried = "\n".join(f"  - {candidate}" for candidate in estimate_candidates(date_str, seq))
    raise FileNotFoundError(f"estimate not found for episode_{date_str}_{seq}; tried:\n{tried}")


def ensure_compatibility_copy(episode_dir: Path, estimate: Path) -> Path:
    """Create pose_data/pose_data_right.csv/json if only the right/ form exists.

    Some older tools expect the pose_data/ directory layout while newer runs keep
    outputs under right/. We keep the content identical and only add a
    compatibility copy when needed.
    """

    if estimate.name != "pose_data.csv":
        return estimate

    pose_dir = episode_dir / "pose_data"
    pose_csv = pose_dir / "pose_data_right.csv"
    pose_json = pose_dir / "pose_data_right.json"
    right_csv = episode_dir / "right" / "pose_data.csv"
    right_json = episode_dir / "right" / "pose_data.json"

    if pose_csv.is_file() and pose_json.is_file():
        return pose_csv
    if not right_csv.is_file():
        return estimate

    pose_dir.mkdir(parents=True, exist_ok=True)
    if not pose_csv.is_file():
        pose_csv.write_text(right_csv.read_text(encoding="utf-8"), encoding="utf-8")
    if right_json.is_file() and not pose_json.is_file():
        pose_json.write_text(right_json.read_text(encoding="utf-8"), encoding="utf-8")
    return pose_csv if pose_csv.is_file() else estimate


def default_gt_path(seq: str) -> Path:
    stem = f"trajectory_sync_rawpose_{int(seq):03d}"
    exact = DEFAULT_GT_ROOT / f"{stem}.json"
    if exact.is_file():
        return exact
    matches = sorted(DEFAULT_GT_ROOT.glob(f"{stem}*.json"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        names = "\n".join(f"  - {path}" for path in matches)
        raise FileNotFoundError(f"multiple ground-truth files matched {stem}:\n{names}")
    raise FileNotFoundError(f"ground truth not found for {stem}")


def default_output_path(date_str: str, seq: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"evo_vio_tcp_{date_str[4:8]}_{seq}_0614mode"


def resolve_estimate_frame(estimate: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    estimate = estimate.resolve()
    for root, frame in AUTO_ESTIMATE_FRAME_BY_ROOT.items():
        try:
            estimate.relative_to(root)
            return frame
        except ValueError:
            continue
    return "vins_base_link"


def run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    date_str, seq = infer_episode(args)

    estimate = (args.estimate.expanduser().resolve() if args.estimate else default_estimate_path(date_str, seq).resolve())
    ground_truth = (
        args.ground_truth.expanduser().resolve() if args.ground_truth else default_gt_path(seq).resolve()
    )
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else default_output_path(date_str, seq).resolve()
    )
    handeye = args.handeye_yaml.expanduser().resolve()

    if not estimate.is_file():
        raise FileNotFoundError(f"estimate not found: {estimate}")
    if not ground_truth.is_file():
        raise FileNotFoundError(f"ground truth not found: {ground_truth}")
    if not handeye.is_file():
        raise FileNotFoundError(f"handeye yaml not found: {handeye}")

    if args.estimate is None:
        # Normalize the path so downstream tools see the same compatibility layout.
        episode_dir = estimate.parents[1]
        estimate = ensure_compatibility_copy(episode_dir, estimate)

    output_dir.mkdir(parents=True, exist_ok=True)
    estimate_frame = resolve_estimate_frame(estimate, args.estimate_frame)
    print(f"[INFO] estimate frame: {estimate_frame} (requested: {args.estimate_frame})", flush=True)

    eval_cmd = [
        sys.executable,
        str(TCP_EVAL_SCRIPT),
        "--estimate",
        str(estimate),
        "--ground-truth",
        str(ground_truth),
        "--output-dir",
        str(output_dir),
        "--handeye-yaml",
        str(handeye),
        "--estimate-frame",
        estimate_frame,
        "--time-association",
        args.time_association,
        "--max-time-gap-ms",
        f"{args.max_time_gap_ms}",
        "--time-offset-sec",
        f"{args.time_offset_sec}",
        "--t-max-diff-sec",
        f"{args.t_max_diff_sec}",
        "--time-offset-auto-span-sec",
        f"{args.time_offset_auto_span_sec}",
        "--time-offset-auto-step-sec",
        f"{args.time_offset_auto_step_sec}",
        "--time-offset-auto-score",
        args.time_offset_auto_score,
        "--time-offset-auto-min-samples",
        str(args.time_offset_auto_min_samples),
        "--rpe-distance-m",
        f"{args.rpe_distance_m}",
    ]
    if args.time_offset_auto:
        eval_cmd.append("--time-offset-auto")
    run_cmd(eval_cmd)

    if not args.skip_viewer:
        viewer_cmd = [
            sys.executable,
            str(VIEWER_SCRIPT),
            "--eval-dir",
            str(output_dir),
            "--output-dir",
            str(output_dir),
            "--max-points",
            str(args.viewer_max_points),
        ]
        run_cmd(viewer_cmd)

    print(f"[OK] raw-pose evaluation finished: {output_dir} (estimate_frame={estimate_frame})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
