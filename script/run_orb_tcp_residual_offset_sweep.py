#!/usr/bin/env python3
"""Sweep small fixed evo time offsets for one existing ORB TCP trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TCP_EVAL = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
VIEWER = REPO_ROOT / "script/visualize_single_tcp_trajectory_3d.py"
DEFAULT_ESTIMATE = REPO_ROOT / (
    "data/evaluation/workbench/orbslam3_fresh_runs_20260623/"
    "episode_20260618_0004_stereo_right_stereo-inertial_low-texture_fastinit0_vins_match/"
    "orb_pose_data_imu_smooth_w7_p2_ts_aligned.csv"
)
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_4.json"
DEFAULT_HAND_EYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"
DEFAULT_CALIBRATION = REPO_ROOT / "data/gripper_data2/episode_20260618_0004/calibration.json"
DEFAULT_SOURCE_RUN_DIR = REPO_ROOT / (
    "data/evaluation/workbench/orbslam3_fresh_runs_20260623/"
    "episode_20260618_0004_stereo_right_stereo-inertial_low-texture_fastinit0_vins_match"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/evaluation/workbench"
DEFAULT_ORB_REPO_ROOT = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_clean")


@dataclass(frozen=True)
class SweepRow:
    offset_ms: float
    eval_dir: Path
    ape_rmse_mm: float
    rpe_rmse_mm: float
    matched_samples: int
    matched_duration_s: float
    matched_first_s: float
    matched_last_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, default=DEFAULT_ESTIMATE)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--handeye-yaml", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--camera-rig", default="stereo_right")
    parser.add_argument("--estimate-frame", choices=("imu", "camera", "vins_base_link"), default="imu")
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--orb-log", type=Path, default=None)
    parser.add_argument("--orb-repo-root", type=Path, default=DEFAULT_ORB_REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--offsets-ms",
        default="-10,-5,-3,-2,-1,0,1,2,3,5,10",
        help="Comma-separated evo --t_offset candidates in milliseconds.",
    )
    parser.add_argument("--tag", default="ts_residual_sweep")
    parser.add_argument("--viewer-max-points", type=int, default=3000)
    parser.add_argument("--t-max-diff-sec", type=float, default=0.01)
    parser.add_argument("--rpe-distance-m", type=float, default=0.05)
    return parser.parse_args()


def parse_offsets_ms(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError("no offsets parsed from --offsets-ms")
    return values


def signed_ms_tag(offset_ms: float) -> str:
    rounded = int(round(offset_ms))
    return f"{'p' if rounded >= 0 else 'm'}{abs(rounded):03d}ms"


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    body = ["$ " + " ".join(cmd), "", proc.stdout]
    if proc.stderr:
        body.extend(["[stderr]", proc.stderr])
    log_path.write_text("\n".join(body), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nsee {log_path}")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary_rmse(summary_path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["metric"]] = float(row["rmse"])
    return result


def tum_time_range(path: Path) -> tuple[float, float]:
    first = None
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            value = float(stripped.split()[0])
            if first is None:
                first = value
            last = value
    if first is None or last is None:
        raise RuntimeError(f"no timestamps in {path}")
    return first, last


def estimate_time_stats(path: Path) -> dict[str, float]:
    times_us = []
    positions = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            times_us.append(float(row["Timestamp_us"]))
            positions.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
    if not times_us:
        raise RuntimeError(f"no estimate rows in {path}")
    pos = np.asarray(positions, dtype=float)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1) if pos.shape[0] > 1 else np.zeros(0, dtype=float)
    if pos.shape[0] >= 7:
        kernel = np.full(7, 1.0 / 7.0, dtype=float)
        padded = np.pad(pos, ((3, 3), (0, 0)), mode="edge")
        smooth = np.empty_like(pos)
        for axis in range(3):
            smooth[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
        residual = np.linalg.norm(pos - smooth, axis=1)
    else:
        residual = np.zeros(pos.shape[0], dtype=float)
    return {
        "rows": float(len(times_us)),
        "first_pose_s": times_us[0] * 1e-6,
        "last_pose_s": times_us[-1] * 1e-6,
        "duration_s": (times_us[-1] - times_us[0]) * 1e-6,
        "mean_step_mm": float(np.mean(step) * 1000.0) if step.size else 0.0,
        "p90_step_mm": float(np.percentile(step, 90) * 1000.0) if step.size else 0.0,
        "smooth_residual_mean_mm": float(np.mean(residual) * 1000.0) if residual.size else 0.0,
        "smooth_residual_p90_mm": float(np.percentile(residual, 90) * 1000.0) if residual.size else 0.0,
    }


def source_export_delay_s(source_run_dir: Path | None, first_pose_s: float) -> float | None:
    if source_run_dir is None:
        return None
    manifest_path = source_run_dir / "export_manifest.json"
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_export_s = float(payload["first_timestamp_ns"]) * 1e-9
    return first_pose_s - first_export_s


def parse_orb_log_stats(log_path: Path | None) -> dict[str, float | int]:
    if log_path is None or not log_path.is_file():
        return {
            "reset_count": 0,
            "tracking_lost_events": 0,
            "tracking_lost_frames": 0,
            "not_enough_accel_count": 0,
        }
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lost_frames = 0
    for match in re.finditer(r"(\d+)\s+Frames set to lost", text):
        lost_frames += int(match.group(1))
    return {
        "reset_count": text.count("Active map reset recieved"),
        "tracking_lost_events": len(re.findall(r"Frames set to lost", text)),
        "tracking_lost_frames": lost_frames,
        "not_enough_accel_count": text.count("not enough acceleration"),
    }


def capture_git_snapshot(repo_dir: Path, output_dir: Path, prefix: str) -> None:
    commands = {
        f"{prefix}_git_status.txt": ["git", "-C", str(repo_dir), "status", "--short"],
        f"{prefix}_git_diff.txt": ["git", "-C", str(repo_dir), "diff"],
    }
    for filename, cmd in commands.items():
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            content = proc.stdout
            if proc.stderr:
                content += ("\n" if content else "") + proc.stderr
        except Exception as exc:
            content = f"[ERROR] {exc}\n"
        (output_dir / filename).write_text(content, encoding="utf-8")


def build_eval_command(args: argparse.Namespace, output_dir: Path, offset_ms: float) -> list[str]:
    return [
        sys.executable,
        str(TCP_EVAL),
        "--estimate",
        str(args.estimate),
        "--ground-truth",
        str(args.ground_truth),
        "--output-dir",
        str(output_dir),
        "--handeye-yaml",
        str(args.handeye_yaml),
        "--calibration-json",
        str(args.calibration_json),
        "--camera-rig",
        args.camera_rig,
        "--estimate-frame",
        args.estimate_frame,
        "--time-association",
        "evo",
        "--time-offset-sec",
        f"{offset_ms * 1e-3:.9f}",
        "--t-max-diff-sec",
        str(args.t_max_diff_sec),
        "--rpe-distance-m",
        str(args.rpe_distance_m),
    ]


def build_viewer_command(eval_dir: Path, max_points: int) -> list[str]:
    return [
        sys.executable,
        str(VIEWER),
        "--eval-dir",
        str(eval_dir),
        "--output-dir",
        str(eval_dir),
        "--max-points",
        str(max_points),
    ]


def format_delta(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.3f}"


def main() -> int:
    args = parse_args()
    args.estimate = args.estimate.expanduser().resolve()
    args.ground_truth = args.ground_truth.expanduser().resolve()
    args.handeye_yaml = args.handeye_yaml.expanduser().resolve()
    args.calibration_json = args.calibration_json.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.source_run_dir = args.source_run_dir.expanduser().resolve() if args.source_run_dir else None
    args.orb_log = args.orb_log.expanduser().resolve() if args.orb_log else None
    args.orb_repo_root = args.orb_repo_root.expanduser().resolve() if args.orb_repo_root else None

    if args.orb_log is None and args.source_run_dir is not None:
        candidate = args.source_run_dir / "orbslam3_native.log"
        args.orb_log = candidate if candidate.is_file() else None

    offsets_ms = parse_offsets_ms(args.offsets_ms)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = args.output_root / f"{args.tag}_{stamp}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    capture_git_snapshot(REPO_ROOT, sweep_root, "vio_eval")
    if args.orb_repo_root is not None:
        capture_git_snapshot(args.orb_repo_root, sweep_root, "orbslam3")

    estimate_stats = estimate_time_stats(args.estimate)
    export_delay_s = source_export_delay_s(args.source_run_dir, estimate_stats["first_pose_s"])
    orb_stats = parse_orb_log_stats(args.orb_log)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "estimate": str(args.estimate),
        "ground_truth": str(args.ground_truth),
        "handeye_yaml": str(args.handeye_yaml),
        "calibration_json": str(args.calibration_json),
        "camera_rig": args.camera_rig,
        "estimate_frame": args.estimate_frame,
        "source_run_dir": str(args.source_run_dir) if args.source_run_dir else "",
        "orb_log": str(args.orb_log) if args.orb_log else "",
        "offsets_ms": offsets_ms,
        "t_max_diff_sec": float(args.t_max_diff_sec),
        "rpe_distance_m": float(args.rpe_distance_m),
        "estimate_stats": estimate_stats,
        "first_valid_pose_delay_from_export_s": export_delay_s,
        "orb_log_stats": orb_stats,
    }
    (sweep_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    sweep_rows: list[SweepRow] = []
    for offset_ms in offsets_ms:
        eval_dir = sweep_root / f"offset_{signed_ms_tag(offset_ms)}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        eval_cmd = build_eval_command(args, eval_dir, offset_ms)
        viewer_cmd = build_viewer_command(eval_dir, args.viewer_max_points)
        run_logged(eval_cmd, eval_dir / "run_eval.log")
        run_logged(viewer_cmd, eval_dir / "run_viewer.log")

        metrics = read_metrics(eval_dir / "metrics.json")
        summary = read_summary_rmse(eval_dir / "summary.csv")
        matched_first_s, matched_last_s = tum_time_range(eval_dir / "gt_tcp_matched.tum")

        (eval_dir / "command.json").write_text(
            json.dumps(
                {
                    "eval_command": eval_cmd,
                    "viewer_command": viewer_cmd,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        sweep_rows.append(
            SweepRow(
                offset_ms=float(offset_ms),
                eval_dir=eval_dir,
                ape_rmse_mm=float(summary["ape_translation_se3"]),
                rpe_rmse_mm=float(summary["rpe_translation_5cm"]),
                matched_samples=int(metrics["matched_samples"]),
                matched_duration_s=float(metrics["matched_duration_s"]),
                matched_first_s=matched_first_s,
                matched_last_s=matched_last_s,
            )
        )

        rows.append(
            {
                "offset_ms": float(offset_ms),
                "evo_t_offset_sec": float(metrics["time_offset_sec"]),
                "ape_rmse_mm": float(summary["ape_translation_se3"]),
                "rpe_rmse_mm": float(summary["rpe_translation_5cm"]),
                "matched_samples": int(metrics["matched_samples"]),
                "matched_duration_s": float(metrics["matched_duration_s"]),
                "matched_first_s": matched_first_s,
                "matched_last_s": matched_last_s,
                "first_valid_pose_s": float(estimate_stats["first_pose_s"]),
                "first_valid_pose_delay_from_export_s": export_delay_s if export_delay_s is not None else "",
                "reset_count": orb_stats["reset_count"],
                "tracking_lost_events": orb_stats["tracking_lost_events"],
                "tracking_lost_frames": orb_stats["tracking_lost_frames"],
                "not_enough_accel_count": orb_stats["not_enough_accel_count"],
                "mean_step_mm": float(estimate_stats["mean_step_mm"]),
                "p90_step_mm": float(estimate_stats["p90_step_mm"]),
                "smooth_residual_mean_mm": float(estimate_stats["smooth_residual_mean_mm"]),
                "smooth_residual_p90_mm": float(estimate_stats["smooth_residual_p90_mm"]),
                "eval_dir": str(eval_dir),
                "viewer_html": str(eval_dir / "index.html"),
            }
        )

    summary_path = sweep_root / "td_sweep_summary.csv"
    write_csv(
        summary_path,
        [
            "offset_ms",
            "evo_t_offset_sec",
            "ape_rmse_mm",
            "rpe_rmse_mm",
            "matched_samples",
            "matched_duration_s",
            "matched_first_s",
            "matched_last_s",
            "first_valid_pose_s",
            "first_valid_pose_delay_from_export_s",
            "reset_count",
            "tracking_lost_events",
            "tracking_lost_frames",
            "not_enough_accel_count",
            "mean_step_mm",
            "p90_step_mm",
            "smooth_residual_mean_mm",
            "smooth_residual_p90_mm",
            "eval_dir",
            "viewer_html",
        ],
        rows,
    )

    by_ape = min(sweep_rows, key=lambda row: (row.ape_rmse_mm, row.rpe_rmse_mm, abs(row.offset_ms)))
    zero = next((row for row in sweep_rows if abs(row.offset_ms) < 1e-9), None)
    baseline_ape = zero.ape_rmse_mm if zero is not None else float("nan")
    baseline_rpe = zero.rpe_rmse_mm if zero is not None else float("nan")
    ape_gain = baseline_ape - by_ape.ape_rmse_mm if zero is not None else float("nan")
    rpe_gain = baseline_rpe - by_ape.rpe_rmse_mm if zero is not None else float("nan")

    report_lines = [
        "# Residual Time Offset Sweep",
        "",
        "## Setup",
        "",
        f"- Estimate: `{args.estimate}`",
        f"- Ground truth: `{args.ground_truth}`",
        f"- Camera rig: `{args.camera_rig}`",
        f"- Estimate frame: `{args.estimate_frame}`",
        f"- Fixed offsets (ms): `{', '.join(str(int(v)) if float(v).is_integer() else str(v) for v in offsets_ms)}`",
        f"- Sweep summary: `{summary_path}`",
        "",
        "## Constant Inputs",
        "",
        f"- First valid pose: `{estimate_stats['first_pose_s']:.6f} s`",
        (
            f"- Delay from exported first image to first valid pose: `{export_delay_s:.6f} s`"
            if export_delay_s is not None
            else "- Delay from exported first image to first valid pose: `n/a`"
        ),
        f"- ORB reset count: `{orb_stats['reset_count']}`",
        f"- Tracking lost events: `{orb_stats['tracking_lost_events']}`",
        f"- Tracking lost frames: `{orb_stats['tracking_lost_frames']}`",
        f"- `not enough acceleration` count: `{orb_stats['not_enough_accel_count']}`",
        f"- Mean adjacent step: `{estimate_stats['mean_step_mm']:.3f} mm`",
        f"- P90 adjacent step: `{estimate_stats['p90_step_mm']:.3f} mm`",
        f"- Smooth residual mean: `{estimate_stats['smooth_residual_mean_mm']:.3f} mm`",
        f"- Smooth residual P90: `{estimate_stats['smooth_residual_p90_mm']:.3f} mm`",
        "",
        "## Best Result",
        "",
        f"- Best APE offset: `{by_ape.offset_ms:+.0f} ms`",
        f"- Best APE RMSE: `{by_ape.ape_rmse_mm:.6f} mm`",
        f"- Best RPE RMSE at that offset: `{by_ape.rpe_rmse_mm:.6f} mm`",
        (
            f"- Delta vs 0 ms: APE `{format_delta(ape_gain)} mm`, RPE `{format_delta(rpe_gain)} mm`"
            if zero is not None
            else "- Delta vs 0 ms: `n/a`"
        ),
        f"- Best viewer: `{by_ape.eval_dir / 'index.html'}`",
        "",
        "## Full Sweep",
        "",
        "| offset ms | APE RMSE mm | RPE RMSE mm | matched samples | matched duration s | viewer |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(sweep_rows, key=lambda item: item.offset_ms):
        report_lines.append(
            f"| {row.offset_ms:+.0f} | {row.ape_rmse_mm:.6f} | {row.rpe_rmse_mm:.6f} | "
            f"{row.matched_samples} | {row.matched_duration_s:.6f} | `{row.eval_dir / 'index.html'}` |"
        )

    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This sweep changes only evo timestamp association. The underlying ORB trajectory is identical in every run.",
            "- Any gain here indicates a residual fixed phase mismatch in the evaluation/alignment layer, not improved ORB tracking or initialization.",
            "- If the best point is within a couple of milliseconds of zero and the gain is tiny, the current `ts_aligned` export is already close enough.",
        ]
    )
    (sweep_root / "td_sweep_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {sweep_root / 'td_sweep_report.md'}")
    print(f"[OK] wrote {sweep_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
