#!/usr/bin/env python3
"""Batch-sweep same-episode residual correction for RM75 gripper_data2 episodes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
HAND_EYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"
APPLY_SCRIPT = REPO_ROOT / "script/apply_tcp_residual_correction.py"
EVAL_SCRIPT = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/evaluation/workbench/rm75_gripper_data2_residual_sweep"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    estimate_csv: Path
    baseline_metrics: Path


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    ground_truth: Path
    sources: Sequence[SourceSpec]


EPISODES: Sequence[EpisodeSpec] = (
    EpisodeSpec(
        episode_id="rm75_0001",
        ground_truth=REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_1.json",
        sources=(
            SourceSpec(
                name="raw",
                estimate_csv=REPO_ROOT / "data/gripper_data2/episode_20260618_0001/right/pose_data.csv",
                baseline_metrics=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0001/metrics.json",
            ),
            SourceSpec(
                name="frame",
                estimate_csv=REPO_ROOT / "data/gripper_data2/episode_20260618_0001/right/vio_log/frame_level_optimized_pose.csv",
                baseline_metrics=REPO_ROOT / "data/evaluation/workbench/rm75_frame_eval_0001/metrics.json",
            ),
        ),
    ),
    EpisodeSpec(
        episode_id="rm75_0002",
        ground_truth=REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_2.json",
        sources=(
            SourceSpec(
                name="raw",
                estimate_csv=REPO_ROOT / "data/gripper_data2/episode_20260618_0002/right/pose_data.csv",
                baseline_metrics=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0002/metrics.json",
            ),
            SourceSpec(
                name="frame",
                estimate_csv=REPO_ROOT / "data/gripper_data2/episode_20260618_0002/right/vio_log/frame_level_optimized_pose.csv",
                baseline_metrics=REPO_ROOT / "data/evaluation/workbench/rm75_frame_eval_0002/metrics.json",
            ),
        ),
    ),
    EpisodeSpec(
        episode_id="rm75_0003",
        ground_truth=REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_3.json",
        sources=(
            SourceSpec(
                name="raw",
                estimate_csv=REPO_ROOT / "data/gripper_data2/episode_20260618_0003/right/pose_data.csv",
                baseline_metrics=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0003/metrics.json",
            ),
            SourceSpec(
                name="frame",
                estimate_csv=REPO_ROOT / "data/gripper_data2/episode_20260618_0003/right/vio_log/frame_level_optimized_pose.csv",
                baseline_metrics=REPO_ROOT / "data/evaluation/workbench/rm75_frame_eval_0003/metrics.json",
            ),
        ),
    ),
    EpisodeSpec(
        episode_id="rm75_0004",
        ground_truth=REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_4.json",
        sources=(
            SourceSpec(
                name="raw",
                estimate_csv=REPO_ROOT / "data/gripper_data2/episode_20260618_0004/right/pose_data.csv",
                baseline_metrics=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0004/metrics.json",
            ),
            SourceSpec(
                name="frame",
                estimate_csv=REPO_ROOT / "data/gripper_data2/episode_20260618_0004/right/vio_log/frame_level_optimized_pose.csv",
                baseline_metrics=REPO_ROOT / "data/evaluation/workbench/rm75_frame_eval_0004/metrics.json",
            ),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=[episode.episode_id for episode in EPISODES],
        help="Subset of episode ids to process.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["raw", "frame"],
        choices=["raw", "frame"],
        help="Trajectory sources to sweep.",
    )
    parser.add_argument(
        "--knot-spacings-sec",
        nargs="*",
        type=float,
        default=[20.0, 10.0, 5.0, 2.0],
        help="Residual knot spacing sweep in seconds.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rpe-distance-m", type=float, default=0.05)
    parser.add_argument("--t-max-diff-sec", type=float, default=0.01)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_info(path: Path) -> Dict[str, float]:
    payload = load_json(path)
    evo = payload["evo"]
    return {
        "time_offset_sec": float(payload["time_offset_sec"]),
        "ape_translation_se3_rmse_mm": float(evo["ape_translation_se3"]["rmse"]),
        "ape_rotation_se3_rmse_deg": float(evo["ape_rotation_se3"]["rmse"]),
        "rpe_translation_5cm_rmse_mm": float(evo["rpe_translation_5cm"]["rmse"]),
        "rpe_rotation_5cm_rmse_deg": float(evo["rpe_rotation_5cm"]["rmse"]),
    }


def eval_info(path: Path) -> Dict[str, float]:
    payload = load_json(path)
    evo = payload["evo"]
    return {
        "time_offset_sec": float(payload["time_offset_sec"]),
        "matched_samples": int(payload["matched_samples"]),
        "ape_translation_se3_rmse_mm": float(evo["ape_translation_se3"]["rmse"]),
        "ape_rotation_se3_rmse_deg": float(evo["ape_rotation_se3"]["rmse"]),
        "ape_translation_sim3_rmse_mm": float(evo["ape_translation_sim3"]["rmse"]),
        "rpe_translation_5cm_rmse_mm": float(evo["rpe_translation_5cm"]["rmse"]),
        "rpe_rotation_5cm_rmse_deg": float(evo["rpe_rotation_5cm"]["rmse"]),
    }


def run_command(command: Sequence[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def format_knot_name(knot_spacing_sec: float) -> str:
    text = f"{knot_spacing_sec:g}".replace(".", "p")
    return f"knot_{text}s"


def candidate_rows(rows: Iterable[Dict[str, object]], episode_id: str) -> List[Dict[str, object]]:
    return [row for row in rows if row["episode_id"] == episode_id and row["kind"] == "candidate"]


def best_row(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return min(rows, key=lambda row: (float(row["ape_translation_se3_rmse_mm"]), float(row["ape_rotation_se3_rmse_deg"])))


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "episode_id",
        "kind",
        "source",
        "knot_spacing_sec",
        "time_offset_sec",
        "matched_samples",
        "ape_translation_se3_rmse_mm",
        "ape_rotation_se3_rmse_deg",
        "ape_translation_sim3_rmse_mm",
        "rpe_translation_5cm_rmse_mm",
        "rpe_rotation_5cm_rmse_deg",
        "delta_ape_mm_vs_source_baseline",
        "estimate_csv",
        "metrics_json",
        "report_md",
        "visualizer_html",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, rows: Sequence[Dict[str, object]], knot_spacings_sec: Sequence[float]) -> None:
    lines: List[str] = []
    lines.append("# RM75 gripper_data2 Residual Sweep")
    lines.append("")
    lines.append("Same-episode GT-calibrated smooth residual correction sweep over gripper_data2 RM75 episodes.")
    lines.append("")
    lines.append("## Sweep")
    lines.append("")
    lines.append(f"- Sources: {', '.join(sorted({str(row['source']) for row in rows if row['kind'] == 'candidate'}))}")
    lines.append(f"- Knot spacings (s): {', '.join(f'{value:g}' for value in knot_spacings_sec)}")
    lines.append(f"- Hand-eye: `{HAND_EYE}`")
    lines.append("")
    lines.append("## Best Per Episode")
    lines.append("")
    lines.append("| episode | baseline source | baseline APE mm | best source | knot s | best APE mm | delta mm | best eval |")
    lines.append("|---|---|---:|---|---:|---:|---:|---|")
    episode_ids = sorted({str(row["episode_id"]) for row in rows})
    for episode_id in episode_ids:
        episode_rows = [row for row in rows if row["episode_id"] == episode_id]
        baseline_rows = [row for row in episode_rows if row["kind"] == "baseline"]
        if not baseline_rows:
            continue
        best_baseline = min(baseline_rows, key=lambda row: float(row["ape_translation_se3_rmse_mm"]))
        candidates = candidate_rows(rows, episode_id)
        if not candidates:
            continue
        best_candidate = best_row(candidates)
        lines.append(
            "| "
            + " | ".join(
                [
                    episode_id,
                    str(best_baseline["source"]),
                    f"{float(best_baseline['ape_translation_se3_rmse_mm']):.3f}",
                    str(best_candidate["source"]),
                    f"{float(best_candidate['knot_spacing_sec']):g}",
                    f"{float(best_candidate['ape_translation_se3_rmse_mm']):.3f}",
                    f"{float(best_candidate['delta_ape_mm_vs_source_baseline']):+.3f}",
                    str(best_candidate["visualizer_html"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This is the same trajectory-level residual correction idea that previously pushed rm75_0004 under 10 mm.")
    lines.append("- Results here are same-episode GT-calibrated post-processing, not pure online VINS accuracy.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    selected_episodes = [episode for episode in EPISODES if episode.episode_id in set(args.episodes)]
    rows: List[Dict[str, object]] = []

    for episode in selected_episodes:
        for source in episode.sources:
            if source.name not in args.sources:
                continue
            baseline = baseline_info(source.baseline_metrics)
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "kind": "baseline",
                    "source": source.name,
                    "knot_spacing_sec": "",
                    "time_offset_sec": baseline["time_offset_sec"],
                    "matched_samples": "",
                    "ape_translation_se3_rmse_mm": baseline["ape_translation_se3_rmse_mm"],
                    "ape_rotation_se3_rmse_deg": baseline["ape_rotation_se3_rmse_deg"],
                    "ape_translation_sim3_rmse_mm": "",
                    "rpe_translation_5cm_rmse_mm": baseline["rpe_translation_5cm_rmse_mm"],
                    "rpe_rotation_5cm_rmse_deg": baseline["rpe_rotation_5cm_rmse_deg"],
                    "delta_ape_mm_vs_source_baseline": 0.0,
                    "estimate_csv": str(source.estimate_csv),
                    "metrics_json": str(source.baseline_metrics),
                    "report_md": str(source.baseline_metrics.with_name("REPORT.md")),
                    "visualizer_html": str(source.baseline_metrics.with_name("index.html")),
                }
            )

            source_root = args.output_root / episode.episode_id / source.name
            for knot_spacing_sec in args.knot_spacings_sec:
                candidate_root = source_root / format_knot_name(knot_spacing_sec)
                corrected_csv = candidate_root / "corrected_pose.csv"
                correction_report = candidate_root / "correction_report.json"
                eval_dir = candidate_root / "eval"

                run_command(
                    [
                        "python3",
                        str(APPLY_SCRIPT),
                        "--estimate",
                        str(source.estimate_csv),
                        "--ground-truth",
                        str(episode.ground_truth),
                        "--handeye-yaml",
                        str(HAND_EYE),
                        "--output-csv",
                        str(corrected_csv),
                        "--report-json",
                        str(correction_report),
                        "--estimate-frame",
                        "imu",
                        "--time-offset-sec",
                        str(baseline["time_offset_sec"]),
                        "--t-max-diff-sec",
                        str(args.t_max_diff_sec),
                        "--knot-spacing-sec",
                        str(knot_spacing_sec),
                        "--rpe-distance-m",
                        str(args.rpe_distance_m),
                    ]
                )
                run_command(
                    [
                        "python3",
                        str(EVAL_SCRIPT),
                        "--estimate",
                        str(corrected_csv),
                        "--ground-truth",
                        str(episode.ground_truth),
                        "--output-dir",
                        str(eval_dir),
                        "--handeye-yaml",
                        str(HAND_EYE),
                        "--estimate-frame",
                        "imu",
                        "--time-association",
                        "evo",
                        "--time-offset-sec",
                        str(baseline["time_offset_sec"]),
                        "--t-max-diff-sec",
                        str(args.t_max_diff_sec),
                        "--rpe-distance-m",
                        str(args.rpe_distance_m),
                    ]
                )

                metrics = eval_info(eval_dir / "metrics.json")
                rows.append(
                    {
                        "episode_id": episode.episode_id,
                        "kind": "candidate",
                        "source": source.name,
                        "knot_spacing_sec": knot_spacing_sec,
                        "time_offset_sec": metrics["time_offset_sec"],
                        "matched_samples": metrics["matched_samples"],
                        "ape_translation_se3_rmse_mm": metrics["ape_translation_se3_rmse_mm"],
                        "ape_rotation_se3_rmse_deg": metrics["ape_rotation_se3_rmse_deg"],
                        "ape_translation_sim3_rmse_mm": metrics["ape_translation_sim3_rmse_mm"],
                        "rpe_translation_5cm_rmse_mm": metrics["rpe_translation_5cm_rmse_mm"],
                        "rpe_rotation_5cm_rmse_deg": metrics["rpe_rotation_5cm_rmse_deg"],
                        "delta_ape_mm_vs_source_baseline": metrics["ape_translation_se3_rmse_mm"]
                        - baseline["ape_translation_se3_rmse_mm"],
                        "estimate_csv": str(corrected_csv),
                        "metrics_json": str(eval_dir / "metrics.json"),
                        "report_md": str(eval_dir / "REPORT.md"),
                        "visualizer_html": str(eval_dir / "index.html"),
                    }
                )

    summary_csv = args.output_root / "summary.csv"
    report_md = args.output_root / "REPORT.md"
    write_summary_csv(summary_csv, rows)
    write_report(report_md, rows, args.knot_spacings_sec)
    print(f"[OK] summary -> {summary_csv}")
    print(f"[OK] report  -> {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
