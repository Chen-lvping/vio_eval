#!/usr/bin/env python3
"""Batch-run the optimized ORB-SLAM3 TCP evaluation pipeline on RM75 datasets."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SINGLE = REPO_ROOT / "script/run_orbslam3_tcp_eval.py"
DEFAULT_EPISODE_ROOT = REPO_ROOT / "data/gripper_data2"
DEFAULT_GT_ROOT = REPO_ROOT / "data/ground_truth/rm75"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/evaluation/workbench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--camera-rig", choices=("stereo_left", "stereo_right"), default="stereo_right")
    parser.add_argument("--mode", choices=("stereo", "stereo-inertial"), default="stereo-inertial")
    parser.add_argument("--feature-preset", choices=("baseline", "low-texture", "aggressive"), default="low-texture")
    parser.add_argument("--imu-fast-init", type=int, choices=(0, 1), default=0)
    parser.add_argument("--vins-noise-mode", choices=("copy", "orb_from_vins"), default="orb_from_vins")
    parser.add_argument("--reuse-trajectory", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--viewer-max-points", type=int, default=3000)
    parser.add_argument("--timeout-sec", type=int, default=360)
    parser.add_argument(
        "--strict-sync-offset-json",
        type=Path,
        default=None,
        help="Optional JSON mapping episode dir names to strict-sync offset seconds.",
    )
    parser.add_argument(
        "--episode-pattern",
        default="episode_20260618_*",
        help="Glob under episode-root used to select episodes.",
    )
    return parser.parse_args()


def run_checked(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    content = ["$ " + " ".join(cmd), "", proc.stdout]
    if proc.stderr:
        content.extend(["[stderr]", proc.stderr])
    log_path.write_text("\n".join(content), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nsee {log_path}")


def episode_to_gt(episode_dir: Path, gt_root: Path) -> Path:
    suffix = episode_dir.name.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"cannot infer GT id from {episode_dir.name}")
    gt_path = gt_root / f"rm75_pose_traj_{int(suffix)}.json"
    if not gt_path.is_file():
        raise FileNotFoundError(f"missing GT for {episode_dir.name}: {gt_path}")
    return gt_path


def vins_config_path(episode_dir: Path, camera_rig: str) -> Path:
    rig_side = "left" if camera_rig == "stereo_left" else "right"
    path = episode_dir / rig_side / "vio_log/generated_config/StereoIMU-vinsfusion.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing VINS config: {path}")
    return path


def read_summary(eval_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with (eval_dir / "summary.csv").open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["metric"]] = float(row["rmse"])
    return out


def read_manifest(eval_dir: Path) -> dict:
    return json.loads((eval_dir / "orbslam3_tcp_eval_manifest.json").read_text(encoding="utf-8"))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_offset_overrides(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in payload.items()}


def main() -> int:
    args = parse_args()
    episode_root = args.episode_root.expanduser().resolve()
    gt_root = args.gt_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    offset_overrides = load_offset_overrides(args.strict_sync_offset_json)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = output_root / f"orbslam3_rm75_batch_eval_{stamp}"
    batch_root.mkdir(parents=True, exist_ok=True)

    episodes = sorted(path for path in episode_root.glob(args.episode_pattern) if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"no episodes matched {args.episode_pattern} under {episode_root}")

    rows: list[dict[str, object]] = []
    for episode_dir in episodes:
        gt_path = episode_to_gt(episode_dir, gt_root)
        vins_config = vins_config_path(episode_dir, args.camera_rig)
        run_dir = output_root / f"orbslam3_batch_runs_{stamp}" / (
            f"{episode_dir.name}_{args.camera_rig}_{args.mode}_{args.feature_preset}"
            f"_fastinit{args.imu_fast_init}_smooth_strictsync"
        )
        eval_dir = batch_root / f"eval_{episode_dir.name}"
        log_path = batch_root / "logs" / f"{episode_dir.name}.log"
        strict_sync_offset_sec = offset_overrides.get(episode_dir.name)

        cmd = [
            sys.executable,
            str(RUN_SINGLE),
            "--episode-dir",
            str(episode_dir),
            "--ground-truth",
            str(gt_path),
            "--output-dir",
            str(run_dir),
            "--eval-dir",
            str(eval_dir),
            "--camera-rig",
            args.camera_rig,
            "--mode",
            args.mode,
            "--feature-preset",
            args.feature_preset,
            "--vins-config",
            str(vins_config),
            "--vins-noise-mode",
            args.vins_noise_mode,
            "--imu-fast-init",
            str(args.imu_fast_init),
            "--timeout-sec",
            str(args.timeout_sec),
            "--viewer-max-points",
            str(args.viewer_max_points),
        ]
        if strict_sync_offset_sec is not None:
            cmd.extend(["--strict-sync-offset-sec", str(strict_sync_offset_sec)])
        if args.reuse_trajectory:
            cmd.append("--reuse-trajectory")
        if args.force_export:
            cmd.append("--force-export")
        if args.skip_viewer:
            cmd.append("--skip-viewer")

        run_checked(cmd, log_path)
        summary = read_summary(eval_dir)
        manifest = read_manifest(eval_dir)
        rows.append(
            {
                "episode": episode_dir.name,
                "ground_truth": gt_path.name,
                "ape_translation_se3_rmse_mm": summary["ape_translation_se3"],
                "rpe_translation_5cm_rmse_mm": summary["rpe_translation_5cm"],
                "ape_rotation_se3_rmse_deg": summary["ape_rotation_se3"],
                "rpe_rotation_5cm_rmse_deg": summary["rpe_rotation_5cm"],
                "trajectory_rows": manifest["trajectory_rows"],
                "estimate_csv_for_eval": manifest["estimate_csv_for_eval"],
                "strict_sync_offset_sec": manifest["strict_sync_offset_sec"],
                "smooth_window": manifest["smooth_window"],
                "smooth_passes": manifest["smooth_passes"],
                "eval_dir": str(eval_dir),
                "viewer_html": str(eval_dir / "index.html"),
                "log_path": str(log_path),
            }
        )

    summary_csv = batch_root / "batch_summary.csv"
    write_csv(
        summary_csv,
        [
            "episode",
            "ground_truth",
            "ape_translation_se3_rmse_mm",
            "rpe_translation_5cm_rmse_mm",
            "ape_rotation_se3_rmse_deg",
            "rpe_rotation_5cm_rmse_deg",
            "trajectory_rows",
            "estimate_csv_for_eval",
            "strict_sync_offset_sec",
            "smooth_window",
            "smooth_passes",
            "eval_dir",
            "viewer_html",
            "log_path",
        ],
        rows,
    )

    report_lines = [
        "# ORB-SLAM3 RM75 Batch Evaluation",
        "",
        f"- Episodes root: `{episode_root}`",
        f"- GT root: `{gt_root}`",
        f"- Camera rig: `{args.camera_rig}`",
        f"- Mode: `{args.mode}`",
        f"- Feature preset: `{args.feature_preset}`",
        f"- IMU fast init: `{args.imu_fast_init}`",
        f"- Strict sync offset json: `{args.strict_sync_offset_json.expanduser().resolve() if args.strict_sync_offset_json else ''}`",
        f"- Summary CSV: `{summary_csv}`",
        "",
        "| episode | GT | APE mm | RPE mm | Rot APE deg | Rot RPE deg | viewer |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['episode']} | {row['ground_truth']} | {float(row['ape_translation_se3_rmse_mm']):.6f} | "
            f"{float(row['rpe_translation_5cm_rmse_mm']):.6f} | {float(row['ape_rotation_se3_rmse_deg']):.6f} | "
            f"{float(row['rpe_rotation_5cm_rmse_deg']):.6f} | `{row['viewer_html']}` |"
        )
    (batch_root / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {summary_csv}")
    print(f"[OK] wrote {batch_root / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
