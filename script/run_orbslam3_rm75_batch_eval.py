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

from generate_run_provenance_log import write_batch_run_provenance


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
    parser.add_argument(
        "--vins-noise-only",
        action="store_true",
        help="Use IMU noise from --fallback-vins-config while keeping camera/IMU extrinsics from each episode calibration.",
    )
    parser.add_argument(
        "--fallback-vins-config",
        type=Path,
        default=None,
        help="Optional VINS config to use when the episode has no local generated_config.",
    )
    parser.add_argument(
        "--force-fallback-vins-config",
        action="store_true",
        help="Use one fallback VINS config for every episode to keep Tbc and IMU noise identical.",
    )
    parser.add_argument(
        "--vins-config-json",
        type=Path,
        default=None,
        help=(
            "Optional episode-name to VINS-config mapping. Unlisted episodes use their local config. "
            "This is for dataset calibration profiles, while ORB algorithm parameters remain shared."
        ),
    )
    parser.add_argument("--reuse-trajectory", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--viewer-max-points", type=int, default=3000)
    parser.add_argument("--timeout-sec", type=int, default=360)
    parser.add_argument(
        "--offline-deterministic",
        dest="offline_deterministic",
        action="store_true",
        help="Wait for LocalMapping after each frame in the offline runner.",
    )
    parser.add_argument(
        "--no-offline-deterministic",
        dest="offline_deterministic",
        action="store_false",
        help="Disable per-frame LocalMapping waits.",
    )
    parser.add_argument("--offline-wait-timeout-sec", type=float, default=10.0)
    parser.add_argument("--nfeatures", type=int, default=3000)
    parser.add_argument("--ini-fast", type=int, default=12)
    parser.add_argument("--min-fast", type=int, default=3)
    parser.add_argument("--min-inertial-coverage", type=float, default=0.25)
    parser.add_argument("--no-stereo-fallback", action="store_true")
    parser.add_argument("--stereo-shadow-gate", action="store_true")
    parser.add_argument("--stereo-shadow-max-disagreement-mm", type=float, default=15.0)
    parser.add_argument("--stereo-fallback-offset-json", type=Path, default=None)
    parser.add_argument("--gyro-noise", type=float, default=None)
    parser.add_argument("--acc-noise", type=float, default=None)
    parser.add_argument("--gyro-walk", type=float, default=None)
    parser.add_argument("--acc-walk", type=float, default=None)
    parser.add_argument("--final-ba-iters", type=int, default=0)
    parser.add_argument(
        "--camera-time-shift-sec",
        type=float,
        default=None,
        help="Optional camera timestamp shift forwarded to every single-episode export.",
    )
    parser.add_argument(
        "--apply-vins-td",
        action="store_true",
        help="Apply td from the selected VINS configuration during camera timestamp export.",
    )
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--clahe-clip-limit", type=float, default=None)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=None)
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
    parser.add_argument(
        "--strict-sync-offset-scan-span-ms",
        type=float,
        default=0.0,
        help="If > 0, auto-scan strict-sync offsets around each episode's resolved center offset.",
    )
    parser.add_argument(
        "--strict-sync-offset-scan-step-ms",
        type=float,
        default=0.0,
        help="Step size in milliseconds for the strict-sync offset scan.",
    )
    parser.add_argument(
        "--strict-sync-offset-scan-score",
        choices=("ape", "rpe", "rotation", "composite"),
        default="composite",
        help="Priority used to choose the best strict-sync offset candidate.",
    )
    parser.set_defaults(offline_deterministic=False)
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
    episode_id = int(suffix)
    candidates = (
        gt_root / f"rm75_pose_traj_{episode_id}.json",
        gt_root / f"rm75_pose_traj{episode_id:02d}.json",
        gt_root / f"rm75_pose_traj{episode_id}.json",
    )
    for gt_path in candidates:
        if gt_path.is_file():
            return gt_path
    raise FileNotFoundError(f"missing GT for {episode_dir.name}: {candidates[0]}")


def vins_config_path(episode_dir: Path, camera_rig: str) -> Path:
    rig_side = "left" if camera_rig == "stereo_left" else "right"
    path = episode_dir / rig_side / "vio_log/generated_config/StereoIMU-vinsfusion.yaml"
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


def load_vins_config_overrides(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    overrides: dict[str, Path] = {}
    for episode, raw_path in payload.items():
        candidate = Path(str(raw_path)).expanduser()
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"VINS config override for {episode} does not exist: {candidate}")
        overrides[str(episode)] = candidate
    return overrides


def format_optional_metric(value: object) -> str:
    if value == "":
        return "n/a"
    return f"{float(value):.6f}"


def main() -> int:
    args = parse_args()
    episode_root = args.episode_root.expanduser().resolve()
    gt_root = args.gt_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.fallback_vins_config is not None:
        args.fallback_vins_config = args.fallback_vins_config.expanduser().resolve()
    if args.vins_config_json is not None:
        args.vins_config_json = args.vins_config_json.expanduser().resolve()
    if args.stereo_fallback_offset_json is not None:
        args.stereo_fallback_offset_json = args.stereo_fallback_offset_json.expanduser().resolve()
    if args.vins_noise_only and args.fallback_vins_config is None:
        raise ValueError("--vins-noise-only requires --fallback-vins-config so every episode uses one noise profile")
    if args.force_fallback_vins_config and args.fallback_vins_config is None:
        raise ValueError("--force-fallback-vins-config requires --fallback-vins-config")
    if args.force_fallback_vins_config and args.vins_noise_only:
        raise ValueError("--force-fallback-vins-config and --vins-noise-only are mutually exclusive")
    if args.vins_config_json is not None and (args.force_fallback_vins_config or args.vins_noise_only):
        raise ValueError("--vins-config-json cannot be combined with forced/noise-only fallback modes")
    offset_overrides = load_offset_overrides(args.strict_sync_offset_json)
    vins_config_overrides = load_vins_config_overrides(args.vins_config_json)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = output_root / f"orbslam3_rm75_batch_eval_{stamp}"
    batch_root.mkdir(parents=True, exist_ok=True)

    episodes = sorted(path for path in episode_root.glob(args.episode_pattern) if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"no episodes matched {args.episode_pattern} under {episode_root}")

    rows: list[dict[str, object]] = []
    for episode_dir in episodes:
        gt_path = episode_to_gt(episode_dir, gt_root)
        local_vins_config = vins_config_path(episode_dir, args.camera_rig)
        if episode_dir.name in vins_config_overrides:
            vins_config = vins_config_overrides[episode_dir.name]
        elif args.force_fallback_vins_config or args.vins_noise_only:
            vins_config = args.fallback_vins_config
        else:
            vins_config = local_vins_config if local_vins_config.is_file() else args.fallback_vins_config
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
            "--vins-noise-mode",
            args.vins_noise_mode,
            "--imu-fast-init",
            str(args.imu_fast_init),
            "--timeout-sec",
            str(args.timeout_sec),
            "--viewer-max-points",
            str(args.viewer_max_points),
            "--min-inertial-coverage",
            str(args.min_inertial_coverage),
            "--offline-wait-timeout-sec",
            str(args.offline_wait_timeout_sec),
        ]
        cmd.append("--offline-deterministic" if args.offline_deterministic else "--no-offline-deterministic")
        if args.stereo_shadow_gate:
            cmd.extend(
                [
                    "--stereo-shadow-gate",
                    "--stereo-shadow-max-disagreement-mm",
                    str(args.stereo_shadow_max_disagreement_mm),
                ]
            )
        if args.no_stereo_fallback:
            cmd.append("--no-stereo-fallback")
        if args.stereo_fallback_offset_json is not None:
            cmd.extend(["--stereo-fallback-offset-json", str(args.stereo_fallback_offset_json)])
        if vins_config is not None and vins_config.is_file():
            cmd.extend(["--vins-config", str(vins_config)])
            if args.vins_noise_only:
                cmd.append("--vins-noise-only")
        if args.apply_vins_td:
            cmd.append("--apply-vins-td")
        for option, value in (
            ("--nfeatures", args.nfeatures),
            ("--ini-fast", args.ini_fast),
            ("--min-fast", args.min_fast),
            ("--gyro-noise", args.gyro_noise),
            ("--acc-noise", args.acc_noise),
            ("--gyro-walk", args.gyro_walk),
            ("--acc-walk", args.acc_walk),
            ("--final-ba-iters", args.final_ba_iters),
            ("--camera-time-shift-sec", args.camera_time_shift_sec),
            ("--clahe-clip-limit", args.clahe_clip_limit),
            ("--clahe-tile-grid-size", args.clahe_tile_grid_size),
        ):
            if value is not None:
                cmd.extend([option, str(value)])
        if args.clahe:
            cmd.append("--clahe")
        if strict_sync_offset_sec is not None:
            cmd.extend(["--strict-sync-offset-sec", str(strict_sync_offset_sec)])
        if float(args.strict_sync_offset_scan_span_ms) > 0.0:
            cmd.extend(
                [
                    "--strict-sync-offset-scan-span-ms",
                    str(args.strict_sync_offset_scan_span_ms),
                    "--strict-sync-offset-scan-step-ms",
                    str(args.strict_sync_offset_scan_step_ms),
                    "--strict-sync-offset-scan-score",
                    args.strict_sync_offset_scan_score,
                ]
            )
        if args.reuse_trajectory:
            cmd.append("--reuse-trajectory")
        if args.force_export:
            cmd.append("--force-export")
        if args.skip_viewer:
            cmd.append("--skip-viewer")

        row: dict[str, object] = {
            "episode": episode_dir.name,
            "ground_truth": gt_path.name,
            "status": "ok",
            "requested_mode": args.mode,
            "selected_mode": "",
            "fallback_to_stereo": "",
            "fallback_reason": "",
            "stereo_shadow_disagreement_mm": "",
            "stereo_shadow_scale": "",
            "vins_config": str(vins_config) if vins_config is not None and vins_config.is_file() else "",
            "ape_translation_se3_rmse_mm": "",
            "rpe_translation_5cm_rmse_mm": "",
            "ape_rotation_se3_rmse_deg": "",
            "rpe_rotation_5cm_rmse_deg": "",
            "trajectory_rows": "",
            "estimate_csv_for_eval": "",
            "strict_sync_offset_sec": strict_sync_offset_sec if strict_sync_offset_sec is not None else "",
            "strict_sync_offset_scan_enabled": "",
            "strict_sync_offset_scan_span_ms": "",
            "strict_sync_offset_scan_step_ms": "",
            "strict_sync_offset_scan_score": "",
            "strict_sync_offset_scan_csv": "",
            "smooth_window": "",
            "smooth_passes": "",
            "nfeatures": args.nfeatures if args.nfeatures is not None else "",
            "ini_fast": args.ini_fast if args.ini_fast is not None else "",
            "min_fast": args.min_fast if args.min_fast is not None else "",
            "gyro_noise": args.gyro_noise if args.gyro_noise is not None else "",
            "acc_noise": args.acc_noise if args.acc_noise is not None else "",
            "gyro_walk": args.gyro_walk if args.gyro_walk is not None else "",
            "acc_walk": args.acc_walk if args.acc_walk is not None else "",
            "final_ba_iters": args.final_ba_iters if args.final_ba_iters is not None else "",
            "clahe": int(args.clahe),
            "eval_dir": str(eval_dir),
            "single_run_manifest": str(eval_dir / "orbslam3_tcp_eval_manifest.json"),
            "viewer_html": str(eval_dir / "index.html"),
            "log_path": str(log_path),
            "error": "",
        }
        try:
            run_checked(cmd, log_path)
            summary = read_summary(eval_dir)
            manifest = read_manifest(eval_dir)
            row.update(
                {
                    "ape_translation_se3_rmse_mm": summary["ape_translation_se3"],
                    "rpe_translation_5cm_rmse_mm": summary["rpe_translation_5cm"],
                    "ape_rotation_se3_rmse_deg": summary["ape_rotation_se3"],
                    "rpe_rotation_5cm_rmse_deg": summary["rpe_rotation_5cm"],
                    "trajectory_rows": manifest["trajectory_rows"],
                    "selected_mode": manifest.get("mode", ""),
                    "fallback_to_stereo": manifest.get("fallback_to_stereo", ""),
                    "fallback_reason": manifest.get("fallback_reason", ""),
                    "stereo_shadow_disagreement_mm": manifest.get("stereo_shadow_consistency", {}).get("translation_rmse_mm", ""),
                    "stereo_shadow_scale": manifest.get("stereo_shadow_consistency", {}).get("sim3_scale_stereo_over_inertial", ""),
                    "estimate_csv_for_eval": manifest["estimate_csv_for_eval"],
                    "strict_sync_offset_sec": manifest["strict_sync_offset_sec"],
                    "strict_sync_offset_scan_enabled": manifest.get("strict_sync_offset_scan_enabled", ""),
                    "strict_sync_offset_scan_span_ms": manifest.get("strict_sync_offset_scan_span_ms", ""),
                    "strict_sync_offset_scan_step_ms": manifest.get("strict_sync_offset_scan_step_ms", ""),
                    "strict_sync_offset_scan_score": manifest.get("strict_sync_offset_scan_score", ""),
                    "strict_sync_offset_scan_csv": manifest.get("strict_sync_offset_scan_csv", ""),
                    "smooth_window": manifest["smooth_window"],
                    "smooth_passes": manifest["smooth_passes"],
                    "nfeatures": manifest.get("resolved_parameter_overrides", {}).get("nfeatures", row["nfeatures"]),
                    "ini_fast": manifest.get("resolved_parameter_overrides", {}).get("ini_fast", row["ini_fast"]),
                    "min_fast": manifest.get("resolved_parameter_overrides", {}).get("min_fast", row["min_fast"]),
                    "gyro_noise": manifest.get("resolved_parameter_overrides", {}).get("gyro_noise", row["gyro_noise"]),
                    "acc_noise": manifest.get("resolved_parameter_overrides", {}).get("acc_noise", row["acc_noise"]),
                    "gyro_walk": manifest.get("resolved_parameter_overrides", {}).get("gyro_walk", row["gyro_walk"]),
                    "acc_walk": manifest.get("resolved_parameter_overrides", {}).get("acc_walk", row["acc_walk"]),
                }
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
        rows.append(row)

    summary_csv = batch_root / "batch_summary.csv"
    write_csv(
        summary_csv,
        [
            "episode",
            "ground_truth",
            "status",
            "requested_mode",
            "selected_mode",
            "fallback_to_stereo",
            "fallback_reason",
            "stereo_shadow_disagreement_mm",
            "stereo_shadow_scale",
            "vins_config",
            "ape_translation_se3_rmse_mm",
            "rpe_translation_5cm_rmse_mm",
            "ape_rotation_se3_rmse_deg",
            "rpe_rotation_5cm_rmse_deg",
            "trajectory_rows",
            "estimate_csv_for_eval",
            "strict_sync_offset_sec",
            "strict_sync_offset_scan_enabled",
            "strict_sync_offset_scan_span_ms",
            "strict_sync_offset_scan_step_ms",
            "strict_sync_offset_scan_score",
            "strict_sync_offset_scan_csv",
            "smooth_window",
            "smooth_passes",
            "nfeatures",
            "ini_fast",
            "min_fast",
            "gyro_noise",
            "acc_noise",
            "gyro_walk",
            "acc_walk",
            "final_ba_iters",
            "clahe",
            "eval_dir",
            "single_run_manifest",
            "viewer_html",
            "log_path",
            "error",
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
        f"- nfeatures: `{args.nfeatures}`",
        f"- ini-fast / min-fast: `{args.ini_fast}` / `{args.min_fast}`",
        f"- gyro-noise / acc-noise: `{args.gyro_noise}` / `{args.acc_noise}`",
        f"- gyro-walk / acc-walk: `{args.gyro_walk}` / `{args.acc_walk}`",
        f"- final-ba-iters: `{args.final_ba_iters}`",
        f"- stereo shadow gate: `{args.stereo_shadow_gate}`",
        f"- stereo shadow max disagreement: `{args.stereo_shadow_max_disagreement_mm} mm`",
        f"- CLAHE: `{args.clahe}`",
        f"- Strict sync offset json: `{args.strict_sync_offset_json.expanduser().resolve() if args.strict_sync_offset_json else ''}`",
        f"- Strict sync offset scan span ms: `{args.strict_sync_offset_scan_span_ms}`",
        f"- Strict sync offset scan step ms: `{args.strict_sync_offset_scan_step_ms}`",
        f"- Strict sync offset scan score: `{args.strict_sync_offset_scan_score}`",
        f"- Summary CSV: `{summary_csv}`",
        f"- Run log: `{batch_root / 'RUN_LOG.md'}`",
        f"- Provenance JSON: `{batch_root / 'batch_provenance.json'}`",
        "",
        "| episode | status | selected | fallback | shadow mm | GT | APE mm | RPE mm | Rot APE deg | Rot RPE deg | strict sync | viewer |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        ape = row["ape_translation_se3_rmse_mm"]
        rpe = row["rpe_translation_5cm_rmse_mm"]
        ape_rot = row["ape_rotation_se3_rmse_deg"]
        rpe_rot = row["rpe_rotation_5cm_rmse_deg"]
        strict_sync_desc = format_optional_metric(row["strict_sync_offset_sec"])
        if row["strict_sync_offset_scan_enabled"] not in ("", False, "False", "0"):
            strict_sync_desc += " scan"
        report_lines.append(
            f"| {row['episode']} | {row['status']} | {row['selected_mode']} | {row['fallback_reason']} | "
            f"{format_optional_metric(row['stereo_shadow_disagreement_mm'])} | {row['ground_truth']} | "
            f"{format_optional_metric(ape)} | "
            f"{format_optional_metric(rpe)} | "
            f"{format_optional_metric(ape_rot)} | "
            f"{format_optional_metric(rpe_rot)} | "
            f"{strict_sync_desc} | `{row['viewer_html']}` |"
        )
        if row["error"]:
            report_lines.append(f"| | | | error | | `{row['error']}` |  |  |  |  |  | |")
    (batch_root / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    snapshot_candidates = [
        args.strict_sync_offset_json,
        args.stereo_fallback_offset_json,
        args.fallback_vins_config,
    ]
    snapshot_candidates.extend(
        Path(row["vins_config"]) for row in rows if row.get("vins_config")
    )
    write_batch_run_provenance(
        batch_root,
        label="ORB-SLAM3 RM75 Batch Evaluation",
        batch_args=args,
        rows=rows,
        summary_csv=summary_csv,
        report_md=batch_root / "REPORT.md",
        source_script=Path(__file__),
        argv=sys.argv[1:],
        copy_files=[path for path in snapshot_candidates if path],
    )

    print(f"[OK] wrote {summary_csv}")
    print(f"[OK] wrote {batch_root / 'REPORT.md'}")
    print(f"[OK] wrote {batch_root / 'RUN_LOG.md'}")
    print(f"[OK] wrote {batch_root / 'batch_provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
