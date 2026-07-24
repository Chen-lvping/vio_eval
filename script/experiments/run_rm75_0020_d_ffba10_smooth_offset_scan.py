#!/usr/bin/env python3
"""Evaluate the best 0020 stereo-inertial profile with BA10 and 21/2/9 smoothing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_SINGLE = ROOT / "script/run_orbslam3_tcp_eval.py"
RESAMPLE = ROOT / "script/postprocess/resample_pose_csv_to_gt_timestamps.py"
EVALUATE = ROOT / "script/evaluate_vio_tcp_camera_evo.py"
SMOOTHER = Path(
    "/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_colleague_sync/scripts/smooth_pose_csv.py"
)
EPISODE = ROOT / "data/gripper/gripper_all_seq/episode_gripper_0020"
GROUND_TRUTH = ROOT / "data/gripper/gripper_all_seq/rm75_pose_traj_20.json"
VALIDATED_VINS = (
    ROOT
    / "data/gripper/gripper_data2/episode_20260618_0004/right/vio_log/generated_config/"
    "StereoIMU-vinsfusion.yaml"
)
HAND_EYE = ROOT / "data/calibration/handeye_0615/handeye_result.yaml"

CENTER_OFFSET_SEC = -0.084884090424
REFERENCE_D_APE_MM = 18.083675
REFERENCE_STEREO_APE_MM = 6.982474


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/evaluation/workbench/rm75_0020_d_ffba10_smooth219_scan_20260722",
    )
    parser.add_argument("--scan-span-ms", type=float, default=30.0)
    parser.add_argument("--scan-step-ms", type=float, default=5.0)
    parser.add_argument("--scan-center-sec", type=float, default=CENTER_OFFSET_SEC)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument(
        "--reuse-existing-run",
        action="store_true",
        help="Reuse the existing BA10 run and merge new offset candidates into the prior scan summary.",
    )
    return parser.parse_args()


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files:\n" + "\n".join(missing))


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def read_summary(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["metric"]] = float(row["rmse"])
    return values


def scan_offsets(center_sec: float, span_ms: float, step_ms: float) -> list[float]:
    if span_ms < 0.0:
        raise ValueError("--scan-span-ms must be >= 0")
    if span_ms == 0.0:
        return [center_sec]
    if step_ms <= 0.0:
        raise ValueError("--scan-step-ms must be > 0")
    count = int(math.floor(span_ms / step_ms + 1e-9))
    return [round(center_sec + index * step_ms / 1000.0, 12) for index in range(-count, count + 1)]


def offset_tag(offset_sec: float) -> str:
    value_us = int(round(offset_sec * 1_000_000.0))
    return ("p" if value_us >= 0 else "m") + f"{abs(value_us):06d}us"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    run_dir = output_root / "run"
    raw_eval_dir = output_root / "raw_center_eval"
    scan_root = output_root / "offset_scan"
    output_root.mkdir(parents=True, exist_ok=True)
    scan_root.mkdir(parents=True, exist_ok=True)

    require_files(
        [
            RUN_SINGLE,
            RESAMPLE,
            EVALUATE,
            SMOOTHER,
            EPISODE / "calibration.json",
            GROUND_TRUTH,
            VALIDATED_VINS,
            HAND_EYE,
        ]
    )

    manifest_path = raw_eval_dir / "orbslam3_tcp_eval_manifest.json"
    if not args.reuse_existing_run:
        run(
            [
                sys.executable,
                str(RUN_SINGLE),
                "--episode-dir",
                str(EPISODE),
                "--ground-truth",
                str(GROUND_TRUTH),
                "--output-dir",
                str(run_dir),
                "--eval-dir",
                str(raw_eval_dir),
                "--camera-rig",
                "stereo_right",
                "--mode",
                "stereo-inertial",
                "--feature-preset",
                "low-texture",
                "--vins-config",
                str(VALIDATED_VINS),
                "--vins-noise-mode",
                "orb_from_vins",
                "--nfeatures",
                "3000",
                "--ini-fast",
                "12",
                "--min-fast",
                "3",
                "--imu-fast-init",
                "0",
                "--min-inertial-coverage",
                "0.25",
                "--no-smooth-trajectory",
                "--strict-sync-offset-sec",
                str(CENTER_OFFSET_SEC),
                "--final-ba-iters",
                "10",
                "--timeout-sec",
                str(args.timeout_sec),
                "--no-offline-deterministic",
                "--skip-viewer",
                "--force-export",
            ]
        )
    elif not manifest_path.is_file():
        raise FileNotFoundError(f"cannot reuse missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "stereo-inertial" or manifest.get("fallback_to_stereo"):
        raise RuntimeError(
            f"expected genuine stereo-inertial output, got mode={manifest.get('mode')} "
            f"fallback={manifest.get('fallback_to_stereo')}"
        )

    raw_pose_csv = Path(manifest["pose_csv"])
    smooth_pose_csv = run_dir / "orb_pose_data_imu_savgol_pos21_poly2_rot9.csv"
    if not args.reuse_existing_run or not smooth_pose_csv.is_file():
        run(
            [
                sys.executable,
                str(SMOOTHER),
                "--input-csv",
                str(raw_pose_csv),
                "--output-csv",
                str(smooth_pose_csv),
                "--position-window",
                "21",
                "--position-poly",
                "2",
                "--rotation-window",
                "9",
            ],
            cwd=SMOOTHER.parent.parent,
        )

    resolved_gt = Path(manifest["ground_truth"])
    resolved_episode = Path(manifest["episode_dir"])
    scan_csv = output_root / "offset_scan_summary.csv"
    rows: list[dict[str, object]] = []
    if args.reuse_existing_run and scan_csv.is_file():
        with scan_csv.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    recovered_offsets = {round(float(row["offset_sec"]), 12) for row in rows}
    if args.reuse_existing_run:
        for candidate_dir in sorted(scan_root.iterdir()):
            if not candidate_dir.is_dir():
                continue
            eval_summary = candidate_dir / "eval/summary.csv"
            manifests = sorted(candidate_dir.glob("pose_smooth219_strictsync_*.manifest.json"))
            if not eval_summary.is_file() or not manifests:
                continue
            resample_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            offset_sec = round(float(resample_manifest["time_offset_sec"]), 12)
            if offset_sec in recovered_offsets:
                continue
            summary = read_summary(eval_summary)
            strict_csv = Path(resample_manifest["output_csv"])
            rows.append(
                {
                    "offset_sec": offset_sec,
                    "offset_ms": offset_sec * 1000.0,
                    "ape_translation_se3_rmse_mm": summary["ape_translation_se3"],
                    "rpe_translation_5cm_rmse_mm": summary["rpe_translation_5cm"],
                    "ape_rotation_se3_rmse_deg": summary["ape_rotation_se3"],
                    "rpe_rotation_5cm_rmse_deg": summary["rpe_rotation_5cm"],
                    "ape_translation_sim3_rmse_mm": summary["ape_translation_sim3"],
                    "eval_dir": str(candidate_dir / "eval"),
                    "strict_sync_csv": str(strict_csv),
                }
            )
            recovered_offsets.add(offset_sec)
            print(f"[RECOVER] offset={offset_sec * 1000.0:+.3f} ms", flush=True)
    completed_offsets = {round(float(row["offset_sec"]), 12) for row in rows}
    for offset_sec in scan_offsets(args.scan_center_sec, args.scan_span_ms, args.scan_step_ms):
        if round(offset_sec, 12) in completed_offsets:
            print(f"[SKIP] offset={offset_sec * 1000.0:+.3f} ms already evaluated", flush=True)
            continue
        tag = offset_tag(offset_sec)
        candidate_dir = scan_root / tag
        candidate_dir.mkdir(parents=True, exist_ok=True)
        strict_csv = candidate_dir / f"pose_smooth219_strictsync_{tag}.csv"
        eval_dir = candidate_dir / "eval"
        run(
            [
                sys.executable,
                str(RESAMPLE),
                "--estimate-csv",
                str(smooth_pose_csv),
                "--gt-json",
                str(resolved_gt),
                "--output-csv",
                str(strict_csv),
                "--time-offset-sec",
                str(offset_sec),
                "--max-gap-sec",
                "0.05",
            ]
        )
        run(
            [
                sys.executable,
                str(EVALUATE),
                "--estimate",
                str(strict_csv),
                "--ground-truth",
                str(resolved_gt),
                "--output-dir",
                str(eval_dir),
                "--handeye-yaml",
                str(HAND_EYE),
                "--calibration-json",
                str(resolved_episode / "calibration.json"),
                "--camera-rig",
                "stereo_right",
                "--estimate-frame",
                "imu",
                "--time-association",
                "evo",
                "--rpe-distance-m",
                "0.001",
                "--time-offset-sec",
                "0.0",
                "--t-max-diff-sec",
                "0.0001",
            ]
        )
        summary = read_summary(eval_dir / "summary.csv")
        rows.append(
            {
                "offset_sec": offset_sec,
                "offset_ms": offset_sec * 1000.0,
                "ape_translation_se3_rmse_mm": summary["ape_translation_se3"],
                "rpe_translation_5cm_rmse_mm": summary["rpe_translation_5cm"],
                "ape_rotation_se3_rmse_deg": summary["ape_rotation_se3"],
                "rpe_rotation_5cm_rmse_deg": summary["rpe_rotation_5cm"],
                "ape_translation_sim3_rmse_mm": summary["ape_translation_sim3"],
                "eval_dir": str(eval_dir),
                "strict_sync_csv": str(strict_csv),
            }
        )
        print(
            f"[SCAN] offset={offset_sec * 1000.0:+.3f} ms "
            f"APE={summary['ape_translation_se3']:.6f} mm "
            f"RPE={summary['rpe_translation_5cm']:.6f} mm",
            flush=True,
        )

    rows.sort(key=lambda row: float(row["offset_sec"]))
    best = min(
        rows,
        key=lambda row: (
            float(row["ape_translation_se3_rmse_mm"]),
            float(row["rpe_translation_5cm_rmse_mm"]),
            float(row["ape_rotation_se3_rmse_deg"]),
        ),
    )
    write_csv(scan_csv, rows)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "episode": str(resolved_episode),
        "ground_truth": str(resolved_gt),
        "camera_rig": "stereo_right",
        "mode": "stereo-inertial",
        "estimate_frame": "imu",
        "profile": "D_validated_tbc_validated_noise",
        "final_ba_iters": 10,
        "smoothing": {"position_window": 21, "position_poly": 2, "rotation_window": 9},
        "scan_center_offset_sec": args.scan_center_sec,
        "scan_span_ms": args.scan_span_ms,
        "scan_step_ms": args.scan_step_ms,
        "inertial_coverage": manifest.get("inertial_coverage"),
        "fallback_to_stereo": manifest.get("fallback_to_stereo"),
        "reference_d_ape_mm": REFERENCE_D_APE_MM,
        "reference_stereo_ape_mm": REFERENCE_STEREO_APE_MM,
        "best": best,
        "rows": rows,
        "run_manifest": str(manifest_path),
        "smooth_pose_csv": str(smooth_pose_csv),
    }
    summary_json = output_root / "summary.json"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# RM75 0020 D + Full BA10 + 21/2/9 + Offset Scan",
        "",
        "- Mode: genuine stereo-inertial (no stereo fallback)",
        "- Profile: validated Tbc + validated IMU noise",
        "- Backend: final full BA = 10",
        "- Smoothing: Savitzky-Golay position 21/2, rotation window 9",
        f"- Offset scan: latest center `{args.scan_center_sec * 1000.0:+.3f} ms`, "
        f"span `+/-{args.scan_span_ms:g} ms`, step `{args.scan_step_ms:g} ms`",
        f"- Previous D APE: `{REFERENCE_D_APE_MM:.6f} mm`",
        f"- Pure-stereo reference APE: `{REFERENCE_STEREO_APE_MM:.6f} mm`",
        "",
        "## Best",
        "",
        f"- Offset: `{float(best['offset_ms']):+.3f} ms`",
        f"- APE SE3: `{float(best['ape_translation_se3_rmse_mm']):.6f} mm`",
        f"- RPE: `{float(best['rpe_translation_5cm_rmse_mm']):.6f} mm`",
        f"- Rotation APE: `{float(best['ape_rotation_se3_rmse_deg']):.6f} deg`",
        f"- Beats 7 mm target: `{'yes' if float(best['ape_translation_se3_rmse_mm']) < 7.0 else 'no'}`",
        "",
        "## Scan",
        "",
        "| offset ms | APE mm | RPE mm | rot APE deg | Sim3 APE mm |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report.append(
            f"| {float(row['offset_ms']):+.3f} | "
            f"{float(row['ape_translation_se3_rmse_mm']):.6f} | "
            f"{float(row['rpe_translation_5cm_rmse_mm']):.6f} | "
            f"{float(row['ape_rotation_se3_rmse_deg']):.6f} | "
            f"{float(row['ape_translation_sim3_rmse_mm']):.6f} |"
        )
    report_path = output_root / "REPORT.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"[OK] summary: {summary_json}")
    print(f"[OK] report: {report_path}")
    print(
        f"[BEST] offset={float(best['offset_ms']):+.3f} ms "
        f"APE={float(best['ape_translation_se3_rmse_mm']):.6f} mm",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
