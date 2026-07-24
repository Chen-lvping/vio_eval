#!/usr/bin/env python3
"""Sequentially rerun selected RM75 stereo-inertial episodes with final BA=10."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_SINGLE = ROOT / "script/run_orbslam3_tcp_eval.py"
EVALUATOR = ROOT / "script/evaluate_vio_tcp_camera_evo.py"
RESAMPLER = ROOT / "script/postprocess/resample_pose_csv_to_gt_timestamps.py"
SCREEN_MODULE = ROOT / "script/experiments/optimize_rm75_22_hybrid_postprocess.py"
SCREEN_ROOT = ROOT / "data/evaluation/workbench/rm75_22_hybrid_postprocess_final_20260722"
SCREEN_CSV = SCREEN_ROOT / "postprocess_screening.csv"
SHADOW_ROOT = (
    ROOT
    / "data/evaluation/workbench/rm75_stereo_imu_shadow_gate_full_20260721"
    / "orbslam3_rm75_batch_eval_20260721_213748"
)
HAND_EYE = ROOT / "data/calibration/handeye_0615/handeye_result.yaml"
SMOOTHER = Path(
    "/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_colleague_sync/scripts/smooth_pose_csv.py"
)
DEFAULT_EPISODES = [
    "episode_gripper_0003",
    "episode_gripper_0005",
    "episode_gripper_0009",
    "episode_gripper_0011",
    "episode_gripper_0019",
    "episode_gripper_0021",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "data/evaluation/workbench"
        / f"rm75_selected_si_ba10_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--scan-span-ms", type=float, default=40.0)
    parser.add_argument("--scan-step-ms", type=float, default=5.0)
    parser.add_argument("--fine-span-ms", type=float, default=8.0)
    parser.add_argument("--fine-step-ms", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def run_logged(cmd: list[str], log_path: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\n" + proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}; see {log_path}")


def read_summary(path: Path) -> dict[str, float]:
    return {row["metric"]: float(row["rmse"]) for row in read_rows(path)}


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for path in (RUN_SINGLE, EVALUATOR, RESAMPLER, SCREEN_MODULE, SCREEN_CSV, HAND_EYE, SMOOTHER):
        if not path.is_file():
            raise FileNotFoundError(path)

    screen = load_module("rm75_hybrid_screen_for_ba10", SCREEN_MODULE)
    tcp_eval = screen.load_module("rm75_tcp_eval_for_ba10", screen.TCP_EVALUATOR)
    smoother = screen.load_module("rm75_smoother_for_ba10", SMOOTHER)
    helpers = tcp_eval.load_helpers()
    t_tcp_camera = tcp_eval.load_tcp_camera_transform(HAND_EYE)

    screen_rows = {row["episode_key"]: row for row in read_rows(SCREEN_CSV)}
    manifests: dict[str, dict[str, Any]] = {}
    for path in sorted(SHADOW_ROOT.glob("eval_episode_gripper_*/orbslam3_tcp_eval_manifest.json")):
        key = path.parent.name.removeprefix("eval_")
        manifests[key] = json.loads(path.read_text(encoding="utf-8"))
    selected = args.episode or DEFAULT_EPISODES
    missing = set(selected) - set(screen_rows)
    if missing:
        raise KeyError(f"unknown episodes: {sorted(missing)}")

    results_csv = output_root / "ba10_results.csv"
    completed: dict[str, dict[str, str]] = {}
    if args.resume and results_csv.is_file():
        completed = {row["episode_key"]: row for row in read_rows(results_csv)}
    results: list[dict[str, Any]] = list(completed.values())

    for index, episode_key in enumerate(selected, start=1):
        if episode_key in completed and completed[episode_key].get("status") == "ok":
            print(f"[SKIP] {episode_key} already complete", flush=True)
            continue
        source = screen_rows[episode_key]
        source_manifest = manifests[episode_key]
        episode_dir = Path(source_manifest["episode_dir"])
        gt_path = Path(source_manifest["ground_truth"])
        vins_config = Path(source_manifest["vins_config"])
        calibration_json = episode_dir / "calibration.json"
        center_sec = float(source["si_best_offset_sec"])
        episode_out = output_root / episode_key
        run_dir = episode_out / "run"
        raw_eval_dir = episode_out / "raw_center_eval"
        run_manifest_path = raw_eval_dir / "orbslam3_tcp_eval_manifest.json"
        final_json = episode_out / "final.json"
        print(
            f"[{index}/{len(selected)}] {episode_key} -> {source['episode']} center={center_sec*1000:+.3f} ms",
            flush=True,
        )

        try:
            if not (args.resume and run_manifest_path.is_file()):
                run_logged(
                    [
                        sys.executable,
                        str(RUN_SINGLE),
                        "--episode-dir",
                        str(episode_dir),
                        "--ground-truth",
                        str(gt_path),
                        "--handeye-yaml",
                        str(HAND_EYE),
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
                        str(vins_config),
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
                        str(center_sec),
                        "--final-ba-iters",
                        "10",
                        "--timeout-sec",
                        str(args.timeout_sec),
                        "--no-offline-deterministic",
                        "--skip-viewer",
                        "--force-export",
                    ],
                    episode_out / "run_command.log",
                )

            manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if manifest.get("mode") != "stereo-inertial" or manifest.get("fallback_to_stereo"):
                raise RuntimeError(
                    f"not a genuine SI result: mode={manifest.get('mode')} fallback={manifest.get('fallback_to_stereo')}"
                )
            raw_pose_csv = Path(manifest["pose_csv"])
            smooth_csv = episode_out / "stereo_inertial_ba10_smooth_pos21_poly2_rot9.csv"
            if not (args.resume and smooth_csv.is_file()):
                screen.smooth_pose_csv(smoother, raw_pose_csv, smooth_csv)

            gt_times, gt_positions, gt_rotations = tcp_eval.load_robot_tcp_trajectory(gt_path, helpers)
            t_camera_imu, _, _ = tcp_eval.load_camera_imu_transform(calibration_json, "stereo_right")
            scan_args = SimpleNamespace(
                coarse_span_ms=args.scan_span_ms,
                coarse_step_ms=args.scan_step_ms,
                fine_span_ms=args.fine_span_ms,
                fine_step_ms=args.fine_step_ms,
                max_gap_sec=0.05,
                min_samples=50,
            )
            best, scan_rows = screen.scan_method(
                series=screen.read_pose_csv(smooth_csv),
                center_sec=center_sec,
                gt_times=gt_times,
                gt_positions=gt_positions,
                gt_rotations=gt_rotations,
                estimate_frame="imu",
                t_tcp_camera=t_tcp_camera,
                t_camera_imu=t_camera_imu,
                helpers=helpers,
                args=scan_args,
            )
            screen.write_csv(episode_out / "offset_scan.csv", scan_rows)

            strict_csv = episode_out / "stereo_inertial_ba10_best_strictsync.csv"
            run_logged(
                [
                    sys.executable,
                    str(RESAMPLER),
                    "--estimate-csv",
                    str(smooth_csv),
                    "--gt-json",
                    str(gt_path),
                    "--output-csv",
                    str(strict_csv),
                    "--time-offset-sec",
                    str(best["offset_sec"]),
                    "--max-gap-sec",
                    "0.05",
                ],
                episode_out / "resample.log",
            )
            final_eval_dir = episode_out / "final_eval"
            run_logged(
                [
                    sys.executable,
                    str(EVALUATOR),
                    "--estimate",
                    str(strict_csv),
                    "--ground-truth",
                    str(gt_path),
                    "--output-dir",
                    str(final_eval_dir),
                    "--handeye-yaml",
                    str(HAND_EYE),
                    "--calibration-json",
                    str(calibration_json),
                    "--camera-rig",
                    "stereo_right",
                    "--estimate-frame",
                    "imu",
                    "--time-association",
                    "evo",
                    "--rpe-distance-m",
                    "0.001",
                    "--time-offset-sec",
                    "0",
                    "--t-max-diff-sec",
                    "0.0001",
                ],
                episode_out / "final_eval.log",
            )
            summary = read_summary(final_eval_dir / "summary.csv")
            row: dict[str, Any] = {
                "episode_key": episode_key,
                "episode": source["episode"],
                "status": "ok",
                "source_si_ba0_ape_mm": source["si_ape_mm"],
                "source_stereo_ape_mm": source["stereo_ape_mm"],
                "best_offset_sec": best["offset_sec"],
                "ape_translation_se3_mm": summary["ape_translation_se3"],
                "rpe_translation_1mm_mm": summary["rpe_translation_5cm"],
                "ape_rotation_deg": summary["ape_rotation_se3"],
                "rpe_rotation_1mm_deg": summary["rpe_rotation_5cm"],
                "ape_translation_sim3_mm": summary["ape_translation_sim3"],
                "inertial_coverage": manifest.get("inertial_coverage", ""),
                "vins_config": str(vins_config),
                "run_manifest": str(run_manifest_path),
                "final_eval_dir": str(final_eval_dir),
                "error": "",
            }
            final_json.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[BEST] {episode_key} APE={summary['ape_translation_se3']:.6f} mm "
                f"RPE={summary['rpe_translation_5cm']:.6f} mm offset={best['offset_sec']*1000:+.3f} ms",
                flush=True,
            )
        except Exception as exc:
            row = {
                "episode_key": episode_key,
                "episode": source["episode"],
                "status": "failed",
                "source_si_ba0_ape_mm": source.get("si_ape_mm", ""),
                "source_stereo_ape_mm": source.get("stereo_ape_mm", ""),
                "best_offset_sec": "",
                "ape_translation_se3_mm": "",
                "rpe_translation_1mm_mm": "",
                "ape_rotation_deg": "",
                "rpe_rotation_1mm_deg": "",
                "ape_translation_sim3_mm": "",
                "inertial_coverage": "",
                "vins_config": str(vins_config),
                "run_manifest": str(run_manifest_path),
                "final_eval_dir": "",
                "error": str(exc),
            }
            print(f"[FAILED] {episode_key}: {exc}", flush=True)

        results = [entry for entry in results if entry["episode_key"] != episode_key]
        results.append(row)
        results.sort(key=lambda entry: entry["episode_key"])
        write_rows(results_csv, results)

    ok_rows = [row for row in results if row.get("status") == "ok"]
    payload = {
        "selected": selected,
        "completed_ok": len(ok_rows),
        "failed": len(results) - len(ok_rows),
        "results_csv": str(results_csv),
        "results": results,
    }
    (output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(ok_rows) == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
