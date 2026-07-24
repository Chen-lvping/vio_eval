#!/usr/bin/env python3
"""Run the deployable RM75 shadow gate on single-MCAP ugripper episodes.

The gate is intentionally GT-free: it selects the stereo-inertial BA0 candidate
only when the independent BA10 stereo shadow is available and agrees with it.
GT is read only after that selection to score each selected trajectory and scan
the offline strict-sync offset for reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER = REPO_ROOT / "script/run_orbslam3_ugripper_mcap.py"
COMPARE = REPO_ROOT / "script/diagnose/compare_orb_stereo_inertial_consistency.py"
CONVERT_TUM = REPO_ROOT / "script/postprocess/convert_tum_to_pose_csv.py"
RESAMPLE = REPO_ROOT / "script/postprocess/resample_pose_csv_to_gt_timestamps.py"
EVALUATOR = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
COLLEAGUE_ROOT = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_colleague_sync")
COLLEAGUE_RUNNER = COLLEAGUE_ROOT / "scripts/run_fays_orbslam3_stereo_right.py"
COLLEAGUE_SMOOTHER = COLLEAGUE_ROOT / "scripts/smooth_pose_csv.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode-pattern", default="episode_20260708_*")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--min-inertial-coverage", type=float, default=0.25)
    parser.add_argument("--max-disagreement-mm", type=float, default=30.0)
    parser.add_argument("--min-scale", type=float, default=0.70)
    parser.add_argument("--max-scale", type=float, default=1.30)
    parser.add_argument("--offset-span-ms", type=float, default=180.0)
    parser.add_argument("--offset-step-ms", type=float, default=10.0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_log(log_path: Path, command: list[str], *, cwd: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n[{}] $ {}\n".format(utc_now(), " ".join(command)))
        handle.flush()
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        handle.write("[exit={}]\n".format(result.returncode))
    return int(result.returncode)


def has_rows(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return any(line.strip() and not line.startswith("#") for line in handle)


def row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def find_mcap(episode: Path) -> Path:
    candidates = sorted(episode.glob("*.mcap"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one MCAP under {episode}, found {len(candidates)}")
    return candidates[0]


def gt_path(episode: Path, root: Path) -> Path:
    suffix = episode.name.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise RuntimeError(f"cannot infer episode id from {episode.name}")
    episode_id = int(suffix)
    candidates = [
        root / f"rm75_pose_traj{episode_id:02d}.json",
        root / f"rm75_pose_traj_{episode_id}.json",
        root / f"rm75_pose_traj{episode_id}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing GT for {episode.name}: {candidates}")


def adapter_command(episode: Path, mcap: Path, work: Path, mode: str, trajectory: str, timeout_sec: int) -> list[str]:
    return [
        sys.executable,
        str(ADAPTER),
        "--episode-dir", str(episode),
        "--mcap-path", str(mcap),
        "--output-dir", str(work),
        "--camera-rig", "stereo_right",
        "--mode", mode,
        "--feature-preset", "low-texture",
        "--trajectory-name", trajectory,
        "--nfeatures", "3000",
        "--ini-fast", "12",
        "--min-fast", "3",
        "--imu-fast-init", "0",
        "--timeout-sec", str(timeout_sec),
    ]


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink() if destination.is_symlink() or destination.is_file() else shutil.rmtree(destination)
    try:
        destination.symlink_to(source)
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def run_ba10_shadow(episode: Path, work: Path, log_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    shadow = work / "stereo_shadow"
    shadow.mkdir(parents=True, exist_ok=True)
    link_or_copy(work / "mav0", shadow / "euroc_sequence")
    link_or_copy(work / "times.txt", shadow / "timestamps.txt")
    shutil.copy2(work / "orbslam3_stereo_right_stereo-inertial.yaml", shadow / "orbslam3_stereo_right.yaml")
    raw_csv = shadow / "pose_raw.csv"
    smooth_csv = shadow / "pose_smooth.csv"
    shadow_name = "shadow_ba10"
    shadow_tum = shadow / f"f_{shadow_name}.txt"
    rc = append_log(
        log_path,
        [
            sys.executable, str(COLLEAGUE_RUNNER),
            "--skip-prepare", "--offline-accurate", "--tracking-mode", "stereo",
            "--episode-dir", str(episode), "--work-dir", str(shadow),
            "--output-csv", str(raw_csv), "--trajectory-name", shadow_name,
            "--gba-iterations", "100", "--full-frame-ba-iterations", "10",
            "--orb-features", "1200", "--orb-init-fast", "20", "--orb-min-fast", "7", "--overwrite",
        ],
        cwd=COLLEAGUE_ROOT,
    )
    if rc != 0 or not has_rows(shadow_tum):
        raise RuntimeError(f"BA10 stereo shadow failed (exit={rc})")
    rc = append_log(
        log_path,
        [
            sys.executable, str(COLLEAGUE_SMOOTHER),
            "--input-csv", str(raw_csv), "--output-csv", str(smooth_csv),
            "--position-window", "21", "--position-poly", "2", "--rotation-window", "9",
        ],
        cwd=COLLEAGUE_ROOT,
    )
    if rc != 0 or not has_rows(smooth_csv):
        raise RuntimeError(f"BA10 stereo shadow smoothing failed (exit={rc})")
    consistency_path = shadow / "stereo_inertial_consistency.json"
    rc = append_log(
        log_path,
        [
            sys.executable, str(COMPARE),
            "--inertial-tum", str(work / "f_inertial.txt"),
            "--stereo-tum", str(shadow_tum),
            "--inertial-settings", str(work / "orbslam3_stereo_right_stereo-inertial.yaml"),
            "--output-json", str(consistency_path),
        ],
    )
    if rc != 0 or not consistency_path.is_file():
        raise RuntimeError(f"stereo/IMU consistency diagnostic failed (exit={rc})")
    return shadow_tum, smooth_csv, json.loads(consistency_path.read_text(encoding="utf-8"))


def read_summary(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["metric"]] = float(row["rmse"])
    return values


def score(values: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        values["ape_translation_se3"],
        values["rpe_translation_5cm"],
        values["ape_rotation_se3"],
        values["rpe_rotation_5cm"],
    )


def scan_strict_sync(
    pose_csv: Path,
    estimate_frame: str,
    calibration: Path,
    ground_truth: Path,
    eval_root: Path,
    span_ms: float,
    step_ms: float,
    log_path: Path,
) -> dict[str, Any]:
    if step_ms <= 0.0:
        raise ValueError("offset step must be positive")
    steps = int(math.floor(span_ms / step_ms + 1e-9))
    candidates = [index * step_ms / 1000.0 for index in range(-steps, steps + 1)]
    scan_root = eval_root / "strict_sync_offset_scan"
    rows: list[dict[str, Any]] = []
    for offset_sec in candidates:
        label = f"offset_{offset_sec * 1000.0:+.1f}ms".replace("+", "p").replace("-", "m")
        estimate = scan_root / f"estimate_{label}.csv"
        candidate_eval = scan_root / label
        rc = append_log(
            log_path,
            [
                sys.executable, str(RESAMPLE), "--estimate-csv", str(pose_csv),
                "--gt-json", str(ground_truth), "--output-csv", str(estimate),
                "--time-offset-sec", str(offset_sec), "--max-gap-sec", "0.05",
            ],
        )
        if rc != 0:
            continue
        rc = append_log(
            log_path,
            [
                sys.executable, str(EVALUATOR), "--estimate", str(estimate),
                "--ground-truth", str(ground_truth), "--output-dir", str(candidate_eval),
                "--calibration-json", str(calibration), "--camera-rig", "stereo_right",
                "--estimate-frame", estimate_frame, "--time-association", "evo",
                "--time-offset-sec", "0.0", "--t-max-diff-sec", "0.0001", "--rpe-distance-m", "0.001",
            ],
        )
        summary_path = candidate_eval / "summary.csv"
        if rc == 0 and summary_path.is_file():
            values = read_summary(summary_path)
            rows.append({"offset_sec": offset_sec, "offset_ms": offset_sec * 1000.0, **values, "estimate_csv": str(estimate)})
    if not rows:
        raise RuntimeError("strict-sync scoring produced no valid candidates")
    best = min(rows, key=score)
    write_json(eval_root / "strict_sync_offset_scan.json", {"candidates": rows, "selected": best})
    final_estimate = Path(str(best["estimate_csv"]))
    rc = append_log(
        log_path,
        [
            sys.executable, str(EVALUATOR), "--estimate", str(final_estimate),
            "--ground-truth", str(ground_truth), "--output-dir", str(eval_root),
            "--calibration-json", str(calibration), "--camera-rig", "stereo_right",
            "--estimate-frame", estimate_frame, "--time-association", "evo",
            "--time-offset-sec", "0.0", "--t-max-diff-sec", "0.0001", "--rpe-distance-m", "0.001",
        ],
    )
    if rc != 0:
        raise RuntimeError(f"final selected trajectory evaluation failed (exit={rc})")
    return best


def run_episode(args: argparse.Namespace, episode: Path) -> dict[str, Any]:
    episode_root = args.output_root / episode.name
    work = episode_root / "run"
    log_path = episode_root / "episode.log"
    mcap = find_mcap(episode)
    result: dict[str, Any] = {
        "episode": episode.name,
        "source_mcap": str(mcap),
        "status": "running",
        "selected_method": "",
        "fallback_reason": "",
        "inertial_coverage": None,
        "shadow_consistency": {},
        "strict_sync_offset_sec": None,
    }
    try:
        rc = append_log(log_path, adapter_command(episode, mcap, work, "stereo-inertial", "inertial", args.timeout_sec))
        inertial = work / "f_inertial.txt"
        coverage = row_count(inertial) / max(row_count(work / "times.txt"), 1) if rc == 0 and has_rows(inertial) else 0.0
        result["inertial_coverage"] = coverage
        selected_csv: Path | None = None
        selected_frame = "camera"
        if rc == 0 and coverage >= args.min_inertial_coverage:
            try:
                _shadow_tum, shadow_csv, consistency = run_ba10_shadow(episode, work, log_path)
                result["shadow_consistency"] = consistency
                disagreement = float(consistency["translation_rmse_mm"])
                scale = float(consistency["sim3_scale_stereo_over_inertial"])
                if disagreement <= args.max_disagreement_mm and args.min_scale <= scale <= args.max_scale:
                    selected_csv = episode_root / "selected_inertial.csv"
                    rc = append_log(log_path, [sys.executable, str(CONVERT_TUM), str(inertial), str(selected_csv)])
                    if rc != 0 or not has_rows(selected_csv):
                        raise RuntimeError(f"inertial TUM conversion failed (exit={rc})")
                    result["selected_method"] = "stereo_inertial_ba0"
                    selected_frame = "imu"
                else:
                    selected_csv = shadow_csv
                    result["selected_method"] = "stereo_ba10_shadow"
                    result["fallback_reason"] = "shadow_disagreement" if disagreement > args.max_disagreement_mm else "extreme_scale"
            except Exception as exc:
                result["fallback_reason"] = f"shadow_failure:{exc}"
        else:
            result["fallback_reason"] = "inertial_failure" if rc != 0 else "low_inertial_coverage"
        if selected_csv is None:
            rc = append_log(log_path, adapter_command(episode, mcap, work, "stereo", "native_stereo", args.timeout_sec))
            native_tum = work / "f_native_stereo.txt"
            if rc != 0 or not has_rows(native_tum):
                raise RuntimeError(f"native stereo fail-closed fallback failed (exit={rc})")
            selected_csv = episode_root / "selected_native_stereo.csv"
            rc = append_log(log_path, [sys.executable, str(CONVERT_TUM), str(native_tum), str(selected_csv)])
            if rc != 0 or not has_rows(selected_csv):
                raise RuntimeError(f"native stereo TUM conversion failed (exit={rc})")
            result["selected_method"] = "stereo_native_fallback"
            selected_frame = "camera"
        # The method selection above has no GT dependency. GT appears only here.
        best = scan_strict_sync(
            selected_csv, selected_frame, work / "episode_shim/calibration.json", gt_path(episode, args.gt_root),
            episode_root / "evaluation", args.offset_span_ms, args.offset_step_ms, log_path,
        )
        result.update({
            "status": "ok",
            "selected_pose_csv": str(selected_csv),
            "estimate_frame": selected_frame,
            "strict_sync_offset_sec": best["offset_sec"],
            "ape_translation_se3_rmse_mm": best["ape_translation_se3"],
            "rpe_translation_5cm_rmse_mm": best["rpe_translation_5cm"],
            "ape_rotation_se3_rmse_deg": best["ape_rotation_se3"],
            "rpe_rotation_5cm_rmse_deg": best["rpe_rotation_5cm"],
        })
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[ERROR] {exc}\n")
    return result


def write_progress(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    fields = sorted({field for row in rows for field in row})
    with (output_root / "progress.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_root / "run_manifest.json", {
        "updated_at": utc_now(),
        "driver": str(Path(__file__).resolve()),
        "gate": {
            "min_inertial_coverage": args.min_inertial_coverage,
            "max_disagreement_mm": args.max_disagreement_mm,
            "extreme_scale_range": [args.min_scale, args.max_scale],
            "fail_closed": True,
            "gt_used_for_selection": False,
        },
        "strict_sync_scoring": {"span_ms": args.offset_span_ms, "step_ms": args.offset_step_ms},
        "episodes": rows,
    })


def main() -> int:
    args = parse_args()
    args.episode_root = args.episode_root.expanduser().resolve()
    args.gt_root = args.gt_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    required = [ADAPTER, COMPARE, CONVERT_TUM, RESAMPLE, EVALUATOR, COLLEAGUE_RUNNER, COLLEAGUE_SMOOTHER]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files: " + ", ".join(missing))
    episodes = sorted(path for path in args.episode_root.glob(args.episode_pattern) if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"no episodes matched {args.episode_pattern} under {args.episode_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        rows.append(run_episode(args, episode))
        write_progress(args.output_root, rows, args)
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
