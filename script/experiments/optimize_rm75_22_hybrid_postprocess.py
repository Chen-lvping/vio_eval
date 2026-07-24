#!/usr/bin/env python3
"""Screen the 22 RM75 episodes with per-episode strict-sync optimization.

The script reuses existing trajectories only; it does not rerun ORB-SLAM3.
For every episode it compares:

* the historical pure-stereo BA10 trajectory, smoothed with 21/2/9; and
* the retained raw stereo-inertial trajectory, smoothed with 21/2/9.

Each method gets an independent coarse/fine strict-sync offset scan.  The
screen is intentionally fast and writes enough provenance to choose the
small subset worth rerunning with final BA=10.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "script"
BASELINE_ROOT = ROOT / "data/evaluation/workbench/rm75_best_all_gt_20260716"
BASELINE_SUMMARY = BASELINE_ROOT / "aggregate_summary.csv"
SHADOW_ROOT = (
    ROOT
    / "data/evaluation/workbench/rm75_stereo_imu_shadow_gate_full_20260721"
    / "orbslam3_rm75_batch_eval_20260721_213748"
)
SI_OFFSET_JSON = ROOT / "data/evaluation/config/rm75_gripper_22_strict_sync_offsets.json"
HAND_EYE = ROOT / "data/calibration/handeye_0615/handeye_result.yaml"
SMOOTHER = Path(
    "/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_colleague_sync/scripts/smooth_pose_csv.py"
)
CONVERTER = SCRIPT_DIR / "postprocess/convert_tum_to_pose_csv.py"
TCP_EVALUATOR = SCRIPT_DIR / "evaluate_vio_tcp_camera_evo.py"


@dataclass(frozen=True)
class PoseSeries:
    times_s: np.ndarray
    positions: np.ndarray
    quats_xyzw: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "data/evaluation/workbench"
        / f"rm75_22_hybrid_postprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--episode", action="append", default=[], help="Optional episode_gripper_#### filter")
    parser.add_argument("--coarse-span-ms", type=float, default=150.0)
    parser.add_argument("--coarse-step-ms", type=float, default=10.0)
    parser.add_argument("--fine-span-ms", type=float, default=10.0)
    parser.add_argument("--fine-step-ms", type=float, default=1.0)
    parser.add_argument("--max-gap-sec", type=float, default=0.05)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--shadow-max-disagreement-mm", type=float, default=15.0)
    parser.add_argument("--shadow-scale-min", type=float, default=0.90)
    parser.add_argument("--shadow-scale-max", type=float, default=1.10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Load the aggregate CSV but recompute and replace the requested episode rows.",
    )
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files:\n" + "\n".join(missing))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_pose_csv(path: Path) -> PoseSeries:
    times: list[float] = []
    positions: list[list[float]] = []
    quats: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            times.append(float(row["Timestamp_us"]) * 1e-6)
            positions.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
            quats.append(
                [float(row["Quat_X"]), float(row["Quat_Y"]), float(row["Quat_Z"]), float(row["Quat_W"])]
            )
    if not times:
        raise RuntimeError(f"no poses in {path}")
    times_arr = np.asarray(times, dtype=float)
    order = np.argsort(times_arr, kind="stable")
    times_arr = times_arr[order]
    pos_arr = np.asarray(positions, dtype=float)[order]
    quat_arr = np.asarray(quats, dtype=float)[order]
    # Slerp requires strictly increasing timestamps. Keep the final sample for
    # duplicate timestamps, matching the effective behavior of interpolation.
    _, reverse_unique = np.unique(times_arr[::-1], return_index=True)
    keep = np.sort(times_arr.size - 1 - reverse_unique)
    return PoseSeries(times_arr[keep], pos_arr[keep], quat_arr[keep])


def convert_tum_to_csv(converter, tum_path: Path, csv_path: Path) -> None:
    rows = converter.convert_rows(tum_path)
    if not rows:
        raise RuntimeError(f"no valid TUM poses in {tum_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(converter.HEADER)
        writer.writerows(rows)


def smooth_pose_csv(smoother, input_csv: Path, output_csv: Path) -> None:
    stamps, positions, quats = smoother.load_pose_csv(input_csv)
    smoothed_positions = smoother.smooth_positions(positions, 21, 2)
    smoothed_quats = smoother.smooth_rotations(quats, 9)
    smoother.write_pose_csv(output_csv, stamps, smoothed_positions, smoothed_quats)


def grid(center: float, span_ms: float, step_ms: float) -> list[float]:
    if span_ms < 0.0 or step_ms <= 0.0:
        raise ValueError("scan span must be >= 0 and step must be > 0")
    count = int(math.floor(span_ms / step_ms + 1e-9))
    return [round(center + idx * step_ms * 1e-3, 12) for idx in range(-count, count + 1)]


def transform_estimate_to_tcp(
    positions: np.ndarray,
    rotations: np.ndarray,
    estimate_frame: str,
    t_tcp_camera: np.ndarray,
    t_camera_imu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t_camera_tcp = np.linalg.inv(t_tcp_camera)
    if estimate_frame == "camera":
        t_est_tcp = t_camera_tcp
    elif estimate_frame == "imu":
        t_est_tcp = np.linalg.inv(t_camera_imu) @ t_camera_tcp
    else:
        raise ValueError(f"unsupported estimate frame: {estimate_frame}")
    offset = t_est_tcp[:3, 3]
    rotation = t_est_tcp[:3, :3]
    tcp_positions = positions + np.einsum("nij,j->ni", rotations, offset)
    tcp_rotations = np.einsum("nij,jk->nik", rotations, rotation)
    return tcp_positions, tcp_rotations


def evaluate_offset(
    series: PoseSeries,
    offset_sec: float,
    gt_times: np.ndarray,
    gt_positions: np.ndarray,
    gt_rotations: np.ndarray,
    estimate_frame: str,
    t_tcp_camera: np.ndarray,
    t_camera_imu: np.ndarray,
    helpers: dict[str, Any],
    max_gap_sec: float,
    min_samples: int,
) -> dict[str, float] | None:
    shifted = series.times_s + float(offset_sec)
    valid = (gt_times >= shifted[0]) & (gt_times <= shifted[-1])
    query = gt_times[valid]
    if query.size < min_samples:
        return None
    left = np.searchsorted(shifted, query, side="right") - 1
    left = np.clip(left, 0, shifted.size - 1)
    right = np.clip(left + 1, 0, shifted.size - 1)
    bracket_gap = np.maximum(np.abs(query - shifted[left]), np.abs(shifted[right] - query))
    keep = bracket_gap <= max_gap_sec
    query = query[keep]
    if query.size < min_samples:
        return None
    gt_indices = np.flatnonzero(valid)[keep]
    interp_positions = np.column_stack(
        [np.interp(query, shifted, series.positions[:, axis]) for axis in range(3)]
    )
    interp_rotations = Slerp(shifted, Rotation.from_quat(series.quats_xyzw))(query).as_matrix()
    tcp_positions, tcp_rotations = transform_estimate_to_tcp(
        interp_positions, interp_rotations, estimate_frame, t_tcp_camera, t_camera_imu
    )
    gt_pos = gt_positions[gt_indices]
    gt_rot = gt_rotations[gt_indices]
    se3, _, _, _, _ = helpers["evaluate_alignment"](
        "se3", False, gt_pos, gt_rot, tcp_positions, tcp_rotations, query, 1.0, 30
    )
    sim3, _, _, _, _ = helpers["evaluate_alignment"](
        "sim3", True, gt_pos, gt_rot, tcp_positions, tcp_rotations, query, 1.0, 30
    )
    return {
        "offset_sec": float(offset_sec),
        "offset_ms": float(offset_sec * 1000.0),
        "samples": int(query.size),
        "duration_sec": float(query[-1] - query[0]) if query.size > 1 else 0.0,
        "ape_se3_mm": float(se3.translation_metrics_m["rmse"] * 1000.0),
        "ape_rotation_deg": float((se3.rotation_metrics_deg or {}).get("rmse", float("nan"))),
        "ape_sim3_mm": float(sim3.translation_metrics_m["rmse"] * 1000.0),
        "sim3_scale": float(sim3.scale),
    }


def scan_method(
    *,
    series: PoseSeries,
    center_sec: float,
    gt_times: np.ndarray,
    gt_positions: np.ndarray,
    gt_rotations: np.ndarray,
    estimate_frame: str,
    t_tcp_camera: np.ndarray,
    t_camera_imu: np.ndarray,
    helpers: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    cache: dict[float, dict[str, float]] = {}

    def run_offsets(offsets: list[float]) -> None:
        for offset in offsets:
            key = round(offset, 12)
            if key in cache:
                continue
            result = evaluate_offset(
                series,
                offset,
                gt_times,
                gt_positions,
                gt_rotations,
                estimate_frame,
                t_tcp_camera,
                t_camera_imu,
                helpers,
                args.max_gap_sec,
                args.min_samples,
            )
            if result is not None:
                cache[key] = result

    run_offsets(grid(center_sec, args.coarse_span_ms, args.coarse_step_ms))
    if not cache:
        raise RuntimeError("offset scan produced no valid candidates")
    coarse_best = min(cache.values(), key=lambda row: (row["ape_se3_mm"], -row["samples"], row["offset_sec"]))
    run_offsets(grid(float(coarse_best["offset_sec"]), args.fine_span_ms, args.fine_step_ms))
    rows = sorted(cache.values(), key=lambda row: row["offset_sec"])
    best = min(rows, key=lambda row: (row["ape_se3_mm"], -row["samples"], row["offset_sec"]))
    return best, rows


def find_baseline_pose(actual_episode: str) -> Path:
    matches = sorted(BASELINE_ROOT.glob(f"orbslam3_rm75_best_batch_*/{actual_episode}/pose_smooth.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one stereo pose_smooth.csv for {actual_episode}, got {len(matches)}")
    return matches[0]


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, float]]:
    baseline_rows = {row["episode"]: row for row in read_csv_rows(BASELINE_SUMMARY)}
    si_offsets = {key: float(value) for key, value in json.loads(SI_OFFSET_JSON.read_text()).items()}
    inputs: list[dict[str, Any]] = []
    for manifest_path in sorted(SHADOW_ROOT.glob("eval_episode_gripper_*/orbslam3_tcp_eval_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episode_key = manifest_path.parent.name.removeprefix("eval_")
        actual_episode = Path(manifest["episode_dir"]).name
        if actual_episode not in baseline_rows:
            raise KeyError(f"{episode_key}: no baseline row for {actual_episode}")
        shadow = manifest.get("stereo_shadow_consistency") or {}
        inputs.append(
            {
                "episode_key": episode_key,
                "actual_episode": actual_episode,
                "manifest_path": manifest_path,
                "manifest": manifest,
                "baseline": baseline_rows[actual_episode],
                "stereo_pose": find_baseline_pose(actual_episode),
                "si_tum": Path(shadow["inertial_tum"]) if shadow.get("inertial_tum") else None,
                "si_center_sec": si_offsets[episode_key],
            }
        )
    if len(inputs) != 22:
        raise RuntimeError(f"expected 22 manifests, found {len(inputs)}")
    return inputs, si_offsets


def select_deployable(row: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    if row.get("si_ape_mm", "") == "":
        return "stereo", "no_inertial_trajectory"
    coverage = float(row["inertial_coverage"])
    disagreement = row.get("shadow_disagreement_mm", "")
    scale = row.get("shadow_scale", "")
    if coverage < 0.25:
        return "stereo", "low_inertial_coverage"
    if disagreement == "" or float(disagreement) > args.shadow_max_disagreement_mm:
        return "stereo", "shadow_disagreement"
    if scale == "" or not (args.shadow_scale_min <= float(scale) <= args.shadow_scale_max):
        return "stereo", "shadow_scale"
    return "stereo-inertial", "shadow_gate_pass"


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    require_files([BASELINE_SUMMARY, SI_OFFSET_JSON, HAND_EYE, SMOOTHER, CONVERTER, TCP_EVALUATOR])

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    tcp_eval = load_module("rm75_tcp_eval_screen", TCP_EVALUATOR)
    converter = load_module("rm75_tum_converter_screen", CONVERTER)
    smoother = load_module("rm75_pose_smoother_screen", SMOOTHER)
    helpers = tcp_eval.load_helpers()
    t_tcp_camera = tcp_eval.load_tcp_camera_transform(HAND_EYE)
    inputs, _ = load_inputs()
    requested = set(args.episode)
    if requested:
        inputs = [item for item in inputs if item["episode_key"] in requested]
        missing = requested - {item["episode_key"] for item in inputs}
        if missing:
            raise KeyError(f"unknown episode filters: {sorted(missing)}")

    aggregate_path = output_root / "postprocess_screening.csv"
    existing: dict[str, dict[str, str]] = {}
    if (args.resume or args.replace_existing) and aggregate_path.is_file():
        existing = {row["episode_key"]: row for row in read_csv_rows(aggregate_path)}
    if args.replace_existing:
        for episode_key in requested:
            existing.pop(episode_key, None)
    aggregate: list[dict[str, Any]] = list(existing.values())

    for index, item in enumerate(inputs, start=1):
        episode_key = item["episode_key"]
        if episode_key in existing:
            print(f"[SKIP] {episode_key} already complete", flush=True)
            continue
        manifest = item["manifest"]
        episode_dir = Path(manifest["episode_dir"])
        gt_path = Path(manifest["ground_truth"])
        calibration_json = episode_dir / "calibration.json"
        require_files([gt_path, calibration_json, item["stereo_pose"]])
        t_camera_imu, _, _ = tcp_eval.load_camera_imu_transform(calibration_json, "stereo_right")
        gt_times, gt_positions, gt_rotations = tcp_eval.load_robot_tcp_trajectory(gt_path, helpers)
        episode_out = output_root / episode_key
        episode_out.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(inputs)}] {episode_key} -> {item['actual_episode']}", flush=True)

        stereo_series = read_pose_csv(item["stereo_pose"])
        stereo_center = float(item["baseline"]["offset"])
        stereo_best, stereo_scan = scan_method(
            series=stereo_series,
            center_sec=stereo_center,
            gt_times=gt_times,
            gt_positions=gt_positions,
            gt_rotations=gt_rotations,
            estimate_frame="camera",
            t_tcp_camera=t_tcp_camera,
            t_camera_imu=t_camera_imu,
            helpers=helpers,
            args=args,
        )
        write_csv(episode_out / "stereo_offset_scan.csv", stereo_scan)

        si_best: dict[str, float] | None = None
        si_tum: Path | None = item["si_tum"]
        si_smooth_csv = episode_out / "stereo_inertial_smooth_pos21_poly2_rot9.csv"
        if si_tum is not None and si_tum.is_file():
            si_raw_csv = episode_out / "stereo_inertial_raw.csv"
            if not (args.resume and si_smooth_csv.is_file()):
                convert_tum_to_csv(converter, si_tum, si_raw_csv)
                smooth_pose_csv(smoother, si_raw_csv, si_smooth_csv)
            si_series = read_pose_csv(si_smooth_csv)
            si_best, si_scan = scan_method(
                series=si_series,
                center_sec=float(item["si_center_sec"]),
                gt_times=gt_times,
                gt_positions=gt_positions,
                gt_rotations=gt_rotations,
                estimate_frame="imu",
                t_tcp_camera=t_tcp_camera,
                t_camera_imu=t_camera_imu,
                helpers=helpers,
                args=args,
            )
            write_csv(episode_out / "stereo_inertial_offset_scan.csv", si_scan)

        shadow = manifest.get("stereo_shadow_consistency") or {}
        row: dict[str, Any] = {
            "episode_key": episode_key,
            "episode": item["actual_episode"],
            "ground_truth": str(gt_path),
            "calibration_json": str(calibration_json),
            "stereo_source": str(item["stereo_pose"]),
            "stereo_center_offset_sec": stereo_center,
            "stereo_best_offset_sec": stereo_best["offset_sec"],
            "stereo_ape_mm": stereo_best["ape_se3_mm"],
            "stereo_rotation_ape_deg": stereo_best["ape_rotation_deg"],
            "stereo_sim3_ape_mm": stereo_best["ape_sim3_mm"],
            "stereo_samples": stereo_best["samples"],
            "si_source": str(si_tum) if si_tum else "",
            "si_center_offset_sec": item["si_center_sec"],
            "si_best_offset_sec": si_best["offset_sec"] if si_best else "",
            "si_ape_mm": si_best["ape_se3_mm"] if si_best else "",
            "si_rotation_ape_deg": si_best["ape_rotation_deg"] if si_best else "",
            "si_sim3_ape_mm": si_best["ape_sim3_mm"] if si_best else "",
            "si_samples": si_best["samples"] if si_best else "",
            "inertial_coverage": manifest.get("inertial_coverage", ""),
            "shadow_disagreement_mm": shadow.get("translation_rmse_mm", ""),
            "shadow_scale": shadow.get("sim3_scale_stereo_over_inertial", ""),
        }
        if si_best is None or stereo_best["ape_se3_mm"] <= si_best["ape_se3_mm"]:
            row["oracle_method"] = "stereo"
            row["oracle_ape_mm"] = stereo_best["ape_se3_mm"]
            row["oracle_offset_sec"] = stereo_best["offset_sec"]
        else:
            row["oracle_method"] = "stereo-inertial"
            row["oracle_ape_mm"] = si_best["ape_se3_mm"]
            row["oracle_offset_sec"] = si_best["offset_sec"]
        deployable_method, deployable_reason = select_deployable(row, args)
        row["deployable_method"] = deployable_method
        row["deployable_reason"] = deployable_reason
        if deployable_method == "stereo-inertial":
            row["deployable_ape_mm"] = row["si_ape_mm"]
            row["deployable_offset_sec"] = row["si_best_offset_sec"]
        else:
            row["deployable_ape_mm"] = row["stereo_ape_mm"]
            row["deployable_offset_sec"] = row["stereo_best_offset_sec"]
        aggregate.append(row)
        aggregate.sort(key=lambda value: value["episode_key"])
        write_csv(aggregate_path, aggregate)
        (episode_out / "best.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"  stereo={float(row['stereo_ape_mm']):.3f} mm @ {float(row['stereo_best_offset_sec'])*1000:+.3f} ms"
            + (
                f", SI={float(row['si_ape_mm']):.3f} mm @ {float(row['si_best_offset_sec'])*1000:+.3f} ms"
                if row["si_ape_mm"] != ""
                else ", SI=unavailable"
            ),
            flush=True,
        )

    aggregate.sort(key=lambda value: value["episode_key"])
    if not aggregate:
        raise RuntimeError("no completed episode results")
    oracle = np.asarray([float(row["oracle_ape_mm"]) for row in aggregate], dtype=float)
    deployable = np.asarray([float(row["deployable_ape_mm"]) for row in aggregate], dtype=float)
    summary = {
        "episodes": len(aggregate),
        "scan": {
            "coarse_span_ms": args.coarse_span_ms,
            "coarse_step_ms": args.coarse_step_ms,
            "fine_span_ms": args.fine_span_ms,
            "fine_step_ms": args.fine_step_ms,
            "max_gap_sec": args.max_gap_sec,
        },
        "oracle": {
            "mean_ape_mm": float(np.mean(oracle)),
            "median_ape_mm": float(np.median(oracle)),
            "p90_ape_mm": float(np.percentile(oracle, 90)),
            "count_le_10mm": int(np.sum(oracle <= 10.0)),
        },
        "deployable": {
            "mean_ape_mm": float(np.mean(deployable)),
            "median_ape_mm": float(np.median(deployable)),
            "p90_ape_mm": float(np.percentile(deployable, 90)),
            "count_le_10mm": int(np.sum(deployable <= 10.0)),
        },
        "baseline_historical": {"mean_ape_mm": 15.040, "count_le_10mm": 12},
        "output_csv": str(aggregate_path),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_root / "REPORT.md").open("w", encoding="utf-8") as handle:
        handle.write("# RM75 22-Episode Hybrid Postprocess Screen\n\n")
        handle.write("Every stereo and stereo-inertial candidate uses an independent per-episode strict-sync scan.\n\n")
        handle.write(f"- Completed episodes: {len(aggregate)}\n")
        handle.write(f"- Oracle mean APE: {summary['oracle']['mean_ape_mm']:.3f} mm\n")
        handle.write(f"- Oracle APE <= 10 mm: {summary['oracle']['count_le_10mm']}/{len(aggregate)}\n")
        handle.write(f"- Deployable mean APE: {summary['deployable']['mean_ape_mm']:.3f} mm\n")
        handle.write(f"- Deployable APE <= 10 mm: {summary['deployable']['count_le_10mm']}/{len(aggregate)}\n")
        handle.write(f"- Detailed CSV: `{aggregate_path}`\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
