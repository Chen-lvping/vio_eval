#!/usr/bin/env python3
"""Sweep one shared postprocess config across data_ok episodes with per-episode time alignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_OK_ROOT = REPO_ROOT / "data/data_ok"
EVAL_SCRIPT = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/evaluation/workbench"
PREFERRED_CHAIN_MANIFESTS = {
    "episode_20260618_0004": REPO_ROOT
    / "data/evaluation/workbench/orbslam3_rm75_batch_eval_20260623_113153/eval_episode_20260618_0004/orbslam3_tcp_eval_manifest.json",
}


@dataclass(frozen=True)
class EpisodeSpec:
    name: str
    source_manifest: Path
    episode_dir: Path
    ground_truth: Path
    calibration_json: Path
    trajectory_txt: Path
    camera_rig: str
    estimate_frame: str
    selected_offset_sec: float
    selected_ape_mm: float


@dataclass(frozen=True)
class SmoothedTrajectory:
    times_s: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-ok-root", type=Path, default=DATA_OK_ROOT)
    parser.add_argument(
        "--manifest-map",
        type=Path,
        default=None,
        help="Optional JSON mapping output episode names to existing ORB evaluation manifests.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=[],
        help="Optional explicit episode names under data_ok, e.g. episode_20260618_0001",
    )
    parser.add_argument(
        "--windows",
        default="0,5,7,9,11,13,15",
        help="Comma-separated smoothing windows. 0 means no smoothing.",
    )
    parser.add_argument(
        "--passes",
        default="0,1,2,3",
        help="Comma-separated smoothing passes. For window=0, pass is forced to 0.",
    )
    parser.add_argument("--strict-sync-max-gap-sec", type=float, default=0.05)
    parser.add_argument("--coarse-span-ms", type=float, default=300.0)
    parser.add_argument("--coarse-step-ms", type=float, default=15.0)
    parser.add_argument("--fine-span-ms", type=float, default=20.0)
    parser.add_argument("--fine-step-ms", type=float, default=1.0)
    parser.add_argument(
        "--center-mode",
        choices=("selected", "midpoint"),
        default="selected",
        help="How to seed the time-offset search center.",
    )
    parser.add_argument("--handeye-yaml", type=Path, default=REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml")
    return parser.parse_args()


def parse_number_list(raw: str, *, as_int: bool) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        value = int(round(float(text)))
        values.append(value if as_int else value)
    if not values:
        raise ValueError("empty numeric list")
    return values


def load_eval_api() -> dict[str, Any]:
    return runpy.run_path(str(EVAL_SCRIPT), run_name="__shared_algo_sweep__")


def normalize_timestamp(value: float) -> float:
    value = float(value)
    av = abs(value)
    if av > 1e17:
        return value * 1e-9
    if av > 1e13:
        return value * 1e-6
    if av > 1e10:
        return value * 1e-3
    return value


def load_index_rows(data_ok_root: Path) -> dict[str, dict[str, str]]:
    with (data_ok_root / "index.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = {row["episode"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise RuntimeError(f"no rows found in {data_ok_root / 'index.csv'}")
    return rows


def load_episode_specs(args: argparse.Namespace) -> list[EpisodeSpec]:
    if args.manifest_map is not None:
        manifest_map_path = args.manifest_map.expanduser().resolve()
        payload = json.loads(manifest_map_path.read_text(encoding="utf-8"))
        specs: list[EpisodeSpec] = []
        for episode_name, raw_manifest_path in sorted(payload.items()):
            manifest_path = Path(str(raw_manifest_path)).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = REPO_ROOT / manifest_path
            manifest_path = manifest_path.resolve()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            episode_dir = Path(manifest["episode_dir"]).resolve()
            trajectory_txt = Path(manifest["trajectory"]).resolve()
            if not trajectory_txt.is_file():
                raise FileNotFoundError(f"trajectory missing for {episode_name}: {trajectory_txt}")
            specs.append(
                EpisodeSpec(
                    name=str(episode_name),
                    source_manifest=manifest_path,
                    episode_dir=episode_dir,
                    ground_truth=Path(manifest["ground_truth"]).resolve(),
                    calibration_json=(episode_dir / "calibration.json").resolve(),
                    trajectory_txt=trajectory_txt,
                    camera_rig=str(manifest.get("camera_rig") or "stereo_right"),
                    estimate_frame=str(manifest.get("estimate_frame") or "imu"),
                    selected_offset_sec=float(manifest.get("strict_sync_offset_sec", 0.0)),
                    selected_ape_mm=float(manifest.get("summary_rmse", {}).get("ape_translation_se3_rmse", float("nan"))),
                )
            )
        if not specs:
            raise RuntimeError(f"no manifests found in {manifest_map_path}")
        return specs

    data_ok_root = args.data_ok_root.expanduser().resolve()
    index_rows = load_index_rows(data_ok_root)
    wanted = set(args.episodes or index_rows.keys())
    specs: list[EpisodeSpec] = []
    for episode_name in sorted(wanted):
        row = index_rows.get(episode_name)
        if row is None:
            raise FileNotFoundError(f"{episode_name} not found in {data_ok_root / 'index.csv'}")
        episode_symlink_dir = data_ok_root / episode_name
        chain_manifest_path = PREFERRED_CHAIN_MANIFESTS.get(
            episode_name,
            (episode_symlink_dir / "chain_manifest.json").resolve(),
        )
        manifest = json.loads(chain_manifest_path.read_text(encoding="utf-8"))
        episode_dir = Path(manifest["episode_dir"]).resolve()
        trajectory_txt = Path(manifest["trajectory"]).resolve()
        if not trajectory_txt.is_file():
            raise FileNotFoundError(f"trajectory missing for {episode_name}: {trajectory_txt}")
        specs.append(
            EpisodeSpec(
                name=episode_name,
                source_manifest=chain_manifest_path.resolve(),
                episode_dir=episode_dir,
                ground_truth=Path(manifest["ground_truth"]).resolve(),
                calibration_json=(episode_dir / "calibration.json").resolve(),
                trajectory_txt=trajectory_txt,
                camera_rig=str(manifest.get("camera_rig") or "stereo_right"),
                estimate_frame=str(manifest.get("estimate_frame") or "imu"),
                selected_offset_sec=float(manifest.get("strict_sync_offset_sec", 0.0)),
                selected_ape_mm=float(manifest.get("summary_rmse", {}).get("ape_translation_se3_rmse", row["ape_mm"])),
            )
        )
    return specs


def read_tum_trajectory(path: Path) -> SmoothedTrajectory:
    times = []
    positions = []
    rotations = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 8:
                continue
            times.append(normalize_timestamp(float(parts[0])))
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rotations.append([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
    if not times:
        raise RuntimeError(f"no trajectory rows found in {path}")
    order = np.argsort(np.asarray(times, dtype=float))
    times_arr = np.asarray(times, dtype=float)[order]
    pos_arr = np.asarray(positions, dtype=float)[order]
    rot_arr = Rotation.from_quat(np.asarray(rotations, dtype=float)[order]).as_matrix()
    return SmoothedTrajectory(times_arr, pos_arr, rot_arr)


def smooth_positions(positions: np.ndarray, window: int, passes: int) -> np.ndarray:
    if window <= 0 or passes <= 0:
        return positions.copy()
    if window < 3 or window % 2 == 0:
        raise ValueError(f"invalid smoothing window: {window}")
    radius = window // 2
    kernel = np.full(window, 1.0 / float(window), dtype=float)
    out = positions.astype(float, copy=True)
    for _ in range(passes):
        padded = np.pad(out, ((radius, radius), (0, 0)), mode="edge")
        next_out = np.empty_like(out)
        for axis in range(3):
            next_out[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
        out = next_out
    return out


def load_ground_truth(api: dict[str, Any], gt_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return api["load_robot_tcp_trajectory"](gt_path, api["load_helpers"]())


def transform_estimate_to_tcp(
    api: dict[str, Any],
    calibration_json: Path,
    camera_rig: str,
    handeye_yaml: Path,
    estimate_frame: str,
    trajectory: SmoothedTrajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    helpers = api["load_helpers"]()
    t_tcp_camera = api["load_tcp_camera_transform"](handeye_yaml)
    t_camera_imu, _, _ = api["load_camera_imu_transform"](calibration_json, camera_rig)
    t_imu_camera = np.linalg.inv(t_camera_imu)
    t_camera_tcp = np.linalg.inv(t_tcp_camera)

    positions_tcp: list[np.ndarray] = []
    rotations_tcp: list[np.ndarray] = []
    for pos, rot in zip(trajectory.positions, trajectory.rotations):
        t_world_estimate = np.eye(4, dtype=float)
        t_world_estimate[:3, :3] = rot
        t_world_estimate[:3, 3] = pos
        if estimate_frame == "imu":
            t_world_tcp = t_world_estimate @ t_imu_camera @ t_camera_tcp
        elif estimate_frame == "camera":
            t_world_tcp = t_world_estimate @ t_camera_tcp
        elif estimate_frame == "vins_base_link":
            t_world_imu = t_world_estimate @ api["T_VINS_BASE_LINK_IMU"]
            t_world_tcp = t_world_imu @ t_imu_camera @ t_camera_tcp
        else:
            raise ValueError(f"unsupported estimate_frame: {estimate_frame}")
        positions_tcp.append(t_world_tcp[:3, 3].copy())
        rotations_tcp.append(t_world_tcp[:3, :3].copy())
    return trajectory.times_s.copy(), np.asarray(positions_tcp, dtype=float), np.asarray(rotations_tcp, dtype=float)


def default_offset_guess(gt_times: np.ndarray, est_times: np.ndarray) -> float:
    return 0.5 * ((gt_times[0] - est_times[0]) + (gt_times[-1] - est_times[-1]))


def offset_candidates(center_sec: float, span_ms: float, step_ms: float) -> list[float]:
    span_sec = float(span_ms) / 1000.0
    step_sec = float(step_ms) / 1000.0
    if step_sec <= 0.0:
        raise ValueError("step must be > 0")
    start = center_sec - span_sec
    stop = center_sec + span_sec
    count = int(math.floor((stop - start) / step_sec + 0.5)) + 1
    values = [start + idx * step_sec for idx in range(count)]
    values.append(center_sec)
    uniq: dict[int, float] = {}
    for value in values:
        uniq[int(round(value * 1_000_000_000.0))] = float(value)
    return sorted(uniq.values())


def interpolate_rotations(times: np.ndarray, rotations: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    slerp = Slerp(times, Rotation.from_matrix(rotations))
    return slerp(query_times).as_matrix()


def strict_sync_match(
    gt_times: np.ndarray,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_times: np.ndarray,
    est_pos: np.ndarray,
    est_rot: np.ndarray,
    offset_sec: float,
    max_gap_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shifted_times = np.asarray(est_times, dtype=float) + float(offset_sec)
    valid_mask = (gt_times >= shifted_times[0]) & (gt_times <= shifted_times[-1])
    overlap_times = gt_times[valid_mask]
    overlap_gt_pos = gt_pos[valid_mask]
    overlap_gt_rot = gt_rot[valid_mask]
    if overlap_times.size == 0:
        raise RuntimeError("no overlap")

    left_indices = np.searchsorted(shifted_times, overlap_times, side="right") - 1
    right_indices = np.clip(left_indices + 1, 0, shifted_times.size - 1)
    left_indices = np.clip(left_indices, 0, shifted_times.size - 1)
    left_dt = np.abs(overlap_times - shifted_times[left_indices])
    right_dt = np.abs(shifted_times[right_indices] - overlap_times)
    bracket_gap = np.maximum(left_dt, right_dt)
    keep_mask = bracket_gap <= float(max_gap_sec)
    match_times = overlap_times[keep_mask]
    match_gt_pos = overlap_gt_pos[keep_mask]
    match_gt_rot = overlap_gt_rot[keep_mask]
    if match_times.size == 0:
        raise RuntimeError("no samples after max-gap filter")

    interp_pos = np.column_stack(
        [
            np.interp(match_times, shifted_times, est_pos[:, 0]),
            np.interp(match_times, shifted_times, est_pos[:, 1]),
            np.interp(match_times, shifted_times, est_pos[:, 2]),
        ]
    )
    interp_rot = interpolate_rotations(shifted_times, est_rot, match_times)
    return match_times, match_gt_pos, match_gt_rot, interp_pos, interp_rot


def evaluate_rmse(
    helpers: dict[str, Any],
    times: np.ndarray,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_pos: np.ndarray,
    est_rot: np.ndarray,
) -> dict[str, float]:
    se3, _, _, _, _ = helpers["evaluate_alignment"](
        "se3",
        False,
        gt_pos,
        gt_rot,
        est_pos,
        est_rot,
        times,
        1.0,
        30,
    )
    return {
        "ape_mm": float(se3.translation_metrics_m["rmse"] * 1000.0),
        "rpe_mm": float((se3.rpe_translation_metrics_m or {})["rmse"] * 1000.0),
        "ape_deg": float((se3.rotation_metrics_deg or {})["rmse"]),
        "rpe_deg": float((se3.rpe_rotation_metrics_deg or {})["rmse"]),
    }


def search_best_offset(
    helpers: dict[str, Any],
    gt_times: np.ndarray,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_times: np.ndarray,
    est_pos: np.ndarray,
    est_rot: np.ndarray,
    center_sec: float,
    coarse_span_ms: float,
    coarse_step_ms: float,
    fine_span_ms: float,
    fine_step_ms: float,
    max_gap_sec: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows: list[dict[str, float]] = []

    def evaluate_offset(offset_sec: float) -> dict[str, float] | None:
        try:
            times, m_gt_pos, m_gt_rot, m_est_pos, m_est_rot = strict_sync_match(
                gt_times,
                gt_pos,
                gt_rot,
                est_times,
                est_pos,
                est_rot,
                offset_sec,
                max_gap_sec,
            )
        except RuntimeError:
            return None
        metrics = evaluate_rmse(helpers, times, m_gt_pos, m_gt_rot, m_est_pos, m_est_rot)
        return {
            "offset_sec": float(offset_sec),
            "matched_samples": float(times.size),
            "matched_duration_s": float(times[-1] - times[0]) if times.size > 1 else 0.0,
            **metrics,
        }

    coarse_results = []
    for offset_sec in offset_candidates(center_sec, coarse_span_ms, coarse_step_ms):
        row = evaluate_offset(offset_sec)
        if row is not None:
            coarse_results.append(row)
            rows.append(row)
    if not coarse_results:
        raise RuntimeError("coarse offset search found no valid candidates")
    coarse_best = min(coarse_results, key=lambda item: (item["ape_mm"], item["rpe_mm"], abs(item["offset_sec"] - center_sec)))

    fine_results = []
    for offset_sec in offset_candidates(coarse_best["offset_sec"], fine_span_ms, fine_step_ms):
        row = evaluate_offset(offset_sec)
        if row is not None:
            fine_results.append(row)
            rows.append(row)
    if not fine_results:
        raise RuntimeError("fine offset search found no valid candidates")
    best = min(fine_results, key=lambda item: (item["ape_mm"], item["rpe_mm"], abs(item["offset_sec"] - coarse_best["offset_sec"])))

    uniq_rows: dict[int, dict[str, float]] = {}
    for row in rows:
        uniq_rows[int(round(row["offset_sec"] * 1_000_000_000.0))] = row
    return best, [uniq_rows[key] for key in sorted(uniq_rows)]


def main() -> int:
    args = parse_args()
    specs = load_episode_specs(args)
    output_root = args.output_root.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"shared_algo_sweep_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    api = load_eval_api()
    helpers = api["load_helpers"]()
    handeye_yaml = args.handeye_yaml.expanduser().resolve()
    windows = parse_number_list(args.windows, as_int=True)
    passes_list = parse_number_list(args.passes, as_int=True)

    gt_cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    traj_cache: dict[Path, SmoothedTrajectory] = {}
    episode_prepared: dict[str, dict[str, Any]] = {}
    for spec in specs:
        raw_traj = traj_cache.setdefault(spec.trajectory_txt, read_tum_trajectory(spec.trajectory_txt))
        gt_data = gt_cache.setdefault(spec.ground_truth, load_ground_truth(api, spec.ground_truth))
        midpoint_guess = default_offset_guess(gt_data[0], raw_traj.times_s)
        episode_prepared[spec.name] = {
            "spec": spec,
            "raw_traj": raw_traj,
            "gt": gt_data,
            "midpoint_guess_sec": float(midpoint_guess),
        }

    combo_rows: list[dict[str, Any]] = []
    best_combo: dict[str, Any] | None = None

    for window in windows:
        combo_passes = [0] if window <= 0 else passes_list
        for passes in combo_passes:
            combo_key = f"w{window}_p{passes}" if window > 0 else "raw"
            combo_dir = out_dir / combo_key
            combo_dir.mkdir(parents=True, exist_ok=True)
            episode_rows: list[dict[str, Any]] = []
            ape_values: list[float] = []
            ok_all = True

            for episode_name in sorted(episode_prepared):
                prepared = episode_prepared[episode_name]
                spec: EpisodeSpec = prepared["spec"]
                raw_traj: SmoothedTrajectory = prepared["raw_traj"]
                gt_times, gt_pos, gt_rot = prepared["gt"]
                center_sec = spec.selected_offset_sec if args.center_mode == "selected" else prepared["midpoint_guess_sec"]
                smoothed_positions = smooth_positions(raw_traj.positions, window, passes) if window > 0 and passes > 0 else raw_traj.positions.copy()
                smoothed_traj = SmoothedTrajectory(raw_traj.times_s, smoothed_positions, raw_traj.rotations)
                est_times, est_pos_tcp, est_rot_tcp = transform_estimate_to_tcp(
                    api,
                    spec.calibration_json,
                    spec.camera_rig,
                    handeye_yaml,
                    spec.estimate_frame,
                    smoothed_traj,
                )
                try:
                    best_offset, scan_rows = search_best_offset(
                        helpers,
                        gt_times,
                        gt_pos,
                        gt_rot,
                        est_times,
                        est_pos_tcp,
                        est_rot_tcp,
                        center_sec,
                        args.coarse_span_ms,
                        args.coarse_step_ms,
                        args.fine_span_ms,
                        args.fine_step_ms,
                        args.strict_sync_max_gap_sec,
                    )
                except Exception as exc:
                    ok_all = False
                    episode_rows.append(
                        {
                            "episode": episode_name,
                            "status": "failed",
                            "error": str(exc),
                            "window": window,
                            "passes": passes,
                        }
                    )
                    continue

                ape_values.append(float(best_offset["ape_mm"]))
                result_row = {
                    "episode": episode_name,
                    "status": "ok",
                    "window": window,
                    "passes": passes,
                    "best_offset_sec": best_offset["offset_sec"],
                    "ape_mm": best_offset["ape_mm"],
                    "rpe_mm": best_offset["rpe_mm"],
                    "ape_deg": best_offset["ape_deg"],
                    "rpe_deg": best_offset["rpe_deg"],
                    "matched_samples": int(best_offset["matched_samples"]),
                    "matched_duration_s": best_offset["matched_duration_s"],
                    "selected_offset_sec": spec.selected_offset_sec,
                    "selected_ape_mm": spec.selected_ape_mm,
                    "delta_vs_selected_mm": best_offset["ape_mm"] - spec.selected_ape_mm,
                    "search_center_sec": center_sec,
                    "scan_json": str(combo_dir / f"{episode_name}_offset_scan.json"),
                }
                episode_rows.append(result_row)
                (combo_dir / f"{episode_name}_offset_scan.json").write_text(
                    json.dumps({"best": best_offset, "rows": scan_rows}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            ok_episode_rows = [row for row in episode_rows if row.get("status") == "ok"]
            summary = {
                "combo": combo_key,
                "window": window,
                "passes": passes,
                "episodes_total": len(specs),
                "episodes_ok": len(ok_episode_rows),
                "all_ok": ok_all and len(ok_episode_rows) == len(specs),
                "mean_ape_mm": float(np.mean(ape_values)) if ape_values else float("nan"),
                "max_ape_mm": float(np.max(ape_values)) if ape_values else float("nan"),
                "episodes_under_10mm": int(sum(1 for value in ape_values if value < 10.0)),
                "episodes": episode_rows,
            }
            (combo_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            combo_rows.append(
                {
                    "combo": combo_key,
                    "window": window,
                    "passes": passes,
                    "episodes_ok": len(ok_episode_rows),
                    "all_ok": summary["all_ok"],
                    "episodes_under_10mm": summary["episodes_under_10mm"],
                    "mean_ape_mm": summary["mean_ape_mm"],
                    "max_ape_mm": summary["max_ape_mm"],
                }
            )
            if best_combo is None:
                best_combo = summary
            else:
                candidate_key = (
                    0 if summary["all_ok"] else 1,
                    -summary["episodes_under_10mm"],
                    summary["max_ape_mm"],
                    summary["mean_ape_mm"],
                    window if window > 0 else -1,
                    passes,
                )
                best_key = (
                    0 if best_combo["all_ok"] else 1,
                    -best_combo["episodes_under_10mm"],
                    best_combo["max_ape_mm"],
                    best_combo["mean_ape_mm"],
                    best_combo["window"] if best_combo["window"] > 0 else -1,
                    best_combo["passes"],
                )
                if candidate_key < best_key:
                    best_combo = summary

    summary_csv = out_dir / "combo_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "combo",
                "window",
                "passes",
                "episodes_ok",
                "all_ok",
                "episodes_under_10mm",
                "mean_ape_mm",
                "max_ape_mm",
            ],
        )
        writer.writeheader()
        writer.writerows(combo_rows)

    final_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "summary_csv": str(summary_csv),
        "best_combo": best_combo,
        "combo_rows": combo_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {summary_csv}")
    print(f"[OK] wrote {out_dir / 'summary.json'}")
    if best_combo is not None:
        print(json.dumps(best_combo, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
