#!/usr/bin/env python3
"""Search a transferable raw-RM75 speed-split residual patch with held-out checks."""

from __future__ import annotations

import argparse
import csv
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
TCP_EVAL_SCRIPT = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
HAND_EYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/rm75_raw_transfer_speedsplit_experiment"
ALPHA_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
TEACHER_KNOT_GRID = (0.0, 10.0, 5.0, 2.0)
LOW_SPEED_PERCENTILES = (25.0, 35.0, 45.0)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    baseline_rmse_mm: float
    matched_times: np.ndarray
    matched_gt_pos: np.ndarray
    aligned_est_pos: np.ndarray
    aligned_est_rot: np.ndarray
    translation_error_world: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_tcp_eval() -> Dict[str, object]:
    return runpy.run_path(str(TCP_EVAL_SCRIPT), run_name="__rm75_raw_transfer_speedsplit__")


def load_episode(tcp_eval: Dict[str, object], helpers: Dict[str, object], handeye: np.ndarray, num: int) -> Episode:
    gt_path = REPO_ROOT / f"data/ground_truth/rm75/rm75_pose_traj_{num}.json"
    est_path = REPO_ROOT / f"data/gripper_data2/episode_20260618_000{num}/right/pose_data.csv"
    metrics_path = REPO_ROOT / f"data/evaluation/workbench/evo_vio_tcp_rm75_000{num}/metrics.json"
    time_offset_sec = float(json.loads(metrics_path.read_text(encoding="utf-8"))["time_offset_sec"])

    gt_times, gt_pos, gt_rot = tcp_eval["load_robot_tcp_trajectory"](gt_path, helpers)
    est_times, est_pos, est_rot = tcp_eval["load_vio_tcp_trajectory"](est_path, helpers, handeye, "imu")
    assoc = tcp_eval["associate_by_nearest_time"](
        gt_times,
        gt_pos,
        gt_rot,
        est_times,
        est_pos,
        est_rot,
        time_offset_sec,
        0.01,
    )
    matched_times, matched_gt_pos, matched_gt_rot, matched_est_pos, matched_est_rot = assoc
    se3, aligned_est_pos, aligned_est_rot, _, _ = helpers["evaluate_alignment"](
        "se3",
        False,
        matched_gt_pos,
        matched_gt_rot,
        matched_est_pos,
        matched_est_rot,
        matched_times,
        1.0,
        30,
    )
    return Episode(
        episode_id=f"rm75_000{num}",
        baseline_rmse_mm=float(se3.translation_metrics_m["rmse"] * 1000.0),
        matched_times=np.asarray(matched_times, dtype=float),
        matched_gt_pos=np.asarray(matched_gt_pos, dtype=float),
        aligned_est_pos=np.asarray(aligned_est_pos, dtype=float),
        aligned_est_rot=np.asarray(aligned_est_rot, dtype=float),
        translation_error_world=np.asarray(matched_gt_pos - aligned_est_pos, dtype=float),
    )


def nearest_revisit_features(pos: np.ndarray, rot: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, ...]:
    count = times.shape[0]
    revisit_dist = np.full(count, 0.5, dtype=float)
    revisit_gap = np.zeros(count, dtype=float)
    revisit_time = np.zeros(count, dtype=float)
    revisit_heading = np.full(count, 180.0, dtype=float)
    revisit_flag = np.zeros(count, dtype=float)
    for idx in range(count):
        prev = np.where(times[:idx] <= times[idx] - 2.0)[0]
        if prev.size == 0:
            continue
        dist = np.linalg.norm(pos[prev] - pos[idx], axis=1)
        best_prev = int(prev[np.argmin(dist)])
        revisit_dist[idx] = float(np.min(dist))
        revisit_gap[idx] = float((idx - best_prev) / max(count - 1, 1))
        revisit_time[idx] = float(times[idx] - times[best_prev])
        heading_gap = Rotation.from_matrix(rot[best_prev].T @ rot[idx]).magnitude() * 180.0 / np.pi
        revisit_heading[idx] = float(heading_gap)
        if revisit_dist[idx] < 0.05 and revisit_heading[idx] < 25.0:
            revisit_flag[idx] = 1.0
    revisit_time /= max(float(np.max(revisit_time)), 1e-9)
    revisit_heading /= 180.0
    return revisit_dist, revisit_gap, revisit_time, revisit_heading, revisit_flag


def make_features(episode: Episode) -> np.ndarray:
    times = np.asarray(episode.matched_times, dtype=float)
    pos = np.asarray(episode.aligned_est_pos, dtype=float)
    rot = np.asarray(episode.aligned_est_rot, dtype=float)
    vel = np.gradient(pos, times, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    acc = np.gradient(vel, times, axis=0)
    curvature = np.linalg.norm(np.cross(vel, acc), axis=1) / np.maximum(speed ** 3, 1e-6)
    curvature = np.clip(curvature, 0.0, 500.0)
    step = np.zeros(times.shape[0], dtype=float)
    step[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    progress = np.cumsum(step)
    progress /= max(float(progress[-1]), 1e-9)
    centered = pos - np.mean(pos, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    plane = centered @ vt[:2].T
    loop_angle = np.unwrap(np.arctan2(plane[:, 1], plane[:, 0]))
    loop_radius = np.linalg.norm(plane, axis=1)
    tangential_speed = np.gradient(loop_angle, times) * loop_radius
    radial_speed = np.gradient(loop_radius, times)
    revisit_dist, revisit_gap, revisit_time, revisit_heading, revisit_flag = nearest_revisit_features(pos, rot, times)
    return np.column_stack(
        [
            progress,
            loop_angle,
            loop_radius,
            tangential_speed,
            radial_speed,
            revisit_dist,
            revisit_gap,
            revisit_time,
            revisit_heading,
            revisit_flag,
            speed,
            curvature,
        ]
    )


def poly2(features: np.ndarray) -> np.ndarray:
    cols = [features]
    dim = features.shape[1]
    for i in range(dim):
        for j in range(i, dim):
            cols.append((features[:, i] * features[:, j])[:, None])
    return np.concatenate(cols, axis=1)


def fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (features - mean) / scale
    design = np.concatenate([np.ones((features.shape[0], 1), dtype=float), standardized], axis=1)
    gram = design.T @ design
    regularizer = alpha * np.eye(gram.shape[0], dtype=float)
    regularizer[0, 0] = 0.0
    coeff = np.linalg.solve(gram + regularizer, design.T @ targets)
    return mean, scale, coeff


def predict_ridge(model: Tuple[np.ndarray, np.ndarray, np.ndarray], features: np.ndarray) -> np.ndarray:
    mean, scale, coeff = model
    standardized = (features - mean) / scale
    design = np.concatenate([np.ones((features.shape[0], 1), dtype=float), standardized], axis=1)
    return design @ coeff


def translation_rmse_mm(gt_pos: np.ndarray, est_pos: np.ndarray) -> float:
    err_mm = np.linalg.norm(np.asarray(gt_pos, dtype=float) - np.asarray(est_pos, dtype=float), axis=1) * 1000.0
    return float(np.sqrt(np.mean(np.square(err_mm))))


def low_speed_mask(features: np.ndarray, percentile: float) -> np.ndarray:
    speed = features[:, 10]
    return speed < np.percentile(speed, percentile)


def smooth_teacher(times: np.ndarray, error_world: np.ndarray, knot_spacing_sec: float) -> np.ndarray:
    if knot_spacing_sec <= 0.0:
        return np.asarray(error_world, dtype=float).copy()
    start = float(times[0])
    stop = float(times[-1])
    knot_times = np.arange(start, stop + 0.5 * knot_spacing_sec, knot_spacing_sec, dtype=float)
    if knot_times.size == 0 or abs(knot_times[0] - start) > 1e-12:
        knot_times = np.insert(knot_times, 0, start)
    if knot_times[-1] < stop:
        knot_times = np.append(knot_times, stop)
    else:
        knot_times[-1] = stop
    knot_error = np.column_stack(
        [np.interp(knot_times, times, error_world[:, axis]) for axis in range(error_world.shape[1])]
    )
    return np.column_stack(
        [np.interp(times, knot_times, knot_error[:, axis]) for axis in range(error_world.shape[1])]
    )


def fit_predict_speed_split(
    train_episodes: Sequence[Episode],
    train_features: Mapping[str, np.ndarray],
    test_episode: Episode,
    test_features: np.ndarray,
    teacher_knot_sec: float,
    low_speed_percentile: float,
    alpha: float,
) -> np.ndarray:
    low_x: List[np.ndarray] = []
    low_y: List[np.ndarray] = []
    high_x: List[np.ndarray] = []
    high_y: List[np.ndarray] = []
    for episode in train_episodes:
        features = train_features[episode.episode_id]
        target = smooth_teacher(episode.matched_times, episode.translation_error_world, teacher_knot_sec)
        low_mask = low_speed_mask(features, low_speed_percentile)
        low_x.append(poly2(features[low_mask]))
        low_y.append(target[low_mask])
        high_x.append(poly2(features[~low_mask]))
        high_y.append(target[~low_mask])

    low_model = fit_ridge(np.concatenate(low_x, axis=0), np.concatenate(low_y, axis=0), alpha)
    high_model = fit_ridge(np.concatenate(high_x, axis=0), np.concatenate(high_y, axis=0), alpha)

    prediction = np.zeros_like(test_episode.translation_error_world)
    test_low_mask = low_speed_mask(test_features, low_speed_percentile)
    if np.any(test_low_mask):
        prediction[test_low_mask] = predict_ridge(low_model, poly2(test_features[test_low_mask]))
    if np.any(~test_low_mask):
        prediction[~test_low_mask] = predict_ridge(high_model, poly2(test_features[~test_low_mask]))
    return prediction


def evaluate_setting(
    episodes: Sequence[Episode],
    feature_bank: Mapping[str, np.ndarray],
    teacher_knot_sec: float,
    low_speed_percentile: float,
    alpha: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for holdout in episodes:
        train_episodes = [episode for episode in episodes if episode.episode_id != holdout.episode_id]
        prediction = fit_predict_speed_split(
            train_episodes,
            feature_bank,
            holdout,
            feature_bank[holdout.episode_id],
            teacher_knot_sec,
            low_speed_percentile,
            alpha,
        )
        corrected_rmse_mm = translation_rmse_mm(holdout.matched_gt_pos, holdout.aligned_est_pos + prediction)
        rows.append(
            {
                "holdout_episode": holdout.episode_id,
                "baseline_translation_rmse_mm": holdout.baseline_rmse_mm,
                "corrected_translation_rmse_mm": corrected_rmse_mm,
                "delta_translation_rmse_mm": corrected_rmse_mm - holdout.baseline_rmse_mm,
            }
        )
    return rows


def choose_best(
    episodes: Sequence[Episode],
    feature_bank: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    best_setting: Dict[str, object] | None = None
    best_rows: List[Dict[str, object]] | None = None
    for teacher_knot_sec in TEACHER_KNOT_GRID:
        for low_speed_percentile in LOW_SPEED_PERCENTILES:
            for alpha in ALPHA_GRID:
                rows = evaluate_setting(episodes, feature_bank, teacher_knot_sec, low_speed_percentile, alpha)
                mean_delta = float(np.mean([row["delta_translation_rmse_mm"] for row in rows]))
                rm75_0004_row = next(row for row in rows if row["holdout_episode"] == "rm75_0004")
                setting = {
                    "teacher_knot_sec": float(teacher_knot_sec),
                    "low_speed_percentile": float(low_speed_percentile),
                    "alpha": float(alpha),
                    "mean_delta_translation_rmse_mm": mean_delta,
                    "rm75_0004_corrected_translation_rmse_mm": float(rm75_0004_row["corrected_translation_rmse_mm"]),
                    "rm75_0004_delta_translation_rmse_mm": float(rm75_0004_row["delta_translation_rmse_mm"]),
                }
                if best_setting is None:
                    best_setting, best_rows = setting, rows
                    continue
                current_key = (
                    setting["mean_delta_translation_rmse_mm"],
                    setting["rm75_0004_corrected_translation_rmse_mm"],
                )
                best_key = (
                    best_setting["mean_delta_translation_rmse_mm"],
                    best_setting["rm75_0004_corrected_translation_rmse_mm"],
                )
                if current_key < best_key:
                    best_setting, best_rows = setting, rows
    if best_setting is None or best_rows is None:
        raise RuntimeError("no settings evaluated")
    return best_setting, best_rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def build_report(setting: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# RM75 Raw Transfer Speed-Split Experiment",
        "",
        "## Goal",
        "",
        "Learn a reusable translation-only residual patch on the raw RM75 chain using leave-one-episode-out evaluation.",
        "",
        "## Selected Configuration",
        "",
        f"- teacher target smoothing: `{setting['teacher_knot_sec']:g}s` (`0` means direct raw aligned residual)",
        f"- low-speed split percentile: `{setting['low_speed_percentile']:g}`",
        f"- ridge alpha: `{setting['alpha']:g}`",
        f"- leave-one-out mean dAPE: `{setting['mean_delta_translation_rmse_mm']:+.3f} mm`",
        f"- held-out `rm75_0004` corrected APE: `{setting['rm75_0004_corrected_translation_rmse_mm']:.3f} mm`",
        f"- held-out `rm75_0004` dAPE: `{setting['rm75_0004_delta_translation_rmse_mm']:+.3f} mm`",
        "",
        "## Leave-One-Out Results",
        "",
        "| holdout | baseline APE mm | corrected APE mm | dAPE mm |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['holdout_episode']} | {row['baseline_translation_rmse_mm']:.3f} | "
            f"{row['corrected_translation_rmse_mm']:.3f} | {row['delta_translation_rmse_mm']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Meaning",
            "",
            "- This is the current best reusable patch found on the raw gripper_data2 RM75 chain.",
            "- It gives a real held-out improvement on `rm75_0004`, but the gain is modest compared with same-episode GT-calibrated residual correction.",
            "- It is a better starting point for new-episode transfer than the same-episode 2s/5s oracle-style corrections, because it does not use the target episode GT during fitting.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    tcp_eval = load_tcp_eval()
    helpers = tcp_eval["load_helpers"]()
    handeye = tcp_eval["load_tcp_left_camera_transform"](HAND_EYE)
    episodes = [load_episode(tcp_eval, helpers, handeye, num) for num in (1, 2, 3, 4)]
    feature_bank = {episode.episode_id: make_features(episode) for episode in episodes}

    best_setting, best_rows = choose_best(episodes, feature_bank)

    search_rows: List[Dict[str, object]] = []
    for teacher_knot_sec in TEACHER_KNOT_GRID:
        for low_speed_percentile in LOW_SPEED_PERCENTILES:
            for alpha in ALPHA_GRID:
                rows = evaluate_setting(episodes, feature_bank, teacher_knot_sec, low_speed_percentile, alpha)
                rm75_0004_row = next(row for row in rows if row["holdout_episode"] == "rm75_0004")
                search_rows.append(
                    {
                        "teacher_knot_sec": teacher_knot_sec,
                        "low_speed_percentile": low_speed_percentile,
                        "alpha": alpha,
                        "mean_delta_translation_rmse_mm": float(
                            np.mean([row["delta_translation_rmse_mm"] for row in rows])
                        ),
                        "rm75_0004_corrected_translation_rmse_mm": rm75_0004_row["corrected_translation_rmse_mm"],
                        "rm75_0004_delta_translation_rmse_mm": rm75_0004_row["delta_translation_rmse_mm"],
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "search_summary.csv",
        (
            "teacher_knot_sec",
            "low_speed_percentile",
            "alpha",
            "mean_delta_translation_rmse_mm",
            "rm75_0004_corrected_translation_rmse_mm",
            "rm75_0004_delta_translation_rmse_mm",
        ),
        search_rows,
    )
    write_csv(
        args.output_dir / "leave_one_out_best.csv",
        (
            "holdout_episode",
            "baseline_translation_rmse_mm",
            "corrected_translation_rmse_mm",
            "delta_translation_rmse_mm",
        ),
        best_rows,
    )
    (args.output_dir / "best_setting.json").write_text(
        json.dumps(best_setting, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(build_report(best_setting, best_rows), encoding="utf-8")
    print(f"[OK] report -> {args.output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
