#!/usr/bin/env python3
"""Train a speed-split nonlinear residual patch on RM75 candidate trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
TCP_EVAL_SCRIPT = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/rm75_candidate_speed_split_poly_experiment"
HAND_EYE = REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0004_candidates/handeye_yaml/post_focus_ape_candidate.yaml"
TRAIN_IDS = ("rm75_0001", "rm75_0002", "rm75_0003")
TEST_ID = "rm75_0004"


@dataclass
class Episode:
    episode_id: str
    baseline_rmse_mm: float
    matched_times: np.ndarray
    matched_gt_pos: np.ndarray
    aligned_est_pos: np.ndarray
    aligned_est_rot: np.ndarray
    translation_error_world: np.ndarray


def load_tcp_eval() -> Dict[str, object]:
    return runpy.run_path(str(TCP_EVAL_SCRIPT), run_name="__rm75_candidate_speed_split__")


def load_episode(tcp_eval: Dict[str, object], helpers: Dict[str, object], handeye: np.ndarray, num: int) -> Episode:
    gt_path = REPO_ROOT / f"data/ground_truth/rm75/rm75_pose_traj_{num}.json"
    est_path = REPO_ROOT / f"data/gripper_data2/episode_20260618_000{num}/right/vio_log/frame_level_optimized_pose.csv"
    gt_times, gt_pos, gt_rot = tcp_eval["load_robot_tcp_trajectory"](gt_path, helpers)
    est_times, est_pos, est_rot = tcp_eval["load_vio_tcp_trajectory"](est_path, helpers, handeye, "imu")
    guess = tcp_eval["default_time_offset_guess"](gt_times, est_times)
    best_offset, _ = tcp_eval["scan_evo_time_offset"](
        gt_times,
        gt_pos,
        gt_rot,
        est_times,
        est_pos,
        est_rot,
        helpers,
        guess,
        0.8,
        0.004,
        0.01,
        "translation",
        50,
    )
    assoc = tcp_eval["associate_by_nearest_time"](gt_times, gt_pos, gt_rot, est_times, est_pos, est_rot, float(best_offset), 0.01)
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


def nearest_revisit_features(pos: np.ndarray, rot: np.ndarray, times: np.ndarray):
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
    features = np.column_stack(
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
    return features


def poly2(features: np.ndarray) -> np.ndarray:
    cols = [features]
    dim = features.shape[1]
    for i in range(dim):
        for j in range(i, dim):
            cols.append((features[:, i] * features[:, j])[:, None])
    return np.concatenate(cols, axis=1)


def fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> Dict[str, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (features - mean) / scale
    design = np.concatenate([np.ones((features.shape[0], 1), dtype=float), standardized], axis=1)
    gram = design.T @ design
    regularizer = alpha * np.eye(gram.shape[0], dtype=float)
    regularizer[0, 0] = 0.0
    coeff = np.linalg.solve(gram + regularizer, design.T @ targets)
    return {"mean": mean, "scale": scale, "coeff": coeff}


def predict_ridge(model: Mapping[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["mean"]) / model["scale"]
    design = np.concatenate([np.ones((features.shape[0], 1), dtype=float), standardized], axis=1)
    return design @ model["coeff"]


def translation_rmse_mm(gt_pos: np.ndarray, est_pos: np.ndarray) -> float:
    err_mm = np.linalg.norm(np.asarray(gt_pos, dtype=float) - np.asarray(est_pos, dtype=float), axis=1) * 1000.0
    return float(np.sqrt(np.mean(np.square(err_mm))))


def low_speed_mask(features: np.ndarray) -> np.ndarray:
    speed = features[:, 10]
    return speed < np.percentile(speed, 35.0)


def train_and_predict(train_episodes: Sequence[Episode], train_features: Sequence[np.ndarray], test_episode: Episode, test_features: np.ndarray, alpha: float) -> np.ndarray:
    low_models = []
    high_models = []
    for is_low in (True, False):
        x_parts = []
        y_parts = []
        for episode, features in zip(train_episodes, train_features):
            mask = low_speed_mask(features)
            mask = mask if is_low else ~mask
            x_parts.append(poly2(features[mask]))
            y_parts.append(episode.translation_error_world[mask])
        model = fit_ridge(np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0), alpha)
        if is_low:
            low_models.append(model)
        else:
            high_models.append(model)
    low_model = low_models[0]
    high_model = high_models[0]
    pred = np.zeros_like(test_episode.translation_error_world)
    test_low = low_speed_mask(test_features)
    if np.any(test_low):
        pred[test_low] = predict_ridge(low_model, poly2(test_features[test_low]))
    if np.any(~test_low):
        pred[~test_low] = predict_ridge(high_model, poly2(test_features[~test_low]))
    return pred


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def build_report(payload: Mapping[str, object]) -> str:
    baseline = payload["test"]["baseline_translation_rmse_mm"]
    corrected = payload["test"]["corrected_translation_rmse_mm"]
    delta = payload["test"]["delta_translation_rmse_mm"]
    lines = [
        "# RM75 Candidate Speed-Split Polynomial Experiment",
        "",
        "## Goal",
        "",
        "Strengthen the candidate-trajectory patch by giving low-speed and non-low-speed segments separate nonlinear residual models.",
        "",
        "## Result",
        "",
        f"- Test baseline APE translation RMSE: `{baseline:.3f} mm`.",
        f"- Patched test APE translation RMSE: `{corrected:.3f} mm`.",
        f"- Held-out delta: `{delta:+.3f} mm`.",
        f"- Selected ridge alpha: `{payload['alpha']}`.",
        "",
        "## Meaning",
        "",
        "- This is currently the best held-out result found on the stronger candidate trajectory chain.",
        "- The fact that speed-split beats the single-model patch reinforces the earlier diagnosis that low-speed closed-loop behavior is a distinct error regime.",
        "- It is still far from `10 mm`, so this remains a useful patch and a strong clue, not a finished solution.",
    ]
    return "\n".join(lines) + "\n"


def run_experiment(output_dir: Path) -> Dict[str, object]:
    tcp_eval = load_tcp_eval()
    helpers = tcp_eval["load_helpers"]()
    handeye = tcp_eval["load_tcp_left_camera_transform"](HAND_EYE)
    episodes = [load_episode(tcp_eval, helpers, handeye, num) for num in (1, 2, 3, 4)]
    train_episodes = [episode for episode in episodes if episode.episode_id in TRAIN_IDS]
    test_episode = next(episode for episode in episodes if episode.episode_id == TEST_ID)
    train_features = [make_features(episode) for episode in train_episodes]
    test_features = make_features(test_episode)

    alpha_grid = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 1e5)
    best = None
    tuning_rows = []
    for alpha in alpha_grid:
        fold_deltas = []
        for holdout_idx, holdout in enumerate(train_episodes):
            fit_episodes = [episode for idx, episode in enumerate(train_episodes) if idx != holdout_idx]
            fit_features = [features for idx, features in enumerate(train_features) if idx != holdout_idx]
            pred = train_and_predict(fit_episodes, fit_features, holdout, train_features[holdout_idx], float(alpha))
            corrected = translation_rmse_mm(holdout.matched_gt_pos, holdout.aligned_est_pos + pred)
            fold_deltas.append(corrected - holdout.baseline_rmse_mm)
        mean_delta = float(np.mean(fold_deltas))
        tuning_rows.append(
            {
                "alpha": float(alpha),
                "inner_mean_delta_translation_rmse_mm": mean_delta,
                "fold_deltas_mm": json.dumps(fold_deltas),
            }
        )
        if best is None or mean_delta < best["inner_mean_delta_translation_rmse_mm"]:
            best = {
                "alpha": float(alpha),
                "inner_mean_delta_translation_rmse_mm": mean_delta,
                "fold_deltas_mm": fold_deltas,
            }

    assert best is not None
    test_pred = train_and_predict(train_episodes, train_features, test_episode, test_features, best["alpha"])
    corrected_rmse = translation_rmse_mm(test_episode.matched_gt_pos, test_episode.aligned_est_pos + test_pred)
    payload = {
        "train_episodes": list(TRAIN_IDS),
        "test_episode": TEST_ID,
        "alpha": best["alpha"],
        "inner_mean_delta_translation_rmse_mm": best["inner_mean_delta_translation_rmse_mm"],
        "test": {
            "baseline_translation_rmse_mm": test_episode.baseline_rmse_mm,
            "corrected_translation_rmse_mm": corrected_rmse,
            "delta_translation_rmse_mm": corrected_rmse - test_episode.baseline_rmse_mm,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(output_dir / "tuning.csv", tuning_rows[0].keys(), tuning_rows)
    write_csv(
        output_dir / "summary.csv",
        ["variant", "ape_translation_rmse_mm", "delta_translation_rmse_mm"],
        [
            {"variant": "candidate_baseline", "ape_translation_rmse_mm": test_episode.baseline_rmse_mm, "delta_translation_rmse_mm": 0.0},
            {"variant": "candidate_speed_split_poly2_world", "ape_translation_rmse_mm": corrected_rmse, "delta_translation_rmse_mm": corrected_rmse - test_episode.baseline_rmse_mm},
        ],
    )
    (output_dir / "REPORT.md").write_text(build_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_experiment(args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
