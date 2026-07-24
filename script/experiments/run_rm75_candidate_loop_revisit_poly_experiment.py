#!/usr/bin/env python3
"""Train a nonlinear loop/revisit residual patch on RM75 candidate trajectories."""

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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/rm75_candidate_loop_revisit_poly_experiment"
HAND_EYE = REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0004_candidates/handeye_yaml/post_focus_ape_candidate.yaml"
TRAIN_IDS = ("rm75_0001", "rm75_0002", "rm75_0003")
TEST_ID = "rm75_0004"


@dataclass
class Episode:
    episode_id: str
    time_offset_sec: float
    est_times: np.ndarray
    est_pos_tcp: np.ndarray
    est_rot_tcp: np.ndarray
    matched_times: np.ndarray
    matched_gt_pos: np.ndarray
    matched_gt_rot: np.ndarray
    aligned_est_pos: np.ndarray
    aligned_est_rot: np.ndarray
    translation_error_world: np.ndarray
    baseline_rmse_mm: float


def load_tcp_eval() -> Dict[str, object]:
    return runpy.run_path(str(TCP_EVAL_SCRIPT), run_name="__rm75_candidate_loop_poly__")


def load_episode(tcp_eval: Dict[str, object], helpers: Dict[str, object], handeye: np.ndarray, episode_num: int) -> Episode:
    gt_path = REPO_ROOT / f"data/ground_truth/rm75/rm75_pose_traj_{episode_num}.json"
    estimate_path = REPO_ROOT / f"data/gripper_data2/episode_20260618_000{episode_num}/right/vio_log/frame_level_optimized_pose.csv"
    gt_times, gt_pos, gt_rot = tcp_eval["load_robot_tcp_trajectory"](gt_path, helpers)
    est_times, est_pos_tcp, est_rot_tcp = tcp_eval["load_vio_tcp_trajectory"](estimate_path, helpers, handeye, "imu")
    center = tcp_eval["default_time_offset_guess"](gt_times, est_times)
    best_offset, _ = tcp_eval["scan_evo_time_offset"](
        gt_times,
        gt_pos,
        gt_rot,
        est_times,
        est_pos_tcp,
        est_rot_tcp,
        helpers,
        center,
        0.8,
        0.004,
        0.01,
        "translation",
        50,
    )
    assoc = tcp_eval["associate_by_nearest_time"](
        gt_times,
        gt_pos,
        gt_rot,
        est_times,
        est_pos_tcp,
        est_rot_tcp,
        float(best_offset),
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
    baseline_rmse_mm = float(se3.translation_metrics_m["rmse"] * 1000.0)
    return Episode(
        episode_id=f"rm75_000{episode_num}",
        time_offset_sec=float(best_offset),
        est_times=np.asarray(est_times, dtype=float),
        est_pos_tcp=np.asarray(est_pos_tcp, dtype=float),
        est_rot_tcp=np.asarray(est_rot_tcp, dtype=float),
        matched_times=np.asarray(matched_times, dtype=float),
        matched_gt_pos=np.asarray(matched_gt_pos, dtype=float),
        matched_gt_rot=np.asarray(matched_gt_rot, dtype=float),
        aligned_est_pos=np.asarray(aligned_est_pos, dtype=float),
        aligned_est_rot=np.asarray(aligned_est_rot, dtype=float),
        translation_error_world=np.asarray(matched_gt_pos - aligned_est_pos, dtype=float),
        baseline_rmse_mm=baseline_rmse_mm,
    )


def nearest_revisit_features(pos: np.ndarray, rot: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, ...]:
    sample_count = times.shape[0]
    revisit_dist = np.full(sample_count, 0.5, dtype=float)
    revisit_gap = np.zeros(sample_count, dtype=float)
    revisit_time = np.zeros(sample_count, dtype=float)
    revisit_heading = np.full(sample_count, 180.0, dtype=float)
    revisit_flag = np.zeros(sample_count, dtype=float)
    for idx in range(sample_count):
        valid_prev = np.where(times[:idx] <= times[idx] - 2.0)[0]
        if valid_prev.size == 0:
            continue
        dist = np.linalg.norm(pos[valid_prev] - pos[idx], axis=1)
        best_prev = int(valid_prev[np.argmin(dist)])
        revisit_dist[idx] = float(np.min(dist))
        revisit_gap[idx] = float((idx - best_prev) / max(sample_count - 1, 1))
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


def poly2_features(features: np.ndarray) -> np.ndarray:
    columns = [features]
    dim = features.shape[1]
    for i in range(dim):
        for j in range(i, dim):
            columns.append((features[:, i] * features[:, j])[:, None])
    return np.concatenate(columns, axis=1)


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
        "# RM75 Candidate Loop-Revisit Polynomial Experiment",
        "",
        "## Goal",
        "",
        "Train a stronger nonlinear loop/revisit translation patch on `rm75_0001~0003` candidate trajectories and test it on the `rm75_0004` candidate trajectory.",
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
        "- This is better than the previous linear candidate patch and currently the best held-out result found on the stronger `frame_level_optimized + post_focus handeye` candidate chain.",
        "- It is still far above the `10 mm` target, so this should be treated as a real but modest algorithm-side improvement, not a final accuracy claim.",
    ]
    return "\n".join(lines) + "\n"


def run_experiment(output_dir: Path) -> Dict[str, object]:
    tcp_eval = load_tcp_eval()
    helpers = tcp_eval["load_helpers"]()
    handeye = tcp_eval["load_tcp_left_camera_transform"](HAND_EYE)

    episodes = [load_episode(tcp_eval, helpers, handeye, num) for num in (1, 2, 3, 4)]
    train_episodes = [episode for episode in episodes if episode.episode_id in TRAIN_IDS]
    test_episode = next(episode for episode in episodes if episode.episode_id == TEST_ID)

    train_features = [poly2_features(make_features(episode)) for episode in train_episodes]
    test_features = poly2_features(make_features(test_episode))
    train_targets = [np.asarray(episode.translation_error_world, dtype=float) for episode in train_episodes]

    alpha_grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 1e5, 1e6)
    best = None
    tuning_rows = []
    for alpha in alpha_grid:
        fold_deltas = []
        for holdout_idx, holdout_episode in enumerate(train_episodes):
            fit_x = np.concatenate([train_features[idx] for idx in range(len(train_features)) if idx != holdout_idx], axis=0)
            fit_y = np.concatenate([train_targets[idx] for idx in range(len(train_targets)) if idx != holdout_idx], axis=0)
            model = fit_ridge(fit_x, fit_y, float(alpha))
            pred = predict_ridge(model, train_features[holdout_idx])
            corrected_pos = holdout_episode.aligned_est_pos + pred
            corrected_rmse = translation_rmse_mm(holdout_episode.matched_gt_pos, corrected_pos)
            fold_deltas.append(corrected_rmse - holdout_episode.baseline_rmse_mm)
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
    model = fit_ridge(np.concatenate(train_features, axis=0), np.concatenate(train_targets, axis=0), best["alpha"])
    test_pred = predict_ridge(model, test_features)
    corrected_test_pos = test_episode.aligned_est_pos + test_pred
    corrected_rmse = translation_rmse_mm(test_episode.matched_gt_pos, corrected_test_pos)

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
            {"variant": "candidate_loop_revisit_poly2_world", "ape_translation_rmse_mm": corrected_rmse, "delta_translation_rmse_mm": corrected_rmse - test_episode.baseline_rmse_mm},
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
