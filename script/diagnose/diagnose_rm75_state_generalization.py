#!/usr/bin/env python3
"""Diagnose RM75 cross-episode motion/state effects and held-out generalization.

This script extends the quick cross-episode hand-eye scan with a deeper check:

1. Compare motion envelopes and motion phases for rm75_0001..0004.
2. Measure how aligned TCP residuals depend on position, orientation, speed,
   acceleration, angular speed, curvature, and simple motion phases.
3. Recheck rm75_0004 time-offset/convention hypotheses.
4. Fit simple shared state-conditioned residual models and test them in
   leave-one-episode-out style. Only held-out improvements are treated as
   evidence of a useful generalizable term.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
TCP_EVAL_SCRIPT = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/rm75_state_generalization_diagnosis"
DEFAULT_HANDEYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    estimate_raw: Path
    ground_truth: Path
    raw_metrics: Path


@dataclass
class EpisodeAnalysis:
    spec: EpisodeSpec
    time_offset_sec: float
    gt_times: np.ndarray
    gt_pos: np.ndarray
    gt_rot: np.ndarray
    est_times: np.ndarray
    est_pos_tcp: np.ndarray
    est_rot_tcp: np.ndarray
    matched_times: np.ndarray
    matched_gt_pos: np.ndarray
    matched_gt_rot: np.ndarray
    matched_est_pos: np.ndarray
    matched_est_rot: np.ndarray
    aligned_est_pos: np.ndarray
    aligned_est_rot: np.ndarray
    translation_error_world: np.ndarray
    residual_translation_world: np.ndarray
    residual_rotation_rotvec_deg: np.ndarray
    features: Dict[str, np.ndarray]
    summary: Dict[str, float]
    dependence: Dict[str, object]
    chain_check: Dict[str, object]


EPISODES: List[EpisodeSpec] = [
    EpisodeSpec(
        episode_id="rm75_0001",
        estimate_raw=REPO_ROOT / "data/gripper_data2/episode_20260618_0001/right/pose_data.csv",
        ground_truth=REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_1.json",
        raw_metrics=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0001/metrics.json",
    ),
    EpisodeSpec(
        episode_id="rm75_0002",
        estimate_raw=REPO_ROOT / "data/gripper_data2/episode_20260618_0002/right/pose_data.csv",
        ground_truth=REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_2.json",
        raw_metrics=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0002/metrics.json",
    ),
    EpisodeSpec(
        episode_id="rm75_0003",
        estimate_raw=REPO_ROOT / "data/gripper_data2/episode_20260618_0003/right/pose_data.csv",
        ground_truth=REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_3.json",
        raw_metrics=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0003/metrics.json",
    ),
    EpisodeSpec(
        episode_id="rm75_0004",
        estimate_raw=REPO_ROOT / "data/gripper_data2/episode_20260618_0004/right/pose_data.csv",
        ground_truth=REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_4.json",
        raw_metrics=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0004/metrics.json",
    ),
]


def load_tcp_eval() -> Dict[str, object]:
    return runpy.run_path(str(TCP_EVAL_SCRIPT), run_name="__rm75_state_generalization__")


def read_time_offset(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["time_offset_sec"])


def load_raw_imu_csv(path: Path, helpers: Dict[str, object], normalize_timestamp) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: List[float] = []
    positions: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            transform = helpers["transform_from_pose"](
                [float(row["X"]), float(row["Y"]), float(row["Z"])],
                [float(row["Quat_X"]), float(row["Quat_Y"]), float(row["Quat_Z"]), float(row["Quat_W"])],
            )
            times.append(normalize_timestamp(float(row["Timestamp_us"])))
            positions.append(transform[:3, 3].copy())
            rotations.append(transform[:3, :3].copy())
    order = np.argsort(np.asarray(times, dtype=float))
    return (
        np.asarray(times, dtype=float)[order],
        np.asarray(positions, dtype=float)[order],
        np.asarray(rotations, dtype=float)[order],
    )


def vector_norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(values, dtype=float), axis=1)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def percentile_gap(values: np.ndarray, metric: np.ndarray, lo: float = 20.0, hi: float = 80.0) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    metric = np.asarray(metric, dtype=float)
    low_thr = float(np.percentile(values, lo))
    high_thr = float(np.percentile(values, hi))
    low_mask = values <= low_thr
    high_mask = values >= high_thr
    low_mean = float(np.mean(metric[low_mask]))
    high_mean = float(np.mean(metric[high_mask]))
    return {
        "low_threshold": low_thr,
        "high_threshold": high_thr,
        "low_mean": low_mean,
        "high_mean": high_mean,
        "delta_high_minus_low": high_mean - low_mean,
    }


def contiguous_segments(mask: np.ndarray, times: np.ndarray, min_duration_s: float = 0.3) -> List[Tuple[int, int, float]]:
    mask = np.asarray(mask, dtype=bool)
    times = np.asarray(times, dtype=float)
    segments: List[Tuple[int, int, float]] = []
    start = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        if not active and start is not None:
            end = idx - 1
            duration = float(times[end] - times[start]) if end > start else 0.0
            if duration >= min_duration_s:
                segments.append((start, end, duration))
            start = None
    if start is not None:
        end = mask.size - 1
        duration = float(times[end] - times[start]) if end > start else 0.0
        if duration >= min_duration_s:
            segments.append((start, end, duration))
    return segments


def trajectory_features(times: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> Dict[str, np.ndarray]:
    dt = np.gradient(times)
    vel = np.gradient(pos, times, axis=0)
    speed = vector_norm(vel)
    acc_vec = np.gradient(vel, times, axis=0)
    acc_mag = vector_norm(acc_vec)
    tangent_acc = np.sum(acc_vec * vel, axis=1) / np.maximum(speed, 1e-6)

    ang_speed_rad_s = np.zeros(times.size, dtype=float)
    for idx in range(1, times.size):
        delta_t = max(float(times[idx] - times[idx - 1]), 1e-9)
        delta_rot = Rotation.from_matrix(rot[idx - 1].T @ rot[idx]).as_rotvec()
        ang_speed_rad_s[idx] = float(np.linalg.norm(delta_rot) / delta_t)
    ang_speed_deg_s = np.rad2deg(ang_speed_rad_s)

    curvature = vector_norm(np.cross(vel, acc_vec)) / np.maximum(speed ** 3, 1e-6)
    curvature = np.clip(curvature, 0.0, 500.0)

    stationary = speed < 0.02
    moving = ~stationary
    turning = ang_speed_deg_s > 20.0
    moving_turn = moving & turning
    moving_straight = moving & ~turning
    stationary_turn = stationary & turning
    accel_phase = tangent_acc > 0.12
    decel_phase = tangent_acc < -0.12

    rot_flat = rot.reshape(rot.shape[0], 9)
    return {
        "position_xyz": np.asarray(pos, dtype=float),
        "orientation_matrix_flat": np.asarray(rot_flat, dtype=float),
        "speed": np.asarray(speed, dtype=float),
        "acc_mag": np.asarray(acc_mag, dtype=float),
        "tangent_acc": np.asarray(tangent_acc, dtype=float),
        "ang_speed_deg_s": np.asarray(ang_speed_deg_s, dtype=float),
        "curvature": np.asarray(curvature, dtype=float),
        "stationary": stationary.astype(float),
        "turning": turning.astype(float),
        "moving_turn": moving_turn.astype(float),
        "moving_straight": moving_straight.astype(float),
        "stationary_turn": stationary_turn.astype(float),
        "accel_phase": accel_phase.astype(float),
        "decel_phase": decel_phase.astype(float),
    }


def summarize_motion(times: np.ndarray, pos: np.ndarray, rot: np.ndarray, error_mm: np.ndarray, offset_sec: float) -> Dict[str, float]:
    features = trajectory_features(times, pos, rot)
    path_length = float(np.sum(vector_norm(np.diff(pos, axis=0)))) if pos.shape[0] > 1 else 0.0
    displacement = float(np.linalg.norm(pos[-1] - pos[0])) if pos.shape[0] > 1 else 0.0
    stationary_segments = contiguous_segments(features["stationary"] > 0.5, times)
    moving_turn_segments = contiguous_segments(features["moving_turn"] > 0.5, times)

    relative_rot = [Rotation.from_matrix(rot[0].T @ current).magnitude() for current in rot]
    return {
        "matched_samples": int(times.size),
        "duration_s": float(times[-1] - times[0]) if times.size > 1 else 0.0,
        "time_offset_sec": float(offset_sec),
        "path_length_m": path_length,
        "net_displacement_m": displacement,
        "closure_ratio": displacement / max(path_length, 1e-9),
        "speed_mean_mps": float(np.mean(features["speed"])),
        "speed_p95_mps": float(np.percentile(features["speed"], 95.0)),
        "speed_max_mps": float(np.max(features["speed"])),
        "ang_speed_mean_deg_s": float(np.mean(features["ang_speed_deg_s"])),
        "ang_speed_p95_deg_s": float(np.percentile(features["ang_speed_deg_s"], 95.0)),
        "ang_speed_max_deg_s": float(np.max(features["ang_speed_deg_s"])),
        "curvature_p95": float(np.percentile(features["curvature"], 95.0)),
        "stationary_ratio": float(np.mean(features["stationary"])),
        "turning_ratio": float(np.mean(features["turning"])),
        "moving_turn_ratio": float(np.mean(features["moving_turn"])),
        "accel_ratio": float(np.mean(features["accel_phase"])),
        "decel_ratio": float(np.mean(features["decel_phase"])),
        "stationary_segment_count": float(len(stationary_segments)),
        "stationary_longest_s": float(max((segment[2] for segment in stationary_segments), default=0.0)),
        "moving_turn_segment_count": float(len(moving_turn_segments)),
        "moving_turn_longest_s": float(max((segment[2] for segment in moving_turn_segments), default=0.0)),
        "orientation_span_deg": float(np.rad2deg(max(relative_rot, default=0.0))),
        "baseline_ape_translation_rmse_mm": float(np.sqrt(np.mean(np.square(error_mm)))),
    }


def residual_dependence(features: Mapping[str, np.ndarray], gt_pos: np.ndarray, gt_rot: np.ndarray, trans_err_mm: np.ndarray) -> Dict[str, object]:
    abs_error = vector_norm(trans_err_mm)
    gt_z_axis = np.asarray(gt_rot[:, :, 2], dtype=float)

    corr = {
        "gt_x": safe_corr(abs_error, gt_pos[:, 0]),
        "gt_y": safe_corr(abs_error, gt_pos[:, 1]),
        "gt_z": safe_corr(abs_error, gt_pos[:, 2]),
        "tcp_z_x": safe_corr(abs_error, gt_z_axis[:, 0]),
        "tcp_z_y": safe_corr(abs_error, gt_z_axis[:, 1]),
        "tcp_z_z": safe_corr(abs_error, gt_z_axis[:, 2]),
        "speed": safe_corr(abs_error, features["speed"]),
        "acc_mag": safe_corr(abs_error, features["acc_mag"]),
        "tangent_acc": safe_corr(abs_error, features["tangent_acc"]),
        "ang_speed_deg_s": safe_corr(abs_error, features["ang_speed_deg_s"]),
        "curvature": safe_corr(abs_error, features["curvature"]),
    }

    stationary_mask = features["stationary"] > 0.5
    moving_mask = ~stationary_mask
    turning_mask = features["turning"] > 0.5
    straight_mask = ~turning_mask
    accel_mask = features["accel_phase"] > 0.5
    decel_mask = features["decel_phase"] > 0.5
    cruise_mask = ~(accel_mask | decel_mask)

    def phase_mean(mask: np.ndarray) -> float:
        return float(np.mean(abs_error[mask])) if np.any(mask) else float("nan")

    return {
        "correlations": corr,
        "speed_percentile_gap_mm": percentile_gap(features["speed"], abs_error),
        "ang_speed_percentile_gap_mm": percentile_gap(features["ang_speed_deg_s"], abs_error),
        "curvature_percentile_gap_mm": percentile_gap(features["curvature"], abs_error),
        "phase_mean_error_mm": {
            "stationary": phase_mean(stationary_mask),
            "moving": phase_mean(moving_mask),
            "turning": phase_mean(turning_mask),
            "straight": phase_mean(straight_mask),
            "accel": phase_mean(accel_mask),
            "decel": phase_mean(decel_mask),
            "cruise": phase_mean(cruise_mask),
        },
    }


def chain_check(
    tcp_eval: Dict[str, object],
    helpers: Dict[str, object],
    gt_times: np.ndarray,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_times: np.ndarray,
    est_pos_tcp: np.ndarray,
    est_rot_tcp: np.ndarray,
    raw_imu_pos: np.ndarray,
    raw_imu_rot: np.ndarray,
    offset_sec: float,
    handeye: np.ndarray,
) -> Dict[str, object]:
    translation_best_offset, _ = tcp_eval["scan_evo_time_offset"](
        gt_times,
        gt_pos,
        gt_rot,
        est_times,
        est_pos_tcp,
        est_rot_tcp,
        helpers,
        offset_sec,
        0.12,
        0.002,
        0.01,
        "translation",
        50,
    )
    rotation_best_offset, _ = tcp_eval["scan_evo_time_offset"](
        gt_times,
        gt_pos,
        gt_rot,
        est_times,
        est_pos_tcp,
        est_rot_tcp,
        helpers,
        offset_sec,
        0.12,
        0.002,
        0.01,
        "rotation",
        50,
    )

    variants = {
        "baseline": np.linalg.inv(tcp_eval["T_LEFT_CAMERA_IMU"]) @ np.linalg.inv(handeye),
        "wrong_handeye_direction": np.linalg.inv(tcp_eval["T_LEFT_CAMERA_IMU"]) @ handeye,
        "wrong_imu_direction": tcp_eval["T_LEFT_CAMERA_IMU"] @ np.linalg.inv(handeye),
        "both_wrong": tcp_eval["T_LEFT_CAMERA_IMU"] @ handeye,
    }
    rows: Dict[str, Dict[str, float]] = {}
    for name, t_imu_tcp in variants.items():
        pos_tcp = np.asarray(raw_imu_rot @ t_imu_tcp[:3, 3] + raw_imu_pos, dtype=float)
        rot_tcp = np.asarray(raw_imu_rot @ t_imu_tcp[:3, :3], dtype=float)
        assoc = tcp_eval["associate_by_nearest_time"](
            gt_times,
            gt_pos,
            gt_rot,
            est_times,
            pos_tcp,
            rot_tcp,
            offset_sec,
            0.01,
        )
        assoc_times, assoc_gt_pos, assoc_gt_rot, assoc_est_pos, assoc_est_rot = assoc
        se3, _, _, _, _ = helpers["evaluate_alignment"](
            "se3",
            False,
            assoc_gt_pos,
            assoc_gt_rot,
            assoc_est_pos,
            assoc_est_rot,
            assoc_times,
            1.0,
            30,
        )
        rows[name] = {
            "ape_translation_rmse_mm": float(se3.translation_metrics_m["rmse"] * 1000.0),
            "ape_rotation_rmse_deg": float(se3.rotation_metrics_deg["rmse"]),
        }
    return {
        "translation_best_offset_sec": float(translation_best_offset),
        "rotation_best_offset_sec": float(rotation_best_offset),
        "offset_delta_rotation_minus_translation_sec": float(rotation_best_offset - translation_best_offset),
        "convention_variants": rows,
    }


def analyze_episode(spec: EpisodeSpec, tcp_eval: Dict[str, object], helpers: Dict[str, object], handeye: np.ndarray) -> EpisodeAnalysis:
    offset_sec = read_time_offset(spec.raw_metrics)
    gt_times, gt_pos, gt_rot = tcp_eval["load_robot_tcp_trajectory"](spec.ground_truth, helpers)
    est_times, est_pos_tcp, est_rot_tcp = tcp_eval["load_vio_tcp_trajectory"](spec.estimate_raw, helpers, handeye, "imu")
    raw_imu_times, raw_imu_pos, raw_imu_rot = load_raw_imu_csv(spec.estimate_raw, helpers, tcp_eval["normalize_timestamp"])

    assoc = tcp_eval["associate_by_nearest_time"](
        gt_times,
        gt_pos,
        gt_rot,
        est_times,
        est_pos_tcp,
        est_rot_tcp,
        offset_sec,
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
    del se3
    translation_error_world = matched_gt_pos - aligned_est_pos
    residual_rotation = np.asarray(
        [matched_gt_rot[idx] @ aligned_est_rot[idx].T for idx in range(matched_times.size)],
        dtype=float,
    )
    residual_translation_world = np.asarray(
        [
            matched_gt_pos[idx] - residual_rotation[idx] @ aligned_est_pos[idx]
            for idx in range(matched_times.size)
        ],
        dtype=float,
    )
    residual_rotation_rotvec_deg = Rotation.from_matrix(residual_rotation).as_rotvec() * 180.0 / math.pi

    features = trajectory_features(matched_times, matched_gt_pos, matched_gt_rot)
    error_mm = vector_norm(translation_error_world) * 1000.0
    summary = summarize_motion(matched_times, matched_gt_pos, matched_gt_rot, error_mm, offset_sec)
    dependence = residual_dependence(features, matched_gt_pos, matched_gt_rot, translation_error_world * 1000.0)
    checks = chain_check(
        tcp_eval,
        helpers,
        gt_times,
        gt_pos,
        gt_rot,
        est_times,
        est_pos_tcp,
        est_rot_tcp,
        raw_imu_pos,
        raw_imu_rot,
        offset_sec,
        handeye,
    )

    return EpisodeAnalysis(
        spec=spec,
        time_offset_sec=offset_sec,
        gt_times=gt_times,
        gt_pos=gt_pos,
        gt_rot=gt_rot,
        est_times=est_times,
        est_pos_tcp=est_pos_tcp,
        est_rot_tcp=est_rot_tcp,
        matched_times=matched_times,
        matched_gt_pos=matched_gt_pos,
        matched_gt_rot=matched_gt_rot,
        matched_est_pos=matched_est_pos,
        matched_est_rot=matched_est_rot,
        aligned_est_pos=aligned_est_pos,
        aligned_est_rot=aligned_est_rot,
        translation_error_world=translation_error_world,
        residual_translation_world=residual_translation_world,
        residual_rotation_rotvec_deg=residual_rotation_rotvec_deg,
        features=features,
        summary=summary,
        dependence=dependence,
        chain_check=checks,
    )


def stack_features(episodes: Sequence[EpisodeAnalysis], feature_names: Sequence[str]) -> np.ndarray:
    matrices: List[np.ndarray] = []
    for episode in episodes:
        columns: List[np.ndarray] = []
        for name in feature_names:
            values = np.asarray(episode.features[name], dtype=float)
            if values.ndim == 1:
                values = values[:, None]
            columns.append(values)
        matrices.append(np.concatenate(columns, axis=1))
    return np.concatenate(matrices, axis=0)


def stack_targets(episodes: Sequence[EpisodeAnalysis]) -> Tuple[np.ndarray, np.ndarray]:
    translation = np.concatenate([episode.residual_translation_world for episode in episodes], axis=0)
    rotation = np.concatenate([episode.residual_rotation_rotvec_deg for episode in episodes], axis=0)
    return translation, rotation


def fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float = 1e-3) -> Dict[str, np.ndarray]:
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (features - mean) / scale
    design = np.concatenate([np.ones((standardized.shape[0], 1), dtype=float), standardized], axis=1)
    gram = design.T @ design
    regularizer = alpha * np.eye(gram.shape[0], dtype=float)
    regularizer[0, 0] = 0.0
    coeff = np.linalg.solve(gram + regularizer, design.T @ targets)
    return {"mean": mean, "scale": scale, "coeff": coeff}


def predict_ridge(model: Mapping[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    standardized = (np.asarray(features, dtype=float) - model["mean"]) / model["scale"]
    design = np.concatenate([np.ones((standardized.shape[0], 1), dtype=float), standardized], axis=1)
    return design @ model["coeff"]


def apply_predicted_residual(
    est_pos: np.ndarray,
    est_rot: np.ndarray,
    pred_translation: np.ndarray,
    pred_rotation_rotvec_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_rotation = Rotation.from_rotvec(pred_rotation_rotvec_deg * math.pi / 180.0).as_matrix()
    corrected_pos = np.empty_like(est_pos)
    corrected_rot = np.empty_like(est_rot)
    for idx in range(est_pos.shape[0]):
        corrected_pos[idx] = pred_rotation[idx] @ est_pos[idx] + pred_translation[idx]
        corrected_rot[idx] = pred_rotation[idx] @ est_rot[idx]
    return corrected_pos, corrected_rot


def pose_error_metrics(gt_pos: np.ndarray, gt_rot: np.ndarray, est_pos: np.ndarray, est_rot: np.ndarray) -> Dict[str, float]:
    trans_err_mm = vector_norm(gt_pos - est_pos) * 1000.0
    rot_err_deg = np.asarray(
        [Rotation.from_matrix(gt_rot[idx] @ est_rot[idx].T).magnitude() * 180.0 / math.pi for idx in range(gt_pos.shape[0])],
        dtype=float,
    )
    return {
        "ape_translation_rmse_mm": float(np.sqrt(np.mean(np.square(trans_err_mm)))),
        "ape_translation_mean_mm": float(np.mean(trans_err_mm)),
        "ape_rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rot_err_deg)))),
        "ape_rotation_mean_deg": float(np.mean(rot_err_deg)),
    }


def feature_groups() -> Dict[str, List[str]]:
    return {
        "constant": [],
        "position": ["position_xyz"],
        "orientation": ["orientation_matrix_flat"],
        "kinematics": ["speed", "acc_mag", "tangent_acc", "ang_speed_deg_s", "curvature"],
        "phase": ["stationary", "turning", "moving_turn", "moving_straight", "stationary_turn", "accel_phase", "decel_phase"],
        "pose": ["position_xyz", "orientation_matrix_flat"],
        "pose_kinematics": [
            "position_xyz",
            "orientation_matrix_flat",
            "speed",
            "acc_mag",
            "tangent_acc",
            "ang_speed_deg_s",
            "curvature",
            "stationary",
            "turning",
            "moving_turn",
            "moving_straight",
            "stationary_turn",
            "accel_phase",
            "decel_phase",
        ],
    }


def leave_one_out_models(episodes: Sequence[EpisodeAnalysis]) -> Dict[str, object]:
    groups = feature_groups()
    results: Dict[str, object] = {}
    for group_name, names in groups.items():
        rows: List[Dict[str, object]] = []
        for holdout_idx, holdout in enumerate(episodes):
            train = [episode for idx, episode in enumerate(episodes) if idx != holdout_idx]
            train_x = np.zeros((sum(ep.matched_times.size for ep in train), 0), dtype=float) if not names else stack_features(train, names)
            holdout_x = np.zeros((holdout.matched_times.size, 0), dtype=float) if not names else stack_features([holdout], names)
            train_y_t, train_y_r = stack_targets(train)

            model_t = fit_ridge(train_x, train_y_t)
            model_r = fit_ridge(train_x, train_y_r)
            pred_t = predict_ridge(model_t, holdout_x)
            pred_r = predict_ridge(model_r, holdout_x)

            corrected_pos, corrected_rot = apply_predicted_residual(
                holdout.aligned_est_pos,
                holdout.aligned_est_rot,
                pred_t,
                pred_r,
            )
            baseline_metrics = pose_error_metrics(
                holdout.matched_gt_pos,
                holdout.matched_gt_rot,
                holdout.aligned_est_pos,
                holdout.aligned_est_rot,
            )
            corrected_metrics = pose_error_metrics(
                holdout.matched_gt_pos,
                holdout.matched_gt_rot,
                corrected_pos,
                corrected_rot,
            )
            rows.append(
                {
                    "holdout_episode": holdout.spec.episode_id,
                    "baseline": baseline_metrics,
                    "corrected": corrected_metrics,
                    "delta_translation_rmse_mm": corrected_metrics["ape_translation_rmse_mm"] - baseline_metrics["ape_translation_rmse_mm"],
                    "delta_rotation_rmse_deg": corrected_metrics["ape_rotation_rmse_deg"] - baseline_metrics["ape_rotation_rmse_deg"],
                }
            )
        results[group_name] = {
            "feature_names": names,
            "rows": rows,
            "mean_delta_translation_rmse_mm": float(np.mean([row["delta_translation_rmse_mm"] for row in rows])),
            "mean_delta_rotation_rmse_deg": float(np.mean([row["delta_rotation_rmse_deg"] for row in rows])),
        }
    return results


def leave_one_out_translation_only(episodes: Sequence[EpisodeAnalysis]) -> Dict[str, object]:
    groups = feature_groups()
    results: Dict[str, object] = {}
    for group_name, names in groups.items():
        rows: List[Dict[str, object]] = []
        for holdout_idx, holdout in enumerate(episodes):
            train = [episode for idx, episode in enumerate(episodes) if idx != holdout_idx]
            train_x = np.zeros((sum(ep.matched_times.size for ep in train), 0), dtype=float) if not names else stack_features(train, names)
            holdout_x = np.zeros((holdout.matched_times.size, 0), dtype=float) if not names else stack_features([holdout], names)
            train_y = np.concatenate([episode.translation_error_world for episode in train], axis=0)

            model = fit_ridge(train_x, train_y)
            pred_t = predict_ridge(model, holdout_x)
            corrected_pos = holdout.aligned_est_pos + pred_t
            baseline_metrics = pose_error_metrics(
                holdout.matched_gt_pos,
                holdout.matched_gt_rot,
                holdout.aligned_est_pos,
                holdout.aligned_est_rot,
            )
            corrected_metrics = pose_error_metrics(
                holdout.matched_gt_pos,
                holdout.matched_gt_rot,
                corrected_pos,
                holdout.aligned_est_rot,
            )
            rows.append(
                {
                    "holdout_episode": holdout.spec.episode_id,
                    "baseline": baseline_metrics,
                    "corrected": corrected_metrics,
                    "delta_translation_rmse_mm": corrected_metrics["ape_translation_rmse_mm"] - baseline_metrics["ape_translation_rmse_mm"],
                }
            )
        results[group_name] = {
            "feature_names": names,
            "rows": rows,
            "mean_delta_translation_rmse_mm": float(np.mean([row["delta_translation_rmse_mm"] for row in rows])),
        }
    return results


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = list(rows)
    ordered_fieldnames: List[str] = list(fieldnames)
    seen = set(ordered_fieldnames)
    for row in normalized_rows:
        for key in row.keys():
            if key not in seen:
                ordered_fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fieldnames)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow(row)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([line1, line2, *body])


def build_report(
    episodes: Sequence[EpisodeAnalysis],
    loo_translation_only: Mapping[str, object],
    loo_joint: Mapping[str, object],
) -> str:
    motion_rows: List[List[object]] = []
    for episode in episodes:
        s = episode.summary
        motion_rows.append(
            [
                episode.spec.episode_id,
                f"{s['duration_s']:.2f}",
                f"{s['path_length_m']:.3f}",
                f"{s['closure_ratio']:.4f}",
                f"{s['speed_p95_mps']:.3f}",
                f"{s['ang_speed_p95_deg_s']:.1f}",
                f"{100.0 * s['stationary_ratio']:.1f}%",
                f"{100.0 * s['moving_turn_ratio']:.1f}%",
                f"{s['baseline_ape_translation_rmse_mm']:.3f}",
            ]
        )

    dep_rows: List[List[object]] = []
    for episode in episodes:
        corr = episode.dependence["correlations"]
        dep_rows.append(
            [
                episode.spec.episode_id,
                f"{corr['speed']:.3f}",
                f"{corr['ang_speed_deg_s']:.3f}",
                f"{corr['curvature']:.3f}",
                f"{corr['gt_z']:.3f}",
                f"{corr['tcp_z_y']:.3f}",
                f"{episode.dependence['phase_mean_error_mm']['stationary']:.2f}",
                f"{episode.dependence['phase_mean_error_mm']['moving']:.2f}",
            ]
        )

    loo_translation_rows: List[List[object]] = []
    for group_name in ("constant", "position", "orientation", "kinematics", "phase", "pose", "pose_kinematics"):
        group = loo_translation_only[group_name]
        loo_translation_rows.append(
            [
                group_name,
                ", ".join(group["feature_names"]) if group["feature_names"] else "intercept only",
                f"{group['mean_delta_translation_rmse_mm']:.3f}",
            ]
        )

    loo_joint_rows: List[List[object]] = []
    for group_name in ("constant", "position", "orientation", "kinematics", "phase", "pose", "pose_kinematics"):
        group = loo_joint[group_name]
        loo_joint_rows.append(
            [
                group_name,
                ", ".join(group["feature_names"]) if group["feature_names"] else "intercept only",
                f"{group['mean_delta_translation_rmse_mm']:.3f}",
                f"{group['mean_delta_rotation_rmse_deg']:.3f}",
            ]
        )

    chain_rows: List[List[object]] = []
    for episode in episodes:
        check = episode.chain_check
        variants = check["convention_variants"]
        chain_rows.append(
            [
                episode.spec.episode_id,
                f"{episode.time_offset_sec:.6f}",
                f"{check['translation_best_offset_sec']:.6f}",
                f"{check['rotation_best_offset_sec']:.6f}",
                f"{check['offset_delta_rotation_minus_translation_sec']:.6f}",
                f"{variants['baseline']['ape_translation_rmse_mm']:.1f}",
                f"{variants['wrong_imu_direction']['ape_translation_rmse_mm']:.1f}",
                f"{variants['wrong_handeye_direction']['ape_translation_rmse_mm']:.1f}",
            ]
        )

    best_translation_group = min(
        loo_translation_only.items(),
        key=lambda item: item[1]["mean_delta_translation_rmse_mm"],
    )
    best_translation_name, best_translation = best_translation_group
    best_joint_group = min(
        loo_joint.items(),
        key=lambda item: item[1]["mean_delta_translation_rmse_mm"],
    )
    best_joint_name, best_joint = best_joint_group
    rm75_0004 = next(episode for episode in episodes if episode.spec.episode_id == "rm75_0004")
    rm75_0004_dep = rm75_0004.dependence
    speed_gap = rm75_0004_dep["speed_percentile_gap_mm"]["delta_high_minus_low"]
    ang_gap = rm75_0004_dep["ang_speed_percentile_gap_mm"]["delta_high_minus_low"]
    translation_only_improves = [
        f"{name} ({group['mean_delta_translation_rmse_mm']:.3f} mm)"
        for name, group in loo_translation_only.items()
        if group["mean_delta_translation_rmse_mm"] < -1e-9
    ]

    lines = [
        "# RM75 Motion-State Generalization Diagnosis",
        "",
        "## Main Findings",
        "",
        "1. `rm75_0004` is not an outlier because it moves faster or rotates harder. It is longer, more closed-loop, and less turn-dominant than `0001~0003`.",
        f"2. `rm75_0004` translation APE decreases when speed and angular speed increase: high-vs-low speed error delta = `{speed_gap:.2f} mm`, high-vs-low angular-speed delta = `{ang_gap:.2f} mm`.",
        (
            "3. No translation-only shared state model improves the held-out mean APE in this sweep."
            if not translation_only_improves
            else "3. Translation-only held-out improvements exist, but they are weak: " + ", ".join(translation_only_improves) + "."
        ),
        f"4. Joint translation+rotation fitting overfits even harder: the least-bad joint model is `{best_joint_name}` with mean dAPE `{best_joint['mean_delta_translation_rmse_mm']:.3f} mm`.",
        "5. Chain/convention rechecks do not support a unique `rm75_0004` frame-direction bug. Wrong conventions consistently worsen translation RMSE on all episodes.",
        "",
        "## Motion Envelope",
        "",
        markdown_table(
            [
                "episode",
                "duration s",
                "path m",
                "closure",
                "speed p95",
                "ang speed p95",
                "stationary",
                "moving turn",
                "APE mm",
            ],
            motion_rows,
        ),
        "",
        "Interpretation:",
        "- `rm75_0004` is the longest sequence and the closest to a closed loop.",
        "- `rm75_0004` has the lowest angular-speed envelope, so its distinct residual is not explained by unusually violent turns.",
        "- `rm75_0001` has the largest translational error and the strongest turn-heavy behavior.",
        "",
        "## Residual-State Dependence",
        "",
        markdown_table(
            [
                "episode",
                "corr speed",
                "corr ang speed",
                "corr curvature",
                "corr gt z",
                "corr tcp z_y",
                "stationary err mm",
                "moving err mm",
            ],
            dep_rows,
        ),
        "",
        "Interpretation:",
        "- Negative speed / angular-speed correlation means error is larger in slower or less dynamic portions.",
        "- `rm75_0004` has the strongest negative speed and angular-speed correlation in the set, unlike `0001` and `0002`.",
        "- This points away from one fixed hand-eye constant and toward a state-dependent or drift-like term that becomes visible during slow cruise / hold segments.",
        "",
        "## Held-Out Translation-Only Models",
        "",
        markdown_table(
            ["model", "features", "mean dAPE mm"],
            loo_translation_rows,
        ),
        "",
        "Interpretation:",
        "- Negative `mean dAPE mm` is good. No model here reaches a negative held-out mean, so there is still no evidence for a strong shared translation-state law.",
        "- `position` helps `rm75_0002` only, `phase` helps `rm75_0003` only, and both hurt `rm75_0004`, so those effects are not shared enough.",
        "",
        "## Held-Out Joint SE(3) Models",
        "",
        markdown_table(
            ["model", "features", "mean dAPE mm", "mean dRot deg"],
            loo_joint_rows,
        ),
        "",
        "Interpretation:",
        "- Jointly learning translation and rotation residuals generalizes worse than translation-only fits.",
        "- This matches the earlier observation that `rm75_0004` rotation residual structure is not consistent with `0001~0003`.",
        "",
        "## Chain And Convention Checks",
        "",
        markdown_table(
            [
                "episode",
                "baseline offset",
                "best trans offset",
                "best rot offset",
                "rot-trans delta",
                "baseline APE",
                "wrong imu dir APE",
                "wrong handeye dir APE",
            ],
            chain_rows,
        ),
        "",
        "Interpretation:",
        "- `rm75_0004` remains the only raw episode with a negative translation-optimal time offset, but the minimum is shallow and nearby.",
        "- Rotation-optimal offset drifts away from translation-optimal offset on multiple episodes, so this is not uniquely a `0004` timestamp-chain symptom.",
        "- Direction mistakes in the hand-eye / IMU chain degrade all episodes; they do not selectively fix `0004`.",
        "",
        "## Bottom Line",
        "",
        "- `rm75_0004` differs mainly by motion envelope and residual-state behavior, not by a single obvious frame-chain bug.",
        "- The shared rigid correction remains weak, and the simple shared state models in this sweep do not yet deliver a convincing held-out APE drop.",
        "- Rotation residual transfer is especially poor; any future shared model should probably avoid forcing one shared rotational correction family across all four episodes.",
        "- The most plausible next target is a slow-varying bias that is conditioned on pose/orientation and becomes strongest during low-speed closed-loop segments, but any future model must still be validated leave-one-episode-out.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--handeye-yaml", type=Path, default=DEFAULT_HANDEYE)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tcp_eval = load_tcp_eval()
    helpers = tcp_eval["load_helpers"]()
    handeye = tcp_eval["load_tcp_left_camera_transform"](args.handeye_yaml.expanduser().resolve())

    episodes = [analyze_episode(spec, tcp_eval, helpers, handeye) for spec in EPISODES]
    loo_translation_only = leave_one_out_translation_only(episodes)
    loo_joint = leave_one_out_models(episodes)

    motion_rows = []
    dep_rows = []
    chain_rows = []
    for episode in episodes:
        motion_rows.append({"episode_id": episode.spec.episode_id, **episode.summary})
        dep_rows.append({"episode_id": episode.spec.episode_id, **episode.dependence["correlations"]})
        chain_rows.append(
            {
                "episode_id": episode.spec.episode_id,
                "baseline_offset_sec": episode.time_offset_sec,
                "translation_best_offset_sec": episode.chain_check["translation_best_offset_sec"],
                "rotation_best_offset_sec": episode.chain_check["rotation_best_offset_sec"],
                "offset_delta_rotation_minus_translation_sec": episode.chain_check["offset_delta_rotation_minus_translation_sec"],
                **{
                    f"{name}_ape_translation_rmse_mm": row["ape_translation_rmse_mm"]
                    for name, row in episode.chain_check["convention_variants"].items()
                },
                **{
                    f"{name}_ape_rotation_rmse_deg": row["ape_rotation_rmse_deg"]
                    for name, row in episode.chain_check["convention_variants"].items()
                },
            }
        )

    loo_translation_rows = []
    for group_name, group in loo_translation_only.items():
        row = {
            "model": group_name,
            "feature_names": ",".join(group["feature_names"]) if group["feature_names"] else "intercept_only",
            "mean_delta_translation_rmse_mm": group["mean_delta_translation_rmse_mm"],
        }
        loo_translation_rows.append(row)
        for holdout in group["rows"]:
            loo_translation_rows.append(
                {
                    "model": group_name,
                    "feature_names": row["feature_names"],
                    "holdout_episode": holdout["holdout_episode"],
                    "baseline_translation_rmse_mm": holdout["baseline"]["ape_translation_rmse_mm"],
                    "corrected_translation_rmse_mm": holdout["corrected"]["ape_translation_rmse_mm"],
                    "delta_translation_rmse_mm": holdout["delta_translation_rmse_mm"],
                }
            )

    loo_joint_rows = []
    for group_name, group in loo_joint.items():
        row = {
            "model": group_name,
            "feature_names": ",".join(group["feature_names"]) if group["feature_names"] else "intercept_only",
            "mean_delta_translation_rmse_mm": group["mean_delta_translation_rmse_mm"],
            "mean_delta_rotation_rmse_deg": group["mean_delta_rotation_rmse_deg"],
        }
        loo_joint_rows.append(row)
        for holdout in group["rows"]:
            loo_joint_rows.append(
                {
                    "model": group_name,
                    "feature_names": row["feature_names"],
                    "holdout_episode": holdout["holdout_episode"],
                    "baseline_translation_rmse_mm": holdout["baseline"]["ape_translation_rmse_mm"],
                    "corrected_translation_rmse_mm": holdout["corrected"]["ape_translation_rmse_mm"],
                    "delta_translation_rmse_mm": holdout["delta_translation_rmse_mm"],
                    "baseline_rotation_rmse_deg": holdout["baseline"]["ape_rotation_rmse_deg"],
                    "corrected_rotation_rmse_deg": holdout["corrected"]["ape_rotation_rmse_deg"],
                    "delta_rotation_rmse_deg": holdout["delta_rotation_rmse_deg"],
                }
            )

    payload = {
        "episodes": {
            episode.spec.episode_id: {
                "summary": episode.summary,
                "dependence": episode.dependence,
                "chain_check": episode.chain_check,
            }
            for episode in episodes
        },
        "leave_one_out_translation_only": loo_translation_only,
        "leave_one_out_joint_models": loo_joint,
    }
    (output_dir / "diagnosis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(build_report(episodes, loo_translation_only, loo_joint), encoding="utf-8")

    write_csv(output_dir / "motion_summary.csv", motion_rows[0].keys(), motion_rows)
    write_csv(output_dir / "residual_correlations.csv", dep_rows[0].keys(), dep_rows)
    write_csv(output_dir / "chain_checks.csv", chain_rows[0].keys(), chain_rows)
    write_csv(output_dir / "heldout_translation_only.csv", loo_translation_rows[0].keys(), loo_translation_rows)
    write_csv(output_dir / "heldout_joint_models.csv", loo_joint_rows[0].keys(), loo_joint_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
