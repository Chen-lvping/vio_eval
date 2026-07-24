#!/usr/bin/env python3
"""Diagnose cross-episode RM75 TCP system error and fit shared hand-eye corrections.

This script focuses on the 2026-06-18 RM75 episodes where:

1. the same robot / camera / IMU system is reused,
2. the GT format is consistent (`rm75_pose_traj_{1..4}.json`), and
3. previous same-episode residual correction showed the error is smooth/systematic.

The goal here is not to overfit one episode. Instead, we fit one shared TCP
hand-eye correction and test whether it improves held-out episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
TCP_EVAL_SCRIPT = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/rm75_cross_episode_diagnosis"
DEFAULT_HANDEYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    estimate_raw: Path
    estimate_frame: Path
    ground_truth: Path
    raw_metrics: Path
    frame_metrics: Path


@dataclass
class BaselineMetrics:
    time_offset_sec: float
    ape_translation_rmse_mm: float
    ape_rotation_rmse_deg: float
    matched_samples: int


@dataclass
class EpisodeData:
    spec: EpisodeSpec
    gt_times: np.ndarray
    gt_pos: np.ndarray
    gt_rot: np.ndarray
    est_times_by_mode: Dict[str, np.ndarray]
    est_pos_by_mode: Dict[str, np.ndarray]
    est_rot_by_mode: Dict[str, np.ndarray]
    baselines: Dict[str, BaselineMetrics]
    matched_by_mode: Dict[str, Dict[str, np.ndarray]]


EPISODES: List[EpisodeSpec] = [
    EpisodeSpec(
        episode_id="rm75_0001",
        estimate_raw=REPO_ROOT / "data/gripper_data2/episode_20260618_0001/right/pose_data.csv",
        estimate_frame=REPO_ROOT / "data/gripper_data2/episode_20260618_0001/right/vio_log/frame_level_optimized_pose.csv",
        ground_truth=REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_1.json",
        raw_metrics=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0001/metrics.json",
        frame_metrics=REPO_ROOT / "data/evaluation/workbench/rm75_frame_eval_0001/metrics.json",
    ),
    EpisodeSpec(
        episode_id="rm75_0002",
        estimate_raw=REPO_ROOT / "data/gripper_data2/episode_20260618_0002/right/pose_data.csv",
        estimate_frame=REPO_ROOT / "data/gripper_data2/episode_20260618_0002/right/vio_log/frame_level_optimized_pose.csv",
        ground_truth=REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_2.json",
        raw_metrics=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0002/metrics.json",
        frame_metrics=REPO_ROOT / "data/evaluation/workbench/rm75_frame_eval_0002/metrics.json",
    ),
    EpisodeSpec(
        episode_id="rm75_0003",
        estimate_raw=REPO_ROOT / "data/gripper_data2/episode_20260618_0003/right/pose_data.csv",
        estimate_frame=REPO_ROOT / "data/gripper_data2/episode_20260618_0003/right/vio_log/frame_level_optimized_pose.csv",
        ground_truth=REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_3.json",
        raw_metrics=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0003/metrics.json",
        frame_metrics=REPO_ROOT / "data/evaluation/workbench/rm75_frame_eval_0003/metrics.json",
    ),
    EpisodeSpec(
        episode_id="rm75_0004",
        estimate_raw=REPO_ROOT / "data/gripper_data2/episode_20260618_0004/right/pose_data.csv",
        estimate_frame=REPO_ROOT / "data/gripper_data2/episode_20260618_0004/right/vio_log/frame_level_optimized_pose.csv",
        ground_truth=REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_4.json",
        raw_metrics=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0004/metrics.json",
        frame_metrics=REPO_ROOT / "data/evaluation/workbench/rm75_frame_eval_0004/metrics.json",
    ),
]


def read_metrics(path: Path) -> BaselineMetrics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    evo = payload["evo"]
    return BaselineMetrics(
        time_offset_sec=float(payload["time_offset_sec"]),
        ape_translation_rmse_mm=float(evo["ape_translation_se3"]["rmse"]),
        ape_rotation_rmse_deg=float(evo["ape_rotation_se3"]["rmse"]),
        matched_samples=int(payload["matched_samples"]),
    )


def load_module() -> Dict[str, object]:
    return runpy.run_path(str(TCP_EVAL_SCRIPT), run_name="__rm75_cross_episode__")


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


def load_raw_imu_csv(path: Path, helpers: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: List[float] = []
    positions: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                timestamp = normalize_timestamp(float(row["Timestamp_us"]))
                position = [float(row["X"]), float(row["Y"]), float(row["Z"])]
                quaternion = [float(row["Quat_X"]), float(row["Quat_Y"]), float(row["Quat_Z"]), float(row["Quat_W"])]
            except KeyError as exc:  # pragma: no cover - input format is fixed in repo
                raise KeyError(f"{path}: missing required column {exc}") from exc
            transform = helpers["transform_from_pose"](position, quaternion)
            times.append(timestamp)
            positions.append(transform[:3, 3].copy())
            rotations.append(transform[:3, :3].copy())
    if not times:
        raise RuntimeError(f"{path}: no valid rows loaded")
    order = np.argsort(np.asarray(times, dtype=float))
    return (
        np.asarray(times, dtype=float)[order],
        np.asarray(positions, dtype=float)[order],
        np.asarray(rotations, dtype=float)[order],
    )


def load_episodes(tcp_eval: Dict[str, object]) -> List[EpisodeData]:
    helpers = tcp_eval["load_helpers"]()
    episodes: List[EpisodeData] = []
    for spec in EPISODES:
        gt_times, gt_pos, gt_rot = tcp_eval["load_robot_tcp_trajectory"](spec.ground_truth, helpers)
        raw_times, raw_pos, raw_rot = load_raw_imu_csv(spec.estimate_raw, helpers)
        frame_times, frame_pos, frame_rot = load_raw_imu_csv(spec.estimate_frame, helpers)
        baselines = {"raw": read_metrics(spec.raw_metrics), "frame": read_metrics(spec.frame_metrics)}
        matched_by_mode: Dict[str, Dict[str, np.ndarray]] = {}
        for mode, est_times, est_pos, est_rot in (
            ("raw", raw_times, raw_pos, raw_rot),
            ("frame", frame_times, frame_pos, frame_rot),
        ):
            assoc_times, assoc_gt_pos, assoc_gt_rot, assoc_est_pos, assoc_est_rot = tcp_eval["associate_by_nearest_time"](
                gt_times,
                gt_pos,
                gt_rot,
                est_times,
                est_pos,
                est_rot,
                baselines[mode].time_offset_sec,
                0.01,
            )
            matched_by_mode[mode] = {
                "times": assoc_times,
                "gt_pos": assoc_gt_pos,
                "gt_rot": assoc_gt_rot,
                "est_pos_imu": assoc_est_pos,
                "est_rot_imu": assoc_est_rot,
            }
        episodes.append(
            EpisodeData(
                spec=spec,
                gt_times=gt_times,
                gt_pos=gt_pos,
                gt_rot=gt_rot,
                est_times_by_mode={"raw": raw_times, "frame": frame_times},
                est_pos_by_mode={"raw": raw_pos, "frame": frame_pos},
                est_rot_by_mode={"raw": raw_rot, "frame": frame_rot},
                baselines=baselines,
                matched_by_mode=matched_by_mode,
            )
        )
    return episodes


def mode_summary(episodes: Sequence[EpisodeData], mode: str) -> Dict[str, float]:
    ape = [episode.baselines[mode].ape_translation_rmse_mm for episode in episodes]
    rot = [episode.baselines[mode].ape_rotation_rmse_deg for episode in episodes]
    return {
        "mean_ape_mm": float(np.mean(ape)),
        "median_ape_mm": float(np.median(ape)),
        "max_ape_mm": float(np.max(ape)),
        "mean_rot_deg": float(np.mean(rot)),
    }


def choose_mode(episodes: Sequence[EpisodeData]) -> str:
    raw = mode_summary(episodes, "raw")
    frame = mode_summary(episodes, "frame")
    return "frame" if frame["mean_ape_mm"] < raw["mean_ape_mm"] else "raw"


def apply_handeye(
    est_pos_imu: np.ndarray,
    est_rot_imu: np.ndarray,
    t_imu_tcp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rot_tcp = np.asarray(est_rot_imu @ t_imu_tcp[:3, :3], dtype=float)
    pos_tcp = np.asarray(est_rot_imu @ t_imu_tcp[:3, 3] + est_pos_imu, dtype=float)
    return pos_tcp, rot_tcp


def build_handeye(base_handeye: np.ndarray, delta_translation_mm: Sequence[float], delta_rotation_deg: Sequence[float]) -> np.ndarray:
    delta = np.eye(4, dtype=float)
    delta[:3, :3] = Rotation.from_euler("xyz", delta_rotation_deg, degrees=True).as_matrix()
    delta[:3, 3] = np.asarray(delta_translation_mm, dtype=float) * 1e-3
    return np.asarray(base_handeye @ delta, dtype=float)


def evaluate_episode(
    tcp_eval: Dict[str, object],
    helpers: Dict[str, object],
    episode: EpisodeData,
    mode: str,
    handeye: np.ndarray,
    time_offset_sec: float,
) -> Dict[str, float]:
    t_imu_tcp = np.linalg.inv(tcp_eval["T_LEFT_CAMERA_IMU"]) @ np.linalg.inv(handeye)
    est_pos_tcp, est_rot_tcp = apply_handeye(
        episode.est_pos_by_mode[mode],
        episode.est_rot_by_mode[mode],
        t_imu_tcp,
    )
    assoc_times, assoc_gt_pos, assoc_gt_rot, assoc_est_pos, assoc_est_rot = tcp_eval["associate_by_nearest_time"](
        episode.gt_times,
        episode.gt_pos,
        episode.gt_rot,
        episode.est_times_by_mode[mode],
        est_pos_tcp,
        est_rot_tcp,
        time_offset_sec,
        0.01,
    )
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
    return {
        "ape_translation_rmse_mm": float(se3.translation_metrics_m["rmse"] * 1000.0),
        "ape_rotation_rmse_deg": float(se3.rotation_metrics_deg["rmse"]),
        "matched_samples": int(assoc_times.size),
        "matched_duration_s": float(assoc_times[-1] - assoc_times[0]) if assoc_times.size > 1 else 0.0,
    }


def evaluate_episode_cached(
    tcp_eval: Dict[str, object],
    helpers: Dict[str, object],
    episode: EpisodeData,
    mode: str,
    handeye: np.ndarray,
) -> Dict[str, float]:
    matched = episode.matched_by_mode[mode]
    t_imu_tcp = np.linalg.inv(tcp_eval["T_LEFT_CAMERA_IMU"]) @ np.linalg.inv(handeye)
    est_pos_tcp, est_rot_tcp = apply_handeye(
        matched["est_pos_imu"],
        matched["est_rot_imu"],
        t_imu_tcp,
    )
    se3, _, _, _, _ = helpers["evaluate_alignment"](
        "se3",
        False,
        matched["gt_pos"],
        matched["gt_rot"],
        est_pos_tcp,
        est_rot_tcp,
        matched["times"],
        1.0,
        30,
    )
    return {
        "ape_translation_rmse_mm": float(se3.translation_metrics_m["rmse"] * 1000.0),
        "ape_rotation_rmse_deg": float(se3.rotation_metrics_deg["rmse"]),
        "matched_samples": int(matched["times"].size),
        "matched_duration_s": float(matched["times"][-1] - matched["times"][0]) if matched["times"].size > 1 else 0.0,
    }


def rescan_time_offset(
    tcp_eval: Dict[str, object],
    helpers: Dict[str, object],
    episode: EpisodeData,
    mode: str,
    handeye: np.ndarray,
    center_offset_sec: float,
    span_sec: float,
    step_sec: float,
) -> Tuple[float, Dict[str, float]]:
    t_imu_tcp = np.linalg.inv(tcp_eval["T_LEFT_CAMERA_IMU"]) @ np.linalg.inv(handeye)
    est_pos_tcp, est_rot_tcp = apply_handeye(
        episode.est_pos_by_mode[mode],
        episode.est_rot_by_mode[mode],
        t_imu_tcp,
    )
    best_offset, _rows = tcp_eval["scan_evo_time_offset"](
        episode.gt_times,
        episode.gt_pos,
        episode.gt_rot,
        episode.est_times_by_mode[mode],
        est_pos_tcp,
        est_rot_tcp,
        helpers,
        float(center_offset_sec),
        float(span_sec),
        float(step_sec),
        0.01,
        "translation",
        50,
    )
    return float(best_offset), evaluate_episode(tcp_eval, helpers, episode, mode, handeye, float(best_offset))


def aggregate_metrics(metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(metrics)
    return {
        "mean_ape_mm": float(np.mean([row["ape_translation_rmse_mm"] for row in rows])),
        "median_ape_mm": float(np.median([row["ape_translation_rmse_mm"] for row in rows])),
        "max_ape_mm": float(np.max([row["ape_translation_rmse_mm"] for row in rows])),
        "mean_rot_deg": float(np.mean([row["ape_rotation_rmse_deg"] for row in rows])),
    }


def score_handeye(
    tcp_eval: Dict[str, object],
    helpers: Dict[str, object],
    episodes: Sequence[EpisodeData],
    mode: str,
    handeye: np.ndarray,
) -> Tuple[float, List[Dict[str, float]]]:
    rows = [
        {
            "episode_id": episode.spec.episode_id,
            **evaluate_episode_cached(
                tcp_eval,
                helpers,
                episode,
                mode,
                handeye,
            ),
        }
        for episode in episodes
    ]
    score = float(np.mean([row["ape_translation_rmse_mm"] for row in rows]))
    return score, rows


def coarse_search_translation(
    tcp_eval: Dict[str, object],
    helpers: Dict[str, object],
    base_handeye: np.ndarray,
    episodes: Sequence[EpisodeData],
    mode: str,
    initial_translation_mm: Sequence[float],
) -> Tuple[np.ndarray, List[float], List[Dict[str, float]]]:
    best_translation = list(initial_translation_mm)
    best_handeye = build_handeye(base_handeye, best_translation, [0.0, 0.0, 0.0])
    best_score, best_rows = score_handeye(tcp_eval, helpers, episodes, mode, best_handeye)

    for step_mm in (10.0, 3.0):
        offsets = (-step_mm, 0.0, step_mm)
        improved = True
        while improved:
            improved = False
            for dx in offsets:
                for dy in offsets:
                    for dz in offsets:
                        candidate_translation = [
                            best_translation[0] + dx,
                            best_translation[1] + dy,
                            best_translation[2] + dz,
                        ]
                        handeye = build_handeye(base_handeye, candidate_translation, [0.0, 0.0, 0.0])
                        score, rows = score_handeye(tcp_eval, helpers, episodes, mode, handeye)
                        if score + 1e-9 < best_score:
                            best_score = score
                            best_translation = candidate_translation
                            best_handeye = handeye
                            best_rows = rows
                            improved = True
    return best_handeye, best_translation, best_rows


def coarse_search_rotation(
    tcp_eval: Dict[str, object],
    helpers: Dict[str, object],
    base_handeye: np.ndarray,
    translation_mm: Sequence[float],
    episodes: Sequence[EpisodeData],
    mode: str,
    initial_rotation_deg: Sequence[float],
) -> Tuple[np.ndarray, List[float], List[Dict[str, float]]]:
    best_rotation = list(initial_rotation_deg)
    best_handeye = build_handeye(base_handeye, translation_mm, best_rotation)
    best_score, best_rows = score_handeye(tcp_eval, helpers, episodes, mode, best_handeye)

    for step_deg in (4.0, 1.0):
        offsets = (-step_deg, 0.0, step_deg)
        improved = True
        while improved:
            improved = False
            for rx in offsets:
                for ry in offsets:
                    for rz in offsets:
                        candidate_rotation = [
                            best_rotation[0] + rx,
                            best_rotation[1] + ry,
                            best_rotation[2] + rz,
                        ]
                        handeye = build_handeye(base_handeye, translation_mm, candidate_rotation)
                        score, rows = score_handeye(tcp_eval, helpers, episodes, mode, handeye)
                        if score + 1e-9 < best_score:
                            best_score = score
                            best_rotation = candidate_rotation
                            best_handeye = handeye
                            best_rows = rows
                            improved = True
    return best_handeye, best_rotation, best_rows


def fit_shared_handeye(
    tcp_eval: Dict[str, object],
    helpers: Dict[str, object],
    base_handeye: np.ndarray,
    episodes: Sequence[EpisodeData],
    mode: str,
) -> Dict[str, object]:
    handeye_after_translation, translation_mm, train_rows_after_translation = coarse_search_translation(
        tcp_eval,
        helpers,
        base_handeye,
        episodes,
        mode,
        [0.0, 0.0, 0.0],
    )
    final_handeye, rotation_deg, final_rows = coarse_search_rotation(
        tcp_eval,
        helpers,
        base_handeye,
        translation_mm,
        episodes,
        mode,
        [0.0, 0.0, 0.0],
    )
    return {
        "translation_mm": translation_mm,
        "rotation_deg": rotation_deg,
        "train_rows_after_translation": train_rows_after_translation,
        "train_rows_final": final_rows,
        "handeye": final_handeye,
    }


def leave_one_out(
    tcp_eval: Dict[str, object],
    helpers: Dict[str, object],
    base_handeye: np.ndarray,
    episodes: Sequence[EpisodeData],
    mode: str,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for holdout_index, holdout_episode in enumerate(episodes):
        train_episodes = [episode for idx, episode in enumerate(episodes) if idx != holdout_index]
        fit = fit_shared_handeye(tcp_eval, helpers, base_handeye, train_episodes, mode)
        corrected_handeye = fit["handeye"]

        baseline_offset = holdout_episode.baselines[mode].time_offset_sec
        baseline_metrics = evaluate_episode(tcp_eval, helpers, holdout_episode, mode, base_handeye, baseline_offset)
        corrected_offset, corrected_metrics = rescan_time_offset(
            tcp_eval,
            helpers,
            holdout_episode,
            mode,
            corrected_handeye,
            baseline_offset,
            0.06,
            0.004,
        )
        results.append(
            {
                "holdout_episode": holdout_episode.spec.episode_id,
                "train_episodes": [episode.spec.episode_id for episode in train_episodes],
                "fitted_translation_mm": fit["translation_mm"],
                "fitted_rotation_deg": fit["rotation_deg"],
                "baseline_time_offset_sec": baseline_offset,
                "corrected_time_offset_sec": corrected_offset,
                "baseline": baseline_metrics,
                "corrected": corrected_metrics,
                "delta_ape_mm": corrected_metrics["ape_translation_rmse_mm"] - baseline_metrics["ape_translation_rmse_mm"],
            }
        )
    return results


def build_all_episode_rows(
    tcp_eval: Dict[str, object],
    helpers: Dict[str, object],
    episodes: Sequence[EpisodeData],
    mode: str,
    base_handeye: np.ndarray,
    corrected_handeye: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for episode in episodes:
        baseline_offset = episode.baselines[mode].time_offset_sec
        baseline = evaluate_episode(tcp_eval, helpers, episode, mode, base_handeye, baseline_offset)
        corrected_offset, corrected = rescan_time_offset(
            tcp_eval,
            helpers,
            episode,
            mode,
            corrected_handeye,
            baseline_offset,
            0.06,
            0.004,
        )
        rows.append(
            {
                "episode_id": episode.spec.episode_id,
                "baseline_time_offset_sec": baseline_offset,
                "corrected_time_offset_sec": corrected_offset,
                "baseline_ape_mm": baseline["ape_translation_rmse_mm"],
                "corrected_ape_mm": corrected["ape_translation_rmse_mm"],
                "baseline_rot_deg": baseline["ape_rotation_rmse_deg"],
                "corrected_rot_deg": corrected["ape_rotation_rmse_deg"],
                "delta_ape_mm": corrected["ape_translation_rmse_mm"] - baseline["ape_translation_rmse_mm"],
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    episodes: Sequence[EpisodeData],
    preferred_mode: str,
    raw_summary: Dict[str, float],
    frame_summary: Dict[str, float],
    fit_all: Dict[str, object],
    fit_all_rows: Sequence[Dict[str, object]],
    loo_rows: Sequence[Dict[str, object]],
) -> str:
    loo_baseline = aggregate_metrics([row["baseline"] for row in loo_rows])
    loo_corrected = aggregate_metrics([row["corrected"] for row in loo_rows])
    lines = [
        "# RM75 Cross-Episode System Error Diagnosis",
        "",
        "## Scope",
        "",
        "This report only targets the 2026-06-18 RM75 episode family (`rm75_pose_traj_1..4`).",
        "The 2026-06-17 episodes are intentionally excluded from the shared fit because their best time offsets live in a very different regime (multi-second shift), which indicates a different timestamp chain rather than the same small geometric bias.",
        "",
        "## Baseline Mode Comparison",
        "",
        f"- Raw mean APE: `{raw_summary['mean_ape_mm']:.3f} mm`",
        f"- Frame-level mean APE: `{frame_summary['mean_ape_mm']:.3f} mm`",
        f"- Preferred mode for shared diagnosis: `{preferred_mode}`",
        "",
        "| episode | raw APE mm | frame APE mm | raw rot deg | frame rot deg |",
        "|---|---:|---:|---:|---:|",
    ]
    for episode in episodes:
        lines.append(
            f"| {episode.spec.episode_id} | "
            f"{episode.baselines['raw'].ape_translation_rmse_mm:.3f} | "
            f"{episode.baselines['frame'].ape_translation_rmse_mm:.3f} | "
            f"{episode.baselines['raw'].ape_rotation_rmse_deg:.3f} | "
            f"{episode.baselines['frame'].ape_rotation_rmse_deg:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Shared Hand-Eye Fit On All RM75 Episodes",
            "",
            f"- Shared translation correction (local TCP frame): `{fit_all['translation_mm']}` mm",
            f"- Shared rotation correction (local TCP frame): `{fit_all['rotation_deg']}` deg",
            "",
            "| episode | baseline APE mm | corrected APE mm | delta mm | baseline rot deg | corrected rot deg |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fit_all_rows:
        lines.append(
            f"| {row['episode_id']} | {row['baseline_ape_mm']:.3f} | {row['corrected_ape_mm']:.3f} | "
            f"{row['delta_ape_mm']:.3f} | {row['baseline_rot_deg']:.3f} | {row['corrected_rot_deg']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Leave-One-Out Generalization",
            "",
            f"- Hold-out baseline mean APE: `{loo_baseline['mean_ape_mm']:.3f} mm`",
            f"- Hold-out corrected mean APE: `{loo_corrected['mean_ape_mm']:.3f} mm`",
            "",
            "| holdout | baseline APE mm | corrected APE mm | delta mm | corrected offset s | fitted translation mm | fitted rotation deg |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in loo_rows:
        lines.append(
            f"| {row['holdout_episode']} | {row['baseline']['ape_translation_rmse_mm']:.3f} | "
            f"{row['corrected']['ape_translation_rmse_mm']:.3f} | {row['delta_ape_mm']:.3f} | "
            f"{row['corrected_time_offset_sec']:.6f} | "
            f"{np.round(np.asarray(row['fitted_translation_mm'], dtype=float), 3).tolist()} | "
            f"{np.round(np.asarray(row['fitted_rotation_deg'], dtype=float), 3).tolist()} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If the all-episode fit improves every sequence but leave-one-out does not, the correction is still too episode-specific.",
            "If leave-one-out improves held-out APE consistently, then the dominant error is a shared geometric/system calibration term rather than a one-off trajectory warp.",
            "",
            "## Files",
            "",
            "- `summary.json`: machine-readable summary",
            "- `all_episode_fit.csv`: baseline vs all-episode shared-fit results",
            "- `leave_one_out.csv`: held-out generalization results",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--handeye-yaml", type=Path, default=DEFAULT_HANDEYE)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tcp_eval = load_module()
    helpers = tcp_eval["load_helpers"]()
    episodes = load_episodes(tcp_eval)
    preferred_mode = choose_mode(episodes)
    raw_summary = mode_summary(episodes, "raw")
    frame_summary = mode_summary(episodes, "frame")
    base_handeye = tcp_eval["load_tcp_left_camera_transform"](args.handeye_yaml.expanduser().resolve())

    fit_all = fit_shared_handeye(tcp_eval, helpers, base_handeye, episodes, preferred_mode)
    fit_all_rows = build_all_episode_rows(
        tcp_eval,
        helpers,
        episodes,
        preferred_mode,
        base_handeye,
        fit_all["handeye"],
    )
    loo_rows = leave_one_out(tcp_eval, helpers, base_handeye, episodes, preferred_mode)

    summary = {
        "preferred_mode": preferred_mode,
        "raw_summary": raw_summary,
        "frame_summary": frame_summary,
        "fit_all": {
            "translation_mm": fit_all["translation_mm"],
            "rotation_deg": fit_all["rotation_deg"],
        },
        "fit_all_rows": fit_all_rows,
        "leave_one_out": loo_rows,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "all_episode_fit.csv", fit_all_rows)
    write_csv(output_dir / "leave_one_out.csv", loo_rows)
    (output_dir / "REPORT.md").write_text(
        build_report(
            episodes,
            preferred_mode,
            raw_summary,
            frame_summary,
            fit_all,
            fit_all_rows,
            loo_rows,
        ),
        encoding="utf-8",
    )

    print(f"[OK] wrote {output_dir / 'summary.json'}")
    print(f"[OK] wrote {output_dir / 'all_episode_fit.csv'}")
    print(f"[OK] wrote {output_dir / 'leave_one_out.csv'}")
    print(f"[OK] wrote {output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
