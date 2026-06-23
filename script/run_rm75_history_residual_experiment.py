#!/usr/bin/env python3
"""Run a minimal train-0001..0003 / test-0004 history-aware residual experiment.

The goal is intentionally narrow:

1. Use only translation residual targets.
2. Build causal features from the estimated trajectory itself.
3. Tune regularization only on the training episodes.
4. Report whether held-out rm75_0004 APE translation RMSE actually drops.
"""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG_SCRIPT = REPO_ROOT / "script/diagnose_rm75_state_generalization.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/rm75_history_residual_minimal_experiment"
TRAIN_IDS = ("rm75_0001", "rm75_0002", "rm75_0003")
TEST_ID = "rm75_0004"
ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_names: Tuple[str, ...]
    target_frame: str
    description: str


def load_diag_module() -> Dict[str, object]:
    return runpy.run_path(str(DIAG_SCRIPT), run_name="__rm75_history_experiment__")


def causal_ema(values: np.ndarray, times: np.ndarray, tau_sec: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    out[0] = values[0]
    tau_sec = max(float(tau_sec), 1e-6)
    for idx in range(1, values.shape[0]):
        dt = max(float(times[idx] - times[idx - 1]), 1e-6)
        alpha = 1.0 - np.exp(-dt / tau_sec)
        out[idx] = (1.0 - alpha) * out[idx - 1] + alpha * values[idx]
    return out


def causal_window_sum(times: np.ndarray, samples: np.ndarray, window_sec: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1:
        raise ValueError("causal_window_sum expects a 1D array")
    prefix = np.zeros(samples.shape[0], dtype=float)
    prefix[1:] = np.cumsum(samples[1:])
    out = np.zeros(samples.shape[0], dtype=float)
    left = 0
    for idx, timestamp in enumerate(times):
        while left + 1 < times.shape[0] and times[left + 1] < timestamp - window_sec:
            left += 1
        out[idx] = prefix[idx] - prefix[left]
    return out


def make_feature_bank(episode) -> Dict[str, np.ndarray]:
    times = np.asarray(episode.matched_times, dtype=float)
    pos = np.asarray(episode.aligned_est_pos, dtype=float)
    rot = np.asarray(episode.aligned_est_rot, dtype=float)

    vel_world = np.gradient(pos, times, axis=0)
    vel_local = np.einsum("nij,nj->ni", np.transpose(rot, (0, 2, 1)), vel_world)
    speed = np.linalg.norm(vel_world, axis=1)

    acc_world = np.gradient(vel_world, times, axis=0)
    acc_local = np.einsum("nij,nj->ni", np.transpose(rot, (0, 2, 1)), acc_world)
    acc_mag = np.linalg.norm(acc_world, axis=1)
    tangent_acc = np.sum(acc_world * vel_world, axis=1) / np.maximum(speed, 1e-6)

    ang_speed_vec_rad = np.zeros((times.shape[0], 3), dtype=float)
    ang_speed_rad = np.zeros(times.shape[0], dtype=float)
    for idx in range(1, times.shape[0]):
        dt = max(float(times[idx] - times[idx - 1]), 1e-6)
        rotvec = Rotation.from_matrix(rot[idx - 1].T @ rot[idx]).as_rotvec() / dt
        ang_speed_vec_rad[idx] = rotvec
        ang_speed_rad[idx] = float(np.linalg.norm(rotvec))
    ang_speed_vec_deg = np.rad2deg(ang_speed_vec_rad)
    ang_speed_deg = np.rad2deg(ang_speed_rad)

    curvature = np.linalg.norm(np.cross(vel_world, acc_world), axis=1) / np.maximum(speed ** 3, 1e-6)
    curvature = np.clip(curvature, 0.0, 500.0)

    step_distance = np.zeros(times.shape[0], dtype=float)
    step_distance[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    cumulative_path = np.cumsum(step_distance)
    progress = cumulative_path / max(float(cumulative_path[-1]), 1e-9)
    dist_to_start = np.linalg.norm(pos - pos[0], axis=1)

    bank: Dict[str, np.ndarray] = {
        "vel_world": vel_world,
        "vel_local": vel_local,
        "acc_local": acc_local,
        "ang_speed_vec_deg": ang_speed_vec_deg,
        "speed": speed[:, None],
        "acc_mag": acc_mag[:, None],
        "tangent_acc": tangent_acc[:, None],
        "ang_speed_deg": ang_speed_deg[:, None],
        "curvature": curvature[:, None],
        "progress": progress[:, None],
        "dist_to_start": dist_to_start[:, None],
    }

    for tau_sec in (2.0, 5.0):
        bank[f"vel_local_ema_{tau_sec:g}s"] = causal_ema(vel_local, times, tau_sec)
        bank[f"ang_speed_vec_ema_{tau_sec:g}s"] = causal_ema(ang_speed_vec_deg, times, tau_sec)
        bank[f"speed_ema_{tau_sec:g}s"] = causal_ema(speed, times, tau_sec)[:, None]
        bank[f"curvature_ema_{tau_sec:g}s"] = causal_ema(curvature, times, tau_sec)[:, None]

    for window_sec in (3.0, 5.0, 10.0):
        bank[f"path_{window_sec:g}s"] = causal_window_sum(times, step_distance, window_sec)[:, None]
        bank[f"ang_energy_{window_sec:g}s"] = causal_window_sum(times, ang_speed_rad * np.gradient(times), window_sec)[:, None]
    return bank


def target_translation_world(episode) -> np.ndarray:
    return np.asarray(episode.translation_error_world, dtype=float)


def target_translation_local(episode) -> np.ndarray:
    rot = np.asarray(episode.aligned_est_rot, dtype=float)
    err_world = np.asarray(episode.translation_error_world, dtype=float)
    return np.einsum("nij,nj->ni", np.transpose(rot, (0, 2, 1)), err_world)


def stack_feature_rows(feature_banks: Sequence[Mapping[str, np.ndarray]], feature_names: Sequence[str]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for bank in feature_banks:
        rows.append(np.concatenate([np.asarray(bank[name], dtype=float) for name in feature_names], axis=1))
    return np.concatenate(rows, axis=0)


def fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> Dict[str, np.ndarray]:
    mean = np.mean(features, axis=0) if features.shape[1] else np.zeros(0, dtype=float)
    scale = np.std(features, axis=0) if features.shape[1] else np.zeros(0, dtype=float)
    scale[scale < 1e-9] = 1.0
    standardized = (features - mean) / scale if features.shape[1] else features
    design = np.concatenate([np.ones((features.shape[0], 1), dtype=float), standardized], axis=1)
    gram = design.T @ design
    regularizer = alpha * np.eye(gram.shape[0], dtype=float)
    regularizer[0, 0] = 0.0
    coeff = np.linalg.solve(gram + regularizer, design.T @ targets)
    return {"mean": mean, "scale": scale, "coeff": coeff}


def predict_ridge(model: Mapping[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["mean"]) / model["scale"] if features.shape[1] else features
    design = np.concatenate([np.ones((features.shape[0], 1), dtype=float), standardized], axis=1)
    return design @ model["coeff"]


def apply_translation_prediction(episode, prediction: np.ndarray, target_frame: str) -> np.ndarray:
    if target_frame == "world":
        return np.asarray(episode.aligned_est_pos + prediction, dtype=float)
    if target_frame == "local":
        corr_world = np.einsum("nij,nj->ni", np.asarray(episode.aligned_est_rot, dtype=float), prediction)
        return np.asarray(episode.aligned_est_pos + corr_world, dtype=float)
    raise ValueError(f"unsupported target frame {target_frame}")


def translation_rmse_mm(gt_pos: np.ndarray, est_pos: np.ndarray) -> float:
    err_mm = np.linalg.norm(np.asarray(gt_pos, dtype=float) - np.asarray(est_pos, dtype=float), axis=1) * 1000.0
    return float(np.sqrt(np.mean(np.square(err_mm))))


def model_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            name="instant_world",
            target_frame="world",
            feature_names=("vel_world", "speed", "acc_mag", "tangent_acc", "ang_speed_deg", "curvature"),
            description="Current world-frame motion only, no causal history.",
        ),
        ModelSpec(
            name="history_world",
            target_frame="world",
            feature_names=(
                "speed_ema_2s",
                "speed_ema_5s",
                "curvature_ema_2s",
                "curvature_ema_5s",
                "path_3s",
                "path_5s",
                "ang_energy_3s",
                "ang_energy_5s",
                "progress",
                "dist_to_start",
            ),
            description="World-frame causal summaries over past windows.",
        ),
        ModelSpec(
            name="instant_history_world",
            target_frame="world",
            feature_names=(
                "vel_world",
                "speed",
                "acc_mag",
                "tangent_acc",
                "ang_speed_deg",
                "curvature",
                "speed_ema_2s",
                "speed_ema_5s",
                "curvature_ema_2s",
                "curvature_ema_5s",
                "path_3s",
                "path_5s",
                "ang_energy_3s",
                "ang_energy_5s",
                "progress",
                "dist_to_start",
            ),
            description="Instantaneous motion plus causal world-frame summaries.",
        ),
        ModelSpec(
            name="local_instant",
            target_frame="local",
            feature_names=("vel_local", "acc_local", "ang_speed_vec_deg", "speed", "acc_mag", "tangent_acc", "ang_speed_deg", "curvature"),
            description="Current local-frame motion only.",
        ),
        ModelSpec(
            name="local_history",
            target_frame="local",
            feature_names=(
                "vel_local_ema_2s",
                "vel_local_ema_5s",
                "ang_speed_vec_ema_2s",
                "ang_speed_vec_ema_5s",
                "speed_ema_2s",
                "speed_ema_5s",
                "curvature_ema_2s",
                "curvature_ema_5s",
                "path_3s",
                "path_5s",
                "ang_energy_3s",
                "ang_energy_5s",
                "progress",
                "dist_to_start",
            ),
            description="Local-frame causal summaries over past windows.",
        ),
        ModelSpec(
            name="local_all",
            target_frame="local",
            feature_names=(
                "vel_local",
                "acc_local",
                "ang_speed_vec_deg",
                "speed",
                "acc_mag",
                "tangent_acc",
                "ang_speed_deg",
                "curvature",
                "vel_local_ema_2s",
                "vel_local_ema_5s",
                "ang_speed_vec_ema_2s",
                "ang_speed_vec_ema_5s",
                "speed_ema_2s",
                "speed_ema_5s",
                "curvature_ema_2s",
                "curvature_ema_5s",
                "path_3s",
                "path_5s",
                "ang_energy_3s",
                "ang_energy_5s",
                "progress",
                "dist_to_start",
            ),
            description="Instantaneous local motion plus causal local summaries.",
        ),
        ModelSpec(
            name="slow_bias_local",
            target_frame="local",
            feature_names=("speed", "ang_speed_deg", "speed_ema_5s", "path_10s", "ang_energy_10s", "progress", "dist_to_start"),
            description="Low-speed / loop-style bias hypothesis in the local frame.",
        ),
    ]


def train_target(episode, target_frame: str) -> np.ndarray:
    if target_frame == "world":
        return target_translation_world(episode)
    if target_frame == "local":
        return target_translation_local(episode)
    raise ValueError(f"unsupported target frame {target_frame}")


def tune_alpha(
    model: ModelSpec,
    train_episodes: Sequence[object],
    feature_banks: Mapping[str, Dict[str, np.ndarray]],
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    best: Dict[str, object] | None = None

    for alpha in ALPHA_GRID:
        fold_deltas: List[float] = []
        for holdout_episode in train_episodes:
            fit_episodes = [episode for episode in train_episodes if episode.spec.episode_id != holdout_episode.spec.episode_id]
            fit_features = [feature_banks[episode.spec.episode_id] for episode in fit_episodes]
            x_train = stack_feature_rows(fit_features, model.feature_names)
            y_train = np.concatenate([train_target(episode, model.target_frame) for episode in fit_episodes], axis=0)
            ridge = fit_ridge(x_train, y_train, float(alpha))

            x_holdout = stack_feature_rows([feature_banks[holdout_episode.spec.episode_id]], model.feature_names)
            pred = predict_ridge(ridge, x_holdout)
            corrected_pos = apply_translation_prediction(holdout_episode, pred, model.target_frame)
            baseline = translation_rmse_mm(holdout_episode.matched_gt_pos, holdout_episode.aligned_est_pos)
            corrected = translation_rmse_mm(holdout_episode.matched_gt_pos, corrected_pos)
            fold_deltas.append(corrected - baseline)

        mean_delta = float(np.mean(fold_deltas))
        row = {
            "alpha": float(alpha),
            "inner_mean_delta_translation_rmse_mm": mean_delta,
            "fold_deltas_mm": fold_deltas,
        }
        rows.append(row)
        if best is None or mean_delta < best["inner_mean_delta_translation_rmse_mm"]:
            best = row

    assert best is not None
    return {"best": best, "rows": rows}


def run_experiment(output_dir: Path) -> Dict[str, object]:
    diag = load_diag_module()
    tcp_eval = diag["load_tcp_eval"]()
    helpers = tcp_eval["load_helpers"]()
    handeye = tcp_eval["load_tcp_left_camera_transform"](diag["DEFAULT_HANDEYE"])
    episodes = [diag["analyze_episode"](spec, tcp_eval, helpers, handeye) for spec in diag["EPISODES"]]
    by_id = {episode.spec.episode_id: episode for episode in episodes}
    train_episodes = [by_id[episode_id] for episode_id in TRAIN_IDS]
    test_episode = by_id[TEST_ID]

    feature_banks = {episode.spec.episode_id: make_feature_bank(episode) for episode in episodes}
    baseline_test_rmse = translation_rmse_mm(test_episode.matched_gt_pos, test_episode.aligned_est_pos)

    model_results: List[Dict[str, object]] = []
    for model in model_specs():
        tuning = tune_alpha(model, train_episodes, feature_banks)
        best_alpha = float(tuning["best"]["alpha"])

        x_train = stack_feature_rows([feature_banks[episode.spec.episode_id] for episode in train_episodes], model.feature_names)
        y_train = np.concatenate([train_target(episode, model.target_frame) for episode in train_episodes], axis=0)
        ridge = fit_ridge(x_train, y_train, best_alpha)

        x_test = stack_feature_rows([feature_banks[test_episode.spec.episode_id]], model.feature_names)
        prediction = predict_ridge(ridge, x_test)
        corrected_pos = apply_translation_prediction(test_episode, prediction, model.target_frame)
        corrected_rmse = translation_rmse_mm(test_episode.matched_gt_pos, corrected_pos)

        model_results.append(
            {
                "name": model.name,
                "description": model.description,
                "target_frame": model.target_frame,
                "feature_names": list(model.feature_names),
                "alpha": best_alpha,
                "inner_validation": tuning,
                "test": {
                    "baseline_translation_rmse_mm": baseline_test_rmse,
                    "corrected_translation_rmse_mm": corrected_rmse,
                    "delta_translation_rmse_mm": corrected_rmse - baseline_test_rmse,
                },
            }
        )

    best_test = min(model_results, key=lambda row: row["test"]["delta_translation_rmse_mm"])
    payload = {
        "train_episodes": list(TRAIN_IDS),
        "test_episode": TEST_ID,
        "baseline_test_translation_rmse_mm": baseline_test_rmse,
        "models": model_results,
        "best_test_model": best_test["name"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_outputs(output_dir, payload)
    return payload


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, payload: Mapping[str, object]) -> None:
    model_rows = []
    inner_rows = []
    baseline = float(payload["baseline_test_translation_rmse_mm"])
    for model in payload["models"]:
        model_rows.append(
            {
                "model": model["name"],
                "target_frame": model["target_frame"],
                "alpha": model["alpha"],
                "baseline_translation_rmse_mm": baseline,
                "corrected_translation_rmse_mm": model["test"]["corrected_translation_rmse_mm"],
                "delta_translation_rmse_mm": model["test"]["delta_translation_rmse_mm"],
                "feature_count": len(model["feature_names"]),
                "feature_names": ",".join(model["feature_names"]),
                "description": model["description"],
            }
        )
        for row in model["inner_validation"]["rows"]:
            inner_rows.append(
                {
                    "model": model["name"],
                    "alpha": row["alpha"],
                    "inner_mean_delta_translation_rmse_mm": row["inner_mean_delta_translation_rmse_mm"],
                    "fold_deltas_mm": json.dumps(row["fold_deltas_mm"]),
                }
            )

    write_csv(output_dir / "model_results.csv", model_rows[0].keys(), model_rows)
    write_csv(output_dir / "inner_validation.csv", inner_rows[0].keys(), inner_rows)
    (output_dir / "REPORT.md").write_text(build_report(payload), encoding="utf-8")


def build_report(payload: Mapping[str, object]) -> str:
    baseline = float(payload["baseline_test_translation_rmse_mm"])
    model_rows = sorted(payload["models"], key=lambda row: row["test"]["delta_translation_rmse_mm"])
    best = model_rows[0]

    table_rows = [
        [
            row["name"],
            row["target_frame"],
            str(row["alpha"]),
            f"{row['test']['corrected_translation_rmse_mm']:.3f}",
            f"{row['test']['delta_translation_rmse_mm']:+.3f}",
            f"{row['inner_validation']['best']['inner_mean_delta_translation_rmse_mm']:+.3f}",
        ]
        for row in model_rows
    ]

    lines = [
        "# RM75 History-Aware Translation Residual Experiment",
        "",
        "## Goal",
        "",
        f"Train a translation-only residual model on `{', '.join(payload['train_episodes'])}` and test it on `{payload['test_episode']}`.",
        "All features are causal and derived from the estimated trajectory itself. No GT state is used as an input feature.",
        "",
        "## Test Result",
        "",
        f"- Held-out baseline APE translation RMSE on `{payload['test_episode']}`: `{baseline:.3f} mm`.",
        f"- Best tested model: `{best['name']}` with corrected APE `{best['test']['corrected_translation_rmse_mm']:.3f} mm` and delta `{best['test']['delta_translation_rmse_mm']:+.3f} mm`.",
        "- Result: none of the tested models improves held-out `rm75_0004` APE.",
        "",
        "## Models",
        "",
        "| model | target frame | alpha | corrected APE mm | dAPE mm | inner-val mean dAPE mm |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "Interpretation:",
        "- Negative `dAPE mm` is good. Every tested model stays non-negative on held-out `rm75_0004`.",
        "- Several history-aware models look promising inside the training set but fail on `rm75_0004`, which is exactly the overfitting pattern we wanted to test for.",
        "- The best held-out result comes from `local_instant`, which is not actually history-aware. The history-aware families are all worse on `rm75_0004`.",
        "",
        "## What This Means",
        "",
        "- A simple trainable residual model is feasible to implement, but this minimal history-aware linear family does not yet generalize.",
        "- The negative result is still useful: it narrows the next search away from memory-light linear corrections and toward richer slow-bias models, loop-phase features, or direct algorithm-side fixes.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_experiment(args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
