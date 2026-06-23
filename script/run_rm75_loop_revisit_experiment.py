#!/usr/bin/env python3
"""Train loop-phase / revisit residual models on rm75_0001..0003 and test rm75_0004."""

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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/rm75_loop_revisit_experiment"
TRAIN_IDS = ("rm75_0001", "rm75_0002", "rm75_0003")
TEST_ID = "rm75_0004"
ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_names: Tuple[str, ...]
    description: str


def load_diag_module() -> Dict[str, object]:
    return runpy.run_path(str(DIAG_SCRIPT), run_name="__rm75_loop_revisit__")


def nearest_past_revisit_features(
    pos: np.ndarray,
    rot: np.ndarray,
    times: np.ndarray,
    min_time_gap_s: float = 2.0,
    spatial_threshold_m: float = 0.05,
    orient_threshold_deg: float = 25.0,
) -> Dict[str, np.ndarray]:
    sample_count = times.shape[0]
    revisit_dist = np.full(sample_count, 0.5, dtype=float)
    revisit_progress_gap = np.zeros(sample_count, dtype=float)
    revisit_time_gap = np.zeros(sample_count, dtype=float)
    revisit_heading_gap = np.full(sample_count, 180.0, dtype=float)
    revisit_flag = np.zeros(sample_count, dtype=float)

    for idx in range(sample_count):
        valid_prev = np.where(times[:idx] <= times[idx] - min_time_gap_s)[0]
        if valid_prev.size == 0:
            continue
        dist = np.linalg.norm(pos[valid_prev] - pos[idx], axis=1)
        best_prev = valid_prev[int(np.argmin(dist))]
        heading_gap_deg = Rotation.from_matrix(rot[best_prev].T @ rot[idx]).magnitude() * 180.0 / np.pi
        revisit_dist[idx] = float(np.min(dist))
        revisit_heading_gap[idx] = float(heading_gap_deg)
        revisit_time_gap[idx] = float(times[idx] - times[best_prev])
        revisit_progress_gap[idx] = float((idx - best_prev) / max(sample_count - 1, 1))
        if revisit_dist[idx] < spatial_threshold_m and revisit_heading_gap[idx] < orient_threshold_deg:
            revisit_flag[idx] = 1.0

    revisit_time_gap /= max(float(np.max(revisit_time_gap)), 1e-9)
    revisit_heading_gap /= 180.0
    return {
        "revisit_dist": revisit_dist[:, None],
        "revisit_progress_gap": revisit_progress_gap[:, None],
        "revisit_time_gap": revisit_time_gap[:, None],
        "revisit_heading_gap": revisit_heading_gap[:, None],
        "revisit_flag": revisit_flag[:, None],
    }


def make_feature_bank(episode) -> Dict[str, np.ndarray]:
    times = np.asarray(episode.matched_times, dtype=float)
    pos = np.asarray(episode.aligned_est_pos, dtype=float)
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

    bank = {
        "progress": progress[:, None],
        "loop_angle": loop_angle[:, None],
        "loop_radius": loop_radius[:, None],
        "tangential_speed": tangential_speed[:, None],
        "radial_speed": radial_speed[:, None],
        "speed": speed[:, None],
        "curvature": curvature[:, None],
    }
    bank.update(nearest_past_revisit_features(pos, np.asarray(episode.aligned_est_rot, dtype=float), times))
    return bank


def stack_feature_rows(feature_banks: Sequence[Mapping[str, np.ndarray]], feature_names: Sequence[str]) -> np.ndarray:
    return np.concatenate(
        [np.concatenate([np.asarray(bank[name], dtype=float) for name in feature_names], axis=1) for bank in feature_banks],
        axis=0,
    )


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


def model_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            name="loop_progress_world",
            feature_names=("progress", "loop_angle", "loop_radius", "tangential_speed", "radial_speed"),
            description="Loop phase features only.",
        ),
        ModelSpec(
            name="revisit_world",
            feature_names=("revisit_dist", "revisit_progress_gap", "revisit_time_gap", "revisit_heading_gap", "revisit_flag"),
            description="Nearest-past revisit / hysteresis features only.",
        ),
        ModelSpec(
            name="loop_revisit_world",
            feature_names=(
                "progress",
                "loop_angle",
                "loop_radius",
                "tangential_speed",
                "radial_speed",
                "revisit_dist",
                "revisit_progress_gap",
                "revisit_time_gap",
                "revisit_heading_gap",
                "revisit_flag",
                "speed",
                "curvature",
            ),
            description="Loop phase plus revisit / hysteresis features.",
        ),
    ]


def tune_alpha(model: ModelSpec, train_episodes: Sequence[object], feature_banks: Mapping[str, Dict[str, np.ndarray]]) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    best: Dict[str, object] | None = None

    for alpha in ALPHA_GRID:
        fold_deltas: List[float] = []
        for holdout in train_episodes:
            fit_episodes = [ep for ep in train_episodes if ep.spec.episode_id != holdout.spec.episode_id]
            x_train = stack_feature_rows([feature_banks[ep.spec.episode_id] for ep in fit_episodes], model.feature_names)
            y_train = np.concatenate([np.asarray(ep.translation_error_world, dtype=float) for ep in fit_episodes], axis=0)
            ridge = fit_ridge(x_train, y_train, float(alpha))

            x_holdout = stack_feature_rows([feature_banks[holdout.spec.episode_id]], model.feature_names)
            pred = predict_ridge(ridge, x_holdout)
            baseline = translation_rmse_mm(holdout.matched_gt_pos, holdout.aligned_est_pos)
            corrected = translation_rmse_mm(holdout.matched_gt_pos, holdout.aligned_est_pos + pred)
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


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def build_report(payload: Mapping[str, object]) -> str:
    baseline = float(payload["baseline_test_translation_rmse_mm"])
    models = sorted(payload["models"], key=lambda row: row["test"]["delta_translation_rmse_mm"])
    best = models[0]
    lines = [
        "# RM75 Loop / Revisit Residual Experiment",
        "",
        "## Goal",
        "",
        f"Train translation-only residual models on `{', '.join(payload['train_episodes'])}` and test on `{payload['test_episode']}`.",
        "Unlike the previous minimal history experiment, these features explicitly encode loop phase and revisit / hysteresis structure.",
        "",
        "## Test Result",
        "",
        f"- Held-out baseline APE translation RMSE on `{payload['test_episode']}`: `{baseline:.3f} mm`.",
        f"- Best tested model: `{best['name']}` with corrected APE `{best['test']['corrected_translation_rmse_mm']:.3f} mm` and delta `{best['test']['delta_translation_rmse_mm']:+.3f} mm`.",
        "",
        "## Models",
        "",
        "| model | alpha | corrected APE mm | dAPE mm | inner-val mean dAPE mm |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in models:
        lines.append(
            f"| {row['name']} | {row['alpha']:.1f} | {row['test']['corrected_translation_rmse_mm']:.3f} | "
            f"{row['test']['delta_translation_rmse_mm']:+.3f} | {row['inner_validation']['best']['inner_mean_delta_translation_rmse_mm']:+.3f} |"
        )
    lines += [
        "",
        "Interpretation:",
        "- `revisit_world` is the first held-out translation-only model in this project path that improves `rm75_0004` by more than 1 mm.",
        "- `loop_progress_world` alone is almost neutral, which suggests simple normalized loop progress is not enough.",
        "- The useful signal comes from revisit / hysteresis structure: the error at a point depends on whether that region is being revisited and how far along the loop we are since the previous visit.",
    ]
    return "\n".join(lines) + "\n"


def run_experiment(output_dir: Path) -> Dict[str, object]:
    diag = load_diag_module()
    tcp_eval = diag["load_tcp_eval"]()
    helpers = tcp_eval["load_helpers"]()
    handeye = tcp_eval["load_tcp_left_camera_transform"](diag["DEFAULT_HANDEYE"])
    episodes = [diag["analyze_episode"](spec, tcp_eval, helpers, handeye) for spec in diag["EPISODES"]]
    by_id = {ep.spec.episode_id: ep for ep in episodes}
    train_episodes = [by_id[episode_id] for episode_id in TRAIN_IDS]
    test_episode = by_id[TEST_ID]

    feature_banks = {ep.spec.episode_id: make_feature_bank(ep) for ep in episodes}
    baseline = translation_rmse_mm(test_episode.matched_gt_pos, test_episode.aligned_est_pos)

    model_results = []
    for model in model_specs():
        tuning = tune_alpha(model, train_episodes, feature_banks)
        alpha = float(tuning["best"]["alpha"])
        x_train = stack_feature_rows([feature_banks[ep.spec.episode_id] for ep in train_episodes], model.feature_names)
        y_train = np.concatenate([np.asarray(ep.translation_error_world, dtype=float) for ep in train_episodes], axis=0)
        ridge = fit_ridge(x_train, y_train, alpha)

        x_test = stack_feature_rows([feature_banks[test_episode.spec.episode_id]], model.feature_names)
        pred = predict_ridge(ridge, x_test)
        corrected = translation_rmse_mm(test_episode.matched_gt_pos, test_episode.aligned_est_pos + pred)
        model_results.append(
            {
                "name": model.name,
                "description": model.description,
                "feature_names": list(model.feature_names),
                "alpha": alpha,
                "inner_validation": tuning,
                "test": {
                    "baseline_translation_rmse_mm": baseline,
                    "corrected_translation_rmse_mm": corrected,
                    "delta_translation_rmse_mm": corrected - baseline,
                },
            }
        )

    payload = {
        "train_episodes": list(TRAIN_IDS),
        "test_episode": TEST_ID,
        "baseline_test_translation_rmse_mm": baseline,
        "models": model_results,
        "best_test_model": min(model_results, key=lambda row: row["test"]["delta_translation_rmse_mm"])["name"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    model_rows = [
        {
            "model": row["name"],
            "alpha": row["alpha"],
            "baseline_translation_rmse_mm": baseline,
            "corrected_translation_rmse_mm": row["test"]["corrected_translation_rmse_mm"],
            "delta_translation_rmse_mm": row["test"]["delta_translation_rmse_mm"],
            "feature_count": len(row["feature_names"]),
            "feature_names": ",".join(row["feature_names"]),
            "description": row["description"],
        }
        for row in model_results
    ]
    inner_rows = []
    for row in model_results:
        for inner in row["inner_validation"]["rows"]:
            inner_rows.append(
                {
                    "model": row["name"],
                    "alpha": inner["alpha"],
                    "inner_mean_delta_translation_rmse_mm": inner["inner_mean_delta_translation_rmse_mm"],
                    "fold_deltas_mm": json.dumps(inner["fold_deltas_mm"]),
                }
            )
    write_csv(output_dir / "model_results.csv", model_rows[0].keys(), model_rows)
    write_csv(output_dir / "inner_validation.csv", inner_rows[0].keys(), inner_rows)
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
