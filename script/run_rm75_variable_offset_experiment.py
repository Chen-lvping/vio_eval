#!/usr/bin/env python3
"""Evaluate held-out and oracle-style variable time-offset models for RM75."""

from __future__ import annotations

import argparse
import csv
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG_SCRIPT = REPO_ROOT / "script/diagnose_rm75_state_generalization.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/rm75_variable_offset_experiment"
TRAIN_IDS = ("rm75_0001", "rm75_0002", "rm75_0003")
TEST_ID = "rm75_0004"
ALPHA_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


@dataclass(frozen=True)
class OffsetModelSpec:
    name: str
    feature_names: Tuple[str, ...]
    description: str


def load_diag_module() -> Dict[str, object]:
    return runpy.run_path(str(DIAG_SCRIPT), run_name="__rm75_variable_offset__")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def interp_pose(times: np.ndarray, pos: np.ndarray, rot: np.ndarray, query_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    pos = np.asarray(pos, dtype=float)
    rot = np.asarray(rot, dtype=float)
    query_times = np.asarray(query_times, dtype=float)

    valid = np.ones(query_times.shape[0], dtype=bool)
    out_pos = np.empty((query_times.shape[0], 3), dtype=float)
    out_rot = np.empty((query_times.shape[0], 3, 3), dtype=float)

    for idx, query in enumerate(query_times):
        if query < times[0] or query > times[-1]:
            valid[idx] = False
            continue
        insert_at = int(np.searchsorted(times, query, side="left"))
        if insert_at == 0:
            out_pos[idx] = pos[0]
            out_rot[idx] = rot[0]
            continue
        if insert_at >= times.shape[0]:
            out_pos[idx] = pos[-1]
            out_rot[idx] = rot[-1]
            continue
        t0 = float(times[insert_at - 1])
        t1 = float(times[insert_at])
        alpha = 0.0 if abs(t1 - t0) < 1e-9 else float((query - t0) / (t1 - t0))
        out_pos[idx] = (1.0 - alpha) * pos[insert_at - 1] + alpha * pos[insert_at]
        slerp = Slerp([t0, t1], Rotation.from_matrix([rot[insert_at - 1], rot[insert_at]]))
        out_rot[idx] = slerp([query]).as_matrix()[0]
    return valid, out_pos, out_rot


def evaluate_dynamic_offset(episode, helpers: Dict[str, object], ref_times: np.ndarray, ref_pos: np.ndarray, ref_rot: np.ndarray, offsets: np.ndarray) -> Dict[str, float]:
    query_times = np.asarray(ref_times, dtype=float) - np.asarray(offsets, dtype=float)
    valid, est_pos, est_rot = interp_pose(
        np.asarray(episode.est_times, dtype=float),
        np.asarray(episode.est_pos_tcp, dtype=float),
        np.asarray(episode.est_rot_tcp, dtype=float),
        query_times,
    )
    eval_times = np.asarray(ref_times, dtype=float)[valid]
    gt_pos = np.asarray(ref_pos, dtype=float)[valid]
    gt_rot = np.asarray(ref_rot, dtype=float)[valid]
    est_pos = est_pos[valid]
    est_rot = est_rot[valid]
    if eval_times.shape[0] < 3:
        raise ValueError("dynamic offset evaluation requires at least three valid paired samples")
    se3, _, _, _, _ = helpers["evaluate_alignment"](
        "se3",
        False,
        gt_pos,
        gt_rot,
        est_pos,
        est_rot,
        eval_times,
        1.0,
        30,
    )
    return {
        "matched_samples": int(eval_times.shape[0]),
        "ape_translation_rmse_mm": float(se3.translation_metrics_m["rmse"] * 1000.0),
        "ape_rotation_rmse_deg": float(se3.rotation_metrics_deg["rmse"]),
    }


def normalized_path_progress(pos: np.ndarray) -> np.ndarray:
    step = np.zeros(pos.shape[0], dtype=float)
    step[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    progress = np.cumsum(step)
    progress /= max(float(progress[-1]), 1e-9)
    return progress


def nearest_revisit_features(pos: np.ndarray, rot: np.ndarray, times: np.ndarray) -> Dict[str, np.ndarray]:
    revisit_dist = np.full(times.shape[0], 0.5, dtype=float)
    revisit_gap = np.zeros(times.shape[0], dtype=float)
    revisit_time = np.zeros(times.shape[0], dtype=float)
    revisit_flag = np.zeros(times.shape[0], dtype=float)
    for idx in range(times.shape[0]):
        prev = np.where(times[:idx] <= times[idx] - 2.0)[0]
        if prev.size == 0:
            continue
        dist = np.linalg.norm(pos[prev] - pos[idx], axis=1)
        best_prev = int(prev[np.argmin(dist)])
        revisit_dist[idx] = float(np.min(dist))
        revisit_gap[idx] = float((idx - best_prev) / max(times.shape[0] - 1, 1))
        revisit_time[idx] = float(times[idx] - times[best_prev])
        heading_gap = Rotation.from_matrix(rot[best_prev].T @ rot[idx]).magnitude() * 180.0 / np.pi
        if revisit_dist[idx] < 0.05 and heading_gap < 25.0:
            revisit_flag[idx] = 1.0
    revisit_time /= max(float(np.max(revisit_time)), 1e-9)
    return {
        "revisit_dist": revisit_dist[:, None],
        "revisit_gap": revisit_gap[:, None],
        "revisit_time": revisit_time[:, None],
        "revisit_flag": revisit_flag[:, None],
    }


def anchor_feature_bank(episode, anchor_times: np.ndarray) -> np.ndarray:
    times = np.asarray(episode.matched_times, dtype=float)
    pos = np.asarray(episode.aligned_est_pos, dtype=float)
    rot = np.asarray(episode.aligned_est_rot, dtype=float)
    vel = np.gradient(pos, times, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    acc = np.gradient(vel, times, axis=0)
    curvature = np.linalg.norm(np.cross(vel, acc), axis=1) / np.maximum(speed ** 3, 1e-6)
    curvature = np.clip(curvature, 0.0, 500.0)
    progress = normalized_path_progress(pos)

    centered = pos - np.mean(pos, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    plane = centered @ vt[:2].T
    loop_angle = np.unwrap(np.arctan2(plane[:, 1], plane[:, 0]))
    loop_radius = np.linalg.norm(plane, axis=1)
    revisit = nearest_revisit_features(pos, rot, times)

    columns = {
        "progress": progress[:, None],
        "loop_angle": loop_angle[:, None],
        "loop_radius": loop_radius[:, None],
        "speed": speed[:, None],
        "curvature": curvature[:, None],
        **revisit,
    }

    feature_rows = []
    for anchor in anchor_times:
        nearest_idx = int(np.argmin(np.abs(times - anchor)))
        feature_rows.append(
            np.concatenate(
                [
                    columns["progress"][nearest_idx],
                    columns["loop_angle"][nearest_idx],
                    columns["loop_radius"][nearest_idx],
                    columns["speed"][nearest_idx],
                    columns["curvature"][nearest_idx],
                    columns["revisit_dist"][nearest_idx],
                    columns["revisit_gap"][nearest_idx],
                    columns["revisit_time"][nearest_idx],
                    columns["revisit_flag"][nearest_idx],
                ]
            )
        )
    return np.asarray(feature_rows, dtype=float)


def anchor_names() -> Tuple[str, ...]:
    return (
        "progress",
        "loop_angle",
        "loop_radius",
        "speed",
        "curvature",
        "revisit_dist",
        "revisit_gap",
        "revisit_time",
        "revisit_flag",
    )


def local_offset_labels(episode, helpers: Dict[str, object], knot_count: int = 11, window_sec: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    anchors = np.linspace(float(episode.matched_times[0]), float(episode.matched_times[-1]), knot_count)
    labels = []
    for anchor in anchors:
        mask = np.abs(np.asarray(episode.matched_times, dtype=float) - anchor) <= 0.5 * window_sec
        ref_times = np.asarray(episode.matched_times, dtype=float)[mask]
        ref_pos = np.asarray(episode.matched_gt_pos, dtype=float)[mask]
        ref_rot = np.asarray(episode.matched_gt_rot, dtype=float)[mask]

        best_offset = None
        best_rmse = None
        for offset in np.arange(float(episode.time_offset_sec) - 0.08, float(episode.time_offset_sec) + 0.0801, 0.004):
            metrics = evaluate_dynamic_offset(
                episode,
                helpers,
                ref_times,
                ref_pos,
                ref_rot,
                np.full(ref_times.shape[0], float(offset), dtype=float),
            )
            if best_rmse is None or metrics["ape_translation_rmse_mm"] < best_rmse:
                best_rmse = metrics["ape_translation_rmse_mm"]
                best_offset = float(offset)
        labels.append(best_offset)
    return anchors, np.asarray(labels, dtype=float)


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


def tune_alpha(train_anchor_features: Sequence[np.ndarray], train_anchor_targets: Sequence[np.ndarray]) -> Dict[str, object]:
    rows = []
    best = None
    for alpha in ALPHA_GRID:
        fold_rmse_ms = []
        for idx in range(len(train_anchor_features)):
            fit_x = np.vstack([train_anchor_features[j] for j in range(len(train_anchor_features)) if j != idx])
            fit_y = np.concatenate([train_anchor_targets[j] for j in range(len(train_anchor_targets)) if j != idx])
            model = fit_ridge(fit_x, fit_y, float(alpha))
            pred = predict_ridge(model, train_anchor_features[idx])
            fold_rmse_ms.append(float(np.sqrt(np.mean(np.square(pred - train_anchor_targets[idx]))) * 1000.0))
        row = {
            "alpha": float(alpha),
            "mean_anchor_rmse_ms": float(np.mean(fold_rmse_ms)),
            "fold_anchor_rmse_ms": fold_rmse_ms,
        }
        rows.append(row)
        if best is None or row["mean_anchor_rmse_ms"] < best["mean_anchor_rmse_ms"]:
            best = row
    assert best is not None
    return {"best": best, "rows": rows}


def experiment_specs() -> List[OffsetModelSpec]:
    return [
        OffsetModelSpec(
            name="loop_revisit_offset_regression",
            feature_names=anchor_names(),
            description="Anchor-level offset regression from loop/revisit features.",
        )
    ]


def build_report(payload: Mapping[str, object]) -> str:
    baseline = payload["baseline"]
    loop_patch = payload["loop_revisit_reference"]
    heldout = payload["heldout_variable_offset"]
    oracle = payload["same_episode_oracle"]
    lines = [
        "# RM75 Variable Time-Offset Experiment",
        "",
        "## Goal",
        "",
        "Test whether a slowly varying / piecewise time-offset model can beat the current `loop_revisit_world` held-out patch.",
        "",
        "## Results",
        "",
        f"- Baseline `rm75_0004` APE translation RMSE: `{baseline['ape_translation_rmse_mm']:.3f} mm`.",
        f"- Held-out variable-offset model: `{heldout['ape_translation_rmse_mm']:.3f} mm` (`{heldout['delta_translation_rmse_mm']:+.3f} mm`).",
        f"- Existing `loop_revisit_world` patch: `{loop_patch['corrected_translation_rmse_mm']:.3f} mm` (`{loop_patch['delta_translation_rmse_mm']:+.3f} mm`).",
        f"- Same-episode oracle-style smooth offset only: `{oracle['ape_translation_rmse_mm']:.3f} mm` (`{oracle['delta_translation_rmse_mm']:+.3f} mm`).",
        "",
        "Interpretation:",
        "- The held-out variable-offset model barely moves the metric and does not beat `loop_revisit_world`.",
        "- More importantly, even a same-episode smooth offset curve does not help by itself, which means offset variation is not the dominant missing correction term.",
        "- This keeps the algorithm-side diagnosis valuable, but weakens `time-offset-only` as the main fix path.",
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

    baseline = evaluate_dynamic_offset(
        test_episode,
        helpers,
        np.asarray(test_episode.matched_times, dtype=float),
        np.asarray(test_episode.matched_gt_pos, dtype=float),
        np.asarray(test_episode.matched_gt_rot, dtype=float),
        np.full(test_episode.matched_times.shape[0], float(test_episode.time_offset_sec), dtype=float),
    )

    train_anchor_times = []
    train_anchor_features = []
    train_anchor_targets = []
    train_anchor_labels = []
    for episode in train_episodes:
        anchors, labels = local_offset_labels(episode, helpers)
        features = anchor_feature_bank(episode, anchors)
        train_anchor_times.append(anchors)
        train_anchor_features.append(features)
        train_anchor_labels.append(labels)
        train_anchor_targets.append(labels - float(episode.time_offset_sec))

    tuning = tune_alpha(train_anchor_features, train_anchor_targets)
    alpha = float(tuning["best"]["alpha"])
    ridge = fit_ridge(np.vstack(train_anchor_features), np.concatenate(train_anchor_targets), alpha)

    test_anchor_times = np.linspace(float(test_episode.matched_times[0]), float(test_episode.matched_times[-1]), 11)
    test_anchor_features = anchor_feature_bank(test_episode, test_anchor_times)
    test_anchor_delta = predict_ridge(ridge, test_anchor_features)
    test_anchor_offsets = float(test_episode.time_offset_sec) + test_anchor_delta
    test_dynamic_offsets = np.interp(np.asarray(test_episode.matched_times, dtype=float), test_anchor_times, test_anchor_offsets)
    heldout = evaluate_dynamic_offset(
        test_episode,
        helpers,
        np.asarray(test_episode.matched_times, dtype=float),
        np.asarray(test_episode.matched_gt_pos, dtype=float),
        np.asarray(test_episode.matched_gt_rot, dtype=float),
        test_dynamic_offsets,
    )
    heldout["delta_translation_rmse_mm"] = heldout["ape_translation_rmse_mm"] - baseline["ape_translation_rmse_mm"]

    oracle_anchor_times, oracle_labels = local_offset_labels(test_episode, helpers)
    oracle_offsets = np.interp(np.asarray(test_episode.matched_times, dtype=float), oracle_anchor_times, oracle_labels)
    oracle = evaluate_dynamic_offset(
        test_episode,
        helpers,
        np.asarray(test_episode.matched_times, dtype=float),
        np.asarray(test_episode.matched_gt_pos, dtype=float),
        np.asarray(test_episode.matched_gt_rot, dtype=float),
        oracle_offsets,
    )
    oracle["delta_translation_rmse_mm"] = oracle["ape_translation_rmse_mm"] - baseline["ape_translation_rmse_mm"]

    loop_revisit = json.loads((REPO_ROOT / "data/evaluation/workbench/rm75_loop_revisit_experiment/experiment.json").read_text(encoding="utf-8"))
    best_loop = min(loop_revisit["models"], key=lambda row: row["test"]["delta_translation_rmse_mm"])

    payload = {
        "train_episodes": list(TRAIN_IDS),
        "test_episode": TEST_ID,
        "baseline": baseline,
        "heldout_variable_offset": heldout,
        "same_episode_oracle": oracle,
        "loop_revisit_reference": {
            "model": best_loop["name"],
            "corrected_translation_rmse_mm": best_loop["test"]["corrected_translation_rmse_mm"],
            "delta_translation_rmse_mm": best_loop["test"]["delta_translation_rmse_mm"],
        },
        "tuning": tuning,
        "test_anchor_offsets_sec": test_anchor_offsets.tolist(),
        "test_anchor_delta_ms": (test_anchor_delta * 1000.0).tolist(),
        "test_anchor_times": test_anchor_times.tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(
        output_dir / "summary.csv",
        ["variant", "ape_translation_rmse_mm", "delta_translation_rmse_mm"],
        [
            {"variant": "baseline", "ape_translation_rmse_mm": baseline["ape_translation_rmse_mm"], "delta_translation_rmse_mm": 0.0},
            {"variant": "heldout_variable_offset", "ape_translation_rmse_mm": heldout["ape_translation_rmse_mm"], "delta_translation_rmse_mm": heldout["delta_translation_rmse_mm"]},
            {"variant": "same_episode_oracle_offset", "ape_translation_rmse_mm": oracle["ape_translation_rmse_mm"], "delta_translation_rmse_mm": oracle["delta_translation_rmse_mm"]},
            {"variant": "loop_revisit_world", "ape_translation_rmse_mm": best_loop["test"]["corrected_translation_rmse_mm"], "delta_translation_rmse_mm": best_loop["test"]["delta_translation_rmse_mm"]},
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
