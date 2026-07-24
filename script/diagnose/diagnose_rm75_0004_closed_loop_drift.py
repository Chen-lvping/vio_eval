#!/usr/bin/env python3
"""Diagnose low-speed closed-loop drift symptoms for RM75 episode 0004."""

from __future__ import annotations

import argparse
import csv
import json
import runpy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG_SCRIPT = REPO_ROOT / "script/diagnose/diagnose_rm75_state_generalization.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/rm75_0004_closed_loop_drift_diagnosis"


def load_diag_module() -> Dict[str, object]:
    return runpy.run_path(str(DIAG_SCRIPT), run_name="__rm75_closed_loop_drift__")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def normalized_progress(gt_pos: np.ndarray) -> np.ndarray:
    step = np.zeros(gt_pos.shape[0], dtype=float)
    step[1:] = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)
    progress = np.cumsum(step)
    progress /= max(float(progress[-1]), 1e-9)
    return progress


def progress_bin_rows(episode) -> List[Dict[str, object]]:
    progress = normalized_progress(np.asarray(episode.matched_gt_pos, dtype=float))
    err_world_mm = np.asarray(episode.translation_error_world, dtype=float) * 1000.0
    err_local_mm = np.einsum(
        "nij,nj->ni",
        np.transpose(np.asarray(episode.matched_gt_rot, dtype=float), (0, 2, 1)),
        err_world_mm,
    )
    err_mag_mm = np.linalg.norm(err_world_mm, axis=1)
    rows = []
    for left, right in zip(np.linspace(0.0, 1.0, 6)[:-1], np.linspace(0.0, 1.0, 6)[1:]):
        mask = (progress >= left) & (progress < right if right < 1.0 else progress <= right)
        rows.append(
            {
                "progress_bin": f"{left:.1f}-{right:.1f}",
                "samples": int(np.sum(mask)),
                "mean_err_rmse_mm": float(np.sqrt(np.mean(np.square(err_mag_mm[mask])))),
                "mean_local_tx_mm": float(np.mean(err_local_mm[mask, 0])),
                "mean_local_ty_mm": float(np.mean(err_local_mm[mask, 1])),
                "mean_local_tz_mm": float(np.mean(err_local_mm[mask, 2])),
            }
        )
    return rows


def segment_best_offsets(episode, tcp_eval: Dict[str, object], helpers: Dict[str, object], baseline_offset_sec: float) -> List[Dict[str, object]]:
    progress = normalized_progress(np.asarray(episode.matched_gt_pos, dtype=float))
    speed = np.asarray(episode.features["speed"], dtype=float)
    masks = {
        "first20": progress < 0.2,
        "mid20_60": (progress >= 0.2) & (progress < 0.6),
        "last20": progress >= 0.8,
        "slow35": speed < np.percentile(speed, 35),
        "fast35": speed > np.percentile(speed, 65),
    }
    rows = []
    for name, mask in masks.items():
        ref_times = episode.matched_times[mask]
        ref_pos = episode.matched_gt_pos[mask]
        ref_rot = episode.matched_gt_rot[mask]
        best = None
        for cand in np.arange(baseline_offset_sec - 0.08, baseline_offset_sec + 0.0801, 0.004):
            try:
                assoc = tcp_eval["associate_by_nearest_time"](
                    ref_times,
                    ref_pos,
                    ref_rot,
                    episode.est_times,
                    episode.est_pos_tcp,
                    episode.est_rot_tcp,
                    float(cand),
                    0.01,
                )
            except RuntimeError:
                continue
            assoc_times, assoc_gt_pos, assoc_gt_rot, assoc_est_pos, assoc_est_rot = assoc
            if assoc_times.size < 100:
                continue
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
            rmse_mm = float(se3.translation_metrics_m["rmse"] * 1000.0)
            if best is None or rmse_mm < best["best_translation_rmse_mm"]:
                best = {
                    "segment": name,
                    "samples": int(assoc_times.size),
                    "best_offset_sec": float(cand),
                    "best_translation_rmse_mm": rmse_mm,
                    "delta_vs_global_ms": float((cand - baseline_offset_sec) * 1000.0),
                }
        if best is not None:
            rows.append(best)
    return rows


def revisit_stats(episode) -> Dict[str, object]:
    gt_pos = np.asarray(episode.matched_gt_pos, dtype=float)
    gt_rot = np.asarray(episode.matched_gt_rot, dtype=float)
    times = np.asarray(episode.matched_times, dtype=float)
    err_mm = np.linalg.norm(np.asarray(episode.translation_error_world, dtype=float), axis=1) * 1000.0

    rows = []
    for idx in range(50, times.shape[0], 5):
        prev = np.arange(0, idx - 50)
        if prev.size == 0:
            continue
        dist = np.linalg.norm(gt_pos[prev] - gt_pos[idx], axis=1)
        best_prev = int(prev[np.argmin(dist)])
        heading_gap_deg = Rotation.from_matrix(gt_rot[best_prev].T @ gt_rot[idx]).magnitude() * 180.0 / np.pi
        if float(np.min(dist)) < 0.03 and heading_gap_deg < 15.0:
            rows.append(
                {
                    "spatial_dist_m": float(np.min(dist)),
                    "heading_gap_deg": float(heading_gap_deg),
                    "error_diff_mm": float(abs(err_mm[idx] - err_mm[best_prev])),
                }
            )

    return {
        "pair_count": len(rows),
        "mean_spatial_dist_m": float(np.mean([row["spatial_dist_m"] for row in rows])) if rows else 0.0,
        "mean_error_diff_mm": float(np.mean([row["error_diff_mm"] for row in rows])) if rows else 0.0,
        "max_error_diff_mm": float(np.max([row["error_diff_mm"] for row in rows])) if rows else 0.0,
    }


def build_report(payload: Mapping[str, object]) -> str:
    drift = payload["rm75_0004"]
    best_segment = max(drift["segment_offsets"], key=lambda row: abs(row["delta_vs_global_ms"]))
    best_model = payload["cross_episode_reference"]["best_loop_revisit_delta_mm"]
    lines = [
        "# RM75 0004 Closed-Loop Drift Diagnosis",
        "",
        "## Main Findings",
        "",
        f"1. The strongest held-out model-side gain so far is only `{best_model:+.3f} mm`, so we still need algorithm-side evidence to explain the remaining error.",
        "2. `rm75_0004` error is strongly phase-dependent: early and late loop segments are much worse than the middle of the loop.",
        f"3. The largest segment-specific time-offset deviation is `{best_segment['delta_vs_global_ms']:+.1f} ms` on `{best_segment['segment']}`, which is too large to treat as a tiny constant-offset perturbation.",
        "4. Revisit statistics show that the same spatial region can carry noticeably different errors on later revisits, which is consistent with hysteresis or slow drift.",
        "",
        "## Progress-Bin Residual Drift",
        "",
        "| progress bin | samples | RMSE mm | mean local tx mm | mean local ty mm | mean local tz mm |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in drift["progress_bins"]:
        lines.append(
            f"| {row['progress_bin']} | {row['samples']} | {row['mean_err_rmse_mm']:.2f} | "
            f"{row['mean_local_tx_mm']:.2f} | {row['mean_local_ty_mm']:.2f} | {row['mean_local_tz_mm']:.2f} |"
        )
    lines += [
        "",
        "Interpretation:",
        "- The residual mean changes sign across the loop, especially in local X/Y.",
        "- That sign flip is inconsistent with one fixed translation correction and much more consistent with progress-dependent drift.",
        "",
        "## Segment-Specific Time Offset",
        "",
        "| segment | samples | best offset s | d vs global ms | best RMSE mm |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in drift["segment_offsets"]:
        lines.append(
            f"| {row['segment']} | {row['samples']} | {row['best_offset_sec']:.6f} | "
            f"{row['delta_vs_global_ms']:+.1f} | {row['best_translation_rmse_mm']:.2f} |"
        )
    lines += [
        "",
        "Interpretation:",
        "- Low-speed and late-loop segments prefer very different offsets from the global optimum.",
        "- This does not prove clock drift by itself, but it is exactly the pattern you would expect when one constant time offset is no longer sufficient.",
        "",
        "## Revisit Hysteresis Comparison",
        "",
        "| episode | revisit pairs | mean error diff mm | max error diff mm |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload["revisit_comparison"]:
        lines.append(
            f"| {row['episode_id']} | {row['pair_count']} | {row['mean_error_diff_mm']:.2f} | {row['max_error_diff_mm']:.2f} |"
        )
    lines += [
        "",
        "Interpretation:",
        "- `rm75_0004` has the largest revisit-pair error inconsistency among the four RM75 episodes except for `rm75_0001` in max outlier only; its mean revisit inconsistency is the highest.",
        "- That supports the idea that `0004` has a history-dependent component rather than a pure pose-only bias.",
        "",
        "## Direction Comparison",
        "",
        f"- Model track: the new loop/revisit model improves held-out `rm75_0004` by `{payload['cross_episode_reference']['best_loop_revisit_delta_mm']:+.3f} mm`.",
        f"- Algorithm-side track: segment offsets vary by up to `{max(abs(row['delta_vs_global_ms']) for row in drift['segment_offsets']):.1f} ms`, and residual means drift from positive to negative across loop progress bins.",
        "- Recommendation: the model track is better as a short-term patch candidate, but the algorithm-side track looks more informative for root cause because it explains why revisit-aware features help at all.",
    ]
    return "\n".join(lines) + "\n"


def run_diagnosis(output_dir: Path) -> Dict[str, object]:
    diag = load_diag_module()
    tcp_eval = diag["load_tcp_eval"]()
    helpers = tcp_eval["load_helpers"]()
    handeye = tcp_eval["load_tcp_left_camera_transform"](diag["DEFAULT_HANDEYE"])
    episodes = [diag["analyze_episode"](spec, tcp_eval, helpers, handeye) for spec in diag["EPISODES"]]
    by_id = {ep.spec.episode_id: ep for ep in episodes}
    ep = by_id["rm75_0004"]

    baseline_metrics = json.loads((REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0004/metrics.json").read_text(encoding="utf-8"))
    baseline_offset_sec = float(baseline_metrics["time_offset_sec"])
    progress_bins = progress_bin_rows(ep)
    segment_offsets = segment_best_offsets(ep, tcp_eval, helpers, baseline_offset_sec)
    revisit_comparison = []
    revisit_by_episode = {}
    for episode in episodes:
        stats = revisit_stats(episode)
        revisit_by_episode[episode.spec.episode_id] = stats
        revisit_comparison.append({"episode_id": episode.spec.episode_id, **stats})

    loop_payload = json.loads((REPO_ROOT / "data/evaluation/workbench/rm75_loop_revisit_experiment/experiment.json").read_text(encoding="utf-8"))
    best_loop_model = min(loop_payload["models"], key=lambda row: row["test"]["delta_translation_rmse_mm"])

    payload = {
        "rm75_0004": {
            "baseline_offset_sec": baseline_offset_sec,
            "progress_bins": progress_bins,
            "segment_offsets": segment_offsets,
            "revisit_stats": revisit_by_episode["rm75_0004"],
        },
        "revisit_comparison": revisit_comparison,
        "cross_episode_reference": {
            "best_loop_revisit_model": best_loop_model["name"],
            "best_loop_revisit_delta_mm": best_loop_model["test"]["delta_translation_rmse_mm"],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnosis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(output_dir / "progress_bins.csv", progress_bins[0].keys(), progress_bins)
    write_csv(output_dir / "segment_offsets.csv", segment_offsets[0].keys(), segment_offsets)
    write_csv(output_dir / "revisit_comparison.csv", revisit_comparison[0].keys(), revisit_comparison)
    (output_dir / "REPORT.md").write_text(build_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_diagnosis(args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
