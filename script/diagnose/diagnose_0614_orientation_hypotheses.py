#!/usr/bin/env python3
"""Diagnose whether 0614 TCP attitude error looks like axis mapping or GT quaternion interpretation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import runpy
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TCP_EVAL_SCRIPT = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
HELPERS_PATH = REPO_ROOT / "script/evaluate_vins_accuracy.py"

DEFAULT_ESTIMATE = Path("/home/chenlvping/6_data_use/0614 _test/episode_20260614_0239/right/pose_data.csv")
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data/ground_truth/trajectory_samples0614/trajectory_001.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/orientation_hypothesis_scan_0614"
DEFAULT_HANDEYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"


def load_api() -> Dict[str, object]:
    return runpy.run_path(str(TCP_EVAL_SCRIPT), run_name="__orient_hypothesis_tcp_eval__")


def load_helpers() -> Dict[str, object]:
    return runpy.run_path(str(HELPERS_PATH), run_name="__orient_hypothesis_helpers__")


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) * 0.5
    value = max(-1.0, min(1.0, value))
    return math.degrees(math.acos(value))


def stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "rmse": float(math.sqrt(np.mean(values * values))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95.0)),
    }


def best_left_rotation(source_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=float)
    for src, dst in zip(source_rot, target_rot):
        matrix += dst @ src.T
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def best_right_rotation(source_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=float)
    for src, dst in zip(source_rot, target_rot):
        matrix += src.T @ dst
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def rotation_errors(gt_rot: np.ndarray, est_rot: np.ndarray) -> np.ndarray:
    return np.asarray([rotation_angle_deg(g.T @ e) for g, e in zip(gt_rot, est_rot)], dtype=float)


def apply_position_alignment(
    helpers: Dict[str, object],
    gt_pos: np.ndarray,
    est_pos: np.ndarray,
    est_rot: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, align_rot, align_trans = helpers["align_umeyama"](est_pos, gt_pos, with_scale=False)
    aligned_pos = (align_rot @ est_pos.T).T + align_trans
    aligned_rot = np.asarray([align_rot @ rot for rot in est_rot], dtype=float)
    return aligned_pos, aligned_rot, align_rot


def signed_axis_matrices(proper_only: bool) -> Iterable[Tuple[str, np.ndarray]]:
    axes = ("x", "y", "z")
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=float)
            labels = []
            for col, src_axis in enumerate(perm):
                matrix[src_axis, col] = signs[col]
                labels.append(("-" if signs[col] < 0 else "+") + axes[src_axis])
            det = round(float(np.linalg.det(matrix)))
            if proper_only and det < 0:
                continue
            yield f"{''.join(labels)} det={det:+d}", matrix


def quat_variants(quat_xyzw: Sequence[float], helpers: Dict[str, object]) -> Dict[str, np.ndarray]:
    q = np.asarray(quat_xyzw, dtype=float).reshape(4)
    x, y, z, w = q.tolist()
    variants = {
        "gt_quat_xyzw": [x, y, z, w],
        "gt_quat_xyzw_conjugate": [-x, -y, -z, w],
        "gt_quat_wxyz_reinterpreted": [y, z, w, x],
        "gt_quat_wxyz_reinterpreted_conjugate": [-y, -z, -w, x],
        "gt_quat_xyzw_vector_neg_w_same": [-x, -y, -z, w],
        "gt_quat_xyzw_w_neg": [x, y, z, -w],
    }
    out: Dict[str, np.ndarray] = {}
    for name, values in variants.items():
        try:
            out[name] = helpers["quat_xyzw_to_rot"](values)
        except Exception:
            continue
    return out


def load_gt_variants(path: Path, helpers: Dict[str, object]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples", data)
    times: List[float] = []
    positions: List[np.ndarray] = []
    rotations_by_name: Dict[str, List[np.ndarray]] = {}
    tcp_api = load_api()
    normalize_timestamp = tcp_api["normalize_timestamp"]

    for sample in samples:
        times.append(normalize_timestamp(float(sample["timestamp"])))
        positions.append(np.asarray(sample["position_m"], dtype=float))
        for name, rot in quat_variants(sample["quaternion_xyzw"], helpers).items():
            rotations_by_name.setdefault(name, []).append(rot)

    order = np.argsort(np.asarray(times, dtype=float))
    times_arr = np.asarray(times, dtype=float)[order]
    pos_arr = np.asarray(positions, dtype=float)[order]
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, rotations in rotations_by_name.items():
        out[name] = (times_arr, pos_arr, np.asarray(rotations, dtype=float)[order])
    return out


def evaluate_case(
    name: str,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_pos: np.ndarray,
    est_rot: np.ndarray,
    helpers: Dict[str, object],
    category: str,
    detail: str,
) -> Dict[str, object]:
    aligned_pos, pos_aligned_rot, pos_align_rot = apply_position_alignment(helpers, gt_pos, est_pos, est_rot)
    trans_mm = np.linalg.norm(aligned_pos - gt_pos, axis=1) * 1000.0
    pos_rot_deg = rotation_errors(gt_rot, pos_aligned_rot)

    left_rot = best_left_rotation(est_rot, gt_rot)
    left_aligned = np.asarray([left_rot @ rot for rot in est_rot], dtype=float)
    left_rot_deg = rotation_errors(gt_rot, left_aligned)

    right_rot = best_right_rotation(est_rot, gt_rot)
    right_aligned = np.asarray([rot @ right_rot for rot in est_rot], dtype=float)
    right_rot_deg = rotation_errors(gt_rot, right_aligned)

    pos_stats = stats(pos_rot_deg)
    left_stats = stats(left_rot_deg)
    right_stats = stats(right_rot_deg)
    trans_stats = stats(trans_mm)
    return {
        "name": name,
        "category": category,
        "detail": detail,
        "samples": int(gt_pos.shape[0]),
        "translation_rmse_mm": trans_stats["rmse"],
        "position_align_rotation_rmse_deg": pos_stats["rmse"],
        "position_align_rotation_mean_deg": pos_stats["mean"],
        "orientation_left_fit_rmse_deg": left_stats["rmse"],
        "orientation_left_fit_mean_deg": left_stats["mean"],
        "orientation_right_fit_rmse_deg": right_stats["rmse"],
        "orientation_right_fit_mean_deg": right_stats["mean"],
        "best_orientation_rmse_deg": min(left_stats["rmse"], right_stats["rmse"]),
        "position_vs_left_fit_delta_deg": rotation_angle_deg(left_rot @ pos_align_rot.T),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, default=DEFAULT_ESTIMATE)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--handeye-yaml", type=Path, default=DEFAULT_HANDEYE)
    parser.add_argument("--estimate-frame", choices=["imu", "vins_base_link", "camera"], default="imu")
    parser.add_argument("--max-gap-ms", type=float, default=80.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tcp_api = load_api()
    helpers = load_helpers()
    handeye = tcp_api["load_tcp_left_camera_transform"](args.handeye_yaml.expanduser().resolve())
    gt_times, gt_pos, gt_rot = tcp_api["load_robot_tcp_trajectory"](args.ground_truth.expanduser().resolve(), helpers)
    est_times, est_pos, est_rot = tcp_api["load_vio_tcp_trajectory"](
        args.estimate.expanduser().resolve(),
        helpers,
        handeye,
        args.estimate_frame,
    )
    times, matched_gt_pos, matched_gt_rot, matched_est_pos, matched_est_rot = tcp_api["match_ground_truth"](
        gt_times,
        gt_pos,
        gt_rot,
        est_times,
        est_pos,
        est_rot,
        helpers,
        args.max_gap_ms / 1000.0,
    )

    rows: List[Dict[str, object]] = [
        evaluate_case(
            "baseline_current_gt_quat",
            matched_gt_pos,
            matched_gt_rot,
            matched_est_pos,
            matched_est_rot,
            helpers,
            "baseline",
            "current GT quaternion interpreted as xyzw T_base_tcp",
        )
    ]

    for label, axis_map in signed_axis_matrices(proper_only=True):
        corrected_gt_rot = np.asarray([rot @ axis_map for rot in matched_gt_rot], dtype=float)
        rows.append(
            evaluate_case(
                f"gt_local_axis_map_{label}",
                matched_gt_pos,
                corrected_gt_rot,
                matched_est_pos,
                matched_est_rot,
                helpers,
                "axis_mapping_local",
                label,
            )
        )
        corrected_gt_rot_world = np.asarray([axis_map @ rot for rot in matched_gt_rot], dtype=float)
        rows.append(
            evaluate_case(
                f"gt_world_axis_map_{label}",
                matched_gt_pos,
                corrected_gt_rot_world,
                matched_est_pos,
                matched_est_rot,
                helpers,
                "axis_mapping_world",
                label,
            )
        )

    gt_variants = load_gt_variants(args.ground_truth.expanduser().resolve(), helpers)
    for variant_name, (variant_times, variant_pos, variant_rot) in gt_variants.items():
        try:
            _, variant_gt_pos, variant_gt_rot, variant_est_pos, variant_est_rot = tcp_api["match_ground_truth"](
                variant_times,
                variant_pos,
                variant_rot,
                est_times,
                est_pos,
                est_rot,
                helpers,
                args.max_gap_ms / 1000.0,
            )
        except Exception:
            continue
        rows.append(
            evaluate_case(
                variant_name,
                variant_gt_pos,
                variant_gt_rot,
                variant_est_pos,
                variant_est_rot,
                helpers,
                "gt_quaternion_interpretation",
                variant_name,
            )
        )

    rows.sort(key=lambda row: (float(row["best_orientation_rmse_deg"]), float(row["position_align_rotation_rmse_deg"])))
    write_csv(output_dir / "hypothesis_scan.csv", rows)
    (output_dir / "hypothesis_scan.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    top = rows[:12]
    lines = [
        "# 0614 Orientation Hypothesis Scan",
        "",
        "## Top Hypotheses",
        "",
        "| rank | category | detail | trans RMSE mm | rot RMSE after position SE(3) deg | best orientation-only RMSE deg |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for idx, row in enumerate(top, 1):
        lines.append(
            f"| {idx} | {row['category']} | `{row['detail']}` | "
            f"{float(row['translation_rmse_mm']):.3f} | "
            f"{float(row['position_align_rotation_rmse_deg']):.3f} | "
            f"{float(row['best_orientation_rmse_deg']):.3f} |"
        )
    baseline = next(row for row in rows if row["name"] == "baseline_current_gt_quat")
    best_axis = min((row for row in rows if str(row["category"]).startswith("axis_mapping")), key=lambda row: row["best_orientation_rmse_deg"])
    best_quat = min((row for row in rows if row["category"] == "gt_quaternion_interpretation"), key=lambda row: row["best_orientation_rmse_deg"])
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"- Baseline: position SE(3) rotation RMSE `{baseline['position_align_rotation_rmse_deg']:.3f} deg`, orientation-only RMSE `{baseline['best_orientation_rmse_deg']:.3f} deg`.",
            f"- Best axis mapping: `{best_axis['detail']}` with orientation-only RMSE `{best_axis['best_orientation_rmse_deg']:.3f} deg` and position-aligned rotation RMSE `{best_axis['position_align_rotation_rmse_deg']:.3f} deg`.",
            f"- Best GT quaternion interpretation: `{best_quat['detail']}` with orientation-only RMSE `{best_quat['best_orientation_rmse_deg']:.3f} deg` and position-aligned rotation RMSE `{best_quat['position_align_rotation_rmse_deg']:.3f} deg`.",
            "",
            "If a discrete axis mapping were the dominant issue, one of the signed permutation rows should collapse the orientation-only RMSE to a few degrees. If a quaternion order/direction mistake were dominant, one of the GT quaternion interpretation rows should do the same.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {output_dir / 'hypothesis_scan.csv'}")
    print(f"[OK] wrote {output_dir / 'REPORT.md'}")
    print(
        "[BEST]",
        rows[0]["category"],
        rows[0]["detail"],
        f"pos_rot={rows[0]['position_align_rotation_rmse_deg']:.3f}deg",
        f"ori={rows[0]['best_orientation_rmse_deg']:.3f}deg",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
