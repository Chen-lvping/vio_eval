#!/usr/bin/env python3
"""Scan whether 0614 VINS can be axis-aligned to the same GT points and poses."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPERS_PATH = REPO_ROOT / "script/evaluate_vins_accuracy.py"

DEFAULT_ESTIMATE = Path("/home/chenlvping/6_data_use/0614 _test/episode_20260614_0239/right/pose_data.csv")
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data/ground_truth/trajectory_samples0614/trajectory_001.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/axis_aligned_scan_0614"
DEFAULT_HANDEYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"

T_LEFT_CAMERA_IMU = np.array(
    [
        [-0.999638319, 0.0265241228, -0.00445450377, -0.0022427286],
        [-0.0265235156, -0.999648213, -0.000195318818, 0.0140962508],
        [-0.0044581173, -7.70990737e-05, 0.999990046, -0.0155969206],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)

T_TCP_LEFT_CAMERA = np.array(
    [
        [0.854672738, -0.422689316, 0.301443615, 0.020087823],
        [0.519166541, 0.694970001, -0.497476432, -0.097333617],
        [0.000783703, 0.581678983, 0.813418064, 0.052300458],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def load_helpers() -> Dict[str, object]:
    import runpy

    return runpy.run_path(str(HELPERS_PATH), run_name="__axis_aligned_scan__")


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


def first_existing(row: dict, names: Sequence[str]) -> str | None:
    lower = {str(k).strip().lower(): k for k in row.keys()}
    for name in names:
        key = lower.get(name.lower())
        if key is not None and row.get(key) not in (None, ""):
            return str(row[key])
    return None


def quat_xyzw_to_rot(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=float).reshape(4)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 1e-12:
        raise ValueError("invalid quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def transform_from_pose(position: Sequence[float], quaternion_xyzw: Sequence[float]) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, :3] = quat_xyzw_to_rot(quaternion_xyzw)
    out[:3, 3] = np.asarray(position, dtype=float)
    return out


def load_estimate(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            ts = first_existing(row, ["Timestamp_us", "timestamp_us", "Timestamp_ns", "timestamp_ns", "timestamp", "time", "t"])
            xyz = [first_existing(row, names) for names in (["X", "x"], ["Y", "y"], ["Z", "z"])]
            quat = [
                first_existing(row, ["Quat_X", "qx", "q_x"]),
                first_existing(row, ["Quat_Y", "qy", "q_y"]),
                first_existing(row, ["Quat_Z", "qz", "q_z"]),
                first_existing(row, ["Quat_W", "qw", "q_w"]),
            ]
            if any(v is None for v in xyz + quat):
                continue
            timestamp = normalize_timestamp(float(ts)) if ts is not None else float(idx)
            rows.append((timestamp, np.asarray([float(v) for v in xyz], dtype=float), np.asarray([float(v) for v in quat], dtype=float)))
    rows.sort(key=lambda item: item[0])
    times = np.asarray([r[0] for r in rows], dtype=float)
    pos = np.asarray([r[1] for r in rows], dtype=float)
    rot = np.asarray([quat_xyzw_to_rot(r[2]) for r in rows], dtype=float)
    return times, pos, rot


def load_gt(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples", data)
    rows = []
    for sample in samples:
        rows.append(
            (
                normalize_timestamp(float(sample["timestamp"])),
                np.asarray(sample["position_m"], dtype=float),
                quat_xyzw_to_rot(sample["quaternion_xyzw"]),
            )
        )
    rows.sort(key=lambda item: item[0])
    return np.asarray([r[0] for r in rows]), np.asarray([r[1] for r in rows]), np.asarray([r[2] for r in rows])


def interpolate_gt(
    gt_times: np.ndarray,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_times: np.ndarray,
    max_gap_s: float,
    helpers: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matched_indices: List[int] = []
    out_pos: List[np.ndarray] = []
    out_rot: List[np.ndarray] = []
    out_times: List[float] = []
    left = 0
    for idx, t in enumerate(est_times):
        while left + 1 < gt_times.size and gt_times[left + 1] <= t:
            left += 1
        if left + 1 >= gt_times.size or t < gt_times[left]:
            continue
        t0 = gt_times[left]
        t1 = gt_times[left + 1]
        if max(abs(t - t0), abs(t1 - t)) > max_gap_s:
            continue
        alpha = 0.0 if abs(t1 - t0) < 1e-12 else float((t - t0) / (t1 - t0))
        q0 = helpers["rot_to_quat_xyzw"](gt_rot[left])
        q1 = helpers["rot_to_quat_xyzw"](gt_rot[left + 1])
        out_times.append(float(t))
        out_pos.append((1.0 - alpha) * gt_pos[left] + alpha * gt_pos[left + 1])
        out_rot.append(helpers["quat_xyzw_to_rot"](helpers["slerp"](q0, q1, alpha)))
        matched_indices.append(idx)
    return (
        np.asarray(out_times, dtype=float),
        np.asarray(out_pos, dtype=float),
        np.asarray(out_rot, dtype=float),
        np.asarray(matched_indices, dtype=int),
    )


def signed_axis_matrices() -> Iterable[Tuple[str, np.ndarray]]:
    axes = ("x", "y", "z")
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=float)
            labels = []
            for col, src_axis in enumerate(perm):
                matrix[src_axis, col] = signs[col]
                labels.append(("-" if signs[col] < 0 else "+") + axes[src_axis])
            if round(float(np.linalg.det(matrix))) < 0:
                continue
            yield f"{''.join(labels)}", matrix


def apply_axis_map_world(pos: np.ndarray, rot: np.ndarray, axis_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return (axis_map @ pos.T).T, np.asarray([axis_map @ r for r in rot], dtype=float)


def apply_axis_map_local(rot: np.ndarray, axis_map: np.ndarray) -> np.ndarray:
    return np.asarray([r @ axis_map for r in rot], dtype=float)


def evaluate(
    name: str,
    gt_pos: np.ndarray,
    gt_rot: np.ndarray,
    est_pos: np.ndarray,
    est_rot: np.ndarray,
    helpers: Dict[str, object],
    category: str,
) -> Dict[str, object]:
    _, align_rot, align_trans = helpers["align_umeyama"](est_pos, gt_pos, with_scale=False)
    est_pos_aligned = (align_rot @ est_pos.T).T + align_trans
    est_rot_aligned = np.asarray([align_rot @ r for r in est_rot], dtype=float)
    trans_mm = np.linalg.norm(est_pos_aligned - gt_pos, axis=1) * 1000.0
    rot_deg = np.asarray([rotation_angle_deg(g.T @ e) for g, e in zip(gt_rot, est_rot_aligned)], dtype=float)
    return {
        "name": name,
        "category": category,
        "translation_rmse_mm": stats(trans_mm)["rmse"],
        "rotation_rmse_deg": stats(rot_deg)["rmse"],
        "rotation_mean_deg": stats(rot_deg)["mean"],
        "translation_mean_mm": stats(trans_mm)["mean"],
        "align_rotation_deg": rotation_angle_deg(align_rot),
        "align_translation_norm_mm": float(np.linalg.norm(align_trans) * 1000.0),
        "samples": int(gt_pos.shape[0]),
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--estimate-frame", choices=["imu", "vins_base_link"], default="imu")
    parser.add_argument("--max-gap-ms", type=float, default=80.0)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    helpers = load_helpers()

    gt_times, gt_pos, gt_rot = load_gt(args.ground_truth.expanduser().resolve())
    est_times, est_pos_raw, est_rot_raw = load_estimate(args.estimate.expanduser().resolve())

    if args.estimate_frame == "imu":
        t_world_imu = np.asarray([np.eye(4, dtype=float) for _ in range(est_pos_raw.shape[0])], dtype=float)
        for i in range(est_pos_raw.shape[0]):
            t_world_imu[i, :3, :3] = est_rot_raw[i]
            t_world_imu[i, :3, 3] = est_pos_raw[i]
        t_world_tcp = np.asarray([t @ np.linalg.inv(T_LEFT_CAMERA_IMU) @ np.linalg.inv(T_TCP_LEFT_CAMERA) for t in t_world_imu], dtype=float)
    else:
        t_world_tcp = np.asarray([transform_from_pose(est_pos_raw[i], [0, 0, 0, 1]) for i in range(est_pos_raw.shape[0])], dtype=float)

    est_pos = t_world_tcp[:, :3, 3]
    est_rot = t_world_tcp[:, :3, :3]

    _, matched_gt_pos, matched_gt_rot, matched_est_idx = interpolate_gt(
        gt_times,
        gt_pos,
        gt_rot,
        est_times,
        args.max_gap_ms / 1000.0,
        helpers,
    )
    matched_est_pos = est_pos[matched_est_idx]
    matched_est_rot = est_rot[matched_est_idx]

    rows: List[Dict[str, object]] = []
    rows.append(evaluate("baseline", matched_gt_pos, matched_gt_rot, matched_est_pos, matched_est_rot, helpers, "baseline"))

    for label, axis_map in signed_axis_matrices():
        gt_pos_world, gt_rot_world = apply_axis_map_world(matched_gt_pos, matched_gt_rot, axis_map)
        rows.append(
            evaluate(
                f"world_{label}",
                gt_pos_world,
                gt_rot_world,
                matched_est_pos,
                matched_est_rot,
                helpers,
                "world_axis_map",
            )
        )
        gt_rot_local = apply_axis_map_local(matched_gt_rot, axis_map)
        rows.append(
            evaluate(
                f"local_{label}",
                matched_gt_pos,
                gt_rot_local,
                matched_est_pos,
                matched_est_rot,
                helpers,
                "local_axis_map",
            )
        )

    rows.sort(key=lambda r: (float(r["rotation_rmse_deg"]), float(r["translation_rmse_mm"])))
    write_csv(output_dir / "axis_aligned_scan.csv", rows)
    (output_dir / "REPORT.md").write_text(
        "\n".join(
            [
                "# 0614 Axis-Aligned Scan",
                "",
                "| rank | category | name | trans RMSE mm | rot RMSE deg | align rotation deg | align translation norm mm |",
                "|---:|---|---|---:|---:|---:|---:|",
            ]
            + [
                f"| {i} | {r['category']} | `{r['name']}` | {float(r['translation_rmse_mm']):.3f} | {float(r['rotation_rmse_deg']):.3f} | {float(r['align_rotation_deg']):.3f} | {float(r['align_translation_norm_mm']):.3f} |"
                for i, r in enumerate(rows[:16], 1)
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    best = rows[0]
    print(f"[OK] wrote {output_dir / 'axis_aligned_scan.csv'}")
    print(f"[OK] wrote {output_dir / 'REPORT.md'}")
    print(
        "[BEST]",
        best["category"],
        best["name"],
        f"trans={best['translation_rmse_mm']:.3f}mm",
        f"rot={best['rotation_rmse_deg']:.3f}deg",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
