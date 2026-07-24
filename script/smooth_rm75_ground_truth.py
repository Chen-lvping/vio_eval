#!/usr/bin/env python3
"""Smooth RM75 ground-truth TCP poses while preserving timestamps and schema."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from typing import Dict, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_HELPERS = REPO_ROOT / "script/evaluate_vins_accuracy.py"


def load_helpers() -> Dict[str, object]:
    return runpy.run_path(str(EVAL_HELPERS), run_name="__smooth_rm75_gt__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input RM75 ground-truth JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output smoothed JSON")
    parser.add_argument(
        "--window",
        type=int,
        default=11,
        help="Odd moving-average window for position and quaternion smoothing",
    )
    parser.add_argument(
        "--export-tum",
        type=Path,
        default=None,
        help="Optional TUM export path for the smoothed trajectory",
    )
    parser.add_argument(
        "--max-pos-dev-mm",
        type=float,
        default=None,
        help="Optional maximum allowed positional deviation from the original trajectory",
    )
    parser.add_argument(
        "--max-rot-dev-deg",
        type=float,
        default=None,
        help="Optional maximum allowed rotational deviation from the original trajectory",
    )
    return parser.parse_args()


def ensure_odd_window(window: int) -> int:
    if window < 1:
        raise ValueError("--window must be >= 1")
    if window % 2 == 0:
        raise ValueError("--window must be odd")
    return window


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.shape[0] <= 2:
        return values.copy()
    half = window // 2
    padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
    out = np.empty_like(values, dtype=float)
    for idx in range(values.shape[0]):
        out[idx] = padded[idx : idx + window].mean(axis=0)
    return out


def smooth_quaternions_xyzw(quats: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or quats.shape[0] <= 2:
        return quats.copy()

    # Keep quaternions on a consistent hemisphere before averaging.
    aligned = quats.copy()
    for idx in range(1, aligned.shape[0]):
        if float(np.dot(aligned[idx - 1], aligned[idx])) < 0.0:
            aligned[idx] *= -1.0

    half = window // 2
    padded = np.pad(aligned, ((half, half), (0, 0)), mode="edge")
    out = np.empty_like(aligned, dtype=float)
    for idx in range(aligned.shape[0]):
        mean_q = padded[idx : idx + window].mean(axis=0)
        norm = float(np.linalg.norm(mean_q))
        if norm <= 1e-12:
            out[idx] = aligned[idx]
        else:
            out[idx] = mean_q / norm
    return out


def normalize_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return quat.copy()
    return quat / norm


def slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalize_quaternion_xyzw(q0)
    q1 = normalize_quaternion_xyzw(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion_xyzw((1.0 - alpha) * q0 + alpha * q1)
    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * alpha
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(sin_theta / sin_theta_0)
    return normalize_quaternion_xyzw(s0 * q0 + s1 * q1)


def quaternion_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = normalize_quaternion_xyzw(q0)
    q1 = normalize_quaternion_xyzw(q1)
    dot = abs(float(np.dot(q0, q1)))
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def clamp_positions_to_deviation(
    original: np.ndarray,
    smoothed: np.ndarray,
    max_dev_m: float | None,
) -> np.ndarray:
    if max_dev_m is None or max_dev_m <= 0.0:
        return smoothed
    out = smoothed.copy()
    for idx in range(out.shape[0]):
        delta = out[idx] - original[idx]
        dist = float(np.linalg.norm(delta))
        if dist > max_dev_m:
            out[idx] = original[idx] + delta * (max_dev_m / dist)
    return out


def clamp_quaternions_to_deviation(
    original: np.ndarray,
    smoothed: np.ndarray,
    max_rot_deg: float | None,
) -> np.ndarray:
    if max_rot_deg is None or max_rot_deg <= 0.0:
        return smoothed
    out = smoothed.copy()
    for idx in range(out.shape[0]):
        angle_deg = quaternion_angle_deg(original[idx], out[idx])
        if angle_deg > max_rot_deg:
            alpha = max_rot_deg / angle_deg
            out[idx] = slerp_xyzw(original[idx], out[idx], alpha)
    return out


def sample_position_xyz(sample: dict) -> np.ndarray:
    pos = sample["position_m"]
    return np.array([float(pos["x"]), float(pos["y"]), float(pos["z"])], dtype=float)


def sample_quat_xyzw(sample: dict) -> np.ndarray:
    quat = sample["quaternion_wxyz"]
    return np.array(
        [float(quat["x"]), float(quat["y"]), float(quat["z"]), float(quat["w"])],
        dtype=float,
    )


def write_sample_pose(sample: dict, pos_xyz: np.ndarray, quat_xyzw: np.ndarray) -> None:
    sample["position_m"] = {
        "x": float(pos_xyz[0]),
        "y": float(pos_xyz[1]),
        "z": float(pos_xyz[2]),
    }
    sample["quaternion_wxyz"] = {
        "w": float(quat_xyzw[3]),
        "x": float(quat_xyzw[0]),
        "y": float(quat_xyzw[1]),
        "z": float(quat_xyzw[2]),
    }


def export_tum(
    path: Path,
    samples: List[dict],
    helpers: Dict[str, object],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            t = float(sample["timestamp_s"])
            pos = sample["position_m"]
            quat = sample["quaternion_wxyz"]
            rot = helpers["quat_xyzw_to_rot"](
                [float(quat["x"]), float(quat["y"]), float(quat["z"]), float(quat["w"])]
            )
            q = helpers["rot_to_quat_xyzw"](rot)
            handle.write(
                f"{t:.9f} {float(pos['x']):.9f} {float(pos['y']):.9f} {float(pos['z']):.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )


def main() -> int:
    args = parse_args()
    window = ensure_odd_window(int(args.window))
    src = args.input.expanduser().resolve()
    dst = args.output.expanduser().resolve()
    payload = json.loads(src.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{src}: expected non-empty 'samples' list")

    positions = np.asarray([sample_position_xyz(sample) for sample in samples], dtype=float)
    quats = np.asarray([sample_quat_xyzw(sample) for sample in samples], dtype=float)
    pos_smooth = moving_average(positions, window)
    quat_smooth = smooth_quaternions_xyzw(quats, window)
    pos_smooth = clamp_positions_to_deviation(
        positions,
        pos_smooth,
        None if args.max_pos_dev_mm is None else float(args.max_pos_dev_mm) * 1e-3,
    )
    quat_smooth = clamp_quaternions_to_deviation(
        quats,
        quat_smooth,
        None if args.max_rot_dev_deg is None else float(args.max_rot_dev_deg),
    )

    out_payload = json.loads(json.dumps(payload))
    out_payload["smoothing"] = {
        "method": "moving_average_position_and_quaternion_mean",
        "window": window,
        "source_file": str(src),
        "max_pos_dev_mm": None if args.max_pos_dev_mm is None else float(args.max_pos_dev_mm),
        "max_rot_dev_deg": None if args.max_rot_dev_deg is None else float(args.max_rot_dev_deg),
        "max_applied_pos_dev_mm": float(np.max(np.linalg.norm(pos_smooth - positions, axis=1)) * 1000.0),
        "max_applied_rot_dev_deg": float(
            max(quaternion_angle_deg(q0, q1) for q0, q1 in zip(quats, quat_smooth))
        ),
    }
    out_samples = out_payload["samples"]
    for sample, pos_xyz, quat_xyzw in zip(out_samples, pos_smooth, quat_smooth):
        write_sample_pose(sample, pos_xyz, quat_xyzw)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {dst}")

    if args.export_tum:
        tum_path = args.export_tum.expanduser().resolve()
        tum_path.parent.mkdir(parents=True, exist_ok=True)
        export_tum(tum_path, out_samples, load_helpers())
        print(f"[OK] wrote {tum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
