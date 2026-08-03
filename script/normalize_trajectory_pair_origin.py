#!/usr/bin/env python3
"""Rebase two TUM trajectories so both first poses are the identity pose."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from run_sxr_basalt import matrix_to_xyzw
from run_sxr_orb_stereo_baseline import ROOT


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--est", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ref-name", default="Reference")
    parser.add_argument("--est-name", default="Estimate")
    parser.add_argument("--title", default="Origin-normalized trajectory comparison")
    parser.add_argument("--t-max-diff", type=float, default=0.01)
    return parser.parse_args(argv)


def inverse(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return result


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("zero-norm quaternion")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        (
            (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
            (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
            (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
        ),
        dtype=np.float64,
    )


def load_tum(path: Path) -> list[tuple[float, np.ndarray]]:
    poses: list[tuple[float, np.ndarray]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            values = line.split()
            if len(values) != 8 or values[0].startswith("#"):
                continue
            timestamp, x, y, z, qx, qy, qz, qw = (float(value) for value in values)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = quat_to_matrix(qx, qy, qz, qw)
            transform[:3, 3] = (x, y, z)
            poses.append((timestamp, transform))
    if len(poses) < 3:
        raise ValueError(f"expected at least three valid poses in {path}")
    return poses


def rebased(poses: Iterable[tuple[float, np.ndarray]]) -> list[tuple[float, np.ndarray]]:
    sequence = list(poses)
    start_time, start_pose = sequence[0]
    start_inverse = inverse(start_pose)
    return [(timestamp - start_time, start_inverse @ transform) for timestamp, transform in sequence]


def write_tum(path: Path, poses: Iterable[tuple[float, np.ndarray]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for timestamp, transform in poses:
            qx, qy, qz, qw = matrix_to_xyzw(transform)
            handle.write(
                f"{timestamp:.9f} {transform[0, 3]:.9f} {transform[1, 3]:.9f} {transform[2, 3]:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )


def run_logged(command: list[str], path: Path) -> None:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    path.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed; see {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reference, estimate = args.ref.expanduser().resolve(), args.est.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ref_rebased, est_rebased = rebased(load_tum(reference)), rebased(load_tum(estimate))
    ref_output, est_output = output / "reference_origin.tum", output / "estimate_origin.tum"
    write_tum(ref_output, ref_rebased)
    write_tum(est_output, est_rebased)
    evaluation = output / "evaluation"
    evaluation.mkdir(exist_ok=True)
    run_logged(
        ["evo_ape", "tum", str(ref_output), str(est_output), "--t_max_diff", str(args.t_max_diff), "--pose_relation", "trans_part"],
        evaluation / "ape_origin_normalized.log",
    )
    run_logged(
        ["evo_rpe", "tum", str(ref_output), str(est_output), "--t_max_diff", str(args.t_max_diff), "--pose_relation", "trans_part", "--delta", "1", "--delta_unit", "f"],
        evaluation / "rpe_origin_normalized.log",
    )
    subprocess.run(
        [
            "python3", str(ROOT / "script/visualize/visualize_trajectory_pair.py"),
            "--ref", str(ref_output), "--est", str(est_output),
            "--output-dir", str(output / "trajectory_viewer"),
            "--ref-name", args.ref_name, "--est-name", args.est_name,
            "--title", args.title,
        ],
        check=True,
    )
    print(f"[OK] first poses rebased to identity -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
