#!/usr/bin/env python3
"""Render a simple XY/XZ/YZ plot for an ORB-SLAM3 trajectory text file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="ORB-SLAM3 trajectory txt file")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--title", default="", help="Optional plot title")
    return parser.parse_args()


def load_xyz(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not rows:
        raise RuntimeError(f"no trajectory rows found in {path}")
    return np.asarray(rows, dtype=np.float64)


def make_axis(ax, xs: np.ndarray, ys: np.ndarray, xlabel: str, ylabel: str) -> None:
    ax.plot(xs, ys, linewidth=1.2, color="#0f766e")
    ax.scatter(xs[0], ys[0], s=28, color="#16a34a", label="start")
    ax.scatter(xs[-1], ys[-1], s=28, color="#dc2626", label="end")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.axis("equal")


def main() -> int:
    args = parse_args()
    traj_path = args.trajectory.expanduser().resolve()
    xyz = load_xyz(traj_path)
    output = args.output.expanduser().resolve() if args.output else traj_path.with_suffix(".png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    title = args.title or traj_path.name
    fig.suptitle(title, fontsize=13)
    make_axis(axes[0], xyz[:, 0], xyz[:, 1], "x [m]", "y [m]")
    axes[0].set_title("XY")
    make_axis(axes[1], xyz[:, 0], xyz[:, 2], "x [m]", "z [m]")
    axes[1].set_title("XZ")
    make_axis(axes[2], xyz[:, 1], xyz[:, 2], "y [m]", "z [m]")
    axes[2].set_title("YZ")
    axes[0].legend(loc="best")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"[OK] Wrote trajectory plot: {output}")
    print(f"[OK] Samples: {xyz.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
