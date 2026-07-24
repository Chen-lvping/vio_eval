#!/usr/bin/env python3
"""Render the verified RM75 final-cohort APE comparison figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORACLE = "#007f7b"
SHADOW = "#d35f3d"
THRESHOLD = "#59636d"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    with args.results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(row["episode_key"].rsplit("_", 1)[1]))

    labels = [row["episode_key"].rsplit("_", 1)[1] for row in rows]
    oracle = np.array([float(row["ape_translation_se3_mm"]) for row in rows])
    shadow = np.array([float(row["shadow_gate_ape_translation_se3_mm"]) for row in rows])
    x = np.arange(len(rows))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })
    fig = plt.figure(figsize=(14.2, 7.4), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.0])
    ax = fig.add_subplot(grid[0, 0])
    summary = fig.add_subplot(grid[0, 1])

    for i, (o, s) in enumerate(zip(oracle, shadow)):
        ax.plot([i, i], [o, s], color="#b8c1c7", linewidth=1.3, zorder=1)
    ax.scatter(x - 0.12, oracle, s=48, color=ORACLE, edgecolor="white", linewidth=0.7,
               label="Oracle (GT-selected)", zorder=3)
    ax.scatter(x + 0.12, shadow, s=48, color=SHADOW, edgecolor="white", linewidth=0.7,
               label="Shadow-gate (GT-free)", zorder=3)
    ax.axhline(10.0, color=THRESHOLD, linestyle=(0, (4, 3)), linewidth=1.2, label="10 mm target")
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylim(0, max(float(np.max(shadow)), float(np.max(oracle))) + 4)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_xlabel("Episode")
    ax.set_ylabel("APE translation, SE(3) aligned (mm)")
    ax.set_title("Verified RM75 APE by Episode (n=21)", loc="left")
    ax.grid(axis="y", color="#d9dee2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=3)

    methods = ["Oracle", "Shadow-gate"]
    values = [oracle, shadow]
    colors = [ORACLE, SHADOW]
    positions = [1, 0]
    for position, vals, color, method in zip(positions, values, colors, methods):
        summary.hlines(position, np.min(vals), np.max(vals), color="#c9d0d5", linewidth=1.1, zorder=1)
        summary.scatter(vals, np.full_like(vals, position), s=32, color=color, edgecolor="white",
                        linewidth=0.55, alpha=0.92, zorder=2)
        summary.scatter([np.mean(vals)], [position], marker="D", s=62, color="#1f2930",
                        edgecolor="white", linewidth=0.8, zorder=3)
        summary.text(0.02, position + 0.17, f"mean {np.mean(vals):.2f} mm", transform=summary.get_yaxis_transform(),
                     color="#1f2930", fontsize=10)
    summary.axvline(10.0, color=THRESHOLD, linestyle=(0, (4, 3)), linewidth=1.2)
    summary.set_xlim(0, max(float(np.max(shadow)), float(np.max(oracle))) + 4)
    summary.set_yticks(positions, methods)
    summary.set_xlabel("APE (mm)")
    summary.set_title("Distribution", loc="left")
    summary.grid(axis="x", color="#d9dee2", linewidth=0.8)
    summary.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle("ORB-SLAM3 RM75 Final Cohort: Oracle vs Deployable Shadow-Gate", fontsize=16, fontweight="bold", x=0.012, ha="left")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "rm75_final_21_ape_comparison"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
