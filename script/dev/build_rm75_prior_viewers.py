#!/usr/bin/env python3
"""Build the established TCP slider viewer for current RM75 batch results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWER = REPO_ROOT / "script/visualize_single_tcp_trajectory_3d.py"


def metric_row(name: str, stats: dict | None, unit: str, scale: float = 1.0) -> dict[str, object]:
    stats = stats or {}
    def value(key: str) -> float:
        return float(stats.get(key, "nan")) * scale

    return {
        "metric": name,
        "rmse": value("rmse"),
        "mean": value("mean"),
        "median": value("median"),
        "min": value("min"),
        "max": value("max"),
        "std": value("std"),
        "unit": unit,
    }


def read_tum(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) >= 8 and not values[0].startswith("#"):
            rows.append([float(value) for value in values[:8]])
    return rows


def write_matched_tums(eval_dir: Path, adapter: Path) -> None:
    gt = read_tum(eval_dir / "ref_tcp.tum")
    est = read_tum(eval_dir / "est_tcp.tum")
    gt_times = np.asarray([row[0] for row in gt], dtype=float)
    est_times = np.asarray([row[0] for row in est], dtype=float)
    pairs = []
    i = j = 0
    while i < len(gt) and j < len(est):
        delta = gt_times[i] - est_times[j]
        if abs(delta) <= 0.010:
            pairs.append((gt[i], est[j]))
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    if not pairs:
        raise RuntimeError(f"no timestamp pairs within 10 ms for {eval_dir}")
    gt_path = adapter / "gt_tcp_matched.tum"
    est_path = adapter / "vio_tcp_matched.tum"
    with gt_path.open("w", encoding="utf-8") as gt_handle, est_path.open("w", encoding="utf-8") as est_handle:
        for gt_row, est_row in pairs:
            timestamp = est_row[0]
            gt_handle.write(" ".join(f"{value:.10f}" for value in [timestamp, *gt_row[1:]]) + "\n")
            est_handle.write(" ".join(f"{value:.10f}" for value in est_row) + "\n")


def write_adapter(eval_dir: Path, summary: dict) -> Path:
    adapter = eval_dir / "prior_viewer_input"
    adapter.mkdir(exist_ok=True)
    for name in ("gt_tcp_matched.tum", "vio_tcp_matched.tum"):
        target = adapter / name
        if target.exists() or target.is_symlink():
            target.unlink()
    write_matched_tums(eval_dir, adapter)

    se3 = summary["results"]["se3"]["all"]
    sim3 = summary["results"]["sim3"]["all"]
    rows = [
        metric_row("ape_translation_se3", se3.get("trans_part"), "mm", 1000.0),
        metric_row("ape_rotation_se3", se3.get("angle_deg"), "deg"),
        metric_row("ape_translation_sim3", sim3.get("trans_part"), "mm", 1000.0),
        metric_row("rpe_translation_5cm", None, "mm"),
        metric_row("rpe_rotation_5cm", None, "deg"),
        metric_row("rpe_translation_5cm_source", None, "mm"),
        metric_row("rpe_rotation_5cm_source", None, "deg"),
    ]
    with (adapter / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "rmse", "mean", "median", "min", "max", "std", "unit"])
        writer.writeheader()
        writer.writerows(rows)
    return adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=3000)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    eval_dirs = sorted(
        path.parent
        for path in root.glob("orbslam3_rm75_best_batch_*/**/eval/summary.json")
    )
    if not eval_dirs:
        raise SystemExit(f"no eval/summary.json found under {root}")
    built = 0
    for eval_dir in eval_dirs:
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
        adapter = write_adapter(eval_dir, summary)
        output_dir = eval_dir / "viewer_prior"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        command = [
            sys.executable,
            str(VIEWER),
            "--eval-dir",
            str(adapter),
            "--output-dir",
            str(output_dir),
            "--max-points",
            str(args.max_points),
        ]
        subprocess.run(command, check=True)
        shutil.copy2(adapter / "summary.csv", output_dir / "summary.csv")
        built += 1
    print(f"[DONE] prior viewers={built}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
