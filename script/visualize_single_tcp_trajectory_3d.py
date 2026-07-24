#!/usr/bin/env python3
"""Build a static 3D TCP trajectory viewer for one evo-style result directory."""

from __future__ import annotations

import argparse
import csv
import json
import re
import runpy
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_VIEWER = REPO_ROOT / "script/visualize/visualize_tcp_trajectory_comparison.py"
DEFAULT_EVAL_DIR = REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_cam0_0617_0004"


def infer_episode_label(eval_dir: Path) -> str:
    candidates = [str(eval_dir)]
    metrics_path = eval_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key in ("estimate", "ground_truth"):
                value = payload.get(key)
                if isinstance(value, str):
                    candidates.append(value)
        except Exception:
            pass

    for text in candidates:
        match = re.search(r"(episode_\d{8}_\d{4})", text)
        if match:
            return match.group(1)
    return eval_dir.name


def load_comparison_module() -> Dict[str, object]:
    return runpy.run_path(str(COMPARISON_VIEWER), run_name="__single_tcp_viewer__")


def read_summary(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["metric"]] = {k: float(v) for k, v in row.items() if k not in {"metric", "unit"}}
    return out


def stride_indices(count: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=int)
    indices = np.linspace(0, count - 1, max_points, dtype=int)
    return np.unique(np.r_[indices, count - 1])


def resolve_viewer_tums(eval_dir: Path) -> tuple[Path, Path, str]:
    matched_gt = eval_dir / "gt_tcp_matched.tum"
    matched_est = eval_dir / "vio_tcp_matched.tum"
    if matched_gt.is_file() and matched_est.is_file():
        return matched_gt, matched_est, "matched"
    return eval_dir / "gt_tcp.tum", eval_dir / "vio_tcp_from_imu_left_camera.tum", "raw"


def make_algorithm(eval_dir: Path, max_points: int, comparison: Dict[str, object]) -> tuple[dict, dict]:
    helpers = comparison["load_helpers"]()
    gt_tum, est_tum, source_mode = resolve_viewer_tums(eval_dir)
    gt_t, gt_pos, gt_rot = comparison["read_tum"](gt_tum, helpers)
    est_t, est_pos, est_rot = comparison["read_tum"](est_tum, helpers)
    if gt_t.shape != est_t.shape or float(np.max(np.abs(gt_t - est_t))) > 1e-6:
        raise ValueError(f"{eval_dir} viewer TUM files are not timestamp-aligned")

    se3_pos, se3_rot, se3_errors = comparison["align_positions"](
        helpers, gt_pos, gt_rot, est_pos, est_rot, gt_t, False
    )
    sim3_pos, sim3_rot, sim3_errors = comparison["align_positions"](
        helpers, gt_pos, gt_rot, est_pos, est_rot, gt_t, True
    )
    anchor_pos, anchor_rot, anchor_errors = comparison["anchor_start_positions"](
        gt_pos, gt_rot, est_pos, est_rot
    )
    raw_errors = np.linalg.norm(est_pos - gt_pos, axis=1)
    idx = stride_indices(gt_t.size, max_points)
    t0 = float(gt_t[0])
    summary = read_summary(eval_dir / "summary.csv")
    metrics = {
        "ape_translation_se3_rmse_mm": summary["ape_translation_se3"]["rmse"],
        "ape_translation_sim3_rmse_mm": summary["ape_translation_sim3"]["rmse"],
        "rpe_translation_5cm_rmse_mm": summary["rpe_translation_5cm"]["rmse"],
        "ape_rotation_se3_rmse_deg": summary["ape_rotation_se3"]["rmse"],
        "rpe_rotation_5cm_rmse_deg": summary["rpe_rotation_5cm"]["rmse"],
    }

    points: List[dict] = []
    for i in idx:
        points.append(
            {
                "t": float(gt_t[i] - t0),
                "gt": gt_pos[i].tolist(),
                "gt_r": gt_rot[i].tolist(),
                "raw": est_pos[i].tolist(),
                "raw_r": est_rot[i].tolist(),
                "se3": se3_pos[i].tolist(),
                "se3_r": se3_rot[i].tolist(),
                "sim3": sim3_pos[i].tolist(),
                "sim3_r": sim3_rot[i].tolist(),
                "anchor": anchor_pos[i].tolist(),
                "anchor_r": anchor_rot[i].tolist(),
                "raw_error_m": float(raw_errors[i]),
                "se3_error_m": float(se3_errors[i]),
                "sim3_error_m": float(sim3_errors[i]),
                "anchor_error_m": float(anchor_errors[i]),
            }
        )

    algorithm = {
        "name": "VIO TCP",
        "color": "#4ea1ff",
        "eval_dir": str(eval_dir),
        "sample_count": int(gt_t.size),
        "duration_s": float(gt_t[-1] - gt_t[0]),
        "metrics": metrics,
        "points": points,
    }
    sources = {
        "gt_tum": str(gt_tum),
        "estimate_tum": str(est_tum),
        "association_source": source_mode,
    }
    return algorithm, sources


def write_viewer_summary(output_dir: Path, algorithms: Iterable[dict]) -> Path:
    path = output_dir / "viewer_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "algorithm",
            "ape_translation_se3_rmse_mm",
            "ape_translation_sim3_rmse_mm",
            "rpe_translation_5cm_rmse_mm",
            "ape_rotation_se3_rmse_deg",
            "rpe_rotation_5cm_rmse_deg",
            "sample_count",
            "duration_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for algo in algorithms:
            row = {
                "algorithm": algo["name"],
                **algo["metrics"],
                "sample_count": algo["sample_count"],
                "duration_s": algo["duration_s"],
            }
            writer.writerow(row)
    return path


def single_viewer_html(comparison_html: str, episode_label: str) -> str:
    replacements = {
        "<title>TCP trajectory comparison</title>": f"<title>{episode_label} TCP 3D visualization</title>",
        "TCP trajectory comparison: robot GT vs VINS / DynaVINS": f"{episode_label} TCP 3D visualization",
        "<span><i class=\"dot\" style=\"background:var(--vins)\"></i>VINS-derived TCP</span>": "<span><i class=\"dot\" style=\"background:var(--vins)\"></i>VIO-derived TCP</span>",
        "      <span><i class=\"dot\" style=\"background:var(--dynavins)\"></i>DynaVINS-derived TCP</span>\n": "",
        "TCP comparison. Default 3D view follows the dominant motion direction; VIO chain: T_world_tcp = T_world_imu @ inv(T_left_camera_imu) @ inv(T_tcp_left_camera).": f"{episode_label}: robot TCP GT vs VIO-derived TCP. Default 3D view follows the dominant motion direction; drag to rotate; wheel to zoom.",
    }
    html = comparison_html
    for old, new in replacements.items():
        html = html.replace(old, new)
    return html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=3000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_dir = args.eval_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_label = infer_episode_label(eval_dir)

    comparison = load_comparison_module()
    algorithm, sources = make_algorithm(eval_dir, args.max_points, comparison)
    payload = {
        "subtitle": f"{episode_label}: robot TCP GT vs VIO-derived TCP. Default 3D view follows the dominant motion direction; drag to rotate; wheel to zoom.",
        "inputs": {
            "eval_dir": str(eval_dir),
            "gt_tum": sources["gt_tum"],
            "estimate_tum": sources["estimate_tum"],
            "association_source": sources["association_source"],
            "frame": "robot TCP",
            "display_modes": "Raw / SE(3) / Sim(3) / Anchor-start",
        },
        "algorithms": [algorithm],
    }
    data_path = output_dir / "viewer_data_3d.json"
    html_path = output_dir / "index.html"
    summary_path = write_viewer_summary(output_dir, payload["algorithms"])
    html = single_viewer_html(comparison["HTML"], episode_label)
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(html.replace("__VIEWER_DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    print(f"[OK] wrote {html_path}")
    print(f"[OK] wrote {data_path}")
    print(f"[OK] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
