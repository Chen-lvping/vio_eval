#!/usr/bin/env python3
"""Batch entrypoint for the 0617 raw-pose TCP evaluation workflow."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "script/eval/run_0614_style_tcp_eval.py"
ESTIMATE_ROOTS = [
    REPO_ROOT / "data/gripper_data_1",
    REPO_ROOT / "data/gripper_data",
]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/evaluation/workbench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260617", help="Episode date prefix, e.g. 20260617")
    parser.add_argument(
        "--episodes",
        nargs="*",
        help="Optional explicit episode suffixes like 0001 0005. Defaults to all discoverable episodes for --date.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-suffix", default="_rawpose")
    parser.add_argument("--handeye-yaml", type=Path)
    parser.add_argument(
        "--estimate-frame",
        choices=["auto", "vins_base_link", "imu", "camera"],
        default="auto",
    )
    parser.add_argument(
        "--time-association",
        choices=["interpolate", "evo"],
        default="interpolate",
    )
    parser.add_argument("--max-time-gap-ms", type=float, default=80.0)
    parser.add_argument("--time-offset-sec", type=float, default=0.0)
    parser.add_argument("--t-max-diff-sec", type=float, default=0.01)
    parser.add_argument("--time-offset-auto", action="store_true")
    parser.add_argument("--time-offset-auto-span-sec", type=float, default=2.0)
    parser.add_argument("--time-offset-auto-step-sec", type=float, default=0.01)
    parser.add_argument("--time-offset-auto-score", choices=["translation", "rotation"], default="translation")
    parser.add_argument("--time-offset-auto-min-samples", type=int, default=50)
    parser.add_argument("--rpe-distance-m", type=float, default=0.05)
    parser.add_argument("--viewer-max-points", type=int, default=3000)
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes whose output metrics.json already exists.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch on first failure instead of continuing.",
    )
    return parser.parse_args()


def discover_episodes(date_str: str) -> List[Tuple[str, Path]]:
    found: Dict[str, Path] = {}
    pattern = f"episode_{date_str}_*"
    # Later roots overwrite earlier ones only if the seq is still missing; keep first hit.
    for root in ESTIMATE_ROOTS:
        if not root.is_dir():
            continue
        for episode_dir in sorted(root.glob(pattern)):
            pose_data = episode_dir / "right" / "pose_data.csv"
            if not pose_data.is_file():
                continue
            seq = episode_dir.name.rsplit("_", 1)[-1]
            found.setdefault(seq, pose_data)
    return sorted(found.items())


def select_episodes(date_str: str, requested: Optional[Iterable[str]]) -> List[Tuple[str, Path]]:
    available = dict(discover_episodes(date_str))
    if not requested:
        return sorted(available.items())

    selected: List[Tuple[str, Path]] = []
    missing: List[str] = []
    for seq in requested:
        norm = str(seq).zfill(4)
        pose_data = available.get(norm)
        if pose_data is None:
            missing.append(norm)
            continue
        selected.append((norm, pose_data))
    if missing:
        raise FileNotFoundError(f"episodes not found for {date_str}: {', '.join(missing)}")
    return selected


def output_dir_for(output_root: Path, date_str: str, seq: str, suffix: str) -> Path:
    return output_root / f"evo_vio_tcp_{date_str[4:8]}_{seq}{suffix}"


def summary_paths(output_root: Path, date_str: str, suffix: str) -> Tuple[Path, Path]:
    stem = f"evo_vio_tcp_{date_str}_{suffix.strip('_')}_batch_summary"
    return output_root / f"{stem}.json", output_root / f"{stem}.csv"


def run_episode(args: argparse.Namespace, date_str: str, seq: str, estimate: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--episode",
        f"{date_str}_{seq}",
        "--estimate",
        str(estimate),
        "--output-dir",
        str(output_dir),
        "--estimate-frame",
        args.estimate_frame,
        "--time-association",
        args.time_association,
        "--max-time-gap-ms",
        f"{args.max_time_gap_ms}",
        "--time-offset-sec",
        f"{args.time_offset_sec}",
        "--t-max-diff-sec",
        f"{args.t_max_diff_sec}",
        "--time-offset-auto-span-sec",
        f"{args.time_offset_auto_span_sec}",
        "--time-offset-auto-step-sec",
        f"{args.time_offset_auto_step_sec}",
        "--time-offset-auto-score",
        args.time_offset_auto_score,
        "--time-offset-auto-min-samples",
        str(args.time_offset_auto_min_samples),
        "--rpe-distance-m",
        f"{args.rpe_distance_m}",
        "--viewer-max-points",
        str(args.viewer_max_points),
    ]
    if args.time_offset_auto:
        cmd.append("--time-offset-auto")
    if args.handeye_yaml:
        cmd.extend(["--handeye-yaml", str(args.handeye_yaml.expanduser().resolve())])
    if args.skip_viewer:
        cmd.append("--skip-viewer")
    print("[BATCH]", f"episode_{date_str}_{seq}", flush=True)
    return subprocess.run(cmd, text=True, capture_output=True)


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict, name: str) -> Optional[float]:
    return metrics.get("evo", {}).get(name, {}).get("rmse")


def write_summary_csv(path: Path, rows: List[dict]) -> None:
    fields = [
        "episode",
        "status",
        "estimate",
        "ground_truth",
        "output_dir",
        "estimate_frame",
        "time_association",
        "time_offset_sec",
        "matched_samples",
        "matched_duration_s",
        "ape_translation_se3_rmse_mm",
        "ape_rotation_se3_rmse_deg",
        "rpe_translation_5cm_rmse_mm",
        "rpe_rotation_5cm_rmse_deg",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    episodes = select_episodes(args.date, args.episodes)
    if not episodes:
        raise RuntimeError(f"no discoverable episodes found for {args.date}")

    rows: List[dict] = []
    total = len(episodes)
    for idx, (seq, estimate) in enumerate(episodes, start=1):
        episode_name = f"episode_{args.date}_{seq}"
        output_dir = output_dir_for(output_root, args.date, seq, args.output_suffix)
        metrics_path = output_dir / "metrics.json"
        print(f"[{idx}/{total}] {episode_name}", flush=True)

        if args.skip_existing and metrics_path.is_file():
            metrics = load_metrics(metrics_path)
            rows.append(
                {
                    "episode": episode_name,
                    "status": "skipped_existing",
                    "estimate": metrics.get("estimate"),
                    "ground_truth": metrics.get("ground_truth"),
                    "output_dir": str(output_dir),
                    "estimate_frame": metrics.get("estimate_frame"),
                    "time_association": metrics.get("time_association"),
                    "time_offset_sec": metrics.get("time_offset_sec"),
                    "matched_samples": metrics.get("matched_samples"),
                    "matched_duration_s": metrics.get("matched_duration_s"),
                    "ape_translation_se3_rmse_mm": metric_value(metrics, "ape_translation_se3"),
                    "ape_rotation_se3_rmse_deg": metric_value(metrics, "ape_rotation_se3"),
                    "rpe_translation_5cm_rmse_mm": metric_value(metrics, "rpe_translation_5cm"),
                    "rpe_rotation_5cm_rmse_deg": metric_value(metrics, "rpe_rotation_5cm"),
                    "error": "",
                }
            )
            continue

        proc = run_episode(args, args.date, seq, estimate, output_dir)
        if proc.returncode == 0 and metrics_path.is_file():
            metrics = load_metrics(metrics_path)
            rows.append(
                {
                    "episode": episode_name,
                    "status": "ok",
                    "estimate": metrics.get("estimate"),
                    "ground_truth": metrics.get("ground_truth"),
                    "output_dir": str(output_dir),
                    "estimate_frame": metrics.get("estimate_frame"),
                    "time_association": metrics.get("time_association"),
                    "time_offset_sec": metrics.get("time_offset_sec"),
                    "matched_samples": metrics.get("matched_samples"),
                    "matched_duration_s": metrics.get("matched_duration_s"),
                    "ape_translation_se3_rmse_mm": metric_value(metrics, "ape_translation_se3"),
                    "ape_rotation_se3_rmse_deg": metric_value(metrics, "ape_rotation_se3"),
                    "rpe_translation_5cm_rmse_mm": metric_value(metrics, "rpe_translation_5cm"),
                    "rpe_rotation_5cm_rmse_deg": metric_value(metrics, "rpe_rotation_5cm"),
                    "error": "",
                }
            )
        else:
            error_text = (proc.stderr or proc.stdout or "unknown failure").strip()
            rows.append(
                {
                    "episode": episode_name,
                    "status": "failed",
                    "estimate": str(estimate),
                    "ground_truth": "",
                    "output_dir": str(output_dir),
                    "estimate_frame": args.estimate_frame,
                    "time_association": args.time_association,
                    "time_offset_sec": args.time_offset_sec,
                    "matched_samples": "",
                    "matched_duration_s": "",
                    "ape_translation_se3_rmse_mm": "",
                    "ape_rotation_se3_rmse_deg": "",
                    "rpe_translation_5cm_rmse_mm": "",
                    "rpe_rotation_5cm_rmse_deg": "",
                    "error": error_text.splitlines()[-1][:500],
                }
            )
            print(proc.stdout, end="")
            print(proc.stderr, end="", file=sys.stderr)
            if args.stop_on_error:
                break

    json_path, csv_path = summary_paths(output_root, args.date, args.output_suffix)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(csv_path, rows)

    ok = sum(1 for row in rows if row["status"] == "ok")
    failed = sum(1 for row in rows if row["status"] == "failed")
    skipped = sum(1 for row in rows if row["status"] == "skipped_existing")
    print(f"[OK] batch summary: {json_path}")
    print(f"[OK] batch summary: {csv_path}")
    print(f"[OK] episodes processed: {len(rows)} | ok={ok} failed={failed} skipped={skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
