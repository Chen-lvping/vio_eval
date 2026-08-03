#!/usr/bin/env python3
"""Batch the validated offline ORB-SLAM3 stereo configuration on EGO tracking episodes.

This preserves the historical RM75 offline engine (GBA=100 and full-frame BA=10)
but uses the independently measured EGO AprilGrid stereo calibration and the
feature density required by the EGO fisheye tracking stream. Device head poses
are only consumed by the runner after SLAM for scale-gated evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "script/run_sxr_csv_orb_stereo.py"
DEFAULT_ORB_ROOT = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_colleague_sync")
DEFAULT_STEREO_CALIBRATION = ROOT / (
    "data/evaluation/workbench/sxr_tracking_stereo_aprilgrid_calibration_20260731_104007.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, action="append", default=[])
    parser.add_argument("--episode-root", type=Path, help="Discover episode directories under this root.")
    parser.add_argument("--episode-pattern", default="2026*", help="Glob used with --episode-root.")
    parser.add_argument(
        "--episode-list-json",
        type=Path,
        help="Reuse the `episodes` list from an earlier batch_provenance.json; requires --episode-root to resolve directories.",
    )
    parser.add_argument("--exclude-episode", action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--orb-root", type=Path, default=DEFAULT_ORB_ROOT)
    parser.add_argument("--stereo-camera1-from-camera0-json", type=Path, default=DEFAULT_STEREO_CALIBRATION)
    parser.add_argument("--robust-offline", action="store_true", help="Enable the EGO keyframe/map-point robustness preset.")
    parser.add_argument("--clahe", action="store_true", help="Enable experimental local-contrast enhancement during image export.")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--retry-status",
        action="append",
        default=[],
        help="With --episode-list-json, run only prior rows having this status; may be specified more than once.",
    )
    parser.add_argument("--timeout-sec", type=int, default=1800)
    args = parser.parse_args(argv)
    if not args.episode_dir and args.episode_root is None and args.episode_list_json is None:
        parser.error("--episode-dir, --episode-root, or --episode-list-json is required")
    if args.episode_list_json is not None and args.episode_root is None:
        parser.error("--episode-list-json requires --episode-root")
    if args.retry_status and args.episode_list_json is None:
        parser.error("--retry-status requires --episode-list-json")
    return args


def rmse_m(path: Path) -> float | None:
    if not path.is_file():
        return None
    match = re.search(r"^\s*rmse\s+([0-9.eE+-]+)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
    return None if match is None else float(match.group(1))


def collect_result(episode: Path, output: Path, status: str, returncode: int | None) -> dict[str, object]:
    gate_path = output / "evaluation/scale_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else None
    return {
        "episode": episode.name,
        "episode_dir": str(episode),
        "output_dir": str(output),
        "status": status,
        "returncode": returncode,
        "scale_gate": gate,
        "ape_rmse_m": rmse_m(output / "evaluation/ape_translation.log"),
        "rpe_rmse_m": rmse_m(output / "evaluation/rpe_translation.log"),
    }


def load_prior_rows(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows: dict[str, dict[str, object]] = {}
    for row in payload:
        if isinstance(row, dict) and isinstance(row.get("episode"), str):
            rows[row["episode"]] = row
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prior_rows = load_prior_rows(output_root / "batch_summary.json")
    episodes = list(args.episode_dir)
    if args.episode_root is not None:
        root = args.episode_root.expanduser().resolve()
        episodes.extend(sorted(path for path in root.glob(args.episode_pattern) if path.is_dir()))
    prior_episode_names: list[str] = []
    if args.episode_list_json is not None:
        list_payload = json.loads(args.episode_list_json.expanduser().resolve().read_text(encoding="utf-8"))
        names = list_payload.get("episodes") if isinstance(list_payload, dict) else None
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"{args.episode_list_json} lacks a string `episodes` list")
        root = args.episode_root.expanduser().resolve()
        prior_episode_names = names
        episodes.extend(root / name for name in names)
    excluded = set(args.exclude_episode)
    unique_episodes: list[Path] = []
    seen: set[Path] = set()
    for episode in episodes:
        resolved = episode.expanduser().resolve()
        if resolved.name not in excluded and resolved not in seen:
            unique_episodes.append(resolved)
            seen.add(resolved)
    rows = dict(prior_rows)
    for episode in unique_episodes:
        output = output_root / episode.name
        previous = prior_rows.get(episode.name)
        if args.retry_status and (previous is None or previous.get("status") not in set(args.retry_status)):
            continue
        if not episode.is_dir():
            print("[UNAVAILABLE]", episode.name, flush=True)
            continue
        if args.reuse_existing and (output / "run_summary.json").is_file():
            rows[episode.name] = collect_result(episode, output, "reused", 0)
            continue
        command = [
            sys.executable, str(RUNNER), "--episode-dir", str(episode), "--output-dir", str(output),
            "--stream", "tracking", "--orb-root", str(args.orb_root.expanduser().resolve()),
            "--image-scale", "1.0", "--nfeatures", "3500", "--ini-th-fast", "10", "--min-th-fast", "3",
            "--jpeg-qscale", "2", "--stereo-camera1-from-camera0-json",
            str(args.stereo_camera1_from_camera0_json.expanduser().resolve()), "--timeout-sec", str(args.timeout_sec),
        ]
        if args.robust_offline:
            command.append("--robust-offline")
        if args.clahe:
            command.append("--clahe")
        print("[RUN]", episode.name, flush=True)
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (output / "batch_runner.log").parent.mkdir(parents=True, exist_ok=True)
        (output / "batch_runner.log").write_text(completed.stdout, encoding="utf-8")
        status = "ok" if completed.returncode == 0 else "failed"
        rows[episode.name] = collect_result(episode, output, status, completed.returncode)
        print(f"[{status.upper()}] {episode.name}", flush=True)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "ORB-SLAM3 offline pure stereo, EGO-adapted frontend",
        "historical_offline_settings": {"gba_iterations": 100, "full_frame_visual_ba_iterations": 10},
        "ego_frontend_settings": {"nfeatures": 3500, "ini_th_fast": 10, "min_th_fast": 3, "image_scale": 1.0},
        "robust_offline": bool(args.robust_offline),
        "clahe": bool(args.clahe),
        "orb_root": str(args.orb_root.expanduser().resolve()),
        "stereo_camera1_from_camera0_json": str(args.stereo_camera1_from_camera0_json.expanduser().resolve()),
        "head_pose_use": "post-SLAM scale-gated evaluation only",
        "episodes": [path.name for path in unique_episodes],
        "resumed_episode_list": prior_episode_names,
        "retry_status": args.retry_status,
    }
    (output_root / "batch_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    ordered_rows = [rows[name] for name in sorted(rows)]
    (output_root / "batch_summary.json").write_text(json.dumps(ordered_rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
