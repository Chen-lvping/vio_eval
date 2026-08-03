#!/usr/bin/env python3
"""Run the validated tracking-Basalt configuration over CSV SXR episodes."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "script/run_sxr_csv_basalt.py"
DEFAULT_ROTATION = ROOT / (
    "data/evaluation/workbench/"
    "sxr_csv_20260730_205558_tracking_aprilgrid_preflight_v1/"
    "cam_imu_timing_extrinsic_validation.json"
)
RMSE_RE = re.compile(r"^\s*rmse\s+([0-9.eE+-]+)\s*$", re.MULTILINE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--episode-pattern", default="*")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--camera-from-imu-rotation-json", type=Path, default=DEFAULT_ROTATION)
    parser.add_argument("--basalt-bin", type=Path, default=ROOT / "third_party/basalt_release/release/bin/basalt_vio")
    parser.add_argument("--config", type=Path, default=ROOT / "third_party/basalt/data/euroc_config.json")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a Basalt JSON config value; repeat for multiple values.",
    )
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=75, choices=range(50, 101))
    parser.add_argument("--camera-quaternion-order", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument("--stereo-camera1-from-camera0-json", type=Path, default=None)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args(argv)


def is_csv_episode(path: Path) -> bool:
    required = (
        "accel.csv", "gyro.csv", "head_pose.csv", "imu_calibration.json",
        "tracking.mp4", "tracking_metainfo.csv", "camera_params_tracking.json",
    )
    return path.is_dir() and all((path / name).is_file() for name in required)


def metric(path: Path) -> float | None:
    if not path.is_file():
        return None
    match = RMSE_RE.search(path.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else None


def run_one(args: argparse.Namespace, episode: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(RUNNER),
        "--episode-dir", str(episode),
        "--output-dir", str(output),
        "--stream", "tracking",
        "--camera-quaternion-order", args.camera_quaternion_order,
        "--extrinsics-convention", "imu-to-camera",
        "--camera-from-imu-rotation-json", str(args.camera_from_imu_rotation_json),
        "--basalt-bin", str(args.basalt_bin),
        "--config", str(args.config),
        *sum((["--config-override", value] for value in args.config_override), []),
        "--image-scale", str(args.image_scale),
        "--jpeg-quality", str(args.jpeg_quality),
        "--num-threads", str(args.num_threads),
        "--timeout-sec", str(args.timeout_sec),
    ]
    if args.stereo_camera1_from_camera0_json is not None:
        command.extend(("--stereo-camera1-from-camera0-json", str(args.stereo_camera1_from_camera0_json)))
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (output / "batch_runner.log").write_text(result.stdout, encoding="utf-8")
    gate_path = output / "evaluation/scale_gate.json"
    gate: dict[str, Any] | None = None
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if result.returncode == 0:
        status = "ok"
    elif gate is not None and not gate.get("passed", False):
        status = "scale_rejected"
    else:
        status = "failed"
    return {
        "episode": episode.name,
        "episode_dir": str(episode.resolve()),
        "output_dir": str(output.resolve()),
        "status": status,
        "returncode": result.returncode,
        "image_pairs": json.loads((output / "run_summary.json").read_text(encoding="utf-8")).get("image_pairs")
        if (output / "run_summary.json").is_file() else None,
        "scale_gate": gate,
        "ape_rmse_m": metric(output / "evaluation/ape_translation.log"),
        "rpe_rmse_m": metric(output / "evaluation/rpe_translation.log"),
    }


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    (output_root / "batch_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (output_root / "batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("episode", "status", "gate_reason", "reference_path_m", "estimate_path_m", "path_ratio", "ape_rmse_m", "rpe_rmse_m"))
        for row in rows:
            gate = row.get("scale_gate") or {}
            writer.writerow((
                row["episode"], row["status"], gate.get("reason", ""), gate.get("reference_path_length_m"),
                gate.get("estimate_path_length_m"), gate.get("path_length_ratio"),
                row.get("ape_rmse_m"), row.get("rpe_rmse_m"),
            ))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.1 <= args.image_scale <= 1.0:
        raise ValueError("--image-scale must be in [0.1, 1.0]")
    root = args.episode_root.expanduser().resolve()
    episodes = sorted(path for path in root.glob(args.episode_pattern) if is_csv_episode(path))
    if not episodes:
        raise FileNotFoundError(f"no CSV SXR episodes found under {root} matching {args.episode_pattern!r}")
    output_root = (args.output_root or ROOT / "data/evaluation/workbench" / f"sxr_tracking_basalt_batch_{datetime.now():%Y%m%d_%H%M%S}").expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        print(f"[RUN] {episode.name}", flush=True)
        rows.append(run_one(args, episode, output_root / episode.name))
        row = rows[-1]
        gate = row.get("scale_gate") or {}
        print(
            f"[{row['status'].upper()}] {episode.name} "
            f"scale_ratio={gate.get('path_length_ratio', 'n/a')} "
            f"ape_rmse={row.get('ape_rmse_m', 'n/a')}",
            flush=True,
        )
    write_summary(output_root, rows)
    print(f"[OK] wrote {output_root / 'batch_summary.json'}")
    return 0 if all(row["status"] == "ok" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
