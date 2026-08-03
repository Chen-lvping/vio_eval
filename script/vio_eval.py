#!/usr/bin/env python3
"""Stable delivery CLI for RM75/VIO TCP trajectory evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "script" / "evaluate_vio_tcp_camera_evo.py"
EXAMPLE = ROOT / "examples" / "minimal_reproduction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use `demo` to verify a fresh checkout, then use `evaluate` with a real exported trajectory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the versioned, dependency-light minimal reproduction")
    demo.add_argument("--output-dir", type=Path, default=ROOT / "local" / "minimal_reproduction")

    evaluate = subparsers.add_parser("evaluate", help="evaluate one camera, IMU, or VINS base-link pose CSV")
    evaluate.add_argument("--estimate", type=Path, required=True, help="CSV pose trajectory")
    evaluate.add_argument("--ground-truth", type=Path, required=True, help="robot TCP JSON trajectory")
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--handeye-yaml", type=Path, default=ROOT / "data/calibration/handeye_0615/handeye_result.yaml")
    evaluate.add_argument("--estimate-frame", choices=("camera", "imu", "vins_base_link"), default="camera")
    evaluate.add_argument("--calibration-json", type=Path)
    evaluate.add_argument("--camera-rig", type=str)
    evaluate.add_argument("--time-association", choices=("interpolate", "evo"), default="interpolate")
    evaluate.add_argument("--max-time-gap-ms", type=float, default=80.0)
    evaluate.add_argument("--time-offset-sec", type=float, default=0.0)
    evaluate.add_argument("--t-max-diff-sec", type=float, default=0.01)
    evaluate.add_argument("--metrics-backend", choices=("auto", "evo", "internal"), default="auto")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def run_evaluation(args: argparse.Namespace, *, demo: bool) -> int:
    if demo:
        estimate = require_file(EXAMPLE / "estimate_camera.csv", "example estimate")
        ground_truth = require_file(EXAMPLE / "ground_truth.json", "example ground truth")
        handeye = require_file(EXAMPLE / "handeye_identity.yaml", "example hand-eye calibration")
        estimate_frame = "camera"
        backend = "internal"
    else:
        estimate = require_file(args.estimate, "estimate")
        ground_truth = require_file(args.ground_truth, "ground truth")
        handeye = require_file(args.handeye_yaml, "hand-eye calibration")
        estimate_frame = args.estimate_frame
        backend = args.metrics_backend

    command = [
        sys.executable,
        str(EVALUATOR),
        "--estimate", str(estimate),
        "--ground-truth", str(ground_truth),
        "--handeye-yaml", str(handeye),
        "--estimate-frame", estimate_frame,
        "--output-dir", str(args.output_dir.expanduser().resolve()),
        "--metrics-backend", backend,
    ]
    if not demo:
        command.extend([
            "--time-association", args.time_association,
            "--max-time-gap-ms", str(args.max_time_gap_ms),
            "--time-offset-sec", str(args.time_offset_sec),
            "--t-max-diff-sec", str(args.t_max_diff_sec),
        ])
        if args.calibration_json:
            command.extend(["--calibration-json", str(require_file(args.calibration_json, "calibration JSON"))])
        if args.camera_rig:
            command.extend(["--camera-rig", args.camera_rig])
    subprocess.run(command, check=True)
    print(f"[DONE] report: {args.output_dir.expanduser().resolve() / 'REPORT.md'}")
    return 0


def main() -> int:
    args = parse_args()
    return run_evaluation(args, demo=args.command == "demo")


if __name__ == "__main__":
    raise SystemExit(main())
