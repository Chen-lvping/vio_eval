#!/usr/bin/env python3
"""Run a raw Stereo-IMU BA10 trajectory from a preserved Euroc export.

This is for episodes whose original collection directory is gone but whose
previous ORB export (mav0, times.txt, and historical settings) is retained.
It deliberately has no stereo fallback path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORB_ROOT = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_clean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--historical-calibration", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=int, default=720)
    return parser.parse_args()


def nonempty_rows(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def main() -> int:
    args = parse_args()
    source = args.source_run.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    calibration = args.historical_calibration.expanduser().resolve()
    ground_truth = args.ground_truth.expanduser().resolve()
    settings = source / "orbslam3_stereo_right_stereo-inertial.yaml"
    required = [source / "mav0", source / "times.txt", settings, calibration, ground_truth]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing preserved input:\n" + "\n".join(missing))

    work = output_root / "run"
    shim = output_root / "episode_shim"
    work.mkdir(parents=True, exist_ok=True)
    shim.mkdir(parents=True, exist_ok=True)
    for name, target in (("mav0", source / "mav0"), ("times.txt", source / "times.txt")):
        destination = work / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(target)
    historical_settings = work / settings.name
    shutil.copy2(settings, historical_settings)
    shim_calibration = shim / "calibration.json"
    if shim_calibration.exists() or shim_calibration.is_symlink():
        shim_calibration.unlink()
    shim_calibration.symlink_to(calibration)

    trajectory_name = f"{args.episode}_stereo_right_stereo_inertial_ba10"
    trajectory = work / f"f_{trajectory_name}.txt"
    log_path = output_root / "orbslam3_native.log"
    env = os.environ.copy()
    library_paths = [
        "/usr/local/lib",
        str(ORB_ROOT / "lib"),
        str(ORB_ROOT / "Thirdparty/DBoW2/lib"),
        str(ORB_ROOT / "Thirdparty/g2o/lib"),
    ]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(library_paths)
    env["ORB_SLAM3_ENABLE_VIEWER"] = "0"
    env["ORB_SLAM3_OFFLINE_WAIT_LOCAL_MAPPING"] = "0"
    env["ORB_SLAM3_FINAL_BA_ITERS"] = "10"
    command = [
        str(ORB_ROOT / "Examples/Stereo-Inertial/stereo_inertial_euroc"),
        str(ORB_ROOT / "Vocabulary/ORBvoc.txt"),
        str(historical_settings),
        str(work),
        str(work / "times.txt"),
        trajectory_name,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        try:
            completed = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_sec)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"ORB-SLAM3 BA10 timed out after {args.timeout_sec}s") from exc
    if completed.returncode != 0 or not trajectory.is_file() or nonempty_rows(trajectory) == 0:
        raise RuntimeError(f"ORB-SLAM3 Stereo-IMU BA10 failed (exit={completed.returncode}); see {log_path}")

    coverage = nonempty_rows(trajectory) / max(nonempty_rows(work / "times.txt"), 1)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "episode": args.episode,
        "mode": "stereo-inertial",
        "requested_mode": "stereo-inertial",
        "fallback_to_stereo": False,
        "fallback_reason": "",
        "final_ba_iters": 10,
        "trajectory": str(trajectory),
        "trajectory_rows": nonempty_rows(trajectory),
        "inertial_coverage": coverage,
        "historical_source_run": str(source),
        "historical_settings": str(settings),
        "historical_calibration": str(calibration),
        "ground_truth": str(ground_truth),
        "orb_command": command,
    }
    (output_root / "orbslam3_tcp_eval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_map = {args.episode: str((output_root / "orbslam3_tcp_eval_manifest.json").resolve())}
    (output_root / "manifest_map.json").write_text(
        json.dumps(manifest_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
