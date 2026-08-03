#!/usr/bin/env python3
"""Run one low-priority, native-inertial RM75 historical reproduction probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "data/evaluation/config/rm75_historical_reproduction_20260730.json"
SINGLE_RUNNER = ROOT / "script/run_orbslam3_tcp_eval.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", choices=("main/0004", "main/0018"), default="main/0004")
    parser.add_argument("--run", action="store_true", help="Execute the probe; otherwise print the exact command.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_runtime_lock(shared: dict[str, object]) -> dict[str, str]:
    runtime = shared["runtime_lock"]
    if not isinstance(runtime, dict):
        raise ValueError("shared.runtime_lock must be an object")
    resolved: dict[str, str] = {}
    for name in ("executable", "library"):
        item = runtime.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"runtime_lock.{name} must be an object")
        path = resolve_path(str(item["path"]))
        expected = str(item["sha256"]).lower()
        if not path.is_file():
            raise FileNotFoundError(f"locked runtime file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"locked runtime {name} checksum changed: {path}\n"
                f"expected={expected}\nactual={actual}"
            )
        resolved[name] = str(path)
    library_dir = resolve_path(str(runtime["library_dir"]))
    if library_dir != Path(resolved["library"]).parent:
        raise ValueError("runtime_lock.library_dir must be the locked library parent")
    resolved["library_dir"] = str(library_dir)
    return resolved


def verify_final_ba_contract(shared: dict[str, object], runtime: dict[str, str]) -> None:
    iterations = int(shared["final_ba_iters"])
    if iterations <= 0:
        return
    # The iteration count is streamed at runtime, so only this fixed prefix
    # exists in the binary. The single-episode runner verifies the exact count
    # from the native log after execution.
    marker = b"[FINAL_INERTIAL_BA] completed iterations="
    library = Path(runtime["library"])
    if marker not in library.read_bytes():
        raise RuntimeError(
            "locked runtime cannot execute the requested final inertial BA: "
            f"{library} does not contain {marker.decode('ascii')!r}"
        )


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    shared = config["shared"]
    episode = config["episodes"][args.episode]
    episode_dir = resolve_path(episode["episode_dir"])
    ground_truth = resolve_path(episode["ground_truth"])
    vins_config = resolve_path(shared["shared_vins_config"])
    runtime = verify_runtime_lock(shared)
    verify_final_ba_contract(shared, runtime)
    required = (episode_dir, ground_truth, vins_config, Path(shared["orb_root"]), Path(runtime["library_dir"]))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing reproduction input:\n" + "\n".join(missing))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.episode.replace("/", "_")
    root = ROOT / "data/evaluation/workbench/rm75_historical_reproduction_20260730" / f"{label}_{stamp}"
    run_dir = root / "run"
    eval_dir = root / "evaluation"
    cmd = [
        "ionice", "-c3", "nice", "-n", "19", "taskset", "-c", "0",
        sys.executable, str(SINGLE_RUNNER),
        "--episode-dir", str(episode_dir),
        "--ground-truth", str(ground_truth),
        "--output-dir", str(run_dir),
        "--eval-dir", str(eval_dir),
        "--camera-rig", shared["camera_rig"],
        "--mode", shared["mode"],
        "--orb-root", shared["orb_root"],
        "--extra-ld-path", runtime["library_dir"],
        "--feature-preset", shared["feature_preset"],
        "--vins-config", str(vins_config),
        "--vins-noise-mode", shared["vins_noise_mode"],
        "--imu-fast-init", str(shared["imu_fast_init"]),
        "--nfeatures", str(shared["nfeatures"]),
        "--ini-fast", str(shared["ini_fast"]),
        "--min-fast", str(shared["min_fast"]),
        "--final-ba-iters", str(shared["final_ba_iters"]),
        "--offline-wait-timeout-sec", str(shared["offline_wait_timeout_sec"]),
        "--timeout-sec", str(shared["timeout_sec"]),
        "--smooth-window", str(shared["smooth_window"]),
        "--smooth-passes", str(shared["smooth_passes"]),
        "--strict-sync-offset-sec", str(episode["strict_sync_offset_sec"]),
        "--no-stereo-fallback",
        "--skip-viewer",
    ]
    if shared["offline_deterministic"]:
        cmd.append("--offline-deterministic")
    if shared["vins_noise_only"]:
        cmd.append("--vins-noise-only")

    print("[CONFIG]", args.config)
    print("[REFERENCE]", json.dumps(episode, ensure_ascii=True))
    print("[RUNTIME_LOCK]", json.dumps(runtime, ensure_ascii=True))
    print("[RUN]", " ".join(cmd))
    if not args.run:
        return 0
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
