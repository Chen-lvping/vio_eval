#!/usr/bin/env python3
"""Sequentially rerun only promising RM75 stereo-IMU episodes with final BA=10."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_UNIFIED = ROOT / "script/run_rm75_unified_stereo_imu.py"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "data/evaluation/workbench/rm75_22_final_robust_20260722/selected_ba10"
)
DEFAULT_EPISODES = (
    "episode_gripper_0001,episode_gripper_0003,episode_gripper_0005,"
    "episode_gripper_0007,episode_gripper_0009,episode_gripper_0010,"
    "episode_gripper_0011,episode_gripper_0015,episode_gripper_0018,"
    "episode_gripper_0019,episode_gripper_0021,episode_gripper_0022"
)
PREEXISTING_MANIFESTS = {
    "episode_gripper_0020": ROOT
    / "data/evaluation/workbench/rm75_0020_d_ffba10_smooth219_scan_20260722"
    / "raw_center_eval/orbslam3_tcp_eval_manifest.json"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", default=DEFAULT_EPISODES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timeout-sec", type=int, default=720)
    parser.add_argument("--final-ba-iters", type=int, default=10)
    parser.add_argument("--direct-imu", action="store_true", help="Keep raw stereo-IMU outputs; do not select a stereo fallback.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_state(path: Path) -> dict:
    if not path.is_file():
        return {"created_at": datetime.now().isoformat(timespec="seconds"), "manifests": {}, "errors": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def find_manifest(output_root: Path, episode: str, started_at_ns: int) -> Path:
    candidates = sorted(
        output_root.glob(
            f"orbslam3_rm75_batch_eval_*/eval_{episode}/orbslam3_tcp_eval_manifest.json"
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.stat().st_mtime_ns >= started_at_ns:
            return candidate.resolve()
    raise FileNotFoundError(f"new BA10 manifest not found for {episode}")


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "selected_runs.json"
    state = load_state(state_path) if args.resume else {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifests": {},
        "errors": {},
    }
    state["configuration"] = {
        "episodes": args.episodes,
        "final_ba_iters": args.final_ba_iters,
        "timeout_sec": args.timeout_sec,
        "runner": str(RUN_UNIFIED),
        "execution": "sequential",
        "postprocess": "21/2/9 smoothing plus independent per-episode strict-sync scan",
        "direct_imu": args.direct_imu,
    }
    for episode, manifest in PREEXISTING_MANIFESTS.items():
        if manifest.is_file():
            state["manifests"].setdefault(episode, str(manifest.resolve()))
    save_state(state_path, state)

    episodes = [part.strip() for part in args.episodes.split(",") if part.strip()]
    for index, episode in enumerate(episodes, start=1):
        existing = Path(str(state["manifests"].get(episode, "")))
        if args.resume and existing.is_file():
            print(f"[SKIP {index}/{len(episodes)}] {episode}: {existing}", flush=True)
            continue
        log_path = logs_dir / f"{episode}.log"
        cmd = [
            sys.executable,
            str(RUN_UNIFIED),
            "--episode-pattern",
            episode,
            "--output-root",
            str(output_root),
            "--timeout-sec",
            str(args.timeout_sec),
            "--final-ba-iters",
            str(args.final_ba_iters),
            "--skip-viewer",
        ]
        if args.direct_imu:
            cmd.append("--direct-imu")
        print(f"[RUN {index}/{len(episodes)}] {episode}", flush=True)
        started_at_ns = datetime.now().timestamp()
        started_at_ns = int(started_at_ns * 1_000_000_000)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            state["errors"][episode] = {
                "returncode": proc.returncode,
                "log": str(log_path),
            }
            save_state(state_path, state)
            print(f"[FAIL] {episode}: see {log_path}", flush=True)
            continue
        try:
            manifest = find_manifest(output_root, episode, started_at_ns)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            final_ba = int(payload.get("resolved_parameter_overrides", {}).get("final_ba_iters", -1))
            if final_ba != args.final_ba_iters:
                raise RuntimeError(f"unexpected final BA value {final_ba} in {manifest}")
            state["manifests"][episode] = str(manifest)
            state["errors"].pop(episode, None)
            save_state(state_path, state)
            print(
                f"[OK] {episode} mode={payload.get('mode')} "
                f"coverage={payload.get('inertial_coverage')} manifest={manifest}",
                flush=True,
            )
        except Exception as exc:
            state["errors"][episode] = {"error": str(exc), "log": str(log_path)}
            save_state(state_path, state)
            print(f"[FAIL] {episode}: {exc}", flush=True)

    state["completed_at"] = datetime.now().isoformat(timespec="seconds")
    save_state(state_path, state)
    print(f"[DONE] {state_path}", flush=True)
    return 1 if state["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
