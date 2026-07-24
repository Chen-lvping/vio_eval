#!/usr/bin/env python3
"""Prepare and optionally run patched offline VINS reruns for RM75 acceptance episodes."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WS = Path("/home/chenlvping/0_SLAM/vinsfusion_ws")
DEFAULT_BATCH_SCRIPT = DEFAULT_WS / "scripts/run_fanysense_vins_offline_batch_parallel.sh"
DEFAULT_EPISODES = [
    ("data/gripper_data2", "episode_20260618_0004"),
    ("data/gripper_data_1", "episode_20260617_0005"),
    ("data/gripper_data_1", "episode_20260617_0006"),
]


def build_job_commands(workspace: Path, jobs: int) -> List[List[str]]:
    commands: List[List[str]] = []
    batch_script = workspace / "scripts/run_fanysense_vins_offline_batch_parallel.sh"
    for dataset_rel, episode in DEFAULT_EPISODES:
        dataset_root = REPO_ROOT / dataset_rel
        commands.append(
            [
                "bash",
                str(batch_script),
                "--ws",
                str(workspace),
                "--dataset-root",
                str(dataset_root),
                "--episode",
                episode,
                "--jobs",
                str(jobs),
            ]
        )
    return commands


def ros1_ready() -> tuple[bool, List[str]]:
    missing = [name for name in ("rosrun", "roscore", "catkin_make") if shutil.which(name) is None]
    return (len(missing) == 0, missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WS)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--run", action="store_true", help="Actually execute reruns when the environment is ready")
    args = parser.parse_args()

    ready, missing = ros1_ready()
    commands = build_job_commands(args.workspace, args.jobs)

    print("RM75 patched VINS rerun plan")
    print(f"workspace: {args.workspace}")
    print(f"batch_script: {args.workspace / 'scripts/run_fanysense_vins_offline_batch_parallel.sh'}")
    print(f"ros1_ready: {ready}")
    if missing:
        print(f"missing_commands: {', '.join(missing)}")
    print()
    for idx, cmd in enumerate(commands, start=1):
        print(f"[job {idx}] {' '.join(cmd)}")
    print()

    if not args.run:
        print("Dry run only. Use --run after sourcing a ROS1-compatible environment.")
        return

    if not ready:
        raise SystemExit(f"ROS1 runtime not ready. Missing: {', '.join(missing)}")

    env = os.environ.copy()
    for idx, cmd in enumerate(commands, start=1):
        print(f"Running job {idx}/{len(commands)}...")
        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
