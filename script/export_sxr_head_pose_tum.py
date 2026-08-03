#!/usr/bin/env python3
"""Export the SXR ``head_pose`` topic from one or more episodes as TUM files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from check_episode_data import CheckResult, inspect_mcap


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="episode_*")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = sorted(path for path in args.episode_root.glob(args.pattern) if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"No episodes match {args.pattern!r} under {args.episode_root}")
    for episode in episodes:
        head, _hand, _channels, _schemas = inspect_mcap(episode / "sensor.mcap", CheckResult())
        poses = sorted(head.poses)
        if len(poses) < 2:
            print(f"[SKIP] {episode.name}: head_pose samples={len(poses)}")
            continue
        destination = args.output_dir / f"{episode.name}.tum"
        with destination.open("w", encoding="utf-8") as handle:
            for timestamp_ns, position, quaternion in poses:
                handle.write(
                    f"{timestamp_ns * 1e-9:.9f} {position[0]:.9f} {position[1]:.9f} {position[2]:.9f} "
                    f"{quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} {quaternion[3]:.9f}\n"
                )
        print(f"[OK] {episode.name}: {len(poses)} poses -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
