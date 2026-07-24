#!/usr/bin/env python3
"""Create non-mutating episode shims with one canonical calibration file."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = ROOT / "data/evaluation/config/rm75_stereo_right_unified_historical_calibration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode-pattern", default="episode_*")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    return parser.parse_args()


def link(destination: Path, target: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == target.resolve():
            return
        raise FileExistsError(f"refusing to replace existing shim entry: {destination}")
    destination.symlink_to(target, target_is_directory=target.is_dir())


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    calibration = args.calibration.expanduser().resolve()
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    episodes = sorted(path for path in source_root.glob(args.episode_pattern) if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"no episodes match {args.episode_pattern} under {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for source in episodes:
        shim = output_root / source.name
        shim.mkdir(exist_ok=True)
        for entry in source.iterdir():
            if entry.name == "calibration.json":
                continue
            link(shim / entry.name, entry)
        link(shim / "calibration.json", calibration)
        rows.append({"episode": source.name, "source": str(source), "shim": str(shim)})
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "canonical_calibration": str(calibration),
        "episodes": rows,
    }
    (output_root / "unified_calibration_shim_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
