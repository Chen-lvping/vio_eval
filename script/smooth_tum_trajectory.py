#!/usr/bin/env python3
"""Smooth only the translation of a TUM trajectory, preserving timestamps and orientation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import savgol_filter


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, required=True, help="Odd Savitzky-Golay window length.")
    parser.add_argument("--polyorder", type=int, default=2)
    return parser.parse_args(argv)


def read_tum(path: Path) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    comments: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            comments.append(text)
            continue
        values = [float(value) for value in text.split()]
        if len(values) < 8:
            raise ValueError(f"TUM row needs 8 columns: {line!r}")
        rows.append(values[:8])
    if len(rows) < 3:
        raise ValueError("Need at least three trajectory samples")
    return np.asarray(rows, dtype=np.float64), comments


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.window < 3 or args.window % 2 == 0:
        raise ValueError("--window must be an odd integer >= 3")
    if args.polyorder < 1 or args.polyorder >= args.window:
        raise ValueError("--polyorder must be >= 1 and smaller than --window")
    values, comments = read_tum(args.input)
    if args.window > values.shape[0]:
        raise ValueError(f"--window={args.window} exceeds {values.shape[0]} samples")
    smoothed = values.copy()
    smoothed[:, 1:4] = savgol_filter(values[:, 1:4], args.window, args.polyorder, axis=0, mode="interp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(f"# translation Savitzky-Golay window={args.window}, polyorder={args.polyorder}; orientation unchanged\n")
        for comment in comments:
            handle.write(f"{comment}\n")
        for row in smoothed:
            handle.write(" ".join(f"{value:.9f}" for value in row) + "\n")
    print(f"[OK] {values.shape[0]} poses -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
