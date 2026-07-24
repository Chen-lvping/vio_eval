#!/usr/bin/env python3
"""Apply symmetric moving-average smoothing to a text trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input trajectory text file")
    parser.add_argument("--output", type=Path, required=True, help="Output trajectory text file")
    parser.add_argument("--window", type=int, default=7, help="Odd moving-average window size")
    parser.add_argument("--passes", type=int, default=2, help="How many smoothing passes to apply")
    return parser.parse_args()


def validate_window(window: int) -> int:
    if window < 3 or window % 2 == 0:
        raise ValueError("--window must be an odd integer >= 3")
    return window


def smooth_positions(positions: np.ndarray, window: int, passes: int) -> np.ndarray:
    radius = window // 2
    out = positions.astype(float, copy=True)
    kernel = np.full(window, 1.0 / window, dtype=float)
    for _ in range(max(1, passes)):
        padded = np.pad(out, ((radius, radius), (0, 0)), mode="edge")
        next_out = np.empty_like(out)
        for axis in range(3):
            next_out[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
        out = next_out
    return out


def format_row(values: np.ndarray) -> str:
    return " ".join(f"{float(v):.9f}" for v in values)


def main() -> int:
    args = parse_args()
    window = validate_window(args.window)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_rows: List[np.ndarray] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                numeric_rows.append(None)  # type: ignore[arg-type]
                continue
            parts = stripped.replace(",", " ").split()
            values = np.asarray([float(item) for item in parts], dtype=float)
            if values.size < 4:
                raise ValueError(f"{input_path} contains a row with fewer than 4 numeric columns: {stripped}")
            numeric_rows.append(values)

    positions = np.asarray([row[1:4] for row in numeric_rows if row is not None], dtype=float)
    smoothed = smooth_positions(positions, window=window, passes=args.passes)

    smoothed_rows: List[str] = []
    pos_index = 0
    for row in numeric_rows:
        if row is None:
            smoothed_rows.append("")
            continue
        next_row = row.copy()
        next_row[1:4] = smoothed[pos_index]
        smoothed_rows.append(format_row(next_row))
        pos_index += 1

    with output_path.open("w", encoding="utf-8") as handle:
        for line in smoothed_rows:
            handle.write(line + "\n")

    print(f"[OK] wrote {output_path}")
    print(f"[INFO] window={window} passes={args.passes} rows={len(positions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
