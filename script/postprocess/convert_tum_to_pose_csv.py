#!/usr/bin/env python3
"""Convert a TUM trajectory into a VINS-style pose CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


HEADER = ["Timestamp_us", "X", "Y", "Z", "Quat_X", "Quat_Y", "Quat_Z", "Quat_W"]


def timestamp_to_us(value: float) -> int:
    if abs(value) >= 1e15:
        return int(round(value / 1_000.0))
    if abs(value) >= 1e12:
        return int(round(value))
    return int(round(value * 1_000_000.0))


def convert_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            try:
                timestamp_us = timestamp_to_us(float(parts[0]))
                xyzq = [float(value) for value in parts[1:]]
            except ValueError:
                continue
            rows.append(
                [
                    str(timestamp_us),
                    f"{xyzq[0]:.9f}",
                    f"{xyzq[1]:.9f}",
                    f"{xyzq[2]:.9f}",
                    f"{xyzq[3]:.9f}",
                    f"{xyzq[4]:.9f}",
                    f"{xyzq[5]:.9f}",
                    f"{xyzq[6]:.9f}",
                ]
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tum", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    rows = convert_rows(args.input_tum.expanduser().resolve())
    args.output_csv.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.expanduser().resolve().open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    print(f"[OK] wrote {args.output_csv} ({len(rows)} poses)")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
