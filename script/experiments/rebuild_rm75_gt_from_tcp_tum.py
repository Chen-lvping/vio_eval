#!/usr/bin/env python3
"""Recover an evaluator-compatible RM75 GT JSON from a preserved TCP TUM reference.

This is intentionally limited to archival recovery: the source TUM must have
been produced by the standard evaluator from the original robot GT.  The
output records that provenance and is suitable only for reproducing that
episode's historical evaluation chain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-tum", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.reference_tum.expanduser().resolve()
    output = args.output_json.expanduser().resolve()
    samples = []
    for line in source.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if len(parts) != 8:
            raise ValueError(f"expected TUM timestamp + xyz + xyzw, got: {line!r}")
        timestamp, x, y, z, qx, qy, qz, qw = map(float, parts)
        samples.append(
            {
                "timestamp_s": timestamp,
                "position_m": {"x": x, "y": y, "z": z},
                "quaternion_xyzw": {"x": qx, "y": qy, "z": qz, "w": qw},
            }
        )
    if len(samples) < 2:
        raise ValueError(f"insufficient TUM samples: {len(samples)}")
    payload = {
        "frame": "RM75 TCP in evaluator reference frame",
        "timestamp_source": "recovered_from_standard_evaluator_ref_tcp_tum",
        "timestamp_policy": "preserved_historical_reference_samples",
        "recovery": {
            "source_tum": str(source),
            "source_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
            "reason": "original rm75_6_23/rm75_pose_traj_3.json is absent from workspace",
        },
        "sample_count": len(samples),
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {output} samples={len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
