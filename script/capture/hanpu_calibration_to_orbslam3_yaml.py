#!/usr/bin/env python3
"""Convert a YCTC stereo calibration JSON into an ORB-SLAM3 stereo YAML."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def matrix(data: object, rows: int, cols: int, name: str) -> list[list[float]]:
    if not isinstance(data, list) or len(data) != rows:
        raise ValueError(f"{name} must be a {rows}x{cols} matrix")
    result: list[list[float]] = []
    for row in data:
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(f"{name} must be a {rows}x{cols} matrix")
        result.append([float(value) for value in row])
    return result


def vector(data: object, length: int, name: str) -> list[float]:
    if not isinstance(data, list) or len(data) < length:
        raise ValueError(f"{name} must contain at least {length} values")
    return [float(value) for value in data[:length]]


def write_yaml(calibration_json: Path, output: Path, width: int, height: int, fps: float, th_depth: float) -> None:
    document = json.loads(calibration_json.read_text(encoding="utf-8"))
    calibration = document.get("calibration", {})
    k1 = matrix(calibration.get("K1"), 3, 3, "K1")
    k2 = matrix(calibration.get("K2"), 3, 3, "K2")
    d1 = vector(calibration.get("D1"), 5, "D1")
    d2 = vector(calibration.get("D2"), 5, "D2")
    r = matrix(calibration.get("R"), 3, 3, "R")
    t_mm = vector([row[0] for row in matrix(calibration.get("T"), 3, 1, "T")], 3, "T")
    baseline_m = math.sqrt(sum(value * value for value in t_mm)) / 1000.0

    text = f'''%YAML:1.0

File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {k1[0][0]:.12g}
Camera1.fy: {k1[1][1]:.12g}
Camera1.cx: {k1[0][2]:.12g}
Camera1.cy: {k1[1][2]:.12g}
Camera1.k1: {d1[0]:.12g}
Camera1.k2: {d1[1]:.12g}
Camera1.p1: {d1[2]:.12g}
Camera1.p2: {d1[3]:.12g}
Camera1.k3: {d1[4]:.12g}
Camera2.fx: {k2[0][0]:.12g}
Camera2.fy: {k2[1][1]:.12g}
Camera2.cx: {k2[0][2]:.12g}
Camera2.cy: {k2[1][2]:.12g}
Camera2.k1: {d2[0]:.12g}
Camera2.k2: {d2[1]:.12g}
Camera2.p1: {d2[2]:.12g}
Camera2.p2: {d2[3]:.12g}
Camera2.k3: {d2[4]:.12g}
Camera.width: {width}
Camera.height: {height}
Camera.fps: {int(round(fps))}
Camera.RGB: 1
Stereo.ThDepth: {th_depth:.12f}
Stereo.T_c1_c2: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [{r[0][0]:.12g}, {r[0][1]:.12g}, {r[0][2]:.12g}, {t_mm[0] / 1000.0:.12g},
         {r[1][0]:.12g}, {r[1][1]:.12g}, {r[1][2]:.12g}, {t_mm[1] / 1000.0:.12g},
         {r[2][0]:.12g}, {r[2][1]:.12g}, {r[2][2]:.12g}, {t_mm[2] / 1000.0:.12g},
         0.0, 0.0, 0.0, 1.0]
ORBextractor.nFeatures: 4200
ORBextractor.scaleFactor: 1.15
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 7
ORBextractor.minThFAST: 2
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"output={output}")
    print(f"camera_size={width}x{height}")
    print(f"baseline_m={baseline_m:.9f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--th-depth", type=float, default=35.0)
    args = parser.parse_args()
    write_yaml(
        args.calibration_json.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.width,
        args.height,
        args.fps,
        args.th_depth,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
