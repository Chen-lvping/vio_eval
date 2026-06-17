#!/usr/bin/env python3
"""
Evaluate a VINS trajectory against robot hand-eye ground truth.

The default paths match the 2026-06-14/0615 data currently used in this
workspace. The evaluator supports:

  * Robot trajectory JSON from record_trajectory.py.
  * VINS pose_data.csv / pose_data.json files with Timestamp_us,X,Y,Z,Quat_*.
  * TUM-style text trajectories: t x y z qx qy qz qw.
  * Timestamp interpolation or index matching.
  * SE(3) alignment for metric VIO accuracy and Sim(3) alignment as a
    shape/scale diagnostic.
  * APE, orientation error, RPE, drift, scale, CSV/JSON/Markdown/PNG outputs.

Example:
  python3 script/evaluate_vins_accuracy.py

  python3 script/evaluate_vins_accuracy.py \
      --estimate "/path/to/pose_data.csv" \
      --ground-truth "/path/to/trajectory_001.json" \
      --handeye "/path/to/handeye_result.yaml"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import webbrowser

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - handled at runtime with a clear error
    yaml = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAND_EYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data/ground_truth/trajectory_samples0614/trajectory_001.json"
DEFAULT_DATASET = Path("/home/chenlvping/0614 _test/episode_20260614_0239")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/vins_eval"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8766


@dataclass
class Trajectory:
    name: str
    path: Path
    timestamps: Optional[np.ndarray]
    positions: np.ndarray
    rotations: Optional[np.ndarray]
    frame: str
    note: str = ""

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def duration_s(self) -> float:
        if self.timestamps is None or self.timestamps.size < 2:
            return float("nan")
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def path_length_m(self) -> float:
        return path_length(self.positions)


@dataclass
class AlignmentResult:
    mode: str
    scale: float
    rotation: List[List[float]]
    translation: List[float]
    translation_metrics_m: Dict[str, float]
    rotation_metrics_deg: Optional[Dict[str, float]]
    rpe_translation_metrics_m: Optional[Dict[str, float]]
    rpe_rotation_metrics_deg: Optional[Dict[str, float]]
    drift: Dict[str, float]


@dataclass
class WindowScanEntry:
    start_index: int
    end_index_exclusive: int
    sample_count: int
    duration_s: float
    estimate_indices_start_end: List[int]
    time_range_s: List[float]
    se3_rmse_m: float
    se3_mean_m: float
    se3_p95_m: float
    se3_path_error_pct: float
    se3_final_drift_pct: float
    se3_rot_rmse_deg: Optional[float]
    sim3_scale: float
    sim3_rmse_m: float
    sim3_mean_m: float
    sim3_p95_m: float
    sim3_path_error_pct: float
    sim3_final_drift_pct: float
    sim3_rot_rmse_deg: Optional[float]


@dataclass
class ViewerPose:
    t: float
    gt: List[float]
    estimate: List[float]
    se3: List[float]
    sim3: List[float]
    se3_error_m: float
    sim3_error_m: float


class EvalViewerState:
    manifest: dict = {}


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VINS Ground Truth Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #171a1f;
      --panel-2: #20242b;
      --line: #343a44;
      --text: #eef2f7;
      --muted: #9aa6b2;
      --accent: #4cc9f0;
      --accent-2: #7bd88f;
      --warn: #ff6b6b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }
    header {
      height: 58px;
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      background: #12151a;
    }
    h1 {
      font-size: 16px;
      font-weight: 650;
      margin: 0;
      min-width: 220px;
      max-width: 34vw;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    button, select, input[type="number"] {
      background: var(--panel-2);
      border: 1px solid var(--line);
      color: var(--text);
      border-radius: 6px;
      height: 34px;
      padding: 0 10px;
    }
    button { cursor: pointer; }
    button:hover { border-color: var(--accent); }
    button.active {
      border-color: var(--accent);
      background: rgba(76, 201, 240, .16);
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      flex: 1;
    }
    .timebar {
      flex: 1;
      min-width: 160px;
      accent-color: var(--accent);
    }
    main {
      height: calc(100vh - 58px);
      display: grid;
      grid-template-columns: minmax(420px, 1.15fr) minmax(360px, .85fr);
      min-height: 0;
    }
    .stage {
      position: relative;
      min-width: 0;
      background: #0c0f12;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .hud {
      position: absolute;
      left: 14px;
      bottom: 14px;
      display: grid;
      gap: 6px;
      padding: 10px 12px;
      background: rgba(16, 18, 20, .82);
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 8px;
      color: var(--muted);
      min-width: 320px;
      backdrop-filter: blur(6px);
    }
    .hud strong { color: var(--text); font-weight: 600; }
    .inspector {
      background: var(--panel);
      padding: 12px 14px;
      overflow: auto;
      color: var(--muted);
      border-left: 1px solid var(--line);
    }
    .inspector pre {
      white-space: pre-wrap;
      margin: 8px 0 0;
      color: var(--text);
      font-size: 12px;
    }
    @media (max-width: 900px) {
      body { overflow: auto; }
      header { height: auto; min-height: 58px; flex-wrap: wrap; padding: 10px; }
      main { height: auto; grid-template-columns: 1fr; }
      .stage { height: 58vh; }
      h1 { max-width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <h1 id="title">VINS Ground Truth Viewer</h1>
    <div class="toolbar">
      <button id="play">Play</button>
      <select id="mode">
        <option value="se3" selected>SE(3)</option>
        <option value="sim3">Sim(3)</option>
      </select>
      <select id="speed">
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
        <option value="4">4x</option>
      </select>
      <input id="timeline" class="timebar" type="range" min="0" max="1" step="0.001" value="0">
      <span id="clock">0.000 / 0.000s</span>
      <button id="reset">Reset View</button>
    </div>
  </header>
  <main>
    <section class="stage">
      <canvas id="scene"></canvas>
      <div class="hud" id="hud"></div>
    </section>
    <section class="inspector">
      <strong>Evaluation summary</strong>
      <pre id="summary"></pre>
      <strong>Inputs</strong>
      <pre id="inputs"></pre>
    </section>
  </main>
<script>
"use strict";
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const playBtn = document.getElementById("play");
const speedEl = document.getElementById("speed");
const modeEl = document.getElementById("mode");
const timeline = document.getElementById("timeline");
const clock = document.getElementById("clock");
const hud = document.getElementById("hud");
let data = null;
let playing = false;
let currentTime = 0;
let lastTick = 0;
let yaw = -0.38;
let pitch = 0.42;
let zoom = 1.0;
let dragging = false;
let lastMouse = [0, 0];
function fmt(v, n = 3) { return Number.isFinite(v) ? v.toFixed(n) : "0"; }
function fmtMeters(value) { return Math.abs(value) < 0.01 ? `${fmt(value * 1000, 1)} mm` : `${fmt(value, 3)} m`; }
function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
function centerAndScale() {
  const b = data.gt.bounds;
  const c = [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2];
  const span = Math.max(...b.span, 0.05);
  const rect = canvas.getBoundingClientRect();
  return { center: c, scale: Math.min(rect.width, rect.height) * 0.72 / span * zoom };
}
function worldToViewer(world, center = [0, 0, 0]) { return [-(world[0] - center[0]), world[1] - center[1], world[2] - center[2]]; }
function rotateViewer(v) {
  const [x, y, z] = v;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  const y1 = cp * y - sp * z1;
  return [x1, y1, z1];
}
function project(p, view = null) {
  const { center, scale } = view || centerAndScale();
  const [x1, y1] = rotateViewer(worldToViewer(p, center));
  const rect = canvas.getBoundingClientRect();
  return [rect.width / 2 + x1 * scale, rect.height / 2 - y1 * scale];
}
function strokePath(points, color, width, view) {
  if (!points || points.length < 2) return;
  ctx.beginPath();
  const first = project(points[0], view);
  ctx.moveTo(first[0], first[1]);
  for (let i = 1; i < points.length; i++) {
    const p = project(points[i], view);
    ctx.lineTo(p[0], p[1]);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}
function drawPoint(p, color, radius, view) {
  const q = project(p, view);
  ctx.beginPath();
  ctx.arc(q[0], q[1], radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}
function drawSegment(a, b, color, width, view) {
  const pa = project(a, view);
  const pb = project(b, view);
  ctx.beginPath();
  ctx.moveTo(pa[0], pa[1]);
  ctx.lineTo(pb[0], pb[1]);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}
function drawGrid() {
  const rect = canvas.getBoundingClientRect();
  ctx.fillStyle = "#0c0f12";
  ctx.fillRect(0, 0, rect.width, rect.height);
  ctx.strokeStyle = "rgba(255,255,255,.05)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= rect.width; x += 48) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, rect.height);
    ctx.stroke();
  }
  for (let y = 0; y <= rect.height; y += 48) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(rect.width, y);
    ctx.stroke();
  }
}
function draw() {
  if (!data) return;
  drawGrid();
  const view = centerAndScale();
  const key = modeEl.value;
  const idx = nearestIndex(currentTime);
  const gt = data.gt.points;
  const est = key === "sim3" ? data.sim3.points : data.se3.points;
  const raw = data.estimate.points;
  const err = key === "sim3" ? data.sim3.errors : data.se3.errors;
  strokePath(gt, "rgba(123,216,143,.95)", 2.4, view);
  strokePath(raw, "rgba(154,166,178,.35)", 1.2, view);
  strokePath(est, "rgba(76,201,240,.95)", 2.0, view);
  drawSegment(gt[idx], est[idx], "rgba(255,107,107,.88)", 2.0, view);
  drawPoint(gt[idx], "#7bd88f", 5.5, view);
  drawPoint(est[idx], "#4cc9f0", 5.5, view);
  drawPoint(raw[idx], "rgba(238,242,247,.7)", 3.5, view);
  hud.innerHTML =
    `<strong>${key.toUpperCase()} comparison</strong>` +
    `<span>gt path=${fmtMeters(data.gt.path_length_m)}, est path=${fmtMeters(data[key].path_length_m)}</span>` +
    `<span>rmse=${fmtMeters(data[key].rmse_m)}, p95=${fmtMeters(data[key].p95_m)}</span>` +
    `<span>scale=${fmt(data[key].scale, 6)}, matches=${data.coverage.matched_samples}</span>` +
    `<span>sample=${idx + 1}/${gt.length}, t=${fmt(currentTime, 3)}s</span>` +
    `<span>current error=${fmtMeters(err[idx])}</span>` +
    `<span style="color:#7bd88f">green: hand-eye ground truth</span>` +
    `<span style="color:#4cc9f0">blue: aligned VINS, gray: raw VINS</span>`;
}
function nearestIndex(t) {
  const poses = data.gt.points;
  let lo = 0, hi = poses.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (data.gt.times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
function setTime(t) {
  currentTime = Math.max(0, Math.min(data.coverage.matched_duration_s, t));
  timeline.value = data.coverage.matched_duration_s ? String(currentTime / data.coverage.matched_duration_s) : "0";
  clock.textContent = `${fmt(currentTime, 3)} / ${fmt(data.coverage.matched_duration_s, 3)}s`;
  draw();
}
function tick(ts) {
  if (!lastTick) lastTick = ts;
  const dt = (ts - lastTick) / 1000;
  lastTick = ts;
  if (playing) setTime(currentTime + dt * Number(speedEl.value));
  requestAnimationFrame(tick);
}
function bind() {
  timeline.addEventListener("input", () => setTime(Number(timeline.value) * data.coverage.matched_duration_s));
  playBtn.addEventListener("click", () => { playing = !playing; playBtn.textContent = playing ? "Pause" : "Play"; lastTick = 0; });
  speedEl.addEventListener("change", draw);
  modeEl.addEventListener("change", draw);
  document.getElementById("reset").addEventListener("click", () => { yaw = -0.38; pitch = 0.42; zoom = 1.0; draw(); });
  canvas.addEventListener("mousedown", e => { dragging = true; lastMouse = [e.clientX, e.clientY]; });
  window.addEventListener("mouseup", () => dragging = false);
  window.addEventListener("mousemove", e => {
    if (!dragging) return;
    const dx = e.clientX - lastMouse[0];
    const dy = e.clientY - lastMouse[1];
    lastMouse = [e.clientX, e.clientY];
    yaw += dx * 0.006;
    pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.006));
    draw();
  });
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    zoom *= e.deltaY > 0 ? 0.9 : 1.1;
    zoom = Math.max(0.1, Math.min(20, zoom));
    draw();
  }, { passive: false });
  window.addEventListener("resize", resizeCanvas);
}
fetch("/api/manifest").then(r => r.json()).then(manifest => {
  data = manifest;
  document.getElementById("title").textContent = manifest.title || "VINS Ground Truth Viewer";
  document.getElementById("summary").textContent = manifest.summary_text || "";
  document.getElementById("inputs").textContent = JSON.stringify(manifest.inputs, null, 2);
  bind();
  resizeCanvas();
  requestAnimationFrame(tick);
}).catch(err => {
  document.body.innerHTML = `<pre style="padding:24px;color:#ff6b6b">${err.stack || err}</pre>`;
});
</script>
</body>
</html>
"""


class EvalViewerHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        if not getattr(self.server, "quiet", False):  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/manifest"):
            body = json.dumps(EvalViewerState.manifest, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


def normalize_timestamp(value: float) -> float:
    """Return seconds from common timestamp magnitudes."""
    value = float(value)
    av = abs(value)
    if av > 1e17:  # nanoseconds since epoch
        return value * 1e-9
    if av > 1e13:  # microseconds since epoch
        return value * 1e-6
    if av > 1e10:  # milliseconds since epoch
        return value * 1e-3
    return value


def normalize_quaternion_xyzw(q: Sequence[float]) -> np.ndarray:
    arr = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12 or not math.isfinite(norm):
        raise ValueError("invalid quaternion norm")
    return arr / norm


def quat_xyzw_to_rot(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rot_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=float).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + r[1, 1] - r[0, 0] - r[2, 2])) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(max(0.0, 1.0 + r[2, 2] - r[0, 0] - r[1, 1])) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    return normalize_quaternion_xyzw([qx, qy, qz, qw])


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalize_quaternion_xyzw(q0)
    q1 = normalize_quaternion_xyzw(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion_xyzw((1.0 - alpha) * q0 + alpha * q1)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return normalize_quaternion_xyzw(s0 * q0 + s1 * q1)


def transform_from_rt(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    out[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return out


def transform_from_pose(position: Sequence[float], quaternion_xyzw: Sequence[float]) -> np.ndarray:
    return transform_from_rt(quat_xyzw_to_rot(quaternion_xyzw), position)


def transform_trajectory(trajectory: Trajectory, static_transform: np.ndarray, target_frame: str, note: str) -> Trajectory:
    static_transform = np.asarray(static_transform, dtype=float).reshape(4, 4)
    if trajectory.rotations is None:
        raise ValueError("estimate frame conversion requires rotations in the estimate trajectory")

    positions: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    for position, rotation in zip(trajectory.positions, trajectory.rotations):
        pose = transform_from_rt(rotation, position) @ static_transform
        positions.append(pose[:3, 3].copy())
        rotations.append(pose[:3, :3].copy())

    return Trajectory(
        trajectory.name,
        trajectory.path,
        trajectory.timestamps.copy() if trajectory.timestamps is not None else None,
        np.asarray(positions, dtype=float),
        np.asarray(rotations, dtype=float),
        target_frame,
        note,
    )


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
    inv = np.eye(4, dtype=float)
    inv[:3, :3] = matrix[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ matrix[:3, 3]
    return inv


def rotation_angle_deg(rotation: np.ndarray) -> float:
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    value = (float(np.trace(r)) - 1.0) * 0.5
    value = max(-1.0, min(1.0, value))
    return math.degrees(math.acos(value))


def path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def read_yaml_matrix(path: Path, key_path: Sequence[str]) -> np.ndarray:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read hand-eye YAML files")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    node = data
    for key in key_path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"missing YAML key: {'.'.join(key_path)}")
        node = node[key]
    matrix = np.asarray(node, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{'.'.join(key_path)} must be a 4x4 matrix")
    return matrix


def read_handeye_transform(path: Path, key: str) -> np.ndarray:
    return read_yaml_matrix(path, ["result", key])


def rotation_from_zyx(rz: float, ry: float, rx: float) -> np.ndarray:
    cz, sz = math.cos(rz), math.sin(rz)
    cy, sy = math.cos(ry), math.sin(ry)
    cx, sx = math.cos(rx), math.sin(rx)
    rz_matrix = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    ry_matrix = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rx_matrix = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    return rz_matrix @ ry_matrix @ rx_matrix


def read_config_list(path: Path, key: str, expected_len: int) -> List[float]:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"^\s*{re.escape(key)}:\s*\[([^\]]+)\]", text, re.M)
    if not match:
        raise ValueError(f"missing {key} in {path}")
    values = [float(item) for item in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", match.group(1))]
    if len(values) != expected_len:
        raise ValueError(f"{key} in {path} must contain {expected_len} values, got {len(values)}")
    return values


def read_opencv_matrix_4x4(path: Path, key: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    pattern = (
        rf"{re.escape(key)}:\s*!!opencv-matrix\s*"
        rf"rows:\s*4\s*cols:\s*4\s*dt:\s*[df]\s*data:\s*\[([^\]]+)\]"
    )
    match = re.search(pattern, text, re.S)
    if not match:
        raise ValueError(f"missing OpenCV 4x4 matrix {key} in {path}")
    values = [float(item) for item in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", match.group(1))]
    if len(values) != 16:
        raise ValueError(f"{key} in {path} must contain 16 values, got {len(values)}")
    return np.asarray(values, dtype=float).reshape(4, 4)


def base_to_imu_transform_from_config(path: Path) -> np.ndarray:
    rz, ry, rx, tx, ty, tz = read_config_list(path, "base_to_imu", 6)
    return transform_from_rt(rotation_from_zyx(rz, ry, rx), (tx, ty, tz))


def estimate_to_camera_transform(config_path: Path, estimate_frame: str) -> Tuple[np.ndarray, str]:
    body_t_cam0 = read_opencv_matrix_4x4(config_path, "body_T_cam0")
    if estimate_frame == "imu":
        return body_t_cam0, "estimate imu/body -> stereo cam0 via body_T_cam0"
    if estimate_frame == "base_link":
        return base_to_imu_transform_from_config(config_path) @ body_t_cam0, (
            "estimate base_link -> stereo cam0 via base_to_imu @ body_T_cam0"
        )
    if estimate_frame == "camera":
        return np.eye(4, dtype=float), "estimate already in stereo cam0 frame"
    raise ValueError(f"unsupported estimate frame: {estimate_frame}")


def estimate_post_transform(
    handeye_path: Optional[Path],
    estimate_target_frame: str,
) -> Tuple[np.ndarray, str]:
    if estimate_target_frame == "camera":
        return np.eye(4, dtype=float), "estimate target frame kept at stereo cam0"
    if estimate_target_frame == "gripper":
        if handeye_path is None:
            raise ValueError("--handeye is required when --estimate-target-frame gripper")
        return read_handeye_transform(handeye_path, "T_cam_to_gripper"), "estimate stereo cam0 -> gripper via handeye T_cam_to_gripper"
    raise ValueError(f"unsupported estimate target frame: {estimate_target_frame}")


def load_robot_ground_truth(
    path: Path,
    handeye_path: Optional[Path],
    target_frame: str,
    robot_pose_direction: str,
) -> Trajectory:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    if len(samples) < 2:
        raise ValueError(f"{path} must contain at least two samples")

    if target_frame == "camera":
        if handeye_path is None:
            raise ValueError("--handeye is required when --ground-truth-frame camera")
        t_gripper_cam = read_handeye_transform(handeye_path, "T_gripper_to_cam")
        frame = "base_to_camera"
    elif target_frame == "gripper":
        t_gripper_cam = np.eye(4, dtype=float)
        frame = "base_to_gripper"
    else:
        raise ValueError(f"unsupported ground-truth frame: {target_frame}")

    timestamps: List[float] = []
    positions: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    for sample in samples:
        timestamps.append(normalize_timestamp(float(sample["timestamp"])))
        pose = transform_from_pose(sample["position_m"], sample["quaternion_xyzw"])
        if robot_pose_direction == "gripper_to_base":
            pose = invert_transform(pose)
        elif robot_pose_direction != "base_to_gripper":
            raise ValueError("--robot-pose-direction must be base_to_gripper or gripper_to_base")
        pose = pose @ t_gripper_cam
        positions.append(pose[:3, 3].copy())
        rotations.append(pose[:3, :3].copy())

    t = np.asarray(timestamps, dtype=float)
    p = np.asarray(positions, dtype=float)
    r = np.asarray(rotations, dtype=float)
    order = np.argsort(t)
    return Trajectory("ground_truth", path, t[order], p[order], r[order], frame)


def first_existing(row: dict, names: Sequence[str]) -> Optional[str]:
    lower = {str(k).strip().lower(): k for k in row.keys()}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def rows_to_trajectory(name: str, path: Path, rows: Iterable[dict], default_dt: float) -> Trajectory:
    timestamps: List[float] = []
    positions: List[List[float]] = []
    rotations: List[np.ndarray] = []
    has_timestamp = False
    has_rotation = False

    for idx, row in enumerate(rows):
        ts = first_existing(row, ["Timestamp_us", "timestamp_us", "time_us", "Timestamp_ns", "timestamp_ns", "t", "time", "timestamp"])
        if ts is not None:
            timestamps.append(normalize_timestamp(float(ts)))
            has_timestamp = True
        else:
            timestamps.append(idx * default_dt)

        x = first_existing(row, ["X", "x", "tx", "p_x", "pos_x", "position_x"])
        y = first_existing(row, ["Y", "y", "ty", "p_y", "pos_y", "position_y"])
        z = first_existing(row, ["Z", "z", "tz", "p_z", "pos_z", "position_z"])
        if x is None or y is None or z is None:
            continue
        positions.append([float(x), float(y), float(z)])

        qx = first_existing(row, ["Quat_X", "qx", "q_x", "orientation_x"])
        qy = first_existing(row, ["Quat_Y", "qy", "q_y", "orientation_y"])
        qz = first_existing(row, ["Quat_Z", "qz", "q_z", "orientation_z"])
        qw = first_existing(row, ["Quat_W", "qw", "q_w", "orientation_w"])
        if None not in (qx, qy, qz, qw):
            rotations.append(quat_xyzw_to_rot([float(qx), float(qy), float(qz), float(qw)]))
            has_rotation = True
        else:
            rotations.append(np.eye(3, dtype=float))

    if len(positions) < 2:
        raise ValueError(f"{path} contains fewer than two valid pose rows")
    timestamps_arr = np.asarray(timestamps[: len(positions)], dtype=float)
    positions_arr = np.asarray(positions, dtype=float)
    rotations_arr = np.asarray(rotations[: len(positions)], dtype=float) if has_rotation else None
    if has_timestamp:
        order = np.argsort(timestamps_arr)
        timestamps_arr = timestamps_arr[order]
        positions_arr = positions_arr[order]
        if rotations_arr is not None:
            rotations_arr = rotations_arr[order]
    return Trajectory(name, path, timestamps_arr if has_timestamp else None, positions_arr, rotations_arr, "estimate")


def load_csv_trajectory(path: Path, default_dt: float) -> Trajectory:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is not a header CSV")
        return rows_to_trajectory(path.stem, path, reader, default_dt)


def load_json_trajectory(path: Path, default_dt: float) -> Trajectory:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "samples" in data:
        rows = []
        for sample in data["samples"]:
            row = {
                "timestamp": sample.get("timestamp"),
                "X": sample.get("position_m", [None, None, None])[0],
                "Y": sample.get("position_m", [None, None, None])[1],
                "Z": sample.get("position_m", [None, None, None])[2],
            }
            if "quaternion_xyzw" in sample:
                row.update(dict(zip(["Quat_X", "Quat_Y", "Quat_Z", "Quat_W"], sample["quaternion_xyzw"])))
            rows.append(row)
    elif isinstance(data, dict):
        rows = data.get("poses") or data.get("trajectory") or data.get("data")
    else:
        rows = data
    if not isinstance(rows, list):
        raise ValueError(f"cannot find pose rows in {path}")
    return rows_to_trajectory(path.stem, path, rows, default_dt)


def load_text_trajectory(path: Path, default_dt: float) -> Trajectory:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, raw in enumerate(handle):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(v) for v in line.replace(",", " ").split()]
            if len(values) == 12:
                rows.append({"t": idx * default_dt, "X": values[3], "Y": values[7], "Z": values[11]})
            elif len(values) >= 8:
                rows.append(
                    {
                        "timestamp": values[0],
                        "X": values[1],
                        "Y": values[2],
                        "Z": values[3],
                        "Quat_X": values[4],
                        "Quat_Y": values[5],
                        "Quat_Z": values[6],
                        "Quat_W": values[7],
                    }
                )
    return rows_to_trajectory(path.stem, path, rows, default_dt)


def load_estimate(path: Path, default_dt: float) -> Trajectory:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv_trajectory(path, default_dt)
    if suffix == ".json":
        return load_json_trajectory(path, default_dt)
    return load_text_trajectory(path, default_dt)


def discover_estimate(dataset_dir: Path, side: str) -> Path:
    candidates = [
        dataset_dir / "pose_data" / f"pose_data_{side}.csv",
        dataset_dir / side / "pose_data.csv",
        dataset_dir / f"pose_data_{side}.csv",
        dataset_dir / "pose_data.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    recursive = sorted(dataset_dir.rglob(f"*pose*{side}*.csv")) + sorted(dataset_dir.rglob("*pose_data*.csv"))
    if recursive:
        return recursive[0]
    raise FileNotFoundError(f"cannot auto-discover estimate trajectory under {dataset_dir}")


def interpolate_ground_truth(
    gt: Trajectory,
    query_times: np.ndarray,
    max_gap_s: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    if gt.timestamps is None:
        raise ValueError("timestamp matching requires ground-truth timestamps")
    if gt.rotations is not None:
        gt_quats = np.asarray([rot_to_quat_xyzw(r) for r in gt.rotations], dtype=float)
    else:
        gt_quats = None

    out_indices: List[int] = []
    out_positions: List[np.ndarray] = []
    out_rotations: List[np.ndarray] = []
    out_times: List[float] = []
    left = 0
    for idx, t in enumerate(query_times):
        while left + 1 < gt.timestamps.size and gt.timestamps[left + 1] <= t:
            left += 1
        if left + 1 >= gt.timestamps.size:
            break
        t0 = gt.timestamps[left]
        t1 = gt.timestamps[left + 1]
        if t < t0 or t > t1:
            continue
        if max(abs(t - t0), abs(t1 - t)) > max_gap_s:
            continue
        alpha = 0.0 if abs(t1 - t0) <= 1e-12 else float((t - t0) / (t1 - t0))
        out_indices.append(idx)
        out_positions.append((1.0 - alpha) * gt.positions[left] + alpha * gt.positions[left + 1])
        out_times.append(float(t))
        if gt_quats is not None:
            out_rotations.append(quat_xyzw_to_rot(slerp(gt_quats[left], gt_quats[left + 1], alpha)))

    if not out_indices:
        raise ValueError("no timestamp overlap after interpolation")
    rotations = np.asarray(out_rotations, dtype=float) if out_rotations else None
    return (
        np.asarray(out_indices, dtype=int),
        np.asarray(out_positions, dtype=float),
        rotations,
        np.asarray(out_times, dtype=float),
    )


def resample_by_index(traj: Trajectory, n: int) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    if traj.positions.shape[0] == n:
        times = np.arange(n, dtype=float) if traj.timestamps is None else traj.timestamps.copy()
        return traj.positions.copy(), None if traj.rotations is None else traj.rotations.copy(), times
    src = np.linspace(0.0, 1.0, traj.positions.shape[0])
    dst = np.linspace(0.0, 1.0, n)
    positions = np.column_stack([np.interp(dst, src, traj.positions[:, i]) for i in range(3)])
    rotations = None
    if traj.rotations is not None:
        quats = np.asarray([rot_to_quat_xyzw(r) for r in traj.rotations], dtype=float)
        out_rot = []
        for d in dst:
            right = int(np.searchsorted(src, d, side="right"))
            left = max(0, min(right - 1, len(src) - 2))
            right = left + 1
            alpha = 0.0 if src[right] == src[left] else float((d - src[left]) / (src[right] - src[left]))
            out_rot.append(quat_xyzw_to_rot(slerp(quats[left], quats[right], alpha)))
        rotations = np.asarray(out_rot, dtype=float)
    if traj.timestamps is None:
        times = np.arange(n, dtype=float)
    else:
        times = np.interp(dst, src, traj.timestamps)
    return positions, rotations, times


def align_umeyama(source: np.ndarray, target: np.ndarray, with_scale: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    """Estimate target ~= scale * R * source + t."""
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("alignment requires at least three paired 3D points")
    mu_s = np.mean(source, axis=0)
    mu_t = np.mean(target, axis=0)
    xs = source - mu_s
    xt = target - mu_t
    cov = (xt.T @ xs) / source.shape[0]
    u, singular, vt = np.linalg.svd(cov)
    d = np.ones(3, dtype=float)
    if np.linalg.det(u @ vt) < 0.0:
        d[-1] = -1.0
    rotation = u @ np.diag(d) @ vt
    if with_scale:
        var_s = float(np.mean(np.sum(xs * xs, axis=1)))
        scale = float(np.sum(singular * d) / max(var_s, 1e-12))
    else:
        scale = 1.0
    translation = mu_t - scale * rotation @ mu_s
    return scale, rotation, translation


def summary_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "count": int(values.size),
        "rmse": float(math.sqrt(np.mean(values * values))),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "first": float(values[0]),
        "last": float(values[-1]),
    }


def nearest_future_indices(times: np.ndarray, delta_s: float) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if times.size < 2:
        return pairs
    for i, t in enumerate(times[:-1]):
        target = t + delta_s
        j = int(np.searchsorted(times, target))
        if j >= times.size:
            break
        if j > i:
            pairs.append((i, j))
    return pairs


def compute_rpe(
    gt_pos: np.ndarray,
    gt_rot: Optional[np.ndarray],
    est_pos: np.ndarray,
    est_rot: Optional[np.ndarray],
    times: np.ndarray,
    delta_s: float,
    delta_samples: int,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    if gt_pos.shape[0] < 3:
        return None, None
    if times.size == gt_pos.shape[0] and delta_s > 0.0 and np.all(np.diff(times) >= 0.0):
        pairs = nearest_future_indices(times, delta_s)
    else:
        step = max(1, int(delta_samples))
        pairs = [(i, i + step) for i in range(0, gt_pos.shape[0] - step)]
    if not pairs:
        return None, None

    trans_errors: List[float] = []
    rot_errors: List[float] = []
    for i, j in pairs:
        gt_delta = gt_pos[j] - gt_pos[i]
        est_delta = est_pos[j] - est_pos[i]
        if gt_rot is not None and est_rot is not None:
            gt_delta = gt_rot[i].T @ gt_delta
            est_delta = est_rot[i].T @ est_delta
            gt_rel_r = gt_rot[i].T @ gt_rot[j]
            est_rel_r = est_rot[i].T @ est_rot[j]
            rot_errors.append(rotation_angle_deg(gt_rel_r.T @ est_rel_r))
        trans_errors.append(float(np.linalg.norm(gt_delta - est_delta)))
    return summary_stats(np.asarray(trans_errors, dtype=float)), (
        summary_stats(np.asarray(rot_errors, dtype=float)) if rot_errors else None
    )


def evaluate_alignment(
    mode: str,
    with_scale: bool,
    gt_pos: np.ndarray,
    gt_rot: Optional[np.ndarray],
    est_pos: np.ndarray,
    est_rot: Optional[np.ndarray],
    times: np.ndarray,
    rpe_delta_s: float,
    rpe_delta_samples: int,
) -> Tuple[AlignmentResult, np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    scale, rotation, translation = align_umeyama(est_pos, gt_pos, with_scale=with_scale)
    est_aligned = (scale * (rotation @ est_pos.T)).T + translation
    trans_errors = np.linalg.norm(est_aligned - gt_pos, axis=1)

    est_rot_aligned = None
    rot_errors = None
    if est_rot is not None and gt_rot is not None:
        est_rot_aligned = np.asarray([rotation @ r for r in est_rot], dtype=float)
        rot_errors = np.asarray(
            [rotation_angle_deg(gt_rot[i].T @ est_rot_aligned[i]) for i in range(gt_rot.shape[0])],
            dtype=float,
        )

    rpe_t, rpe_r = compute_rpe(gt_pos, gt_rot, est_aligned, est_rot_aligned, times, rpe_delta_s, rpe_delta_samples)
    gt_path = path_length(gt_pos)
    est_path = path_length(est_aligned)
    drift = {
        "gt_path_length_m": gt_path,
        "aligned_est_path_length_m": est_path,
        "path_length_error_m": est_path - gt_path,
        "path_length_error_pct": 100.0 * (est_path - gt_path) / gt_path if gt_path > 1e-12 else float("nan"),
        "final_position_error_m": float(trans_errors[-1]),
        "final_drift_pct_of_path": 100.0 * float(trans_errors[-1]) / gt_path if gt_path > 1e-12 else float("nan"),
    }
    result = AlignmentResult(
        mode=mode,
        scale=scale,
        rotation=rotation.tolist(),
        translation=translation.tolist(),
        translation_metrics_m=summary_stats(trans_errors),
        rotation_metrics_deg=summary_stats(rot_errors) if rot_errors is not None else None,
        rpe_translation_metrics_m=rpe_t,
        rpe_rotation_metrics_deg=rpe_r,
        drift=drift,
    )
    return result, est_aligned, est_rot_aligned, trans_errors, rot_errors


def slice_optional_array(values: Optional[np.ndarray], start: int, end: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    return values[start:end]


def alignment_window_rmse(gt_pos: np.ndarray, est_pos: np.ndarray) -> float:
    if gt_pos.shape[0] < 2:
        return float("inf")
    _, rot, trans = align_umeyama(est_pos, gt_pos, with_scale=False)
    errors = np.linalg.norm((rot @ est_pos.T).T + trans - gt_pos, axis=1)
    return float(math.sqrt(np.mean(errors * errors)))


def select_alignment_window(
    args: argparse.Namespace,
    gt_pos: np.ndarray,
    gt_rot: Optional[np.ndarray],
    est_pos: np.ndarray,
    est_rot: Optional[np.ndarray],
    matched_times: np.ndarray,
    matched_indices: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, dict]:
    total_samples = int(gt_pos.shape[0])
    total_duration = float(matched_times[-1] - matched_times[0]) if matched_times.size > 1 else 0.0
    window_meta = {
        "enabled": bool(args.window_align),
        "applied": False,
        "reason": "disabled",
        "samples_before": total_samples,
        "samples_after": total_samples,
        "duration_before_s": total_duration,
        "duration_after_s": total_duration,
        "score_rmse_m": None,
        "start_index": 0,
        "end_index_exclusive": total_samples,
        "estimate_indices_start_end": [int(matched_indices[0]), int(matched_indices[-1])] if matched_indices.size else None,
        "time_range_s": [float(matched_times[0]), float(matched_times[-1])] if matched_times.size else None,
        "min_duration_s": float(args.window_min_duration_s),
        "min_matches": int(max(args.min_matches, args.window_min_matches)),
    }
    if not args.window_align:
        return gt_pos, gt_rot, est_pos, est_rot, matched_times, matched_indices, window_meta

    min_duration = max(0.0, float(args.window_min_duration_s))
    min_matches = int(max(args.min_matches, args.window_min_matches))
    if total_samples < min_matches:
        window_meta["reason"] = "not_enough_matches"
        return gt_pos, gt_rot, est_pos, est_rot, matched_times, matched_indices, window_meta

    best: Optional[Tuple[float, float, int, int, int]] = None
    for start in range(0, total_samples - min_matches + 1):
        end = start + min_matches
        while end <= total_samples:
            duration = float(matched_times[end - 1] - matched_times[start]) if end - start > 1 else 0.0
            if duration >= min_duration:
                break
            end += 1
        if end > total_samples:
            break
        for stop in range(end, total_samples + 1):
            duration = float(matched_times[stop - 1] - matched_times[start]) if stop - start > 1 else 0.0
            if duration < min_duration:
                continue
            score = alignment_window_rmse(gt_pos[start:stop], est_pos[start:stop])
            candidate = (score, -(stop - start), -int(duration * 1000.0), start, stop)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        window_meta["reason"] = "no_valid_window"
        return gt_pos, gt_rot, est_pos, est_rot, matched_times, matched_indices, window_meta

    score, _, _, start, stop = best
    window_meta.update(
        {
            "applied": bool(start > 0 or stop < total_samples),
            "reason": "best_contiguous_window",
            "samples_after": int(stop - start),
            "duration_after_s": float(matched_times[stop - 1] - matched_times[start]) if stop - start > 1 else 0.0,
            "score_rmse_m": float(score),
            "start_index": int(start),
            "end_index_exclusive": int(stop),
            "estimate_indices_start_end": [int(matched_indices[start]), int(matched_indices[stop - 1])],
            "time_range_s": [float(matched_times[start]), float(matched_times[stop - 1])],
        }
    )
    return (
        gt_pos[start:stop],
        slice_optional_array(gt_rot, start, stop),
        est_pos[start:stop],
        slice_optional_array(est_rot, start, stop),
        matched_times[start:stop],
        matched_indices[start:stop],
        window_meta,
    )


def generate_window_scan(
    args: argparse.Namespace,
    gt_pos: np.ndarray,
    gt_rot: Optional[np.ndarray],
    est_pos: np.ndarray,
    est_rot: Optional[np.ndarray],
    matched_times: np.ndarray,
    matched_indices: np.ndarray,
) -> List[WindowScanEntry]:
    duration = float(args.scan_window_duration_s)
    if duration <= 0.0 or gt_pos.shape[0] < max(3, args.min_matches):
        return []

    min_matches = max(3, int(args.scan_window_min_matches))
    step = max(1, int(args.scan_step_samples))
    total = int(gt_pos.shape[0])
    windows: List[WindowScanEntry] = []

    for start in range(0, total - min_matches + 1, step):
        end = start + min_matches
        while end <= total:
            cur_duration = float(matched_times[end - 1] - matched_times[start]) if end - start > 1 else 0.0
            if cur_duration >= duration:
                break
            end += 1
        if end > total:
            break

        gt_slice = gt_pos[start:end]
        est_slice = est_pos[start:end]
        gt_rot_slice = slice_optional_array(gt_rot, start, end)
        est_rot_slice = slice_optional_array(est_rot, start, end)
        time_slice = matched_times[start:end]

        se3, _, _, _, _ = evaluate_alignment(
            "se3",
            False,
            gt_slice,
            gt_rot_slice,
            est_slice,
            est_rot_slice,
            time_slice,
            args.rpe_delta_s,
            args.rpe_delta_samples,
        )
        sim3, _, _, _, _ = evaluate_alignment(
            "sim3",
            True,
            gt_slice,
            gt_rot_slice,
            est_slice,
            est_rot_slice,
            time_slice,
            args.rpe_delta_s,
            args.rpe_delta_samples,
        )
        windows.append(
            WindowScanEntry(
                start_index=int(start),
                end_index_exclusive=int(end),
                sample_count=int(end - start),
                duration_s=float(time_slice[-1] - time_slice[0]) if time_slice.size > 1 else 0.0,
                estimate_indices_start_end=[int(matched_indices[start]), int(matched_indices[end - 1])],
                time_range_s=[float(time_slice[0]), float(time_slice[-1])],
                se3_rmse_m=float(se3.translation_metrics_m["rmse"]),
                se3_mean_m=float(se3.translation_metrics_m["mean"]),
                se3_p95_m=float(se3.translation_metrics_m["p95"]),
                se3_path_error_pct=float(se3.drift["path_length_error_pct"]),
                se3_final_drift_pct=float(se3.drift["final_drift_pct_of_path"]),
                se3_rot_rmse_deg=None if not se3.rotation_metrics_deg else float(se3.rotation_metrics_deg["rmse"]),
                sim3_scale=float(sim3.scale),
                sim3_rmse_m=float(sim3.translation_metrics_m["rmse"]),
                sim3_mean_m=float(sim3.translation_metrics_m["mean"]),
                sim3_p95_m=float(sim3.translation_metrics_m["p95"]),
                sim3_path_error_pct=float(sim3.drift["path_length_error_pct"]),
                sim3_final_drift_pct=float(sim3.drift["final_drift_pct_of_path"]),
                sim3_rot_rmse_deg=None if not sim3.rotation_metrics_deg else float(sim3.rotation_metrics_deg["rmse"]),
            )
        )
    return windows


def prepare_matches(args: argparse.Namespace, gt: Trajectory, est: Trajectory) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    if args.matching == "index":
        n = min(gt.count, est.count)
        gt_pos, gt_rot, gt_times = resample_by_index(gt, n)
        est_pos, est_rot, est_times = resample_by_index(est, n)
        return gt_pos, gt_rot, est_pos, est_rot, gt_times, np.arange(n, dtype=int)

    if est.timestamps is None:
        raise ValueError("estimate trajectory has no timestamps; use --matching index")
    query_times = est.timestamps + args.time_offset_sec
    indices, gt_pos, gt_rot, matched_times = interpolate_ground_truth(gt, query_times, args.max_time_gap_ms * 1e-3)
    return gt_pos, gt_rot, est.positions[indices], None if est.rotations is None else est.rotations[indices], matched_times, indices


def choose_time_offset(args: argparse.Namespace, gt: Trajectory, est: Trajectory) -> float:
    if args.matching != "timestamp" or args.time_offset_search <= 0.0 or est.timestamps is None:
        return args.time_offset_sec
    original = args.time_offset_sec
    span = float(args.time_offset_search)
    step = max(1e-6, float(args.time_offset_step))
    offsets = np.arange(original - span, original + span + 0.5 * step, step)
    best: Optional[Tuple[float, int, float]] = None
    for offset in offsets:
        args.time_offset_sec = float(offset)
        try:
            gt_pos, _, est_pos, _, _, _ = prepare_matches(args, gt, est)
            if gt_pos.shape[0] < max(10, args.min_matches):
                continue
            _, rot, trans = align_umeyama(est_pos, gt_pos, with_scale=False)
            errors = np.linalg.norm((rot @ est_pos.T).T + trans - gt_pos, axis=1)
            score = float(math.sqrt(np.mean(errors * errors)))
            candidate = (score, -int(gt_pos.shape[0]), float(offset))
            if best is None or candidate < best:
                best = candidate
        except Exception:
            continue
    args.time_offset_sec = original
    if best is None:
        return original
    return best[2]


def write_matched_csv(
    path: Path,
    times: np.ndarray,
    gt_pos: np.ndarray,
    est_pos: np.ndarray,
    se3_est_pos: np.ndarray,
    se3_trans_errors: np.ndarray,
    se3_rot_errors: Optional[np.ndarray],
    sim3_est_pos: np.ndarray,
    sim3_trans_errors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "time_s",
            "gt_x",
            "gt_y",
            "gt_z",
            "estimate_x",
            "estimate_y",
            "estimate_z",
            "se3_x",
            "se3_y",
            "se3_z",
            "se3_trans_error_m",
            "se3_rot_error_deg",
            "sim3_x",
            "sim3_y",
            "sim3_z",
            "sim3_trans_error_m",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(gt_pos.shape[0]):
            writer.writerow(
                {
                    "time_s": f"{times[i]:.9f}",
                    "gt_x": f"{gt_pos[i, 0]:.9f}",
                    "gt_y": f"{gt_pos[i, 1]:.9f}",
                    "gt_z": f"{gt_pos[i, 2]:.9f}",
                    "estimate_x": f"{est_pos[i, 0]:.9f}",
                    "estimate_y": f"{est_pos[i, 1]:.9f}",
                    "estimate_z": f"{est_pos[i, 2]:.9f}",
                    "se3_x": f"{se3_est_pos[i, 0]:.9f}",
                    "se3_y": f"{se3_est_pos[i, 1]:.9f}",
                    "se3_z": f"{se3_est_pos[i, 2]:.9f}",
                    "se3_trans_error_m": f"{se3_trans_errors[i]:.9f}",
                    "se3_rot_error_deg": "" if se3_rot_errors is None else f"{se3_rot_errors[i]:.9f}",
                    "sim3_x": f"{sim3_est_pos[i, 0]:.9f}",
                    "sim3_y": f"{sim3_est_pos[i, 1]:.9f}",
                    "sim3_z": f"{sim3_est_pos[i, 2]:.9f}",
                    "sim3_trans_error_m": f"{sim3_trans_errors[i]:.9f}",
                }
            )


def set_equal_3d_axes(axis, points: np.ndarray) -> None:
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def make_plots(
    output_dir: Path,
    gt_pos: np.ndarray,
    raw_est_pos: np.ndarray,
    se3_est_pos: np.ndarray,
    sim3_est_pos: np.ndarray,
    times: np.ndarray,
    se3_errors: np.ndarray,
    sim3_errors: np.ndarray,
) -> List[str]:
    if plt is None:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], label="Ground truth", linewidth=2.2)
    ax.plot(se3_est_pos[:, 0], se3_est_pos[:, 1], se3_est_pos[:, 2], label="VINS aligned SE(3)", linewidth=1.6)
    ax.plot(sim3_est_pos[:, 0], sim3_est_pos[:, 1], sim3_est_pos[:, 2], label="VINS aligned Sim(3)", linewidth=1.2, alpha=0.75)
    ax.scatter(*gt_pos[0], s=40, label="GT start")
    ax.scatter(*gt_pos[-1], s=40, label="GT end")
    ax.set_title("3D trajectory comparison")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    set_equal_3d_axes(ax, np.vstack([gt_pos, se3_est_pos, sim3_est_pos]))
    fig.tight_layout()
    path = output_dir / "trajectory_3d.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    planes = [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]
    labels = ("X", "Y", "Z")
    for ax, (a, b, title) in zip(axes, planes):
        ax.plot(gt_pos[:, a], gt_pos[:, b], label="GT", linewidth=2.0)
        ax.plot(se3_est_pos[:, a], se3_est_pos[:, b], label="SE(3)", linewidth=1.4)
        ax.plot(sim3_est_pos[:, a], sim3_est_pos[:, b], label="Sim(3)", linewidth=1.0, alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel(f"{labels[a]} [m]")
        ax.set_ylabel(f"{labels[b]} [m]")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    path = output_dir / "trajectory_planes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    t_rel = times - times[0] if times.size else np.arange(se3_errors.size)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    axes[0].plot(t_rel, se3_errors * 1000.0, label="SE(3) APE")
    axes[0].plot(t_rel, sim3_errors * 1000.0, label="Sim(3) APE", alpha=0.8)
    axes[0].set_xlabel("Time from first match [s]")
    axes[0].set_ylabel("Translation error [mm]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].hist(se3_errors * 1000.0, bins=50, alpha=0.7, label="SE(3)")
    axes[1].hist(sim3_errors * 1000.0, bins=50, alpha=0.55, label="Sim(3)")
    axes[1].set_xlabel("Translation error [mm]")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    path = output_dir / "translation_errors.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    coord = ("X", "Y", "Z")
    for i, ax in enumerate(axes):
        ax.plot(t_rel, gt_pos[:, i], label=f"GT {coord[i]}", linewidth=1.8)
        ax.plot(t_rel, se3_est_pos[:, i], label=f"SE(3) {coord[i]}", linewidth=1.2)
        ax.plot(t_rel, raw_est_pos[:, i], label=f"Raw est {coord[i]}", linewidth=0.9, alpha=0.5)
        ax.set_ylabel(f"{coord[i]} [m]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time from first match [s]")
    fig.tight_layout()
    path = output_dir / "xyz_timeseries.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))
    return written


def fmt_m(value: float) -> str:
    return f"{value * 1000.0:.3f} mm"


def format_report(payload: dict) -> str:
    se3 = payload["alignments"]["se3"]
    sim3 = payload["alignments"]["sim3"]
    window = payload.get("window_selection", {})
    lines = [
        "# VINS Accuracy Evaluation",
        "",
        "## Inputs",
        f"- Ground truth: `{payload['inputs']['ground_truth']}`",
        f"- Estimate: `{payload['inputs']['estimate']}`",
        f"- Hand-eye: `{payload['inputs'].get('handeye') or 'not used'}`",
        f"- Ground-truth frame: `{payload['settings']['ground_truth_frame']}`",
        f"- Estimate frame: `{payload['settings']['estimate_frame']}` -> `{payload['settings'].get('estimate_target_frame', payload['settings']['estimate_frame'])}`",
        f"- Matching: `{payload['settings']['matching']}`",
        f"- Time offset applied to estimate: `{payload['settings']['time_offset_sec']:.6f} s`",
        "",
        "## Data Coverage",
        f"- Ground-truth samples: {payload['coverage']['ground_truth_samples']}",
        f"- Estimate samples: {payload['coverage']['estimate_samples']}",
        f"- Matched samples: {payload['coverage']['matched_samples']}",
        f"- Matched duration: {payload['coverage']['matched_duration_s']:.3f} s",
        "",
    ]
    if window.get("enabled"):
        lines.extend(
            [
                "## Window Alignment",
                f"- Mode: `{window.get('reason', 'unknown')}`",
                f"- Samples before / after: {window.get('samples_before')} / {window.get('samples_after')}",
                f"- Duration before / after: {window.get('duration_before_s', 0.0):.3f} s / {window.get('duration_after_s', 0.0):.3f} s",
                f"- Selected time range: `{window['time_range_s'][0]:.6f}` to `{window['time_range_s'][1]:.6f}` s" if window.get("time_range_s") else "- Selected time range: n/a",
                f"- Pre-alignment window score: {fmt_m(window['score_rmse_m'])}" if window.get("score_rmse_m") is not None else "- Pre-alignment window score: n/a",
                "",
            ]
        )
    lines.extend(
        [
            "## SE(3) Metric Evaluation",
            f"- APE RMSE: {fmt_m(se3['translation_metrics_m']['rmse'])}",
            f"- APE mean / median / p95 / max: "
            f"{fmt_m(se3['translation_metrics_m']['mean'])} / "
            f"{fmt_m(se3['translation_metrics_m']['median'])} / "
            f"{fmt_m(se3['translation_metrics_m']['p95'])} / "
            f"{fmt_m(se3['translation_metrics_m']['max'])}",
            f"- Final drift: {fmt_m(se3['drift']['final_position_error_m'])} "
            f"({se3['drift']['final_drift_pct_of_path']:.3f}% of GT path)",
            f"- Path length GT / estimate: {se3['drift']['gt_path_length_m']:.4f} m / "
            f"{se3['drift']['aligned_est_path_length_m']:.4f} m",
        ]
    )
    if se3.get("rotation_metrics_deg"):
        rot = se3["rotation_metrics_deg"]
        lines.append(f"- Rotation RMSE / p95 / max: {rot['rmse']:.3f} / {rot['p95']:.3f} / {rot['max']:.3f} deg")
    if se3.get("rpe_translation_metrics_m"):
        rpe = se3["rpe_translation_metrics_m"]
        lines.append(f"- RPE translation RMSE / p95: {fmt_m(rpe['rmse'])} / {fmt_m(rpe['p95'])}")
    lines.extend(
        [
            "",
            "## Sim(3) Shape Diagnostic",
            f"- Estimated scale: {sim3['scale']:.8f}",
            f"- Sim(3) APE RMSE / p95 / max: "
            f"{fmt_m(sim3['translation_metrics_m']['rmse'])} / "
            f"{fmt_m(sim3['translation_metrics_m']['p95'])} / "
            f"{fmt_m(sim3['translation_metrics_m']['max'])}",
            "",
            "## Interpretation Notes",
            "- Use SE(3) as the primary VIO metric because VINS should be metric scale.",
            "- Use Sim(3) only to diagnose trajectory shape after scale correction.",
            "- Large SE(3)-to-Sim(3) improvement usually means scale bias or frame/timestamp issues.",
        ]
    )
    return "\n".join(lines) + "\n"


def downsample_indices(count: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=int)
    return np.unique(np.linspace(0, count - 1, max_points).astype(int))


def bounds_for(*arrays: np.ndarray) -> dict:
    points = np.vstack([arr for arr in arrays if arr.size])
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "span": (maxs - mins).tolist(),
    }


def points_json(points: np.ndarray, indices: np.ndarray) -> List[List[float]]:
    return [[float(v) for v in points[i]] for i in indices]


def build_viewer_manifest(
    payload: dict,
    gt_pos: np.ndarray,
    est_pos: np.ndarray,
    se3_pos: np.ndarray,
    sim3_pos: np.ndarray,
    matched_times: np.ndarray,
    se3_errors: np.ndarray,
    sim3_errors: np.ndarray,
    max_points: int,
) -> dict:
    indices = downsample_indices(gt_pos.shape[0], max(50, int(max_points)))
    rel_times = matched_times - matched_times[0] if matched_times.size else np.arange(gt_pos.shape[0], dtype=float)
    se3_metrics = payload["alignments"]["se3"]["translation_metrics_m"]
    sim3_metrics = payload["alignments"]["sim3"]["translation_metrics_m"]
    se3_drift = payload["alignments"]["se3"]["drift"]
    sim3_drift = payload["alignments"]["sim3"]["drift"]
    summary_text = "\n".join(
        [
            f"SE(3) RMSE: {se3_metrics['rmse'] * 1000.0:.3f} mm",
            f"SE(3) mean / p95 / max: {se3_metrics['mean'] * 1000.0:.3f} / {se3_metrics['p95'] * 1000.0:.3f} / {se3_metrics['max'] * 1000.0:.3f} mm",
            f"SE(3) final drift: {se3_drift['final_position_error_m'] * 1000.0:.3f} mm",
            f"Sim(3) scale: {payload['alignments']['sim3']['scale']:.8f}",
            f"Sim(3) RMSE / p95: {sim3_metrics['rmse'] * 1000.0:.3f} / {sim3_metrics['p95'] * 1000.0:.3f} mm",
            f"Matched samples shown: {len(indices)} / {gt_pos.shape[0]}",
        ]
    )
    return {
        "title": "VINS vs Hand-eye Ground Truth",
        "inputs": payload["inputs"],
        "settings": payload["settings"],
        "coverage": payload["coverage"],
        "summary_text": summary_text,
        "gt": {
            "points": points_json(gt_pos, indices),
            "times": [float(rel_times[i]) for i in indices],
            "bounds": bounds_for(gt_pos, se3_pos, sim3_pos),
            "path_length_m": float(path_length(gt_pos)),
        },
        "estimate": {
            "points": points_json(est_pos, indices),
            "path_length_m": float(path_length(est_pos)),
        },
        "se3": {
            "points": points_json(se3_pos, indices),
            "errors": [float(se3_errors[i]) for i in indices],
            "scale": float(payload["alignments"]["se3"]["scale"]),
            "rmse_m": float(se3_metrics["rmse"]),
            "p95_m": float(se3_metrics["p95"]),
            "path_length_m": float(se3_drift["aligned_est_path_length_m"]),
        },
        "sim3": {
            "points": points_json(sim3_pos, indices),
            "errors": [float(sim3_errors[i]) for i in indices],
            "scale": float(payload["alignments"]["sim3"]["scale"]),
            "rmse_m": float(sim3_metrics["rmse"]),
            "p95_m": float(sim3_metrics["p95"]),
            "path_length_m": float(sim3_drift["aligned_est_path_length_m"]),
        },
    }


def serve_viewer(manifest: dict, host: str, port: int, open_browser: bool, quiet: bool) -> None:
    EvalViewerState.manifest = manifest
    server = ThreadingHTTPServer((host, port), EvalViewerHandler)
    server.quiet = quiet  # type: ignore[attr-defined]
    url = f"http://{host}:{port}/"
    print(f"VINS ground-truth comparison viewer: {url}")
    print("Press Ctrl-C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH, help="Robot ground-truth trajectory JSON")
    parser.add_argument("--estimate", type=Path, help="Estimated VINS trajectory. If omitted, auto-discover under --dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Episode dataset directory for auto-discovery")
    parser.add_argument("--side", choices=["left", "right"], default="right", help="Side used for auto-discovered pose_data")
    parser.add_argument("--handeye", type=Path, default=DEFAULT_HAND_EYE, help="handeye_result.yaml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for reports and plots")
    parser.add_argument("--ground-truth-frame", choices=["camera", "gripper"], default="camera")
    parser.add_argument(
        "--estimate-frame",
        choices=["camera", "imu", "base_link"],
        default="camera",
        help="Frame of the estimate pose. Use imu/base_link with --estimate-config to convert to stereo cam0.",
    )
    parser.add_argument(
        "--estimate-target-frame",
        choices=["camera", "gripper"],
        default="camera",
        help="Optional static transform applied after --estimate-frame conversion, e.g. cam0 -> gripper via hand-eye.",
    )
    parser.add_argument("--estimate-config", type=Path, help="Generated StereoIMU-vinsfusion.yaml for estimate frame conversion")
    parser.add_argument("--robot-pose-direction", choices=["base_to_gripper", "gripper_to_base"], default="base_to_gripper")
    parser.add_argument("--matching", choices=["timestamp", "index"], default="timestamp")
    parser.add_argument("--max-time-gap-ms", type=float, default=80.0, help="Max bracket gap for GT interpolation")
    parser.add_argument("--time-offset-sec", type=float, default=0.0, help="Offset added to estimate timestamps before matching")
    parser.add_argument("--time-offset-search", type=float, default=0.0, help="Search +/- seconds around --time-offset-sec")
    parser.add_argument("--time-offset-step", type=float, default=0.005, help="Time-offset search step in seconds")
    parser.add_argument("--window-align", action="store_true", help="Select the best contiguous overlap window after timestamp matching")
    parser.add_argument("--window-min-duration-s", type=float, default=12.0, help="Minimum duration for the selected overlap window")
    parser.add_argument("--window-min-matches", type=int, default=120, help="Minimum matched samples kept by window alignment")
    parser.add_argument("--default-estimate-dt", type=float, default=1.0 / 30.0, help="Fallback dt for untimestamped text trajectories")
    parser.add_argument("--rpe-delta-s", type=float, default=1.0, help="RPE interval when timestamps are available")
    parser.add_argument("--rpe-delta-samples", type=int, default=30, help="RPE interval fallback for index matching")
    parser.add_argument("--min-matches", type=int, default=20, help="Minimum paired samples required")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation")
    parser.add_argument("--serve", action="store_true", help="Start a local viewer after evaluation")
    parser.add_argument("--host", default=DEFAULT_SERVER_HOST, help="Viewer host")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT, help="Viewer port")
    parser.add_argument("--open-browser", action="store_true", help="Open the viewer in a browser")
    parser.add_argument("--quiet-server", action="store_true", help="Suppress HTTP logs from the viewer")
    parser.add_argument("--viewer-max-points", type=int, default=2500, help="Maximum points kept in the interactive viewer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gt_path = args.ground_truth.expanduser().resolve()
    dataset_dir = args.dataset.expanduser().resolve()
    estimate_path = args.estimate.expanduser().resolve() if args.estimate else discover_estimate(dataset_dir, args.side).resolve()
    handeye_path = args.handeye.expanduser().resolve() if args.handeye else None
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_robot_ground_truth(gt_path, handeye_path, args.ground_truth_frame, args.robot_pose_direction)
    est = load_estimate(estimate_path, args.default_estimate_dt)
    estimate_frame_note = "estimate frame unchanged"
    if args.estimate_frame != "camera":
        if args.estimate_config is None:
            raise ValueError("--estimate-config is required when --estimate-frame is imu or base_link")
        estimate_config = args.estimate_config.expanduser().resolve()
        t_est_cam0, estimate_frame_note = estimate_to_camera_transform(estimate_config, args.estimate_frame)
        est = transform_trajectory(est, t_est_cam0, "stereo_cam0", estimate_frame_note)
    if args.estimate_target_frame != "camera":
        t_est_target, estimate_target_note = estimate_post_transform(handeye_path, args.estimate_target_frame)
        est = transform_trajectory(est, t_est_target, args.estimate_target_frame, estimate_target_note)
        estimate_frame_note = f"{estimate_frame_note}; {estimate_target_note}"

    chosen_offset = choose_time_offset(args, gt, est)
    args.time_offset_sec = chosen_offset
    gt_pos, gt_rot, est_pos, est_rot, matched_times, matched_indices = prepare_matches(args, gt, est)
    gt_pos, gt_rot, est_pos, est_rot, matched_times, matched_indices, window_meta = select_alignment_window(
        args, gt_pos, gt_rot, est_pos, est_rot, matched_times, matched_indices
    )
    if gt_pos.shape[0] < args.min_matches:
        raise ValueError(f"only {gt_pos.shape[0]} matched samples; require at least {args.min_matches}")

    se3, se3_pos, se3_rot, se3_errors, se3_rot_errors = evaluate_alignment(
        "se3", False, gt_pos, gt_rot, est_pos, est_rot, matched_times, args.rpe_delta_s, args.rpe_delta_samples
    )
    sim3, sim3_pos, _, sim3_errors, _ = evaluate_alignment(
        "sim3", True, gt_pos, gt_rot, est_pos, est_rot, matched_times, args.rpe_delta_s, args.rpe_delta_samples
    )

    plots = [] if args.no_plots else make_plots(output_dir, gt_pos, est_pos, se3_pos, sim3_pos, matched_times, se3_errors, sim3_errors)
    matched_csv = output_dir / "matched_samples.csv"
    write_matched_csv(matched_csv, matched_times, gt_pos, est_pos, se3_pos, se3_errors, se3_rot_errors, sim3_pos, sim3_errors)

    payload = {
        "created_at_unix": time.time(),
        "inputs": {
            "ground_truth": str(gt_path),
            "estimate": str(estimate_path),
            "dataset": str(dataset_dir),
            "handeye": str(handeye_path) if handeye_path else None,
            "estimate_config": str(args.estimate_config.expanduser().resolve()) if args.estimate_config else None,
        },
        "settings": {
            "ground_truth_frame": args.ground_truth_frame,
            "estimate_frame": args.estimate_frame,
            "estimate_target_frame": args.estimate_target_frame,
            "estimate_frame_note": estimate_frame_note,
            "robot_pose_direction": args.robot_pose_direction,
            "matching": args.matching,
            "max_time_gap_ms": args.max_time_gap_ms,
            "time_offset_sec": args.time_offset_sec,
            "window_align": args.window_align,
            "window_min_duration_s": args.window_min_duration_s,
            "window_min_matches": args.window_min_matches,
            "rpe_delta_s": args.rpe_delta_s,
            "rpe_delta_samples": args.rpe_delta_samples,
        },
        "coverage": {
            "ground_truth_samples": gt.count,
            "ground_truth_duration_s": gt.duration_s,
            "ground_truth_path_length_m_full": gt.path_length_m,
            "estimate_samples": est.count,
            "estimate_duration_s": est.duration_s,
            "estimate_path_length_m_full": est.path_length_m,
            "matched_samples": int(gt_pos.shape[0]),
            "matched_duration_s": float(matched_times[-1] - matched_times[0]) if matched_times.size > 1 else 0.0,
            "matched_estimate_indices_start_end": [int(matched_indices[0]), int(matched_indices[-1])],
        },
        "window_selection": window_meta,
        "alignments": {"se3": asdict(se3), "sim3": asdict(sim3)},
        "outputs": {"matched_csv": str(matched_csv), "plots": plots},
    }

    json_path = output_dir / "metrics.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = output_dir / "REPORT.md"
    report_path.write_text(format_report(payload), encoding="utf-8")

    print(f"[OK] ground truth: {gt_path}")
    print(f"[OK] estimate:     {estimate_path}")
    print(f"[OK] matched samples: {gt_pos.shape[0]} / GT {gt.count} / EST {est.count}")
    print(f"[OK] time offset: {args.time_offset_sec:.6f} s")
    if window_meta.get("enabled"):
        print(
            "[OK] window: samples {} -> {} duration {:.3f}s -> {:.3f}s".format(
                window_meta["samples_before"],
                window_meta["samples_after"],
                window_meta["duration_before_s"],
                window_meta["duration_after_s"],
            )
        )
    print(
        "[SE3] APE rmse={:.3f} mm mean={:.3f} mm p95={:.3f} mm max={:.3f} mm".format(
            se3.translation_metrics_m["rmse"] * 1000.0,
            se3.translation_metrics_m["mean"] * 1000.0,
            se3.translation_metrics_m["p95"] * 1000.0,
            se3.translation_metrics_m["max"] * 1000.0,
        )
    )
    print(
        "[Sim3] scale={:.8f} APE rmse={:.3f} mm p95={:.3f} mm".format(
            sim3.scale,
            sim3.translation_metrics_m["rmse"] * 1000.0,
            sim3.translation_metrics_m["p95"] * 1000.0,
        )
    )
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {report_path}")
    print(f"[OK] wrote {matched_csv}")
    for plot in plots:
        print(f"[OK] wrote {plot}")

    if args.serve:
        viewer_manifest = build_viewer_manifest(
            payload,
            gt_pos,
            est_pos,
            se3_pos,
            sim3_pos,
            matched_times,
            se3_errors,
            sim3_errors,
            args.viewer_max_points,
        )
        serve_viewer(viewer_manifest, args.host, args.port, args.open_browser, args.quiet_server)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
