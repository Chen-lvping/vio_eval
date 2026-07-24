#!/usr/bin/env python3
"""Build a static viewer for comparing two arbitrary trajectories.

The script accepts common trajectory formats:
- TUM / whitespace text: `t x y z [qx qy qz qw]`
- CSV with common timestamp / position column names
- JSON with either `samples` or a top-level list of samples

It produces a self-contained `index.html` in the output directory so the
result can be opened directly in a browser and shared with other people.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/trajectory_pair_view"


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>trajectory pair viewer</title>
<style>
:root{color-scheme:dark;--bg:#101214;--panel:#171b1f;--line:#2a3138;--text:#eef2f5;--muted:#9aa7b2;--ref:#35d07f;--raw:#4ea1ff;--se3:#ffb84d;--sim3:#b37cff;--err:#ff5f57;--grid:#263039}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{display:grid;gap:8px;padding:16px 20px;border-bottom:1px solid var(--line);background:#12161a}h1{margin:0;font-size:20px;font-weight:650}.sub{color:var(--muted)}.layout{display:grid;grid-template-columns:410px minmax(0,1fr);min-height:calc(100vh - 82px)}aside{border-right:1px solid var(--line);background:var(--panel);padding:16px;overflow:auto}main{padding:16px;overflow:hidden}.section{margin-bottom:16px}.row{display:flex;gap:8px;flex-wrap:wrap}.legend{display:grid;gap:8px}.legend span{display:inline-flex;gap:8px;align-items:center}.dot{width:10px;height:10px;border-radius:50%;display:inline-block}button{border:1px solid var(--line);background:#20262c;color:var(--text);border-radius:6px;padding:7px 10px}button.active{border-color:#6aa9ff;background:#22344a}input[type=range]{width:100%;accent-color:#6aa9ff}table{width:100%;border-collapse:collapse}th,td{padding:7px 5px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}.pose-table td{font-variant-numeric:tabular-nums}.pose-table td:first-child{color:var(--muted)}pre{white-space:pre-wrap;word-break:break-word;color:var(--muted);background:#111519;border:1px solid var(--line);border-radius:6px;padding:10px;max-height:260px;overflow:auto}.canvas-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-auto-rows:minmax(260px,1fr);gap:12px;height:calc(100vh - 120px)}.plot{position:relative;min-height:240px;border:1px solid var(--line);border-radius:8px;background:#111519;overflow:hidden}.wide{grid-column:1/-1;min-height:340px}.plot-title{position:absolute;left:10px;top:8px;z-index:1;color:var(--muted);font-size:12px}canvas{display:block;width:100%;height:100%}@media(max-width:960px){.layout{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid var(--line)}.canvas-grid{grid-template-columns:1fr;height:auto}.plot{height:300px}}
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div id="subtitle" class="sub"></div>
</header>
<div class="layout">
  <aside>
    <div class="section">
      <div class="row">
        <button id="rawBtn" class="active">Raw</button>
        <button id="se3Btn">SE(3)</button>
        <button id="sim3Btn">Sim(3)</button>
      </div>
    </div>
    <div class="section">
      <label for="timeSlider">时间点 <span id="timeValue"></span></label>
      <input id="timeSlider" type="range" min="0" max="1000" value="0" step="1">
      <table id="poseInfo" class="pose-table"></table>
    </div>
    <div class="section legend">
      <span><i class="dot" style="background:var(--ref)"></i><span id="refLegend"></span></span>
      <span><i class="dot" style="background:var(--raw)"></i><span id="rawLegend"></span></span>
      <span><i class="dot" style="background:var(--se3)"></i><span id="se3Legend"></span></span>
      <span><i class="dot" style="background:var(--sim3)"></i><span id="sim3Legend"></span></span>
    </div>
    <div class="section">
      <table id="stats"></table>
    </div>
    <div class="section">
      <pre id="inputs"></pre>
    </div>
  </aside>
  <main>
    <div class="canvas-grid">
      <div class="plot wide"><div class="plot-title">3D trajectory comparison, drag to rotate, wheel to zoom</div><canvas id="view3d"></canvas></div>
      <div class="plot"><div class="plot-title">XY plane</div><canvas id="xy"></canvas></div>
      <div class="plot"><div class="plot-title">XZ plane</div><canvas id="xz"></canvas></div>
      <div class="plot"><div class="plot-title">YZ plane</div><canvas id="yz"></canvas></div>
      <div class="plot"><div class="plot-title">Path length progress</div><canvas id="progress"></canvas></div>
    </div>
  </main>
</div>
<script>
const data = __VIEWER_DATA__;
let mode = "raw";
let yaw = -0.55;
let pitch = 0.48;
let zoom = 1.0;
let dragging = false;
let lastMouse = [0, 0];
let cursor = 0;

const colors = {
  ref: getComputedStyle(document.documentElement).getPropertyValue("--ref").trim(),
  raw: getComputedStyle(document.documentElement).getPropertyValue("--raw").trim(),
  se3: getComputedStyle(document.documentElement).getPropertyValue("--se3").trim(),
  sim3: getComputedStyle(document.documentElement).getPropertyValue("--sim3").trim(),
  grid: getComputedStyle(document.documentElement).getPropertyValue("--grid").trim(),
  text: "#9aa7b2",
};

function activeEstimate() {
  if (mode === "se3") return data.estimate_se3;
  if (mode === "sim3") return data.estimate_sim3;
  return data.estimate_raw;
}

function nearestSample(arr, t) {
  if (!arr.length) return null;
  let best = arr[0];
  let bestDelta = Math.abs(arr[0].t - t);
  for (const sample of arr) {
    const delta = Math.abs(sample.t - t);
    if (delta < bestDelta) { best = sample; bestDelta = delta; }
  }
  return best;
}

function currentSamples() {
  const ref = data.reference[Math.round(cursor * (data.reference.length - 1))];
  const est = nearestSample(activeEstimate(), ref.t);
  return {ref, est};
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {ctx, w: rect.width, h: rect.height};
}

function drawGrid(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  for (let i = 1; i < 5; i++) {
    const x = (w * i) / 5;
    const y = (h * i) / 5;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
}

function line(ctx, arr, pick, color, width = 2) {
  if (!arr.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  arr.forEach((p, i) => {
    const q = pick(p);
    if (i === 0) ctx.moveTo(q[0], q[1]);
    else ctx.lineTo(q[0], q[1]);
  });
  ctx.stroke();
}

function bounds3d() {
  const pts = data.reference.concat(activeEstimate());
  let mn = [Infinity, Infinity, Infinity];
  let mx = [-Infinity, -Infinity, -Infinity];
  for (const p of pts) {
    for (let i = 0; i < 3; i++) {
      mn[i] = Math.min(mn[i], p.p[i]);
      mx[i] = Math.max(mx[i], p.p[i]);
    }
  }
  const c = mn.map((v, i) => (v + mx[i]) / 2);
  const s = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 0.05);
  return {c, s};
}

function project(p, view, w, h) {
  const x = p[0] - view.c[0];
  const y = p[1] - view.c[1];
  const z = p[2] - view.c[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  const y1 = cp * y - sp * z1;
  return [w / 2 + x1 * view.scale, h / 2 - y1 * view.scale];
}

function drawEndpoint(ctx, p, view, w, h, color, r) {
  const q = project(p, view, w, h);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(q[0], q[1], r, 0, Math.PI * 2);
  ctx.fill();
}

function drawCurrent(ctx, p, view, w, h, color) {
  if (!p) return;
  const q = project(p, view, w, h);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(q[0], q[1], 8, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(q[0] - 12, q[1]); ctx.lineTo(q[0] + 12, q[1]); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(q[0], q[1] - 12); ctx.lineTo(q[0], q[1] + 12); ctx.stroke();
}

function drawCurrentScreen(ctx, q, color) {
  if (!q) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(q[0], q[1], 7, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(q[0] - 10, q[1]); ctx.lineTo(q[0] + 10, q[1]); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(q[0], q[1] - 10); ctx.lineTo(q[0], q[1] + 10); ctx.stroke();
}

function planeBounds(a, b) {
  const pts = data.reference.concat(activeEstimate()).map(x => x.p);
  let mnA = Infinity, mxA = -Infinity, mnB = Infinity, mxB = -Infinity;
  for (const p of pts) {
    mnA = Math.min(mnA, p[a]); mxA = Math.max(mxA, p[a]);
    mnB = Math.min(mnB, p[b]); mxB = Math.max(mxB, p[b]);
  }
  const pa = Math.max((mxA - mnA) * 0.08, 0.02);
  const pb = Math.max((mxB - mnB) * 0.08, 0.02);
  return {mnA: mnA - pa, mxA: mxA + pa, mnB: mnB - pb, mxB: mxB + pb};
}

function draw3d() {
  const {ctx, w, h} = resizeCanvas(document.getElementById("view3d"));
  drawGrid(ctx, w, h);
  const b = bounds3d();
  const view = {c: b.c, s: b.s, scale: Math.min(w, h) * 0.72 / b.s * zoom};
  line(ctx, data.reference, p => project(p.p, view, w, h), colors.ref, 2.5);
  const est = activeEstimate();
  const estColor = mode === "raw" ? colors.raw : mode === "se3" ? colors.se3 : colors.sim3;
  line(ctx, est, p => project(p.p, view, w, h), estColor, 2.1);
  drawEndpoint(ctx, data.reference[0].p, view, w, h, colors.ref, 5);
  drawEndpoint(ctx, data.reference[data.reference.length - 1].p, view, w, h, colors.ref, 7);
  drawEndpoint(ctx, est[0].p, view, w, h, estColor, 5);
  drawEndpoint(ctx, est[est.length - 1].p, view, w, h, estColor, 7);
  const current = currentSamples();
  drawCurrent(ctx, current.ref.p, view, w, h, colors.ref);
  drawCurrent(ctx, current.est.p, view, w, h, estColor);
}

function drawPlane(id, a, b) {
  const {ctx, w, h} = resizeCanvas(document.getElementById(id));
  drawGrid(ctx, w, h);
  const bb = planeBounds(a, b);
  const sx = v => 26 + (v - bb.mnA) / ((bb.mxA - bb.mnA) || 1) * (w - 52);
  const sy = v => h - 26 - (v - bb.mnB) / ((bb.mxB - bb.mnB) || 1) * (h - 52);
  const pr = p => [sx(p.p[a]), sy(p.p[b])];
  const est = activeEstimate();
  const estColor = mode === "raw" ? colors.raw : mode === "se3" ? colors.se3 : colors.sim3;
  line(ctx, data.reference, pr, colors.ref, 2.4);
  line(ctx, est, pr, estColor, 2.0);
  const current = currentSamples();
  drawCurrentScreen(ctx, pr(current.ref), colors.ref);
  drawCurrentScreen(ctx, pr(current.est), estColor);
}

function drawProgress() {
  const {ctx, w, h} = resizeCanvas(document.getElementById("progress"));
  drawGrid(ctx, w, h);
  const est = activeEstimate();
  const maxLen = Math.max(data.reference[data.reference.length - 1].s, est[est.length - 1].s, 0.001);
  line(ctx, data.reference, p => [30 + p.u * (w - 58), h - 28 - (p.s / maxLen) * (h - 56)], colors.ref, 2);
  const estColor = mode === "raw" ? colors.raw : mode === "se3" ? colors.se3 : colors.sim3;
  line(ctx, est, p => [30 + p.u * (w - 58), h - 28 - (p.s / maxLen) * (h - 56)], estColor, 2);
  const current = currentSamples();
  ctx.fillStyle = colors.ref;
  ctx.beginPath(); ctx.arc(30 + current.ref.u * (w - 58), h - 28 - (current.ref.s / maxLen) * (h - 56), 5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = estColor;
  ctx.beginPath(); ctx.arc(30 + current.est.u * (w - 58), h - 28 - (current.est.s / maxLen) * (h - 56), 5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = colors.text;
  ctx.fillText(maxLen.toFixed(3) + " m", 8, 18);
}

function poseCell(label, value) {
  return `<tr><td>${label}</td><td>${Number(value).toFixed(5)}</td></tr>`;
}

function renderPoseInfo() {
  const current = currentSamples();
  const error = Math.sqrt(current.ref.p.reduce((sum, value, index) => sum + (value - current.est.p[index]) ** 2, 0));
  const q = current.est.q || [0, 0, 0, 1];
  timeValue.textContent = `${current.ref.t.toFixed(3)} s`;
  poseInfo.innerHTML = [
    `<tr><th colspan="2">当前估计位姿</th></tr>`,
    poseCell("X (m)", current.est.p[0]), poseCell("Y (m)", current.est.p[1]), poseCell("Z (m)", current.est.p[2]),
    poseCell("Qx", q[0]), poseCell("Qy", q[1]), poseCell("Qz", q[2]), poseCell("Qw", q[3]),
    `<tr><th>参考时间</th><td>${current.ref.t.toFixed(3)} s</td></tr>`,
    `<tr><th>位置差</th><td>${(error * 1000).toFixed(2)} mm</td></tr>`,
  ].join("");
}

function render() {
  draw3d();
  drawPlane("xy", 0, 1);
  drawPlane("xz", 0, 2);
  drawPlane("yz", 1, 2);
  drawProgress();
  renderPoseInfo();
}

function setMode(nextMode) {
  mode = nextMode;
  rawBtn.classList.toggle("active", mode === "raw");
  se3Btn.classList.toggle("active", mode === "se3");
  sim3Btn.classList.toggle("active", mode === "sim3");
  render();
}

title.textContent = data.title;
subtitle.textContent = data.subtitle;
refLegend.textContent = `${data.reference_name} (reference)`;
rawLegend.textContent = `${data.estimate_name} (raw)`;
se3Legend.textContent = `${data.estimate_name} (SE(3) aligned)`;
sim3Legend.textContent = `${data.estimate_name} (Sim(3) aligned)`;
inputs.textContent = JSON.stringify(data.inputs, null, 2);
stats.innerHTML = Object.entries(data.stats).map(([k, v]) => `<tr><td>${k}</td><td>${typeof v === "number" ? v.toFixed(4) : v}</td></tr>`).join("");
rawBtn.onclick = () => setMode("raw");
se3Btn.onclick = () => setMode("se3");
sim3Btn.onclick = () => setMode("sim3");
timeSlider.oninput = event => { cursor = Number(event.target.value) / 1000; render(); };

const view3d = document.getElementById("view3d");
view3d.addEventListener("mousedown", e => { dragging = true; lastMouse = [e.clientX, e.clientY]; });
window.addEventListener("mouseup", () => dragging = false);
window.addEventListener("mousemove", e => {
  if (!dragging) return;
  const dx = e.clientX - lastMouse[0];
  const dy = e.clientY - lastMouse[1];
  lastMouse = [e.clientX, e.clientY];
  yaw += dx * 0.006;
  pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.006));
  render();
});
view3d.addEventListener("wheel", e => {
  e.preventDefault();
  zoom *= e.deltaY > 0 ? 0.9 : 1.1;
  zoom = Math.max(0.15, Math.min(20, zoom));
  render();
}, {passive: false});
window.addEventListener("resize", render);
render();
</script>
</body>
</html>
"""


def normalize_timestamp(value: float) -> float:
    value = float(value)
    av = abs(value)
    if av > 1e17:
        return value * 1e-9
    if av > 1e13:
        return value * 1e-6
    if av > 1e10:
        return value * 1e-3
    return value


def first_existing(row: dict, names: Sequence[str]) -> str | None:
    lower = {str(k).strip().lower(): k for k in row.keys()}
    for name in names:
        key = lower.get(name.lower())
        if key is not None and row.get(key) not in (None, ""):
            return str(row[key])
    return None


def normalize_quat(q: Sequence[float]) -> np.ndarray:
    arr = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("invalid quaternion")
    return arr / norm


def quat_xyzw_to_rot(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quat(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def transform_from_pose(position: Sequence[float], quaternion_xyzw: Sequence[float] | None = None) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, 3] = np.asarray(position, dtype=float)
    if quaternion_xyzw is not None:
        out[:3, :3] = quat_xyzw_to_rot(quaternion_xyzw)
    return out


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=float)
    if points.shape[0] == 1:
        return np.zeros(1, dtype=float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def resample_by_arclength(points: np.ndarray, count: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    if points.shape[0] == 1:
        return np.repeat(points[:1], count, axis=0)
    lengths = cumulative_lengths(points)
    total = float(lengths[-1])
    if total <= 1e-12:
        return np.repeat(points[:1], count, axis=0)
    src = lengths / total
    dst = np.linspace(0.0, 1.0, count)
    return np.column_stack([np.interp(dst, src, points[:, axis]) for axis in range(3)])


def align_umeyama(source: np.ndarray, target: np.ndarray, with_scale: bool) -> Tuple[np.ndarray, np.ndarray, float]:
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("alignment requires paired arrays with at least 3 points")
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    xs = source - mu_s
    xt = target - mu_t
    cov = (xt.T @ xs) / source.shape[0]
    u, sing, vt = np.linalg.svd(cov)
    d = np.ones(3, dtype=float)
    if np.linalg.det(u @ vt) < 0.0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt
    scale = 1.0
    if with_scale:
        var_s = float(np.mean(np.sum(xs * xs, axis=1)))
        if var_s > 1e-12:
            scale = float(np.sum(sing * d) / var_s)
    trans = mu_t - scale * (rot @ mu_s)
    return rot, trans, scale


def apply_transform(points: np.ndarray, rot: np.ndarray, trans: np.ndarray, scale: float) -> np.ndarray:
    return (scale * (rot @ points.T)).T + trans


def trajectory_payload(times: np.ndarray, points: np.ndarray, t0: float, quats: np.ndarray | None = None) -> List[dict]:
    lengths = cumulative_lengths(points)
    total = float(lengths[-1]) if lengths.size else 0.0
    denom = total if total > 1e-12 else 1.0
    return [
        {"t": float(t - t0), "p": points[i].tolist(), "q": (quats[i].tolist() if quats is not None else [0.0, 0.0, 0.0, 1.0]), "s": float(lengths[i]), "u": float(lengths[i] / denom)}
        for i, t in enumerate(times)
    ]


def stats_for(times: np.ndarray, points: np.ndarray) -> dict:
    return {
        "samples": int(points.shape[0]),
        "duration_s": float(times[-1] - times[0]) if times.size > 1 else 0.0,
        "path_length_m": path_length(points),
        "endpoint_distance_m": float(np.linalg.norm(points[-1] - points[0])) if points.shape[0] > 1 else 0.0,
        "x_span_m": float(np.ptp(points[:, 0])) if points.shape[0] else 0.0,
        "y_span_m": float(np.ptp(points[:, 1])) if points.shape[0] else 0.0,
        "z_span_m": float(np.ptp(points[:, 2])) if points.shape[0] else 0.0,
    }


def read_text_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: List[float] = []
    points: List[List[float]] = []
    quats: List[List[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            values = [float(item) for item in parts]
            if len(values) >= 8:
                times.append(normalize_timestamp(values[0]))
                points.append(values[1:4])
                quats.append(values[4:8])
            elif len(values) >= 4:
                times.append(normalize_timestamp(values[0]))
                points.append(values[1:4])
                quats.append([0.0, 0.0, 0.0, 1.0])
            elif len(values) >= 3:
                times.append(float(idx))
                points.append(values[:3])
                quats.append([0.0, 0.0, 0.0, 1.0])
    if len(points) < 2:
        raise ValueError(f"{path} does not contain enough pose rows")
    order = np.argsort(np.asarray(times, dtype=float))
    return np.asarray(times, dtype=float)[order], np.asarray(points, dtype=float)[order], np.asarray(quats, dtype=float)[order]


def read_csv_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: List[float] = []
    points: List[List[float]] = []
    quats: List[List[float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is not a header CSV")
        for idx, row in enumerate(reader):
            ts = first_existing(row, ["Timestamp_us", "timestamp_us", "Timestamp_ns", "timestamp_ns", "timestamp", "time", "t"])
            x = first_existing(row, ["X", "x", "tx", "position_x", "pos_x"])
            y = first_existing(row, ["Y", "y", "ty", "position_y", "pos_y"])
            z = first_existing(row, ["Z", "z", "tz", "position_z", "pos_z"])
            if None in (x, y, z):
                continue
            times.append(normalize_timestamp(float(ts)) if ts is not None else float(idx))
            points.append([float(x), float(y), float(z)])
            quat_values = [first_existing(row, [name]) for name in ("Quat_X", "Quat_Y", "Quat_Z", "Quat_W")]
            quats.append([float(value) for value in quat_values] if all(value is not None for value in quat_values) else [0.0, 0.0, 0.0, 1.0])
    if len(points) < 2:
        raise ValueError(f"{path} does not contain enough pose rows")
    order = np.argsort(np.asarray(times, dtype=float))
    return np.asarray(times, dtype=float)[order], np.asarray(points, dtype=float)[order], np.asarray(quats, dtype=float)[order]


def read_json_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples", data) if isinstance(data, dict) else data
    if not isinstance(samples, list):
        raise ValueError(f"{path} is not a supported JSON trajectory")
    times: List[float] = []
    points: List[List[float]] = []
    quats: List[List[float]] = []
    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        ts = sample.get("timestamp", sample.get("timestamp_s", sample.get("timestamp_us")))
        pos = sample.get("position_m", sample.get("position", sample.get("pos", sample.get("translation"))))
        if isinstance(pos, dict):
            lower = {str(k).strip().lower(): v for k, v in pos.items()}
            if all(k in lower for k in ("x", "y", "z")):
                xyz = [float(lower["x"]), float(lower["y"]), float(lower["z"])]
            else:
                continue
        elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
            xyz = [float(pos[0]), float(pos[1]), float(pos[2])]
        else:
            continue
        times.append(normalize_timestamp(float(ts)) if ts is not None else float(idx))
        points.append(xyz)
        quat = sample.get("quaternion_xyzw", sample.get("quat_xyzw", sample.get("quaternion")))
        if isinstance(quat, dict):
            quat = [quat.get(key, 0.0) for key in ("x", "y", "z", "w")]
        quats.append([float(value) for value in quat] if isinstance(quat, (list, tuple)) and len(quat) >= 4 else [0.0, 0.0, 0.0, 1.0])
    if len(points) < 2:
        raise ValueError(f"{path} does not contain enough pose rows")
    order = np.argsort(np.asarray(times, dtype=float))
    return np.asarray(times, dtype=float)[order], np.asarray(points, dtype=float)[order], np.asarray(quats, dtype=float)[order]


def load_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return read_csv_trajectory(path)
    if suffix in {".json"}:
        return read_json_trajectory(path)
    return read_text_trajectory(path)


def stride_indices(count: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=int)
    idx = np.linspace(0, count - 1, max_points, dtype=int)
    return np.unique(np.r_[idx, count - 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=Path, required=True, help="Reference trajectory file")
    parser.add_argument("--est", type=Path, required=True, help="Estimate trajectory file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ref-name", default="Reference")
    parser.add_argument("--est-name", default="Estimate")
    parser.add_argument("--title", default=None)
    parser.add_argument("--max-points", type=int, default=3000)
    parser.add_argument("--align-samples", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ref_path = args.ref.expanduser().resolve()
    est_path = args.est.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_t, ref_pos, ref_q = load_trajectory(ref_path)
    est_t, est_pos, est_q = load_trajectory(est_path)

    align_count = max(3, min(args.align_samples, ref_pos.shape[0], est_pos.shape[0]))
    ref_fit = resample_by_arclength(ref_pos, align_count)
    est_fit = resample_by_arclength(est_pos, align_count)

    se3_rot, se3_trans, se3_scale = align_umeyama(est_fit, ref_fit, with_scale=False)
    sim3_rot, sim3_trans, sim3_scale = align_umeyama(est_fit, ref_fit, with_scale=True)
    est_se3 = apply_transform(est_pos, se3_rot, se3_trans, se3_scale)
    est_sim3 = apply_transform(est_pos, sim3_rot, sim3_trans, sim3_scale)

    raw_fit_rmse = float(math.sqrt(np.mean(np.sum((est_fit - ref_fit) ** 2, axis=1))))
    se3_fit_rmse = float(math.sqrt(np.mean(np.sum((apply_transform(est_fit, se3_rot, se3_trans, se3_scale) - ref_fit) ** 2, axis=1))))
    sim3_fit_rmse = float(math.sqrt(np.mean(np.sum((apply_transform(est_fit, sim3_rot, sim3_trans, sim3_scale) - ref_fit) ** 2, axis=1))))

    ref_idx = stride_indices(ref_pos.shape[0], args.max_points)
    est_idx = stride_indices(est_pos.shape[0], args.max_points)
    t0 = min(float(ref_t[0]), float(est_t[0]))

    payload = {
        "title": args.title or f"{args.ref_name} vs {args.est_name} trajectory comparison",
        "subtitle": "Compare two trajectories locally. Raw / SE(3) / Sim(3) are shape-alignment views, not timestamp-matched metrics.",
        "reference_name": args.ref_name,
        "estimate_name": args.est_name,
        "inputs": {
            "reference_name": args.ref_name,
            "estimate_name": args.est_name,
            "reference": str(ref_path),
            "estimate": str(est_path),
            "reference_format": ref_path.suffix.lower() or "text",
            "estimate_format": est_path.suffix.lower() or "text",
            "alignment": f"Umeyama on {align_count} arc-length-normalized samples",
        },
        "stats": {
            "reference_samples": int(ref_pos.shape[0]),
            "reference_duration_s": stats_for(ref_t, ref_pos)["duration_s"],
            "reference_path_length_m": stats_for(ref_t, ref_pos)["path_length_m"],
            "estimate_samples": int(est_pos.shape[0]),
            "estimate_duration_s": stats_for(est_t, est_pos)["duration_s"],
            "estimate_path_length_m": stats_for(est_t, est_pos)["path_length_m"],
            "raw_shape_fit_rmse_m": raw_fit_rmse,
            "se3_shape_fit_rmse_m": se3_fit_rmse,
            "sim3_shape_fit_rmse_m": sim3_fit_rmse,
            "sim3_scale": float(sim3_scale),
        },
        "reference": trajectory_payload(ref_t[ref_idx], ref_pos[ref_idx], t0, ref_q[ref_idx]),
        "estimate_raw": trajectory_payload(est_t[est_idx], est_pos[est_idx], t0, est_q[est_idx]),
        "estimate_se3": trajectory_payload(est_t[est_idx], est_se3[est_idx], t0, est_q[est_idx]),
        "estimate_sim3": trajectory_payload(est_t[est_idx], est_sim3[est_idx], t0, est_q[est_idx]),
    }

    data_path = output_dir / "viewer_data.json"
    html_path = output_dir / "index.html"
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(HTML.replace("__VIEWER_DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    print(f"[OK] wrote {html_path}")
    print(f"[OK] wrote {data_path}")
    print(f"[INFO] raw shape-fit RMSE: {raw_fit_rmse:.4f} m")
    print(f"[INFO] SE3 shape-fit RMSE: {se3_fit_rmse:.4f} m")
    print(f"[INFO] Sim3 shape-fit RMSE: {sim3_fit_rmse:.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
