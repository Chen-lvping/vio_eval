#!/usr/bin/env python3
"""Build a static viewer for a recorded robot trajectory JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot trajectory viewer</title>
<style>
:root {
  color-scheme: dark;
  --bg: #101214;
  --panel: #171b1f;
  --line: #2a3138;
  --text: #eef2f5;
  --muted: #9aa7b2;
  --traj: #4ea1ff;
  --rawx: #ff6b6b;
  --rawy: #7bd88f;
  --rawz: #6aa9ff;
  --qx: #f59e0b;
  --qy: #34d399;
  --qz: #60a5fa;
  --qw: #f472b6;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  display: grid;
  gap: 8px;
  padding: 16px 20px;
  border-bottom: 1px solid var(--line);
  background: #12161a;
}
h1 { margin: 0; font-size: 20px; font-weight: 650; }
.sub { color: var(--muted); }
.layout {
  display: grid;
  grid-template-columns: 360px minmax(0, 1fr);
  min-height: calc(100vh - 82px);
}
aside {
  border-right: 1px solid var(--line);
  background: var(--panel);
  padding: 16px;
  overflow: auto;
}
main { padding: 16px; overflow: hidden; }
.section { margin-bottom: 16px; }
label { color: var(--muted); }
input[type="range"] { width: 100%; }
.legend { display: grid; gap: 8px; }
.legend span { display: inline-flex; gap: 8px; align-items: center; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.canvas-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  grid-auto-rows: minmax(260px, 1fr);
  gap: 12px;
  height: calc(100vh - 120px);
}
.plot-wide { grid-column: 1 / -1; min-height: 320px; }
.plot {
  position: relative;
  min-height: 240px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #111519;
  overflow: hidden;
}
.plot-title {
  position: absolute;
  left: 10px;
  top: 8px;
  z-index: 1;
  color: var(--muted);
  font-size: 12px;
}
canvas { display: block; width: 100%; height: 100%; }
pre {
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--muted);
  background: #111519;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
  max-height: 320px;
  overflow: auto;
}
@media (max-width: 960px) {
  .layout { grid-template-columns: 1fr; }
  aside { border-right: 0; border-bottom: 1px solid var(--line); }
  .canvas-grid { grid-template-columns: 1fr; height: auto; }
  .plot { height: 300px; }
}
</style>
</head>
<body>
<header>
  <h1>Robot trajectory viewer</h1>
  <div id="subtitle" class="sub"></div>
</header>
<div class="layout">
  <aside>
    <div class="section">
      <label for="scrub">Sample</label>
      <input id="scrub" type="range" min="0" max="0" value="0">
      <div id="hud" class="sub"></div>
    </div>
    <div class="section legend">
      <span><i class="dot" style="background:var(--traj)"></i>Trajectory path</span>
      <span><i class="dot" style="background:#ff8a80"></i>Pose X axis</span>
      <span><i class="dot" style="background:#9effb0"></i>Pose Y axis</span>
      <span><i class="dot" style="background:#7fb5ff"></i>Pose Z axis</span>
    </div>
    <div class="section">
      <pre id="meta"></pre>
    </div>
    <div class="section">
      <pre id="poseInfo"></pre>
    </div>
  </aside>
  <main>
    <div class="canvas-grid">
      <div class="plot plot-wide"><div class="plot-title">3D trajectory and pose frames</div><canvas id="view3d"></canvas></div>
      <div class="plot"><div class="plot-title">XY plane</div><canvas id="xy"></canvas></div>
      <div class="plot"><div class="plot-title">XZ plane</div><canvas id="xz"></canvas></div>
      <div class="plot"><div class="plot-title">Position over time</div><canvas id="pos"></canvas></div>
      <div class="plot"><div class="plot-title">Orientation raw over time</div><canvas id="raw"></canvas></div>
      <div class="plot"><div class="plot-title">Quaternion over time</div><canvas id="quat"></canvas></div>
      <div class="plot"><div class="plot-title">SDK read duration over time</div><canvas id="dur"></canvas></div>
    </div>
  </main>
</div>
<script>
const data = __VIEWER_DATA__;
let activeIndex = 0;
let yaw = -0.7;
let pitch = 0.55;
let zoom3d = 1.0;
let dragging3d = false;
let lastMouse3d = [0, 0];

const colors = {
  traj: getComputedStyle(document.documentElement).getPropertyValue("--traj").trim(),
  rawx: getComputedStyle(document.documentElement).getPropertyValue("--rawx").trim(),
  rawy: getComputedStyle(document.documentElement).getPropertyValue("--rawy").trim(),
  rawz: getComputedStyle(document.documentElement).getPropertyValue("--rawz").trim(),
  qx: getComputedStyle(document.documentElement).getPropertyValue("--qx").trim(),
  qy: getComputedStyle(document.documentElement).getPropertyValue("--qy").trim(),
  qz: getComputedStyle(document.documentElement).getPropertyValue("--qz").trim(),
  qw: getComputedStyle(document.documentElement).getPropertyValue("--qw").trim(),
  grid: "#263039",
  text: "#9aa7b2",
};

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
    const [x, y] = pick(p);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function bounds3d() {
  let min = [Infinity, Infinity, Infinity];
  let max = [-Infinity, -Infinity, -Infinity];
  for (const p of data.samples) {
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], p.position[i]);
      max[i] = Math.max(max[i], p.position[i]);
    }
  }
  const center = min.map((v, i) => (v + max[i]) / 2);
  const span = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 0.05);
  return {center, span};
}

function project3d(p, view, w, h) {
  const x = p[0] - view.center[0];
  const y = p[1] - view.center[1];
  const z = p[2] - view.center[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  const y1 = cp * y - sp * z1;
  return [w / 2 + x1 * view.scale, h / 2 - y1 * view.scale];
}

function axisEndpoint(origin, rotation, axisIndex, length) {
  return [
    origin[0] + rotation[0][axisIndex] * length,
    origin[1] + rotation[1][axisIndex] * length,
    origin[2] + rotation[2][axisIndex] * length,
  ];
}

function drawPoseFrame(ctx, origin, rotation, view, w, h, length, alpha) {
  const start = project3d(origin, view, w, h);
  const axisColors = ["#ff8a80", "#9effb0", "#7fb5ff"];
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 1.2;
  for (let axis = 0; axis < 3; axis++) {
    const end = project3d(axisEndpoint(origin, rotation, axis, length), view, w, h);
    ctx.strokeStyle = axisColors[axis];
    ctx.beginPath(); ctx.moveTo(start[0], start[1]); ctx.lineTo(end[0], end[1]); ctx.stroke();
  }
  ctx.restore();
}

function draw3d() {
  const canvas = document.getElementById("view3d");
  const {ctx, w, h} = resizeCanvas(canvas);
  drawGrid(ctx, w, h);
  const b = bounds3d();
  const view = {center: b.center, span: b.span, scale: Math.min(w, h) * 0.72 / b.span * zoom3d};
  line(ctx, data.samples, p => project3d(p.position, view, w, h), colors.traj, 2.3);
  const stride = Math.max(1, Math.floor(data.samples.length / 40));
  for (let i = 0; i < data.samples.length; i += stride) {
    drawPoseFrame(ctx, data.samples[i].position, data.samples[i].rotation, view, w, h, view.span * 0.04, 0.35);
  }
  if (data.samples.length) {
    const cur = data.samples[Math.min(activeIndex, data.samples.length - 1)];
    drawPoseFrame(ctx, cur.position, cur.rotation, view, w, h, view.span * 0.07, 1.0);
    const [x, y] = project3d(cur.position, view, w, h);
    ctx.fillStyle = colors.traj;
    ctx.beginPath(); ctx.arc(x, y, 4.2, 0, Math.PI * 2); ctx.fill();
  }
}

function boundsForPlane(axisA, axisB) {
  let minA = Infinity, maxA = -Infinity, minB = Infinity, maxB = -Infinity;
  for (const p of data.samples) {
    minA = Math.min(minA, p.position[axisA]); maxA = Math.max(maxA, p.position[axisA]);
    minB = Math.min(minB, p.position[axisB]); maxB = Math.max(maxB, p.position[axisB]);
  }
  const padA = Math.max((maxA - minA) * 0.08, 0.02);
  const padB = Math.max((maxB - minB) * 0.08, 0.02);
  return {minA: minA - padA, maxA: maxA + padA, minB: minB - padB, maxB: maxB + padB};
}

function drawPlane(canvasId, axisA, axisB) {
  const canvas = document.getElementById(canvasId);
  const {ctx, w, h} = resizeCanvas(canvas);
  drawGrid(ctx, w, h);
  const b = boundsForPlane(axisA, axisB);
  const sx = a => 26 + ((a - b.minA) / (b.maxA - b.minA || 1)) * (w - 52);
  const sy = bb => h - 26 - ((bb - b.minB) / (b.maxB - b.minB || 1)) * (h - 52);
  line(ctx, data.samples, p => [sx(p.position[axisA]), sy(p.position[axisB])], colors.traj, 2.3);
  const cur = data.samples[Math.min(activeIndex, data.samples.length - 1)];
  if (cur) {
    ctx.fillStyle = colors.traj;
    ctx.beginPath(); ctx.arc(sx(cur.position[axisA]), sy(cur.position[axisB]), 4.2, 0, Math.PI * 2); ctx.fill();
  }
}

function drawSeries(canvasId, series, labels, palette, unitLabel) {
  const canvas = document.getElementById(canvasId);
  const {ctx, w, h} = resizeCanvas(canvas);
  drawGrid(ctx, w, h);
  let minY = Infinity;
  let maxY = -Infinity;
  let minT = data.samples[0]?.t ?? 0;
  let maxT = data.samples[data.samples.length - 1]?.t ?? 1;
  for (const arr of series) {
    for (const value of arr) {
      minY = Math.min(minY, value);
      maxY = Math.max(maxY, value);
    }
  }
  if (!isFinite(minY) || !isFinite(maxY)) {
    minY = -1;
    maxY = 1;
  }
  if (Math.abs(maxY - minY) < 1e-9) {
    maxY += 1;
    minY -= 1;
  }
  const x = t => 34 + ((t - minT) / (maxT - minT || 1)) * (w - 62);
  const y = v => h - 28 - ((v - minY) / (maxY - minY || 1)) * (h - 56);
  series.forEach((arr, idx) => {
    const mapped = data.samples.map((p, i) => ({t: p.t, v: arr[i]}));
    line(ctx, mapped, p => [x(p.t), y(p.v)], palette[idx], 2);
  });
  const cur = Math.min(activeIndex, Math.max(0, data.samples.length - 1));
  if (data.samples.length) {
    const xx = x(data.samples[cur].t);
    ctx.strokeStyle = "#ff5f57";
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(xx, 18); ctx.lineTo(xx, h - 22); ctx.stroke();
  }
  ctx.fillStyle = colors.text;
  ctx.fillText(`${maxY.toFixed(3)} ${unitLabel}`, 8, 18);
  ctx.fillText(`${minY.toFixed(3)} ${unitLabel}`, 8, h - 12);
  let legendX = 90;
  labels.forEach((label, idx) => {
    ctx.fillStyle = palette[idx];
    ctx.fillRect(legendX, 10, 10, 10);
    ctx.fillStyle = colors.text;
    ctx.fillText(label, legendX + 16, 18);
    legendX += 68;
  });
}

function renderHud() {
  const cur = data.samples[Math.min(activeIndex, data.samples.length - 1)];
  if (!cur) return;
  document.getElementById("hud").textContent =
    `sample ${activeIndex + 1}/${data.samples.length} | t=${cur.t.toFixed(3)} s | read=${(cur.read_duration_sec * 1000).toFixed(2)} ms`;
}

function fmtVec(v, n = 4) {
  return `[${v.map(x => Number(x).toFixed(n)).join(", ")}]`;
}

function renderPoseInfo() {
  const cur = data.samples[Math.min(activeIndex, data.samples.length - 1)];
  if (!cur) return;
  const lines = [];
  lines.push(`sample = ${activeIndex + 1}/${data.samples.length}`);
  lines.push(`timestamp = ${cur.timestamp.toFixed(6)}`);
  lines.push(`t = ${cur.t.toFixed(6)} s`);
  lines.push(`position_m = ${fmtVec(cur.position, 6)}`);
  lines.push(`quaternion_xyzw = ${fmtVec(cur.quaternion, 6)}`);
  lines.push(`orientation_raw = ${fmtVec(cur.orientation_raw, 6)}`);
  lines.push(`raw_pose = ${fmtVec(cur.raw_pose, 6)}`);
  lines.push(`read_duration_sec = ${cur.read_duration_sec.toFixed(6)}`);
  document.getElementById("poseInfo").textContent = lines.join("\n");
}

function drawAll() {
  draw3d();
  drawPlane("xy", 0, 1);
  drawPlane("xz", 0, 2);
  drawSeries(
    "pos",
    [
      data.samples.map(p => p.position[0]),
      data.samples.map(p => p.position[1]),
      data.samples.map(p => p.position[2]),
    ],
    ["x", "y", "z"],
    [colors.rawx, colors.rawy, colors.rawz],
    "m"
  );
  drawSeries(
    "raw",
    [
      data.samples.map(p => p.orientation_raw[0]),
      data.samples.map(p => p.orientation_raw[1]),
      data.samples.map(p => p.orientation_raw[2]),
    ],
    ["rx", "ry", "rz"],
    [colors.rawx, colors.rawy, colors.rawz],
    "rad"
  );
  drawSeries(
    "quat",
    [
      data.samples.map(p => p.quaternion[0]),
      data.samples.map(p => p.quaternion[1]),
      data.samples.map(p => p.quaternion[2]),
      data.samples.map(p => p.quaternion[3]),
    ],
    ["qx", "qy", "qz", "qw"],
    [colors.qx, colors.qy, colors.qz, colors.qw],
    ""
  );
  drawSeries(
    "dur",
    [data.samples.map(p => p.read_duration_sec * 1000)],
    ["read ms"],
    ["#ffb84d"],
    "ms"
  );
  renderHud();
  renderPoseInfo();
}

document.getElementById("subtitle").textContent = data.subtitle;
document.getElementById("meta").textContent = JSON.stringify(data.meta, null, 2);
const scrub = document.getElementById("scrub");
scrub.max = Math.max(0, data.samples.length - 1);
scrub.addEventListener("input", () => { activeIndex = Number(scrub.value); drawAll(); });
const view3d = document.getElementById("view3d");
view3d.addEventListener("mousedown", event => {
  dragging3d = true;
  lastMouse3d = [event.clientX, event.clientY];
});
window.addEventListener("mouseup", () => { dragging3d = false; });
window.addEventListener("mousemove", event => {
  if (!dragging3d) return;
  const dx = event.clientX - lastMouse3d[0];
  const dy = event.clientY - lastMouse3d[1];
  lastMouse3d = [event.clientX, event.clientY];
  yaw += dx * 0.006;
  pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.006));
  drawAll();
});
view3d.addEventListener("wheel", event => {
  event.preventDefault();
  zoom3d *= event.deltaY > 0 ? 0.9 : 1.1;
  zoom3d = Math.max(0.15, Math.min(20, zoom3d));
  drawAll();
}, {passive: false});
window.addEventListener("resize", drawAll);
drawAll();
</script>
</body>
</html>
"""


def quat_to_rot_xyzw(quaternion: list[float]) -> list[list[float]]:
    x, y, z, w = [float(v) for v in quaternion]
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm < 1e-12:
        raise ValueError("quaternion norm is too small")
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="robot trajectory JSON")
    parser.add_argument("--output-dir", type=Path, help="defaults to sibling viewer directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if len(samples) < 2:
        raise ValueError(f"{input_path} must contain at least two samples")

    if args.output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_viewer"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = float(samples[0]["timestamp"])
    viewer_samples = []
    for sample in samples:
        viewer_samples.append(
            {
                "timestamp": float(sample["timestamp"]),
                "t": float(sample["timestamp"]) - t0,
                "position": [float(v) for v in sample["position_m"]],
                "quaternion": [float(v) for v in sample["quaternion_xyzw"]],
                "rotation": quat_to_rot_xyzw(sample["quaternion_xyzw"]),
                "orientation_raw": [float(v) for v in sample.get("orientation_raw", [0.0, 0.0, 0.0])],
                "raw_pose": [float(v) for v in sample.get("raw_pose", sample["position_m"] + sample.get("orientation_raw", [0.0, 0.0, 0.0]))],
                "read_duration_sec": float(sample.get("read_duration_sec", 0.0)),
            }
        )

    viewer_payload = {
        "subtitle": f"{input_path.name} | frame={payload.get('meta', {}).get('frame', 'unknown')} | samples={len(viewer_samples)}",
        "meta": payload.get("meta", {}),
        "input": str(input_path),
        "samples": viewer_samples,
    }

    data_path = output_dir / "viewer_data.json"
    html_path = output_dir / "index.html"
    data_path.write_text(json.dumps(viewer_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(HTML.replace("__VIEWER_DATA__", json.dumps(viewer_payload, ensure_ascii=False)), encoding="utf-8")
    print(f"[OK] wrote {html_path}")
    print(f"[OK] wrote {data_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
