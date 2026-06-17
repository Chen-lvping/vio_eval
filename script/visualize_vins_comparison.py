#!/usr/bin/env python3
"""
Build a static VINS/DynaVINS trajectory comparison viewer from evaluation output.

The default inputs use the corrected stereo_right/cam0 evaluation directories.
Open the generated index.html directly in a browser, or serve the output folder
with any local HTTP server.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VINS_DIR = REPO_ROOT / "data/evaluation/camera_benchmark/vins_right_cam0_eval_0615"
DEFAULT_DYNAVINS_DIR = REPO_ROOT / "data/evaluation/camera_benchmark/dynavins_right_cam0_eval_0615"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/camera_benchmark/vins_dynavins_right_cam0_compare_0615"


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>stereo_right cam0 VINS comparison</title>
<style>
:root {
  color-scheme: dark;
  --bg: #101214;
  --panel: #171b1f;
  --line: #2a3138;
  --text: #eef2f5;
  --muted: #9aa7b2;
  --gt: #35d07f;
  --vins: #4ea1ff;
  --dynavins: #ffb84d;
  --sim: #b37cff;
  --err: #ff5f57;
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
  gap: 12px;
  padding: 16px 20px;
  border-bottom: 1px solid var(--line);
  background: #12161a;
}
h1 { margin: 0; font-size: 20px; font-weight: 650; }
.sub { color: var(--muted); }
.layout {
  display: grid;
  grid-template-columns: 360px minmax(0, 1fr);
  min-height: calc(100vh - 88px);
}
aside {
  border-right: 1px solid var(--line);
  background: var(--panel);
  padding: 16px;
  overflow: auto;
}
main { padding: 16px; overflow: hidden; }
.controls, .section { margin-bottom: 16px; }
.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
button, select {
  border: 1px solid var(--line);
  background: #20262c;
  color: var(--text);
  border-radius: 6px;
  padding: 7px 10px;
}
button.active { border-color: #6aa9ff; background: #22344a; }
label { color: var(--muted); }
input[type="range"] { width: 100%; }
.legend { display: grid; gap: 8px; }
.legend span { display: inline-flex; gap: 8px; align-items: center; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 7px 5px; border-bottom: 1px solid var(--line); text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; }
.canvas-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  grid-auto-rows: minmax(260px, 1fr);
  gap: 12px;
  height: calc(100vh - 126px);
}
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
  <h1>stereo_right / cam0 VINS accuracy comparison</h1>
  <div id="subtitle" class="sub"></div>
</header>
<div class="layout">
  <aside>
    <div class="controls">
      <div class="row">
        <button id="se3Btn" class="active">SE(3)</button>
        <button id="sim3Btn">Sim(3)</button>
        <button id="rawBtn">Raw</button>
      </div>
    </div>
    <div class="section">
      <label for="scrub">Sample</label>
      <input id="scrub" type="range" min="0" max="0" value="0">
      <div id="hud" class="sub"></div>
    </div>
    <div class="section legend">
      <span><i class="dot" style="background:var(--gt)"></i>Ground truth, hand-eye to stereo_right/cam0</span>
      <span><i class="dot" style="background:var(--vins)"></i>VINS</span>
      <span><i class="dot" style="background:var(--dynavins)"></i>DynaVINS</span>
      <span><i class="dot" style="background:var(--err)"></i>Current error connector</span>
    </div>
    <div class="section">
      <table id="metrics"></table>
    </div>
    <div class="section">
      <pre id="inputs"></pre>
    </div>
  </aside>
  <main>
    <div class="canvas-grid">
      <div class="plot"><div class="plot-title">XY plane</div><canvas id="xy"></canvas></div>
      <div class="plot"><div class="plot-title">XZ plane</div><canvas id="xz"></canvas></div>
      <div class="plot"><div class="plot-title">YZ plane</div><canvas id="yz"></canvas></div>
      <div class="plot"><div class="plot-title">Translation error over time</div><canvas id="err"></canvas></div>
    </div>
  </main>
</div>
<script>
let data = null;
let mode = "se3";
let activeIndex = 0;

const colors = {
  gt: getComputedStyle(document.documentElement).getPropertyValue("--gt").trim(),
  vins: getComputedStyle(document.documentElement).getPropertyValue("--vins").trim(),
  dynavins: getComputedStyle(document.documentElement).getPropertyValue("--dynavins").trim(),
  err: getComputedStyle(document.documentElement).getPropertyValue("--err").trim(),
  grid: "#263039",
  text: "#9aa7b2",
};

function pts(algo) {
  const key = mode === "raw" ? "raw" : mode;
  return algo.points.map(p => ({t: p.t, gt: p.gt, v: p[key], e: p[`${key}_error_m`] ?? 0}));
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

function boundsForPlane(axisA, axisB) {
  const values = [];
  for (const algo of data.algorithms) {
    for (const p of pts(algo)) {
      values.push([p.gt[axisA], p.gt[axisB]]);
      values.push([p.v[axisA], p.v[axisB]]);
    }
  }
  let minA = Infinity, maxA = -Infinity, minB = Infinity, maxB = -Infinity;
  for (const [a, b] of values) {
    minA = Math.min(minA, a); maxA = Math.max(maxA, a);
    minB = Math.min(minB, b); maxB = Math.max(maxB, b);
  }
  const padA = Math.max((maxA - minA) * 0.08, 0.02);
  const padB = Math.max((maxB - minB) * 0.08, 0.02);
  return {minA: minA - padA, maxA: maxA + padA, minB: minB - padB, maxB: maxB + padB};
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

function drawPlane(canvasId, axisA, axisB) {
  const canvas = document.getElementById(canvasId);
  const {ctx, w, h} = resizeCanvas(canvas);
  drawGrid(ctx, w, h);
  const b = boundsForPlane(axisA, axisB);
  const sx = a => 26 + ((a - b.minA) / (b.maxA - b.minA)) * (w - 52);
  const sy = bb => h - 26 - ((bb - b.minB) / (b.maxB - b.minB)) * (h - 52);
  const project = v => [sx(v[axisA]), sy(v[axisB])];

  const gt = pts(data.algorithms[0]).map(p => p.gt);
  line(ctx, gt, p => project(p), colors.gt, 2.4);
  const refLen = pts(data.algorithms[0]).length;
  for (const algo of data.algorithms) {
    const arr = pts(algo);
    line(ctx, arr, p => project(p.v), algo.color, 2);
    const progress = activeIndex / Math.max(1, refLen - 1);
    const idx = Math.min(Math.round(progress * (arr.length - 1)), arr.length - 1);
    const [gx, gy] = project(arr[idx].gt);
    const [ex, ey] = project(arr[idx].v);
    ctx.strokeStyle = colors.err;
    ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(ex, ey); ctx.stroke();
    ctx.fillStyle = algo.color;
    ctx.beginPath(); ctx.arc(ex, ey, 4, 0, Math.PI * 2); ctx.fill();
  }
}

function drawError() {
  const canvas = document.getElementById("err");
  const {ctx, w, h} = resizeCanvas(canvas);
  drawGrid(ctx, w, h);
  let maxE = 0;
  let minT = Infinity, maxT = -Infinity;
  for (const algo of data.algorithms) {
    for (const p of pts(algo)) {
      maxE = Math.max(maxE, p.e);
      minT = Math.min(minT, p.t);
      maxT = Math.max(maxT, p.t);
    }
  }
  maxE = Math.max(maxE, 0.001);
  const x = t => 34 + ((t - minT) / (maxT - minT || 1)) * (w - 62);
  const y = e => h - 28 - (e / maxE) * (h - 56);
  for (const algo of data.algorithms) {
    const arr = pts(algo);
    line(ctx, arr, p => [x(p.t), y(p.e)], algo.color, 2);
  }
  const ref = pts(data.algorithms[0]);
  const idx = Math.min(activeIndex, ref.length - 1);
  const xx = x(ref[idx].t);
  ctx.strokeStyle = colors.err;
  ctx.beginPath(); ctx.moveTo(xx, 18); ctx.lineTo(xx, h - 22); ctx.stroke();
  ctx.fillStyle = colors.text;
  ctx.fillText(`${(maxE * 1000).toFixed(1)} mm`, 8, 18);
  ctx.fillText("0", 10, h - 16);
}

function renderMetrics() {
  const table = document.getElementById("metrics");
  table.innerHTML = `<tr><th>Metric</th>${data.algorithms.map(a => `<th>${a.name}</th>`).join("")}</tr>`;
  const rows = [
    ["APE RMSE", "ape_rmse_mm", "mm"],
    ["APE p95", "ape_p95_mm", "mm"],
    ["Final drift", "final_drift_mm", "mm"],
    ["RPE RMSE", "rpe_rmse_mm", "mm"],
    ["Sim3 scale", "sim3_scale", ""],
  ];
  for (const [label, key, unit] of rows) {
    const cells = data.algorithms.map(a => `<td>${Number(a.metrics[key]).toFixed(key === "sim3_scale" ? 4 : 2)} ${unit}</td>`).join("");
    table.innerHTML += `<tr><td>${label}</td>${cells}</tr>`;
  }
}

function renderHud() {
  const ref = pts(data.algorithms[0]);
  const idx = Math.min(activeIndex, ref.length - 1);
  const parts = data.algorithms.map(algo => {
    const arr = pts(algo);
    const j = Math.min(Math.round((idx / Math.max(1, ref.length - 1)) * (arr.length - 1)), arr.length - 1);
    return `${algo.name}: ${(arr[j].e * 1000).toFixed(1)} mm`;
  });
  document.getElementById("hud").textContent = `t=${ref[idx].t.toFixed(3)} s | ${parts.join(" | ")}`;
}

function drawAll() {
  drawPlane("xy", 0, 1);
  drawPlane("xz", 0, 2);
  drawPlane("yz", 1, 2);
  drawError();
  renderHud();
}

function setMode(next) {
  mode = next;
  for (const id of ["se3Btn", "sim3Btn", "rawBtn"]) document.getElementById(id).classList.remove("active");
  document.getElementById(`${next}Btn`).classList.add("active");
  drawAll();
}

Promise.resolve(__VIEWER_DATA__).then(payload => {
  data = payload;
  document.getElementById("subtitle").textContent = payload.subtitle;
  document.getElementById("inputs").textContent = JSON.stringify(payload.inputs, null, 2);
  renderMetrics();
  const scrub = document.getElementById("scrub");
  scrub.max = Math.max(0, data.algorithms[0].points.length - 1);
  scrub.value = 0;
  scrub.addEventListener("input", () => { activeIndex = Number(scrub.value); drawAll(); });
  document.getElementById("se3Btn").onclick = () => setMode("se3");
  document.getElementById("sim3Btn").onclick = () => setMode("sim3");
  document.getElementById("rawBtn").onclick = () => setMode("raw");
  window.addEventListener("resize", drawAll);
  drawAll();
}).catch(err => {
  document.body.innerHTML = `<pre style="padding:24px;color:#ff5f57">${err.stack || err}</pre>`;
});
</script>
</body>
</html>
"""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stride_sample(rows: Sequence[dict], max_points: int) -> List[dict]:
    if len(rows) <= max_points:
        return list(rows)
    step = max(1, len(rows) // max_points)
    sampled = list(rows[::step])
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    return sampled


def point(row: dict) -> dict:
    gt = [float(row["gt_x"]), float(row["gt_y"]), float(row["gt_z"])]
    raw = [float(row["estimate_x"]), float(row["estimate_y"]), float(row["estimate_z"])]
    return {
        "t": float(row["time_s"]),
        "gt": gt,
        "raw": raw,
        "se3": [float(row["se3_x"]), float(row["se3_y"]), float(row["se3_z"])],
        "sim3": [float(row["sim3_x"]), float(row["sim3_y"]), float(row["sim3_z"])],
        "raw_error_m": math.dist(gt, raw),
        "se3_error_m": float(row["se3_trans_error_m"]),
        "sim3_error_m": float(row["sim3_trans_error_m"]),
    }


def algo_payload(name: str, color: str, eval_dir: Path, max_points: int) -> dict:
    metrics = read_json(eval_dir / "metrics.json")
    rows = stride_sample(read_rows(eval_dir / "matched_samples.csv"), max_points)
    se3 = metrics["alignments"]["se3"]
    sim3 = metrics["alignments"]["sim3"]
    tm = se3["translation_metrics_m"]
    rpe = se3["rpe_translation_metrics_m"]
    drift = se3["drift"]
    return {
        "name": name,
        "color": color,
        "eval_dir": str(eval_dir),
        "metrics": {
            "ape_rmse_mm": tm["rmse"] * 1000.0,
            "ape_p95_mm": tm["p95"] * 1000.0,
            "final_drift_mm": drift["final_position_error_m"] * 1000.0,
            "rpe_rmse_mm": rpe["rmse"] * 1000.0 if rpe else float("nan"),
            "sim3_scale": sim3["scale"],
        },
        "coverage": metrics["coverage"],
        "settings": metrics["settings"],
        "points": [point(row) for row in rows],
    }


def write_summary_csv(output_dir: Path, algorithms: Iterable[dict]) -> Path:
    path = output_dir / "comparison_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "ape_rmse_mm", "ape_p95_mm", "final_drift_mm", "rpe_rmse_mm", "sim3_scale"])
        for algo in algorithms:
            m = algo["metrics"]
            writer.writerow(
                [
                    algo["name"],
                    f"{m['ape_rmse_mm']:.6f}",
                    f"{m['ape_p95_mm']:.6f}",
                    f"{m['final_drift_mm']:.6f}",
                    f"{m['rpe_rmse_mm']:.6f}",
                    f"{m['sim3_scale']:.9f}",
                ]
            )
    return path


def build_payload(args: argparse.Namespace) -> dict:
    vins_dir = args.vins_dir.expanduser().resolve()
    dynavins_dir = args.dynavins_dir.expanduser().resolve()
    algorithms = [
        algo_payload("VINS", "#4ea1ff", vins_dir, args.max_points),
        algo_payload("DynaVINS", "#ffb84d", dynavins_dir, args.max_points),
    ]
    return {
        "subtitle": "Corrected frame: stereo_right/cam0. Ground truth uses hand-eye T_gripper_to_cam.",
        "inputs": {
            "vins_eval": str(vins_dir),
            "dynavins_eval": str(dynavins_dir),
            "frame": "stereo_right/cam0",
            "note": "VINS base_link->cam0, DynaVINS imu/body->cam0 before alignment.",
        },
        "algorithms": algorithms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vins-dir", type=Path, default=DEFAULT_VINS_DIR)
    parser.add_argument("--dynavins-dir", type=Path, default=DEFAULT_DYNAVINS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-points", type=int, default=2500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args)
    data_path = output_dir / "viewer_data.json"
    html_path = output_dir / "index.html"
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html = HTML.replace("__VIEWER_DATA__", json.dumps(payload, ensure_ascii=False))
    html_path.write_text(html, encoding="utf-8")
    summary_path = write_summary_csv(output_dir, payload["algorithms"])
    print(f"[OK] wrote {html_path}")
    print(f"[OK] wrote {data_path}")
    print(f"[OK] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
