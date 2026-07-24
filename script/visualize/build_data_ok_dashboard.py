#!/usr/bin/env python3
"""Build a static dashboard for the curated ORB results under data/data_ok."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_OK_DIR = REPO_ROOT / "data/data_ok"
INDEX_CSV = DATA_OK_DIR / "index.csv"
OUTPUT_HTML = DATA_OK_DIR / "index.html"


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>data_ok ORB dashboard</title>
<style>
:root {
  --bg: #f6f1e8;
  --ink: #182028;
  --muted: #5c6670;
  --panel: rgba(255, 251, 245, 0.88);
  --line: rgba(24, 32, 40, 0.12);
  --shadow: 0 24px 70px rgba(92, 62, 24, 0.12);
  --warm: #d87b38;
  --warm-soft: #f3c9a6;
  --cool: #197278;
  --cool-soft: #9fd4d7;
  --accent: #b83b5e;
  --ok: #2f7d32;
  --grid: rgba(24, 32, 40, 0.08);
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  color: var(--ink);
  font: 15px/1.5 "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(216, 123, 56, 0.18), transparent 28%),
    radial-gradient(circle at top right, rgba(25, 114, 120, 0.16), transparent 30%),
    linear-gradient(180deg, #f9f5ef 0%, #f1ebdf 100%);
}
a { color: inherit; }
.shell {
  width: min(1280px, calc(100vw - 32px));
  margin: 24px auto 36px;
}
.hero {
  padding: 28px 30px;
  border: 1px solid var(--line);
  border-radius: 28px;
  background: linear-gradient(135deg, rgba(255,255,255,0.8), rgba(255,248,238,0.92));
  box-shadow: var(--shadow);
}
.eyebrow {
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 10px;
}
h1 {
  margin: 0;
  font-size: clamp(30px, 5vw, 56px);
  line-height: 0.98;
  letter-spacing: -0.04em;
  max-width: 11ch;
}
.subhead {
  margin-top: 14px;
  color: var(--muted);
  max-width: 72ch;
}
.chain {
  margin-top: 18px;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.chip {
  padding: 7px 11px;
  border-radius: 999px;
  background: rgba(24, 32, 40, 0.05);
  border: 1px solid rgba(24, 32, 40, 0.08);
  font-size: 13px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  gap: 16px;
  margin-top: 18px;
}
.card {
  border: 1px solid var(--line);
  border-radius: 24px;
  background: var(--panel);
  box-shadow: var(--shadow);
}
.metric-card {
  grid-column: span 3;
  padding: 18px 18px 16px;
}
.metric-label {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.metric-value {
  margin-top: 8px;
  font-size: 34px;
  line-height: 1;
  letter-spacing: -0.04em;
}
.metric-note {
  margin-top: 8px;
  color: var(--muted);
  font-size: 13px;
}
.panel {
  padding: 18px 18px 16px;
}
.panel h2 {
  margin: 0 0 4px;
  font-size: 20px;
}
.panel p {
  margin: 0 0 14px;
  color: var(--muted);
}
.scatter-panel { grid-column: span 7; }
.bar-panel { grid-column: span 5; }
.table-panel { grid-column: 1 / -1; }
canvas {
  display: block;
  width: 100%;
  height: 360px;
  border-radius: 18px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.58), rgba(255,255,255,0.9)),
    linear-gradient(180deg, rgba(216,123,56,0.04), rgba(25,114,120,0.04));
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
th, td {
  padding: 10px 8px;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}
th {
  text-align: left;
  color: var(--muted);
  font-weight: 600;
}
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.dataset-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.swatch {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
}
.links {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.links a {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 9px;
  border-radius: 999px;
  border: 1px solid var(--line);
  text-decoration: none;
  background: rgba(255,255,255,0.65);
}
.flag {
  display: inline-block;
  margin-top: 6px;
  padding: 3px 8px;
  border-radius: 999px;
  background: rgba(184, 59, 94, 0.1);
  color: var(--accent);
  font-size: 12px;
}
.foot {
  margin-top: 14px;
  color: var(--muted);
  font-size: 13px;
}
@media (max-width: 1080px) {
  .metric-card { grid-column: span 6; }
  .scatter-panel, .bar-panel { grid-column: 1 / -1; }
}
@media (max-width: 720px) {
  .shell { width: min(100vw - 20px, 1280px); margin-top: 10px; }
  .hero { padding: 20px; border-radius: 20px; }
  .metric-card { grid-column: 1 / -1; }
  .panel { padding: 16px; }
  canvas { height: 300px; }
  table { font-size: 13px; }
}
</style>
</head>
<body>
<div class="shell">
  <section class="hero">
    <div class="eyebrow">data_ok overview</div>
    <h1>Curated ORB results worth keeping close</h1>
    <div class="subhead">
      A single page for the current ORB chain that is repeatedly landing in the single-digit to teen-millimeter range.
      All rows below point back to the preserved episode, run output, selected trajectory, and full TCP evaluation viewer.
    </div>
    <div class="chain" id="chain"></div>
  </section>

  <section class="grid">
    <article class="card metric-card">
      <div class="metric-label">Best APE</div>
      <div class="metric-value" id="bestApe"></div>
      <div class="metric-note" id="bestApeNote"></div>
    </article>
    <article class="card metric-card">
      <div class="metric-label">Best RPE</div>
      <div class="metric-value" id="bestRpe"></div>
      <div class="metric-note" id="bestRpeNote"></div>
    </article>
    <article class="card metric-card">
      <div class="metric-label">Average APE</div>
      <div class="metric-value" id="avgApe"></div>
      <div class="metric-note">Across the six selected episodes</div>
    </article>
    <article class="card metric-card">
      <div class="metric-label">Average RPE</div>
      <div class="metric-value" id="avgRpe"></div>
      <div class="metric-note">Across the six selected episodes</div>
    </article>

    <article class="card panel scatter-panel">
      <h2>APE vs RPE</h2>
      <p>Lower-left is better. Color marks dataset family. Bubble size follows rotation APE.</p>
      <canvas id="scatter"></canvas>
    </article>

    <article class="card panel bar-panel">
      <h2>APE Ranking</h2>
      <p>Sorted absolute TCP translation error. The current keep-set is intentionally compact.</p>
      <canvas id="bars"></canvas>
    </article>

    <article class="card panel table-panel">
      <h2>Episode Index</h2>
      <p>Everything in this table links back to the preserved source directories under <code>data/data_ok</code>.</p>
      <table>
        <thead>
          <tr>
            <th>Episode</th>
            <th>Dataset</th>
            <th>Offset</th>
            <th>APE</th>
            <th>RPE</th>
            <th>Rot APE</th>
            <th>Links</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
      <div class="foot" id="footnote"></div>
    </article>
  </section>
</div>

<script>
const DATA = __DATA__;

function fmtMm(value) {
  return `${value.toFixed(3)} mm`;
}

function fmtDeg(value) {
  return `${value.toFixed(3)} deg`;
}

function fmtOffset(value) {
  const ms = value * 1000.0;
  const sign = ms >= 0 ? "+" : "";
  return `${sign}${ms.toFixed(0)} ms`;
}

function datasetColor(name) {
  return name === "gripper_data2" ? "#d87b38" : "#197278";
}

function sizeForRot(rot) {
  return 7 + Math.min(rot, 6.5) * 2.2;
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {ctx, width: rect.width, height: rect.height};
}

function drawAxes(ctx, width, height, pad, xTicks, yTicks, xMin, xMax, yMin, yMax) {
  ctx.strokeStyle = "rgba(24, 32, 40, 0.12)";
  ctx.fillStyle = "#5c6670";
  ctx.lineWidth = 1;
  ctx.font = '12px "IBM Plex Sans", "Segoe UI", sans-serif';

  for (const tick of xTicks) {
    const x = pad + ((tick - xMin) / (xMax - xMin)) * (width - pad * 2);
    ctx.beginPath();
    ctx.moveTo(x, pad - 4);
    ctx.lineTo(x, height - pad);
    ctx.stroke();
    ctx.fillText(String(tick), x - 8, height - pad + 18);
  }

  for (const tick of yTicks) {
    const y = height - pad - ((tick - yMin) / (yMax - yMin)) * (height - pad * 2);
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad + 4, y);
    ctx.stroke();
    ctx.fillText(String(tick), 8, y + 4);
  }

  ctx.strokeStyle = "rgba(24, 32, 40, 0.22)";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, height - pad);
  ctx.lineTo(width - pad, height - pad);
  ctx.stroke();

  ctx.fillStyle = "#182028";
  ctx.fillText("APE (mm)", width - pad - 50, height - 12);
  ctx.save();
  ctx.translate(14, pad + 28);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("RPE (mm)", 0, 0);
  ctx.restore();
}

function drawScatter() {
  const canvas = document.getElementById("scatter");
  const {ctx, width, height} = resizeCanvas(canvas);
  const pad = 40;
  const xMax = Math.max(...DATA.rows.map(row => row.ape_mm)) * 1.12;
  const yMax = Math.max(...DATA.rows.map(row => row.rpe_mm)) * 1.12;
  const xTicks = [0, 5, 10, 15, 20];
  const yTicks = [0, 5, 10, 15];

  ctx.clearRect(0, 0, width, height);
  drawAxes(ctx, width, height, pad, xTicks, yTicks, 0, xMax, 0, yMax);

  for (const row of DATA.rows) {
    const x = pad + (row.ape_mm / xMax) * (width - pad * 2);
    const y = height - pad - (row.rpe_mm / yMax) * (height - pad * 2);
    const r = sizeForRot(row.rot_deg);

    ctx.fillStyle = datasetColor(row.dataset);
    ctx.globalAlpha = 0.16;
    ctx.beginPath();
    ctx.arc(x, y, r + 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.globalAlpha = 1;
    ctx.fillStyle = datasetColor(row.dataset);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "rgba(24, 32, 40, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "#182028";
    ctx.font = '12px "IBM Plex Sans", "Segoe UI", sans-serif';
    ctx.fillText(row.short_episode, x + r + 6, y - 6);
  }
}

function drawBars() {
  const canvas = document.getElementById("bars");
  const {ctx, width, height} = resizeCanvas(canvas);
  const padTop = 24;
  const padLeft = 90;
  const padRight = 22;
  const padBottom = 26;
  const rows = [...DATA.rows].sort((a, b) => a.ape_mm - b.ape_mm);
  const maxApe = Math.max(...rows.map(row => row.ape_mm)) * 1.08;
  const trackHeight = Math.max(26, (height - padTop - padBottom) / rows.length - 10);

  ctx.clearRect(0, 0, width, height);
  ctx.font = '12px "IBM Plex Sans", "Segoe UI", sans-serif';

  rows.forEach((row, index) => {
    const y = padTop + index * ((height - padTop - padBottom) / rows.length);
    const barWidth = ((width - padLeft - padRight) * row.ape_mm) / maxApe;
    ctx.fillStyle = "rgba(24, 32, 40, 0.08)";
    ctx.fillRect(padLeft, y, width - padLeft - padRight, trackHeight);
    ctx.fillStyle = datasetColor(row.dataset);
    ctx.fillRect(padLeft, y, barWidth, trackHeight);
    ctx.fillStyle = "#182028";
    ctx.fillText(row.short_episode, 12, y + 17);
    ctx.fillText(`${row.ape_mm.toFixed(3)} mm`, padLeft + barWidth + 8, y + 17);
  });
}

function buildTable() {
  const tbody = document.getElementById("rows");
  const ordered = [...DATA.rows].sort((a, b) => a.ape_mm - b.ape_mm);
  tbody.innerHTML = "";

  for (const row of ordered) {
    const tr = document.createElement("tr");

    const notes = [];
    if (row.override_flag) notes.push('<span class="flag">selected eval differs from chain manifest</span>');

    tr.innerHTML = `
      <td>
        <strong>${row.episode}</strong><br>
        <span style="color:#5c6670">${row.source_tag}</span>
        ${notes.join("")}
      </td>
      <td>
        <span class="dataset-badge">
          <i class="swatch" style="background:${datasetColor(row.dataset)}"></i>
          ${row.dataset}
        </span>
      </td>
      <td class="num">${fmtOffset(row.offset_sec)}</td>
      <td class="num">${fmtMm(row.ape_mm)}</td>
      <td class="num">${fmtMm(row.rpe_mm)}</td>
      <td class="num">${fmtDeg(row.rot_deg)}</td>
      <td>
        <div class="links">
          <a href="${row.links.viewer}">viewer</a>
          <a href="${row.links.report}">report</a>
          <a href="${row.links.eval_dir}">eval dir</a>
          <a href="${row.links.run_dir}">run dir</a>
          <a href="${row.links.episode_dir}">episode</a>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  }
}

function hydrate() {
  const bestApe = [...DATA.rows].sort((a, b) => a.ape_mm - b.ape_mm)[0];
  const bestRpe = [...DATA.rows].sort((a, b) => a.rpe_mm - b.rpe_mm)[0];

  document.getElementById("bestApe").textContent = fmtMm(bestApe.ape_mm);
  document.getElementById("bestApeNote").textContent = `${bestApe.episode} / ${bestApe.dataset}`;
  document.getElementById("bestRpe").textContent = fmtMm(bestRpe.rpe_mm);
  document.getElementById("bestRpeNote").textContent = `${bestRpe.episode} / ${bestRpe.dataset}`;
  document.getElementById("avgApe").textContent = fmtMm(DATA.summary.avg_ape_mm);
  document.getElementById("avgRpe").textContent = fmtMm(DATA.summary.avg_rpe_mm);

  const chain = document.getElementById("chain");
  DATA.chain_tags.forEach(tag => {
    const span = document.createElement("span");
    span.className = "chip";
    span.textContent = tag;
    chain.appendChild(span);
  });

  const foot = document.getElementById("footnote");
  foot.textContent =
    `Datasets: ${DATA.summary.dataset_breakdown.map(item => `${item.dataset} avg APE ${item.avg_ape_mm.toFixed(3)} mm`).join(" · ")}. ` +
    "Bubble size in the scatter plot follows rotation APE.";

  buildTable();
  drawScatter();
  drawBars();
}

window.addEventListener("resize", () => {
  drawScatter();
  drawBars();
});

hydrate();
</script>
</body>
</html>
"""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_index_rows(index_csv: Path) -> list[dict]:
    with index_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rel_link(target: Path) -> str:
    return target.relative_to(DATA_OK_DIR).as_posix()


def build_payload() -> dict:
    rows = []
    csv_rows = read_index_rows(INDEX_CSV)
    chain_tags = None

    for row in csv_rows:
        episode_dir = DATA_OK_DIR / row["episode"]
        chain_manifest = read_json(episode_dir / "chain_manifest.json")
        selected_manifest = read_json(episode_dir / "selected_estimate_manifest.json")

        offset_sec = float(selected_manifest["time_offset_sec"])
        chain_offset_sec = chain_manifest.get("strict_sync_offset_sec")
        override_flag = (
            chain_offset_sec is not None
            and abs(float(chain_offset_sec) - offset_sec) > 1e-9
        )

        if chain_tags is None:
            chain_tags = [
                chain_manifest["mode"],
                chain_manifest["camera_rig"],
                chain_manifest["feature_preset"],
                f"fastinit={chain_manifest['imu_fast_init']}",
                chain_manifest["vins_noise_mode"],
                f"smooth w{chain_manifest['smooth_window']} p{chain_manifest['smooth_passes']}",
                "strict-sync",
                "TCP eval + viewer",
            ]

        rows.append(
            {
                "episode": row["episode"],
                "short_episode": row["episode"].split("_")[-1],
                "dataset": row["dataset"],
                "ape_mm": float(row["ape_mm"]),
                "rpe_mm": float(row["rpe_mm"]),
                "rot_deg": float(chain_manifest["summary_rmse"]["ape_rotation_se3_rmse"]),
                "offset_sec": offset_sec,
                "offset_tag": row["selected_offset_tag"],
                "source_tag": (
                    f"{chain_manifest['camera_rig']} / {chain_manifest['mode']} / "
                    f"{chain_manifest['feature_preset']} / fastinit={chain_manifest['imu_fast_init']}"
                ),
                "override_flag": override_flag,
                "links": {
                    "viewer": rel_link(episode_dir / "eval_output/index.html"),
                    "report": rel_link(episode_dir / "eval_output/REPORT.md"),
                    "eval_dir": rel_link(episode_dir / "eval_output"),
                    "run_dir": rel_link(episode_dir / "run_output"),
                    "episode_dir": rel_link(episode_dir / "original_episode"),
                },
            }
        )

    dataset_breakdown = []
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        dataset_breakdown.append(
            {
                "dataset": dataset,
                "count": len(subset),
                "avg_ape_mm": mean(row["ape_mm"] for row in subset),
                "avg_rpe_mm": mean(row["rpe_mm"] for row in subset),
            }
        )

    return {
        "rows": rows,
        "summary": {
            "count": len(rows),
            "avg_ape_mm": mean(row["ape_mm"] for row in rows),
            "avg_rpe_mm": mean(row["rpe_mm"] for row in rows),
            "dataset_breakdown": dataset_breakdown,
        },
        "chain_tags": chain_tags or [],
    }


def main() -> None:
    payload = build_payload()
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=True))
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
