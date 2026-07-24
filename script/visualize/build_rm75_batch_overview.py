#!/usr/bin/env python3
"""Build a compact HTML overview for one RM75 batch evaluation."""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_DIR = REPO_ROOT / "data/evaluation/workbench/orbslam3_rm75_batch_eval_20260708_141821"
DEFAULT_OUTPUT_HTML = DEFAULT_BATCH_DIR / "overview.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    return parser.parse_args()


def read_rows(batch_dir: Path) -> list[dict[str, str]]:
    with (batch_dir / "batch_summary.csv").open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    parsed = float(value)
    return None if math.isnan(parsed) else parsed


def svg_bar_chart(rows: list[dict[str, str]]) -> str:
    w, h = 1240, 460
    left, right, top, bottom = 72, 24, 24, 56
    plot_w = w - left - right
    plot_h = h - top - bottom
    ok_rows = [r for r in rows if r.get("status") == "ok" and metric(r, "ape_translation_se3_rmse_mm") is not None]
    apes = [metric(r, "ape_translation_se3_rmse_mm") or 0.0 for r in ok_rows]
    max_ape = max(apes) if apes else 1.0
    max_ape = max(max_ape, 1.0)
    n = len(ok_rows)
    step = plot_w / max(1, n)
    bar_w = step * 0.72
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="APE bars">']
    parts.append('<style>.lbl{font:12px sans-serif;fill:#5c6670}.val{font:12px sans-serif;fill:#182028;font-weight:600}</style>')
    for tick in [0, 5, 10, 15, 20, 30, 40, 60]:
        if tick > max_ape * 1.05:
            continue
        y = top + plot_h - (tick / max_ape) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" stroke="rgba(24,32,40,0.12)" stroke-width="1"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="lbl">{tick}</text>')
    for idx, row in enumerate(ok_rows):
        ape = metric(row, "ape_translation_se3_rmse_mm") or 0.0
        rpe = metric(row, "rpe_translation_5cm_rmse_mm") or 0.0
        x = left + idx * step + (step - bar_w) / 2
        bar_h = (ape / max_ape) * plot_h
        y = top + plot_h - bar_h
        fill = "#197278" if ape <= 10.0 else "#d87b38"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" class="val">{ape:.2f}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{h-22:.1f}" text-anchor="middle" class="lbl">{html.escape(row["episode"][-4:])}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{h-36:.1f}" text-anchor="middle" class="lbl">{rpe:.2f} rpe</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_scatter(rows: list[dict[str, str]]) -> str:
    w, h = 700, 460
    left, right, top, bottom = 64, 20, 20, 58
    plot_w = w - left - right
    plot_h = h - top - bottom
    ok_rows = [r for r in rows if r.get("status") == "ok" and metric(r, "ape_translation_se3_rmse_mm") is not None]
    apes = [metric(r, "ape_translation_se3_rmse_mm") or 0.0 for r in ok_rows]
    rpes = [metric(r, "rpe_translation_5cm_rmse_mm") or 0.0 for r in ok_rows]
    max_x = max(apes) * 1.08 if apes else 1.0
    max_y = max(rpes) * 1.08 if rpes else 1.0
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="APE vs RPE scatter">']
    for tick in [0, 5, 10, 15, 20, 30, 40, 60]:
        if tick <= max_x * 1.05:
            x = left + (tick / max_x) * plot_w
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h-bottom}" stroke="rgba(24,32,40,0.10)" stroke-width="1"/>')
            parts.append(f'<text x="{x:.1f}" y="{h-16:.1f}" text-anchor="middle" style="font:12px sans-serif;fill:#5c6670">{tick}</text>')
    for tick in [0, 1, 2, 3, 5, 10, 20]:
        if tick <= max_y * 1.05:
            y = top + plot_h - (tick / max_y) * plot_h
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" stroke="rgba(24,32,40,0.10)" stroke-width="1"/>')
            parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" style="font:12px sans-serif;fill:#5c6670">{tick}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h - (10 / max_y) * plot_h:.1f}" x2="{w-right}" y2="{top + plot_h - (10 / max_y) * plot_h:.1f}" stroke="#197278" stroke-dasharray="6 6" stroke-width="1.5"/>')
    parts.append(f'<text x="{w-right}" y="{top + plot_h - (10 / max_y) * plot_h - 6:.1f}" text-anchor="end" style="font:12px sans-serif;fill:#197278">10 mm RPE ref</text>')
    for row in ok_rows:
        ape = metric(row, "ape_translation_se3_rmse_mm") or 0.0
        rpe = metric(row, "rpe_translation_5cm_rmse_mm") or 0.0
        x = left + (ape / max_x) * plot_w
        y = top + plot_h - (rpe / max_y) * plot_h
        fill = "#197278" if ape <= 10.0 else "#d87b38"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{fill}" stroke="white" stroke-width="2"/>')
    parts.append("</svg>")
    return "".join(parts)


def build_html(batch_dir: Path, rows: list[dict[str, str]]) -> str:
    ok = [r for r in rows if r.get("status") == "ok"]
    ape_vals = [metric(r, "ape_translation_se3_rmse_mm") for r in ok if metric(r, "ape_translation_se3_rmse_mm") is not None]
    rpe_vals = [metric(r, "rpe_translation_5cm_rmse_mm") for r in ok if metric(r, "rpe_translation_5cm_rmse_mm") is not None]
    ape_mean = statistics.mean(ape_vals) if ape_vals else 0.0
    rpe_mean = statistics.mean(rpe_vals) if rpe_vals else 0.0
    best = min(ok, key=lambda r: metric(r, "ape_translation_se3_rmse_mm") or 1e9) if ok else None
    worst = max(ok, key=lambda r: metric(r, "ape_translation_se3_rmse_mm") or -1e9) if ok else None
    under_10 = sum(1 for r in ok if (metric(r, "ape_translation_se3_rmse_mm") or 1e9) <= 10.0)

    rows_html = []
    for r in ok:
        ape = metric(r, "ape_translation_se3_rmse_mm") or 0.0
        rpe = metric(r, "rpe_translation_5cm_rmse_mm") or 0.0
        viewer = r.get("viewer_html", "")
        report = r.get("single_run_manifest", "")
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(r['episode'])}</td>"
            f"<td class='num'>{ape:.3f}</td>"
            f"<td class='num'>{rpe:.3f}</td>"
            f"<td>{html.escape(r.get('strict_sync_offset_sec',''))}</td>"
            f"<td><a href='{html.escape(viewer)}'>viewer</a></td>"
            f"<td><a href='{html.escape(report)}'>manifest</a></td>"
            "</tr>"
        )

    best_txt = f"{best['episode']} / {metric(best, 'ape_translation_se3_rmse_mm'):.3f} mm" if best else "n/a"
    worst_txt = f"{worst['episode']} / {metric(worst, 'ape_translation_se3_rmse_mm'):.3f} mm" if worst else "n/a"
    html_text = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RM75 Batch Overview</title>
<style>
:root {{ --bg:#f6f1e8; --ink:#182028; --muted:#5c6670; --panel:rgba(255,251,245,.88); --line:rgba(24,32,40,.12); --shadow:0 24px 70px rgba(92,62,24,.12); --warm:#d87b38; --cool:#197278; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); font:15px/1.5 "IBM Plex Sans","Segoe UI",sans-serif; background:linear-gradient(180deg,#f9f5ef 0%,#f1ebdf 100%); }}
a {{ color:inherit; }}
.shell {{ width:min(1440px, calc(100vw - 28px)); margin:16px auto 28px; }}
.hero,.card {{ border:1px solid var(--line); border-radius:24px; background:var(--panel); box-shadow:var(--shadow); }}
.hero {{ padding:24px 26px; }}
.eyebrow {{ text-transform:uppercase; letter-spacing:.12em; color:var(--muted); font-size:12px; }}
h1 {{ margin:6px 0 10px; font-size:clamp(28px,4vw,48px); line-height:1; }}
.sub {{ color:var(--muted); max-width:80ch; }}
.stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin-top:16px; }}
.stat {{ padding:16px; border:1px solid var(--line); border-radius:20px; background:rgba(255,255,255,.7); }}
.label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
.value {{ font-size:30px; line-height:1; margin-top:8px; }}
.grid {{ display:grid; grid-template-columns:1.5fr 1fr; gap:16px; margin-top:16px; }}
.panel {{ padding:16px; }}
.panel h2 {{ margin:0 0 8px; }}
svg {{ width:100%; height:auto; display:block; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:9px 8px; border-bottom:1px solid var(--line); }}
th {{ text-align:left; color:var(--muted); }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.table-wrap {{ overflow:auto; max-height: 740px; }}
.note {{ color:var(--muted); font-size:13px; margin-top:10px; }}
@media (max-width: 1024px) {{
  .stats {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  .grid {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body>
<div class="shell">
  <section class="hero">
    <div class="eyebrow">RM75 batch overview</div>
    <h1>{html.escape(batch_dir.name)}</h1>
    <div class="sub">This page summarizes the full batch, with APE/RPE charts and direct links to each episode viewer. The batch uses the current ORB mainline, final BA, and per-episode strict-sync scan results.</div>
    <div class="stats">
      <div class="stat"><div class="label">Episodes</div><div class="value">{len(ok)}</div></div>
      <div class="stat"><div class="label">APE mean</div><div class="value">{ape_mean:.3f} mm</div></div>
      <div class="stat"><div class="label">RPE mean</div><div class="value">{rpe_mean:.3f} mm</div></div>
      <div class="stat"><div class="label">APE <= 10 mm</div><div class="value">{under_10}/{len(ok)}</div></div>
    </div>
    <div class="note">Best APE: {html.escape(best_txt)} | Worst APE: {html.escape(worst_txt)}</div>
  </section>

  <section class="grid">
    <article class="card panel">
      <h2>APE bars</h2>
      {svg_bar_chart(rows)}
    </article>
    <article class="card panel">
      <h2>APE vs RPE</h2>
      {svg_scatter(rows)}
    </article>
    <article class="card panel" style="grid-column:1/-1;">
      <h2>Episode table</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Episode</th><th>APE mm</th><th>RPE mm</th><th>Strict sync</th><th>Viewer</th><th>Manifest</th></tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
      </div>
    </article>
  </section>
</div>
</body>
</html>
"""
    return html_text


def main() -> int:
    args = parse_args()
    batch_dir = args.batch_dir.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve()
    rows = read_rows(batch_dir)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(build_html(batch_dir, rows), encoding="utf-8")
    print(f"[OK] wrote {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
