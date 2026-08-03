#!/usr/bin/env python3
"""Build one GT-vs-estimate viewer for every recovered RM75 baseline."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data/evaluation/recovered/rm75_main_clean_portfolio_trajectories_20260730"
DEFAULT_AUDIT = ROOT / "data/evaluation/workbench/rm75_historical_input_audit_20260731/input_audit.csv"
DEFAULT_OUTPUT = ROOT / "data/evaluation/workbench/rm75_recovered_baseline_viewers_20260731"
VIEWER = ROOT / "script/visualize_single_tcp_trajectory_3d.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-points", type=int, default=3000)
    return parser.parse_args()


def audit_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric(path: Path, name: str) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] == name:
                return float(row["rmse"])
    raise KeyError(f"missing {name} in {path}")


def count_tum_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "episode", "tag", "ape_translation_se3_rmse_mm", "pairs",
        "fresh_input_status", "source_dir", "viewer_html",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_overview(path: Path, rows: list[dict[str, object]]) -> None:
    body = []
    for row in rows:
        ready = row["fresh_input_status"] == "ready"
        status = "fresh-ready" if ready else "historical-only"
        status_class = "ready" if ready else "blocked"
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['episode']))}</td>"
            f"<td>{html.escape(str(row['tag']))}</td>"
            f"<td class='number'>{float(row['ape_translation_se3_rmse_mm']):.3f} mm</td>"
            f"<td class='number'>{int(row['pairs'])}</td>"
            f"<td><span class='status {status_class}'>{status}</span></td>"
            f"<td><a href='{html.escape(str(row['viewer_html']))}'>Open 3D viewer</a></td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RM75 recovered baseline trajectories</title>
<style>
:root {{ color-scheme: light; --ink:#1b2428; --muted:#66757b; --line:#d9e1e3; --accent:#087f5b; --warn:#a04422; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#f5f7f7; color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ background:#fff; border-bottom:1px solid var(--line); padding:22px max(18px,calc((100vw - 1180px)/2)); }}
h1 {{ margin:0 0 6px; font-size:24px; letter-spacing:0; }}
p {{ margin:0; color:var(--muted); }}
main {{ max-width:1180px; margin:0 auto; padding:22px 18px 34px; }}
.summary {{ display:flex; gap:24px; margin-bottom:16px; color:var(--muted); flex-wrap:wrap; }}
.summary strong {{ color:var(--ink); }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); background:#fff; }}
table {{ width:100%; border-collapse:collapse; min-width:860px; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ background:#eef2f2; color:var(--muted); font-size:12px; }}
tr:last-child td {{ border-bottom:0; }}
.number {{ text-align:right; font-variant-numeric:tabular-nums; }}
.status {{ font-size:12px; font-weight:600; }}
.ready {{ color:var(--accent); }}
.blocked {{ color:var(--warn); }}
a {{ color:#075d9b; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
</style>
</head>
<body>
<header><h1>RM75 18 条恢复基准轨迹</h1><p>机器人 TCP 真值与历史 ORB 结果，统一 SE(3) 对齐口径。</p></header>
<main>
<div class="summary"><span><strong>{len(rows)}</strong> episodes</span><span><strong>{sum(r['fresh_input_status'] == 'ready' for r in rows)}</strong> fresh-ready</span><span><strong>{sum(r['fresh_input_status'] != 'ready' for r in rows)}</strong> historical-only</span></div>
<div class="table-wrap"><table><thead><tr><th>Episode</th><th>Dataset</th><th class="number">APE SE(3)</th><th class="number">Pairs</th><th>Status</th><th>Comparison</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>
</main>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    audit_csv = args.audit_csv.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    rows = audit_rows(audit_csv)
    if len(rows) != 18:
        raise RuntimeError(f"expected 18 audited baselines, found {len(rows)}")
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for audit in rows:
        episode = audit["episode"]
        tag = audit["tag"]
        source = source_root / tag
        gt = source / "gt_tcp_matched.tum"
        estimate = source / "vio_tcp_matched.tum"
        source_summary = source / "viewer_summary.csv"
        for required in (gt, estimate, source_summary):
            if not required.is_file():
                raise FileNotFoundError(required)

        episode_output = output_root / episode.replace("/", "_")
        adapter = episode_output / "input"
        viewer_output = episode_output / "viewer"
        adapter.mkdir(parents=True, exist_ok=True)
        viewer_output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gt, adapter / gt.name)
        shutil.copy2(estimate, adapter / estimate.name)
        shutil.copy2(source_summary, adapter / "summary.csv")

        command = [
            sys.executable, str(VIEWER), "--eval-dir", str(adapter),
            "--output-dir", str(viewer_output), "--max-points", str(args.max_points),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        pair_count = count_tum_rows(gt)
        if pair_count != count_tum_rows(estimate):
            raise RuntimeError(f"mismatched GT/result row counts for {episode}")
        results.append(
            {
                "episode": episode,
                "tag": tag,
                "ape_translation_se3_rmse_mm": metric(source_summary, "ape_translation_se3"),
                "pairs": pair_count,
                "fresh_input_status": audit["status"],
                "source_dir": str(source),
                "viewer_html": f"{episode.replace('/', '_')}/viewer/index.html",
            }
        )

    write_csv(output_root / "summary.csv", results)
    write_overview(output_root / "index.html", results)
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Recovered historical RM75 GT-vs-ORB trajectory viewers",
        "source_root": str(source_root),
        "audit_csv": str(audit_csv),
        "viewer_script": str(VIEWER),
        "coordinate_frame": "robot TCP",
        "alignment_modes": ["raw", "se3", "sim3", "anchor-start"],
        "entries": results,
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"[DONE] viewers={len(results)} output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
