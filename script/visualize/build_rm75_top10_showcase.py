#!/usr/bin/env python3
"""Build ten full RM75 showcase bundles from the best current GT-backed runs."""

from __future__ import annotations

import argparse
import csv
import html
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = REPO_ROOT / "data/evaluation/workbench/rm75_best_all_gt_20260716/aggregate_summary.csv"
DEFAULT_OUTPUT = Path.home() / "Desktop/rm75_showcase_top10_20260717"
VIDEO_BUILDER = REPO_ROOT / "script/visualize/build_rm75_focus_showcase_video.py"


def find_episode_dir(episode: str) -> Path:
    matches = sorted((REPO_ROOT / "data/gripper").glob(f"*/{episode}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one source episode for {episode}, found: {matches}")
    return matches[0]


def load_selection(summary: Path, limit: int) -> list[dict[str, str]]:
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if float(row["ape_mm"]) < 10.0]
    rows.sort(key=lambda row: float(row["ape_mm"]))
    if len(rows) < limit:
        raise RuntimeError(f"only {len(rows)} episodes have APE < 10 mm; requested {limit}")
    return rows[:limit]


def write_index(output_dir: Path, rows: list[dict[str, str]]) -> None:
    items = "".join(
        f"<li><a href='{html.escape(row['episode'])}/index.html'>{html.escape(row['episode'])}</a>"
        f"<span>{float(row['ape_mm']):.3f} mm</span></li>"
        for row in rows
    )
    page = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>RM75 Top 10 Showcase</title><style>body{{margin:0;background:#10141c;color:#e8edf3;font:16px/1.5 system-ui,sans-serif}}main{{max-width:820px;margin:48px auto;padding:0 20px}}h1{{margin-bottom:8px}}p{{color:#aeb9c7}}ol{{padding:0;list-style:none;display:grid;gap:10px}}li{{background:#18202b;border:1px solid #2b394a;padding:14px 16px;border-radius:8px;display:flex;justify-content:space-between;gap:12px}}a{{color:#84d9c0;text-decoration:none;font-weight:650}}span{{color:#c6d0dc}}</style>
<main><h1>RM75 Top 10 Precision Showcase</h1><p>Current best GT-backed runs, selected by TCP APE SE3 RMSE below 10 mm.</p><ol>{items}</ol></main></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    args = parser.parse_args()

    summary = args.summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = load_selection(summary, args.limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "ape_mm", "pass_10mm", "offset", "viewer"])
        writer.writeheader()
        writer.writerows(rows)

    for number, row in enumerate(rows, start=1):
        episode = row["episode"]
        viewer_path = summary.parent / row["viewer"]
        eval_dir = viewer_path.parent
        episode_dir = find_episode_dir(episode)
        bundle_dir = output_dir / episode
        command = [
            sys.executable, str(VIDEO_BUILDER),
            "--episode-dir", str(episode_dir),
            "--eval-dir", str(eval_dir),
            "--output-dir", str(bundle_dir),
            "--duration-sec", str(args.duration_sec),
        ]
        print(f"[RUN {number}/{len(rows)}] {episode} APE={float(row['ape_mm']):.3f} mm", flush=True)
        subprocess.run(command, check=True)

    write_index(output_dir, rows)
    print(f"[DONE] showcase={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
