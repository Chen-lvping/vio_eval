#!/usr/bin/env python3
"""Aggregate all current-best RM75 batch outputs into one HTML overview."""
from __future__ import annotations

import csv
import html
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2] / "data/evaluation/workbench/rm75_best_all_gt_20260716"
    rows = {}
    for summary in sorted(root.glob("**/batch_summary.csv")):
        batch = summary.parent
        for row in csv.DictReader(summary.open()):
            if row.get("status") != "ok":
                continue
            tag = row["episode"]
            eval_dir = batch / tag / "eval"
            try:
                ape = float(row["ape_translation_se3_rmse_mm"])
            except (KeyError, TypeError, ValueError):
                continue
            rows[tag] = {
                "episode": tag,
                "ape_mm": ape,
                "pass_10mm": row.get("pass_10mm", ""),
                "offset": row.get("strict_sync_offset_sec", "0"),
                "viewer": str(eval_dir.relative_to(root) / "viewer_prior/index.html") if (eval_dir / "viewer_prior/index.html").is_file() else "",
            }
    ordered = sorted(rows.values(), key=lambda row: row["episode"])
    (root / "aggregate_summary.csv").parent.mkdir(parents=True, exist_ok=True)
    with (root / "aggregate_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "ape_mm", "pass_10mm", "offset", "viewer"])
        writer.writeheader(); writer.writerows(ordered)
    if not ordered:
        raise SystemExit("no completed evaluated episodes found")
    mean = sum(row["ape_mm"] for row in ordered) / len(ordered)
    passed = sum(row["ape_mm"] <= 10.0 for row in ordered)
    max_ape = max(row["ape_mm"] for row in ordered)
    rows_html = []
    for row in ordered:
        width = max(4.0, row["ape_mm"] / max_ape * 100.0)
        color = "#16827c" if row["ape_mm"] <= 10.0 else "#c8643d"
        link = f"<a href='{html.escape(row['viewer'])}'>3D viewer</a>" if row["viewer"] else "n/a"
        rows_html.append(f"<tr><td>{html.escape(row['episode'])}</td><td>{row['ape_mm']:.3f} mm</td><td>{float(row['offset']):+.6f} s</td><td><span class='bar' style='width:{width:.1f}%;background:{color}'></span></td><td>{link}</td></tr>")
    page = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>RM75 current best overview</title><style>body{{margin:0;background:#f4f6f5;color:#1b2729;font:15px/1.5 system-ui,sans-serif}}main{{max-width:1160px;margin:28px auto;padding:0 18px}}section{{background:#fff;border:1px solid #dce5e2;border-radius:12px;padding:22px;box-shadow:0 8px 30px #193b3512}}h1{{margin:0 0 8px}}.muted{{color:#62716f}}.stats{{display:flex;gap:28px;margin:22px 0;flex-wrap:wrap}}.stat b{{display:block;font-size:28px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:11px 8px;border-bottom:1px solid #e7eeec;text-align:left}}th{{color:#62716f;font-size:13px}}.bar{{display:block;height:12px;border-radius:6px;min-width:4px}}a{{color:#0b6e68}}.note{{margin-top:16px;color:#62716f;font-size:13px}}</style><main><section><h1>RM75 当前最佳主链路总览</h1><div class='muted'>同事版 ORB-SLAM3 offline + GBA=100 + Full-frame BA=10 + smooth(21/2/9) + per-episode strict-sync。</div><div class='stats'><div class='stat'><b>{len(ordered)}</b> episodes</div><div class='stat'><b>{passed}/{len(ordered)}</b> APE &le; 10 mm</div><div class='stat'><b>{mean:.3f} mm</b>平均 APE</div></div><table><thead><tr><th>Episode</th><th>APE SE3</th><th>Offset</th><th>误差相对大小</th><th>轨迹</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table><div class='note'>当前总览只包含已完成且有真值的 episode；APE 为 SE3 translation RMSE。</div></section></main></html>"""
    (root / "overview.html").write_text(page)
    print(f"[DONE] episodes={len(ordered)} pass_10mm={passed}/{len(ordered)} mean_ape={mean:.3f} mm")
    print(f"[DONE] overview={root / 'overview.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
