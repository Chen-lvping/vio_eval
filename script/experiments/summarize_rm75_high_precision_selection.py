#!/usr/bin/env python3
"""Rebuild a high-precision RM75 selection report from completed offset scans.

The optimizer writes one JSON file per episode/candidate before it writes its
aggregate CSVs.  This utility makes interrupted long runs recoverable without
rerunning SLAM or the expensive scans.  It explicitly separates the shared,
deployable quality gate from the GT-scored oracle choice.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, median


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan-root", action="append", type=Path, required=True, help="Directory containing one fast_screen_*/offset_scans directory; repeatable.")
    p.add_argument("--manifest-map", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    a = args()
    manifest_map_path = a.manifest_map.resolve()
    raw_map = read_json(manifest_map_path).get("manifests", {})
    manifests = {key: (manifest_map_path.parent / value).resolve() for key, value in raw_map.items()}
    candidates: dict[str, dict[str, dict]] = {}
    for root in a.scan_root:
        for scan in root.resolve().glob("fast_screen_*/offset_scans/episode_gripper_*.json"):
            stem = scan.stem
            episode, candidate = stem.split("_", 3)[0] + "_" + stem.split("_", 1)[1], ""
            # Names are episode_gripper_XXXX_<candidate>; retain candidate underscores.
            parts = stem.split("_")
            episode = "_".join(parts[:3])
            candidate = "_".join(parts[3:])
            best = read_json(scan)["best"]
            candidates.setdefault(episode, {})[candidate] = {"candidate": candidate, "scan_json": str(scan), **best}

    rows: list[dict] = []
    candidate_rows: list[dict] = []
    for episode in sorted(candidates):
        manifest_path = manifests.get(episode)
        if manifest_path is None or not manifest_path.is_file():
            raise FileNotFoundError(f"missing manifest for {episode}: {manifest_path}")
        manifest = read_json(manifest_path)
        shadow = manifest.get("stereo_shadow_consistency") or {}
        coverage = float(manifest.get("inertial_coverage") or 0.0)
        shadow_mm = float(shadow.get("translation_rmse_mm") or float("inf"))
        scale = float(shadow.get("sim3_scale_stereo_over_inertial") or float("nan"))
        trusted = coverage >= 0.25 and shadow_mm <= 30.0 and (scale != scale or 0.70 <= scale <= 1.30)
        for data in candidates[episode].values():
            candidate_rows.append({
                "episode": episode, "candidate": data["candidate"], "ape_mm": data["ape_mm"],
                "rpe_mm": data["rpe_mm"], "ape_deg": data["ape_deg"], "rpe_deg": data["rpe_deg"],
                "best_offset_ms": float(data["offset_sec"]) * 1000.0, "matched_samples": int(data["matched_samples"]),
                "inertial_coverage": coverage, "shadow_disagreement_mm": shadow_mm, "shadow_scale": scale,
                "scan_json": data["scan_json"],
            })
        available = candidates[episode]
        oracle = min(available.values(), key=lambda x: (float(x["ape_mm"]), float(x["rpe_mm"])))
        gate_name = "fusion_a0.25" if trusted and "fusion_a0.25" in available else ("stereo_imu" if trusted and "stereo_imu" in available else "stereo")
        gate = available.get(gate_name, available["stereo"])
        stereo = available["stereo"]
        rows.append({
            "episode": episode, "status": "ok", "trusted_imu": trusted,
            "inertial_coverage": coverage, "shadow_disagreement_mm": shadow_mm, "shadow_scale": scale,
            "stereo_ape_mm": stereo["ape_mm"], "stereo_offset_ms": float(stereo["offset_sec"]) * 1000.0,
            "robust_candidate": gate_name, "robust_ape_mm": gate["ape_mm"], "robust_rpe_mm": gate["rpe_mm"], "robust_offset_ms": float(gate["offset_sec"]) * 1000.0,
            "oracle_candidate": oracle["candidate"], "oracle_ape_mm": oracle["ape_mm"], "oracle_rpe_mm": oracle["rpe_mm"], "oracle_offset_ms": float(oracle["offset_sec"]) * 1000.0,
            "offline_safe_candidate": oracle["candidate"], "offline_safe_ape_mm": oracle["ape_mm"],
            "offline_safe_delta_vs_stereo_mm": float(oracle["ape_mm"]) - float(stereo["ape_mm"]),
        })
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("candidate_summary.csv", candidate_rows), ("final_summary.csv", rows)):
        with (a.output_dir / name).open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    robust = [float(row["robust_ape_mm"]) for row in rows]
    oracle = [float(row["oracle_ape_mm"]) for row in rows]
    stereo = [float(row["stereo_ape_mm"]) for row in rows]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "episodes": len(rows), "policy": "shared 75% stereo + 25% IMU only when coverage>=0.25 and stereo/IMU disagreement<=30 mm; otherwise stereo",
        "robust_mean_ape_mm": mean(robust), "robust_median_ape_mm": median(robust), "robust_under_10mm": sum(v <= 10.0 for v in robust),
        "stereo_mean_ape_mm": mean(stereo), "stereo_median_ape_mm": median(stereo), "stereo_under_10mm": sum(v <= 10.0 for v in stereo),
        "oracle_mean_ape_mm": mean(oracle), "oracle_median_ape_mm": median(oracle), "oracle_under_10mm": sum(v <= 10.0 for v in oracle),
        "offline_safe_policy": "GT-evaluated minimum of stereo, native IMU, and fixed fusion candidates; offline benchmark only, not an online SLAM selector",
        "final_summary_csv": str((a.output_dir / "final_summary.csv").resolve()),
        "candidate_summary_csv": str((a.output_dir / "candidate_summary.csv").resolve()),
    }
    (a.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
