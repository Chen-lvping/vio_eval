#!/usr/bin/env python3
"""Build a non-duplicating clean RM75 main benchmark bundle.

The bundle is intentionally a collection of absolute symbolic links plus
machine-readable provenance.  It contains the exact trajectories referenced
by the comparison-table screen summaries; no raw recording or trajectory is
copied or regenerated.

The conservative cleaning rule is applied per episode: retain an episode only
when the three table metrics (Stereo, IMU BA0, IMU BA10) are all <= threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ROBUST_ROOT = ROOT / "data/evaluation/workbench/rm75_22_final_robust_20260722"
TABLE_PATH = ROBUST_ROOT / "full_stereo_vs_stereo_imu_comparison_20260724.md"
DEFAULT_OUTPUT = ROOT / "data/evaluation/curated/rm75_main_clean_ape_le_50mm_20260729"

OLD_BA0 = ROBUST_ROOT / "fast_screen_20260722_144000/candidate_summary.csv"
OLD_BA10 = ROBUST_ROOT / "fast_screen_20260722_154636/candidate_summary.csv"
DIRECT_BA0 = ROBUST_ROOT / "missing_direct_imu_ba0_screen/fast_screen_20260724_130848/candidate_summary.csv"
DIRECT_BA10 = ROBUST_ROOT / "missing_direct_imu_ba10_screen/fast_screen_20260724_130516/candidate_summary.csv"

# The July 24 direct runs supersede the earlier screen only for these methods.
DIRECT_BA0_EPISODES = {"0016", "0017"}
DIRECT_BA10_EPISODES = {"0002", "0004", "0008", "0012", "0013", "0014", "0016", "0017"}

# Explicit curation decisions in addition to the numerical threshold.
MANUAL_EXCLUSIONS = {
    "0005": "user_requested_exclusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-mm", type=float, default=50.0)
    parser.add_argument(
        "--require-no-fallback",
        action="store_true",
        help="also exclude episodes where any selected configuration reports fallback_to_stereo=true",
    )
    parser.add_argument("--replace", action="store_true", help="replace an existing output directory")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_main_rows(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    pattern = re.compile(r"^\| main/(\d{4}) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            episode, stereo, ba0, ba10 = match.groups()
            rows[episode] = {"stereo_mm": float(stereo), "imu_ba0_mm": float(ba0), "imu_ba10_mm": float(ba10)}
    if len(rows) != 22:
        raise RuntimeError(f"expected 22 main rows in {path}, found {len(rows)}")
    return rows


def indexed_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(path):
        if row.get("status") == "ok":
            rows[(row["episode"].removeprefix("episode_gripper_"), row["candidate"])] = row
    return rows


def require_file(path_text: str, label: str) -> Path:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def link(target: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(target)


def load_manifest(row: dict[str, str]) -> tuple[Path, dict[str, Any]]:
    path = require_file(row["imu_manifest"], "evaluation manifest")
    return path, json.loads(path.read_text(encoding="utf-8"))


def chosen_row(
    episode: str,
    candidate: str,
    old: dict[tuple[str, str], dict[str, str]],
    direct: dict[tuple[str, str], dict[str, str]],
    direct_episodes: set[str],
) -> dict[str, str]:
    source = direct if episode in direct_episodes else old
    try:
        return source[(episode, candidate)]
    except KeyError as exc:
        raise KeyError(f"missing {candidate} row for main/{episode}") from exc


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"output exists: {output}; pass --replace to rebuild it")
        if output == ROOT or ROOT not in output.parents:
            raise ValueError(f"refusing to replace unsafe output path: {output}")
        shutil.rmtree(output)

    table_rows = parse_main_rows(TABLE_PATH)
    old_ba0, old_ba10 = indexed_rows(OLD_BA0), indexed_rows(OLD_BA10)
    direct_ba0, direct_ba10 = indexed_rows(DIRECT_BA0), indexed_rows(DIRECT_BA10)

    output.mkdir(parents=True)
    episode_records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for episode, metrics in sorted(table_rows.items()):
        if episode in MANUAL_EXCLUSIONS:
            excluded.append({
                "episode": f"main/{episode}",
                "reason": MANUAL_EXCLUSIONS[episode],
                "metrics_mm": metrics,
            })
            continue
        offending = {name: value for name, value in metrics.items() if value > args.threshold_mm}
        if offending:
            excluded.append({
                "episode": f"main/{episode}",
                "reason": "metric_threshold",
                "metrics_mm": metrics,
                "excluded_metrics_mm": offending,
            })
            continue

        stereo = chosen_row(episode, "stereo", old_ba10, direct_ba10, DIRECT_BA10_EPISODES)
        ba0 = chosen_row(episode, "stereo_imu", old_ba0, direct_ba0, DIRECT_BA0_EPISODES)
        ba10 = chosen_row(episode, "stereo_imu", old_ba10, direct_ba10, DIRECT_BA10_EPISODES)
        config_rows = {"stereo": stereo, "imu_ba0": ba0, "imu_ba10": ba10}
        config_manifests = {name: load_manifest(row) for name, row in config_rows.items()}
        fallback_configs = [name for name, (_, item) in config_manifests.items() if item.get("fallback_to_stereo")]
        if args.require_no_fallback and fallback_configs:
            excluded.append({
                "episode": f"main/{episode}",
                "reason": "fallback_to_stereo",
                "metrics_mm": metrics,
                "fallback_configs": fallback_configs,
            })
            continue

        manifest_path, manifest = config_manifests["imu_ba10"]
        raw_dataset = Path(manifest["episode_dir"])
        ground_truth = require_file(manifest["ground_truth"], "ground truth")
        if not raw_dataset.is_dir():
            raise FileNotFoundError(f"raw dataset is missing: {raw_dataset}")

        episode_root = output / "episodes" / f"main_{episode}"
        episode_root.mkdir(parents=True)
        link(raw_dataset, episode_root / "raw_dataset")
        link(ground_truth, episode_root / "ground_truth.json")
        calibration = raw_dataset / "calibration.json"
        if calibration.is_file():
            link(calibration, episode_root / "calibration.json")

        config_metadata: dict[str, Any] = {}
        for config_name, row in config_rows.items():
            config_root = episode_root / "orb" / config_name
            config_root.mkdir(parents=True)
            trajectory = require_file(row["source"], f"{config_name} trajectory for main/{episode}")
            eval_manifest_path, eval_manifest = config_manifests[config_name]
            link(trajectory, config_root / "trajectory.tum")
            link(eval_manifest_path, config_root / "evaluation_manifest.json")
            scan_path = Path(row.get("scan_json", ""))
            if scan_path.is_file():
                link(scan_path, config_root / "offset_scan.json")
            settings = Path(eval_manifest.get("settings_yaml", ""))
            if settings.is_file():
                link(settings, config_root / "orb_settings.yaml")
            config_metadata[config_name] = {
                "ape_mm": float(row["ape_mm"]),
                "best_offset_ms": float(row["best_offset_ms"]),
                "trajectory": str(trajectory),
                "evaluation_manifest": str(eval_manifest_path),
                "requested_mode": eval_manifest.get("requested_mode"),
                "actual_mode": eval_manifest.get("mode"),
                "fallback_to_stereo": eval_manifest.get("fallback_to_stereo"),
            }

        record = {
            "episode": f"main/{episode}",
            "metrics_from_comparison_table_mm": metrics,
            "raw_dataset": str(raw_dataset),
            "ground_truth": str(ground_truth),
            "configs": config_metadata,
        }
        (episode_root / "provenance.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        episode_records.append(record)

    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "stereo_mm", "imu_ba0_mm", "imu_ba10_mm", "raw_dataset", "ground_truth"])
        writer.writeheader()
        for record in episode_records:
            metrics = record["metrics_from_comparison_table_mm"]
            writer.writerow({
                "episode": record["episode"],
                **metrics,
                "raw_dataset": record["raw_dataset"],
                "ground_truth": record["ground_truth"],
            })
    (output / "excluded_episodes.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comparison_lines = [
        "# RM75 main clean comparison",
        "",
        f"Retained rows have Stereo, IMU BA0, and IMU BA10 translation APE RMSE <= {args.threshold_mm:.1f} mm.",
        "The source table remains unchanged; excluded episodes are listed in `excluded_episodes.json`.",
        "",
        "| Episode | Stereo (mm) | IMU BA0 (mm) | IMU BA10 (mm) |",
        "|---|---:|---:|---:|",
    ]
    for record in episode_records:
        metrics = record["metrics_from_comparison_table_mm"]
        comparison_lines.append(
            f"| {record['episode']} | {metrics['stereo_mm']:.3f} | {metrics['imu_ba0_mm']:.3f} | {metrics['imu_ba10_mm']:.3f} |"
        )
    (output / "comparison_clean.md").write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")
    readme = f"""# RM75 main clean benchmark bundle

This is a symlink-based, non-duplicating curation of the `main` subset from
`{TABLE_PATH.relative_to(ROOT)}`.

Inclusion rule: every table metric (`Stereo`, `IMU BA0`, `IMU BA10`) is at most
{args.threshold_mm:.1f} mm translation APE RMSE{" and no selected configuration reported a stereo fallback" if args.require_no_fallback else ""}.
The exact exclusions are recorded in `excluded_episodes.json`.

Contents per episode:

- `raw_dataset` -> original episode directory
- `ground_truth.json` -> RM75 TCP ground truth used by the recorded evaluation
- `orb/stereo`, `orb/imu_ba0`, `orb/imu_ba10` -> exact evaluated trajectory,
  evaluator manifest, ORB settings, and offset-scan artifact when available
- `provenance.json` -> metrics and source paths, including requested/actual
  runtime mode and any fallback flag

The files are links, not copies. Use `episodes.csv` for the retained-episode
index, `comparison_clean.md` for the cleaned human-readable table, and inspect
each `provenance.json` before treating a trajectory as a native inertial output.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(f"[OK] curated {len(episode_records)} main episodes in {output}")
    print(f"[OK] excluded {len(excluded)} main episodes by curation policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
