#!/usr/bin/env python3
"""Joint robustness sweep for RM75 6_24 and 6_25 ORB-SLAM3 configurations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_BATCH = REPO_ROOT / "script/run_orbslam3_rm75_batch_eval.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/evaluation/workbench"
DEFAULT_VINS_CONFIG = (
    REPO_ROOT
    / "data/gripper_data_6_23/episode_20260623_0004/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml"
)
DEFAULT_6_24_EPISODE_ROOT = REPO_ROOT / "data/gripper_data_6_24"
DEFAULT_6_24_GT_ROOT = REPO_ROOT / "data/ground_truth/rm75_6_24"
DEFAULT_6_25_EPISODE_ROOT = REPO_ROOT / "data/gripper_data_6_25"
DEFAULT_6_25_GT_ROOT = REPO_ROOT / "data/ground_truth/rm75_6_25"


OFFSETS_6_24 = {
    "episode_20260624_0001": 0.00011590957641601562,
    "episode_20260624_0002": -0.014884090423583962,
    "episode_20260624_0003": -0.029884090423583975,
    "episode_20260624_0004": -0.05988409042358398,
    "episode_20260624_0005": -0.05988409042358398,
    "episode_20260624_0006": 0.045115909576416016,
    "episode_20260624_0007": 0.045115909576416016,
}

OFFSETS_6_25 = {
    "episode_20260625_0002": -0.014884090423583962,
    "episode_20260625_0003": -0.014884090423583962,
}


@dataclass(frozen=True)
class SweepCase:
    nfeatures: int
    ini_fast: int
    min_fast: int
    clahe: bool
    tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fallback-vins-config", type=Path, default=DEFAULT_VINS_CONFIG)
    parser.add_argument("--camera-rig", choices=("stereo_left", "stereo_right"), default="stereo_right")
    parser.add_argument("--mode", choices=("stereo", "stereo-inertial"), default="stereo-inertial")
    parser.add_argument("--feature-preset", choices=("baseline", "low-texture", "aggressive"), default="low-texture")
    parser.add_argument("--imu-fast-init", type=int, choices=(0, 1), default=0)
    parser.add_argument("--vins-noise-mode", choices=("copy", "orb_from_vins"), default="orb_from_vins")
    parser.add_argument("--viewer-max-points", type=int, default=3000)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--nfeatures-list", default="3000,3400")
    parser.add_argument("--fast-pairs", default="12:7,10:5,10:3")
    parser.add_argument("--clahe-modes", default="off")
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--reuse-trajectory", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--tag", default="cross_scene_robustness")
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("empty integer list")
    return values


def parse_fast_pairs(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        ini_text, min_text = text.split(":", 1)
        pairs.append((int(ini_text), int(min_text)))
    if not pairs:
        raise ValueError("empty FAST pair list")
    return pairs


def parse_clahe_modes(raw: str) -> list[bool]:
    mapping = {"off": False, "0": False, "false": False, "on": True, "1": True, "true": True}
    modes: list[bool] = []
    for part in raw.split(","):
        text = part.strip().lower()
        if not text:
            continue
        if text not in mapping:
            raise ValueError(f"unsupported CLAHE mode: {part}")
        value = mapping[text]
        if value not in modes:
            modes.append(value)
    if not modes:
        raise ValueError("empty CLAHE mode list")
    return modes


def build_cases(args: argparse.Namespace) -> list[SweepCase]:
    cases: list[SweepCase] = []
    for nfeatures in parse_int_list(args.nfeatures_list):
        for ini_fast, min_fast in parse_fast_pairs(args.fast_pairs):
            for clahe in parse_clahe_modes(args.clahe_modes):
                cases.append(
                    SweepCase(
                        nfeatures=nfeatures,
                        ini_fast=ini_fast,
                        min_fast=min_fast,
                        clahe=clahe,
                        tag=f"nf{nfeatures}_f{ini_fast}_{min_fast}_c{int(clahe)}",
                    )
                )
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    return cases


def write_offsets_json(path: Path, payload: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    return int(proc.returncode)


def load_batch_rows(batch_dir: Path) -> list[dict[str, str]]:
    with (batch_dir / "batch_summary.csv").open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarize_scene(rows: list[dict[str, str]]) -> dict[str, float | int]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    ape_values = [float(row["ape_translation_se3_rmse_mm"]) for row in ok_rows]
    rpe_values = [float(row["rpe_translation_5cm_rmse_mm"]) for row in ok_rows]
    rot_values = [float(row["ape_rotation_se3_rmse_deg"]) for row in ok_rows]
    return {
        "total": len(rows),
        "ok": len(ok_rows),
        "failed": len(rows) - len(ok_rows),
        "ape_mean_mm": mean(ape_values),
        "ape_max_mm": max(ape_values) if ape_values else float("inf"),
        "rpe_mean_mm": mean(rpe_values),
        "rot_ape_mean_deg": mean(rot_values),
    }


def build_joint_row(case: SweepCase, rows_24: list[dict[str, str]], rows_25: list[dict[str, str]], batch_24: Path, batch_25: Path) -> dict[str, object]:
    summary_24 = summarize_scene(rows_24)
    summary_25 = summarize_scene(rows_25)
    robust_score = max(float(summary_24["ape_max_mm"]), float(summary_25["ape_max_mm"]))
    return {
        "case": case.tag,
        "status": "ok" if int(summary_24["failed"]) == 0 and int(summary_25["failed"]) == 0 else "failed",
        "nfeatures": case.nfeatures,
        "ini_fast": case.ini_fast,
        "min_fast": case.min_fast,
        "clahe": int(case.clahe),
        "scene_6_24_ape_mean_mm": summary_24["ape_mean_mm"],
        "scene_6_24_ape_max_mm": summary_24["ape_max_mm"],
        "scene_6_24_rpe_mean_mm": summary_24["rpe_mean_mm"],
        "scene_6_24_failed": summary_24["failed"],
        "scene_6_25_ape_mean_mm": summary_25["ape_mean_mm"],
        "scene_6_25_ape_max_mm": summary_25["ape_max_mm"],
        "scene_6_25_rpe_mean_mm": summary_25["rpe_mean_mm"],
        "scene_6_25_failed": summary_25["failed"],
        "robust_score_mm": robust_score,
        "joint_ape_mean_mm": mean([float(summary_24["ape_mean_mm"]), float(summary_25["ape_mean_mm"])]),
        "batch_6_24_dir": str(batch_24),
        "batch_6_25_dir": str(batch_25),
        "report_6_24": str(batch_24 / "REPORT.md"),
        "report_6_25": str(batch_25 / "REPORT.md"),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]], root: Path) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    ok_rows.sort(key=lambda row: (float(row["robust_score_mm"]), float(row["joint_ape_mean_mm"])))
    lines = [
        "# RM75 Cross-Scene Robustness Sweep",
        "",
        f"- Sweep root: `{root}`",
        f"- Total cases: `{len(rows)}`",
        f"- Success: `{len(ok_rows)}`",
        f"- Failed: `{len(rows) - len(ok_rows)}`",
        "",
        "| rank | case | robust score mm | joint mean mm | 6_24 mean mm | 6_25 mean mm | 6_24 report | 6_25 report |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for index, row in enumerate(ok_rows[:20], start=1):
        lines.append(
            f"| {index} | {row['case']} | {float(row['robust_score_mm']):.3f} | "
            f"{float(row['joint_ape_mean_mm']):.3f} | {float(row['scene_6_24_ape_mean_mm']):.3f} | "
            f"{float(row['scene_6_25_ape_mean_mm']):.3f} | `{row['report_6_24']}` | `{row['report_6_25']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_batch_cmd(
    args: argparse.Namespace,
    case: SweepCase,
    *,
    episode_root: Path,
    gt_root: Path,
    episode_pattern: str,
    offsets_json: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(RUN_BATCH),
        "--episode-root",
        str(episode_root),
        "--gt-root",
        str(gt_root),
        "--episode-pattern",
        episode_pattern,
        "--camera-rig",
        args.camera_rig,
        "--mode",
        args.mode,
        "--feature-preset",
        args.feature_preset,
        "--imu-fast-init",
        str(args.imu_fast_init),
        "--fallback-vins-config",
        str(args.fallback_vins_config),
        "--vins-noise-mode",
        args.vins_noise_mode,
        "--strict-sync-offset-json",
        str(offsets_json),
        "--timeout-sec",
        str(args.timeout_sec),
        "--viewer-max-points",
        str(args.viewer_max_points),
        "--nfeatures",
        str(case.nfeatures),
        "--ini-fast",
        str(case.ini_fast),
        "--min-fast",
        str(case.min_fast),
    ]
    if case.clahe:
        cmd.append("--clahe")
    if args.skip_viewer:
        cmd.append("--skip-viewer")
    if args.reuse_trajectory:
        cmd.append("--reuse-trajectory")
    if args.force_export:
        cmd.append("--force-export")
    return cmd


def newest_batch_dir(before: set[Path], output_root: Path) -> Path:
    after = {path.resolve() for path in output_root.glob("orbslam3_rm75_batch_eval_*") if path.is_dir()}
    created = sorted(after - before)
    if not created:
        raise FileNotFoundError("failed to locate newly created batch directory")
    return created[-1]


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.fallback_vins_config = args.fallback_vins_config.expanduser().resolve()
    if not args.fallback_vins_config.is_file():
        raise FileNotFoundError(f"missing fallback VINS config: {args.fallback_vins_config}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = args.output_root / f"rm75_cross_scene_robustness_{args.tag}_{stamp}"
    logs_root = sweep_root / "logs"
    offsets_root = sweep_root / "offsets"
    sweep_root.mkdir(parents=True, exist_ok=True)

    offsets_6_24 = offsets_root / "offsets_6_24.json"
    offsets_6_25 = offsets_root / "offsets_6_25.json"
    write_offsets_json(offsets_6_24, OFFSETS_6_24)
    write_offsets_json(offsets_6_25, OFFSETS_6_25)

    rows: list[dict[str, object]] = []
    cases = build_cases(args)
    fieldnames = [
        "case",
        "status",
        "nfeatures",
        "ini_fast",
        "min_fast",
        "clahe",
        "scene_6_24_ape_mean_mm",
        "scene_6_24_ape_max_mm",
        "scene_6_24_rpe_mean_mm",
        "scene_6_24_failed",
        "scene_6_25_ape_mean_mm",
        "scene_6_25_ape_max_mm",
        "scene_6_25_rpe_mean_mm",
        "scene_6_25_failed",
        "robust_score_mm",
        "joint_ape_mean_mm",
        "batch_6_24_dir",
        "batch_6_25_dir",
        "report_6_24",
        "report_6_25",
    ]

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.tag}", flush=True)
        before = {path.resolve() for path in args.output_root.glob("orbslam3_rm75_batch_eval_*") if path.is_dir()}
        cmd_24 = build_batch_cmd(
            args,
            case,
            episode_root=DEFAULT_6_24_EPISODE_ROOT,
            gt_root=DEFAULT_6_24_GT_ROOT,
            episode_pattern="episode_20260624_*",
            offsets_json=offsets_6_24,
        )
        rc_24 = run_cmd(cmd_24, logs_root / f"{case.tag}_6_24.log")
        if rc_24 != 0:
            rows.append(
                {
                    "case": case.tag,
                    "status": "failed",
                    "nfeatures": case.nfeatures,
                    "ini_fast": case.ini_fast,
                    "min_fast": case.min_fast,
                    "clahe": int(case.clahe),
                    "scene_6_24_ape_mean_mm": "",
                    "scene_6_24_ape_max_mm": "",
                    "scene_6_24_rpe_mean_mm": "",
                    "scene_6_24_failed": 7,
                    "scene_6_25_ape_mean_mm": "",
                    "scene_6_25_ape_max_mm": "",
                    "scene_6_25_rpe_mean_mm": "",
                    "scene_6_25_failed": 2,
                    "robust_score_mm": "",
                    "joint_ape_mean_mm": "",
                    "batch_6_24_dir": "",
                    "batch_6_25_dir": "",
                    "report_6_24": "",
                    "report_6_25": "",
                }
            )
            write_csv(sweep_root / "summary.csv", fieldnames, rows)
            write_report(sweep_root / "REPORT.md", rows, sweep_root)
            continue
        batch_24 = newest_batch_dir(before, args.output_root)
        rows_24 = load_batch_rows(batch_24)

        before = {path.resolve() for path in args.output_root.glob("orbslam3_rm75_batch_eval_*") if path.is_dir()}
        cmd_25 = build_batch_cmd(
            args,
            case,
            episode_root=DEFAULT_6_25_EPISODE_ROOT,
            gt_root=DEFAULT_6_25_GT_ROOT,
            episode_pattern="episode_20260625_*",
            offsets_json=offsets_6_25,
        )
        rc_25 = run_cmd(cmd_25, logs_root / f"{case.tag}_6_25.log")
        if rc_25 != 0:
            rows.append(
                {
                    "case": case.tag,
                    "status": "failed",
                    "nfeatures": case.nfeatures,
                    "ini_fast": case.ini_fast,
                    "min_fast": case.min_fast,
                    "clahe": int(case.clahe),
                    "scene_6_24_ape_mean_mm": summarize_scene(rows_24)["ape_mean_mm"],
                    "scene_6_24_ape_max_mm": summarize_scene(rows_24)["ape_max_mm"],
                    "scene_6_24_rpe_mean_mm": summarize_scene(rows_24)["rpe_mean_mm"],
                    "scene_6_24_failed": summarize_scene(rows_24)["failed"],
                    "scene_6_25_ape_mean_mm": "",
                    "scene_6_25_ape_max_mm": "",
                    "scene_6_25_rpe_mean_mm": "",
                    "scene_6_25_failed": 2,
                    "robust_score_mm": "",
                    "joint_ape_mean_mm": "",
                    "batch_6_24_dir": str(batch_24),
                    "batch_6_25_dir": "",
                    "report_6_24": str(batch_24 / "REPORT.md"),
                    "report_6_25": "",
                }
            )
            write_csv(sweep_root / "summary.csv", fieldnames, rows)
            write_report(sweep_root / "REPORT.md", rows, sweep_root)
            continue
        batch_25 = newest_batch_dir(before, args.output_root)
        rows_25 = load_batch_rows(batch_25)

        rows.append(build_joint_row(case, rows_24, rows_25, batch_24, batch_25))
        write_csv(sweep_root / "summary.csv", fieldnames, rows)
        write_report(sweep_root / "REPORT.md", rows, sweep_root)

    print(f"[OK] wrote {sweep_root / 'summary.csv'}")
    print(f"[OK] wrote {sweep_root / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
