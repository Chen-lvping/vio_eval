#!/usr/bin/env python3
"""Batch sweep ORB-SLAM3 algo configs on the 2026-06-24 RM75 dataset."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SINGLE = REPO_ROOT / "script/run_orbslam3_tcp_eval.py"
DEFAULT_EPISODE_ROOT = REPO_ROOT / "data/gripper_data_6_24"
DEFAULT_GT_ROOT = REPO_ROOT / "data/ground_truth/rm75_6_24"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/evaluation/workbench"
DEFAULT_VINS_CONFIG = (
    REPO_ROOT
    / "data/gripper_data_6_23/episode_20260623_0004/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml"
)


@dataclass(frozen=True)
class AlgoCase:
    episode_dir: Path
    gt_path: Path
    nfeatures: int
    ini_fast: int
    min_fast: int
    clahe: bool

    @property
    def case_name(self) -> str:
        clahe_tag = "c1" if self.clahe else "c0"
        return f"{self.episode_dir.name}_nf{self.nfeatures}_f{self.ini_fast}_{self.min_fast}_{clahe_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vins-config", type=Path, default=DEFAULT_VINS_CONFIG)
    parser.add_argument("--tag", default="overnight")
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=[],
        help="Optional episode dir names, e.g. episode_20260624_0001",
    )
    parser.add_argument(
        "--episode-pattern",
        default="episode_20260624_*",
        help="Glob under episode-root used when --episodes is empty.",
    )
    parser.add_argument("--nfeatures-list", default="3000,3400")
    parser.add_argument("--fast-pairs", default="12:7,10:5,10:3")
    parser.add_argument("--clahe-modes", default="off")
    parser.add_argument("--camera-rig", choices=("stereo_left", "stereo_right"), default="stereo_right")
    parser.add_argument("--mode", choices=("stereo", "stereo-inertial"), default="stereo-inertial")
    parser.add_argument("--feature-preset", choices=("baseline", "low-texture", "aggressive"), default="low-texture")
    parser.add_argument("--imu-fast-init", type=int, choices=(0, 1), default=0)
    parser.add_argument("--vins-noise-mode", choices=("copy", "orb_from_vins"), default="orb_from_vins")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--viewer-max-points", type=int, default=3000)
    parser.add_argument("--strict-sync-offset-sec", type=float, default=-0.10488409042358399)
    parser.add_argument("--strict-sync-offset-scan-span-ms", type=float, default=150.0)
    parser.add_argument("--strict-sync-offset-scan-step-ms", type=float, default=15.0)
    parser.add_argument(
        "--strict-sync-offset-scan-score",
        choices=("ape", "rpe", "rotation", "composite"),
        default="composite",
    )
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--reuse-trajectory", action="store_true")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional limit for smoke tests. 0 means run all cases.",
    )
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    values = [int(round(float(part.strip()))) for part in raw.split(",") if part.strip()]
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
        pairs.append((int(round(float(ini_text))), int(round(float(min_text)))))
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
        modes.append(mapping[text])
    if not modes:
        raise ValueError("empty CLAHE mode list")
    deduped: list[bool] = []
    for value in modes:
        if value not in deduped:
            deduped.append(value)
    return deduped


def episode_to_gt(episode_dir: Path, gt_root: Path) -> Path:
    suffix = episode_dir.name.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"cannot infer GT id from {episode_dir.name}")
    gt_path = gt_root / f"rm75_pose_traj_{int(suffix)}.json"
    if not gt_path.is_file():
        raise FileNotFoundError(f"missing GT for {episode_dir.name}: {gt_path}")
    return gt_path


def list_episodes(args: argparse.Namespace) -> list[Path]:
    episode_root = args.episode_root.expanduser().resolve()
    if args.episodes:
        episodes = [episode_root / name for name in args.episodes]
    else:
        episodes = sorted(path for path in episode_root.glob(args.episode_pattern) if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"no episodes selected under {episode_root}")
    return [path.resolve() for path in episodes]


def build_cases(args: argparse.Namespace) -> list[AlgoCase]:
    gt_root = args.gt_root.expanduser().resolve()
    nfeatures_list = parse_int_list(args.nfeatures_list)
    fast_pairs = parse_fast_pairs(args.fast_pairs)
    clahe_modes = parse_clahe_modes(args.clahe_modes)
    cases: list[AlgoCase] = []
    for episode_dir in list_episodes(args):
        gt_path = episode_to_gt(episode_dir, gt_root)
        for nfeatures in nfeatures_list:
            for ini_fast, min_fast in fast_pairs:
                for clahe in clahe_modes:
                    cases.append(
                        AlgoCase(
                            episode_dir=episode_dir,
                            gt_path=gt_path,
                            nfeatures=nfeatures,
                            ini_fast=ini_fast,
                            min_fast=min_fast,
                            clahe=clahe,
                        )
                    )
    if args.max_cases and args.max_cases > 0:
        return cases[: args.max_cases]
    return cases


def run_logged(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    return int(proc.returncode)


def read_summary(eval_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with (eval_dir / "summary.csv").open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["metric"]] = float(row["rmse"])
    return out


def read_manifest(eval_dir: Path) -> dict:
    return json.loads((eval_dir / "orbslam3_tcp_eval_manifest.json").read_text(encoding="utf-8"))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(path: Path, rows: list[dict[str, object]], args: argparse.Namespace, sweep_root: Path) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    ok_rows.sort(key=lambda row: (float(row["ape_translation_se3_rmse_mm"]), float(row["rpe_translation_5cm_rmse_mm"])))
    lines = [
        "# RM75 6_24 ORB Algo Sweep",
        "",
        f"- Sweep root: `{sweep_root}`",
        f"- Episode root: `{args.episode_root.expanduser().resolve()}`",
        f"- GT root: `{args.gt_root.expanduser().resolve()}`",
        f"- VINS config: `{args.vins_config.expanduser().resolve()}`",
        f"- nfeatures: `{args.nfeatures_list}`",
        f"- FAST pairs: `{args.fast_pairs}`",
        f"- CLAHE modes: `{args.clahe_modes}`",
        f"- strict-sync offset center: `{args.strict_sync_offset_sec}`",
        f"- strict-sync scan: `span={args.strict_sync_offset_scan_span_ms} ms step={args.strict_sync_offset_scan_step_ms} ms score={args.strict_sync_offset_scan_score}`",
        f"- Total cases: `{len(rows)}`",
        f"- Success: `{len(ok_rows)}`",
        f"- Failed: `{len(rows) - len(ok_rows)}`",
        "",
        "| rank | case | APE mm | RPE mm | Rot APE deg | offset ms | viewer |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(ok_rows[:20], start=1):
        lines.append(
            f"| {rank} | {row['case']} | {float(row['ape_translation_se3_rmse_mm']):.3f} | "
            f"{float(row['rpe_translation_5cm_rmse_mm']):.3f} | {float(row['ape_rotation_se3_rmse_deg']):.3f} | "
            f"{float(row['strict_sync_offset_sec']) * 1000.0:.3f} | `{row['viewer_html']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.episode_root = args.episode_root.expanduser().resolve()
    args.gt_root = args.gt_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.vins_config = args.vins_config.expanduser().resolve()
    if not args.vins_config.is_file():
        raise FileNotFoundError(f"missing VINS config: {args.vins_config}")

    cases = build_cases(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = args.output_root / f"orbslam3_6_24_algo_sweep_{args.tag}_{stamp}"
    runs_root = sweep_root / "runs"
    evals_root = sweep_root / "evals"
    logs_root = sweep_root / "logs"
    sweep_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case",
        "episode",
        "status",
        "nfeatures",
        "ini_fast",
        "min_fast",
        "clahe",
        "ape_translation_se3_rmse_mm",
        "rpe_translation_5cm_rmse_mm",
        "ape_rotation_se3_rmse_deg",
        "rpe_rotation_5cm_rmse_deg",
        "trajectory_rows",
        "strict_sync_offset_sec",
        "estimate_csv_for_eval",
        "eval_dir",
        "viewer_html",
        "log_path",
        "error",
    ]
    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        run_dir = runs_root / case.case_name
        eval_dir = evals_root / case.case_name
        log_path = logs_root / f"{case.case_name}.log"
        cmd = [
            sys.executable,
            str(RUN_SINGLE),
            "--episode-dir",
            str(case.episode_dir),
            "--ground-truth",
            str(case.gt_path),
            "--output-dir",
            str(run_dir),
            "--eval-dir",
            str(eval_dir),
            "--camera-rig",
            args.camera_rig,
            "--mode",
            args.mode,
            "--feature-preset",
            args.feature_preset,
            "--vins-config",
            str(args.vins_config),
            "--vins-noise-mode",
            args.vins_noise_mode,
            "--imu-fast-init",
            str(args.imu_fast_init),
            "--nfeatures",
            str(case.nfeatures),
            "--ini-fast",
            str(case.ini_fast),
            "--min-fast",
            str(case.min_fast),
            "--strict-sync-offset-sec",
            str(args.strict_sync_offset_sec),
            "--strict-sync-offset-scan-span-ms",
            str(args.strict_sync_offset_scan_span_ms),
            "--strict-sync-offset-scan-step-ms",
            str(args.strict_sync_offset_scan_step_ms),
            "--strict-sync-offset-scan-score",
            args.strict_sync_offset_scan_score,
            "--timeout-sec",
            str(args.timeout_sec),
            "--viewer-max-points",
            str(args.viewer_max_points),
        ]
        if case.clahe:
            cmd.append("--clahe")
        if args.skip_viewer:
            cmd.append("--skip-viewer")
        if args.force_export:
            cmd.append("--force-export")
        if args.reuse_trajectory:
            cmd.append("--reuse-trajectory")

        row: dict[str, object] = {
            "case": case.case_name,
            "episode": case.episode_dir.name,
            "status": "ok",
            "nfeatures": case.nfeatures,
            "ini_fast": case.ini_fast,
            "min_fast": case.min_fast,
            "clahe": int(case.clahe),
            "ape_translation_se3_rmse_mm": "",
            "rpe_translation_5cm_rmse_mm": "",
            "ape_rotation_se3_rmse_deg": "",
            "rpe_rotation_5cm_rmse_deg": "",
            "trajectory_rows": "",
            "strict_sync_offset_sec": "",
            "estimate_csv_for_eval": "",
            "eval_dir": str(eval_dir),
            "viewer_html": str(eval_dir / "index.html"),
            "log_path": str(log_path),
            "error": "",
        }
        print(f"[{index}/{len(cases)}] {case.case_name}", flush=True)
        rc = run_logged(cmd, log_path)
        if rc != 0:
            row["status"] = "failed"
            row["error"] = f"exit code {rc}"
        else:
            try:
                summary = read_summary(eval_dir)
                manifest = read_manifest(eval_dir)
                row.update(
                    {
                        "ape_translation_se3_rmse_mm": summary["ape_translation_se3"],
                        "rpe_translation_5cm_rmse_mm": summary["rpe_translation_5cm"],
                        "ape_rotation_se3_rmse_deg": summary["ape_rotation_se3"],
                        "rpe_rotation_5cm_rmse_deg": summary["rpe_rotation_5cm"],
                        "trajectory_rows": manifest["trajectory_rows"],
                        "strict_sync_offset_sec": manifest["strict_sync_offset_sec"],
                        "estimate_csv_for_eval": manifest["estimate_csv_for_eval"],
                    }
                )
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
        rows.append(row)
        write_csv(sweep_root / "batch_summary.csv", fieldnames, rows)
        build_report(sweep_root / "REPORT.md", rows, args, sweep_root)

    print(f"[OK] wrote {sweep_root / 'batch_summary.csv'}")
    print(f"[OK] wrote {sweep_root / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
