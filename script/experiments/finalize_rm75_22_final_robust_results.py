#!/usr/bin/env python3
"""Merge RM75 22-episode robust candidates and formally verify final picks.

Inputs are existing postprocess screens only.  The script does not rerun
ORB-SLAM3.  It rebuilds selected screen candidates as pose CSV files, applies
their independent strict-sync offsets, runs the standard TCP evaluator, and
writes the final oracle and GT-free shadow-gate summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import runpy
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
ROBUST_ROOT = ROOT / "data/evaluation/workbench/rm75_22_final_robust_20260722"
BA0_SCREEN = ROBUST_ROOT / "fast_screen_20260722_144000"
BA10_SCREEN = ROBUST_ROOT / "fast_screen_20260722_154636"
WIDE_0008_SCREEN = ROBUST_ROOT / "fast_screen_20260722_144824"
SUPPLEMENTAL_0006_SCREEN = ROBUST_ROOT / "supplemental_0006/fast_screen_20260724_112616"
RECOVERED_0006_GT = ROBUST_ROOT / "supplemental_0006/recovered_gt/rm75_pose_traj_3_from_ref_tcp.json"
RECOVERED_0006_EPISODE = ROBUST_ROOT / "supplemental_0006/recovered_episode"
HYBRID_FORMAL_ROOT = ROOT / "data/evaluation/workbench/rm75_22_hybrid_final_20260722"
SELECTED_BA10_JSON = ROBUST_ROOT / "selected_ba10/selected_runs.json"
FINAL_OPTIMIZER = ROOT / "script/experiments/run_rm75_22_final_robust_optimizer.py"
RESAMPLER = ROOT / "script/postprocess/resample_pose_csv_to_gt_timestamps.py"
EVALUATOR = ROOT / "script/evaluate_vio_tcp_camera_evo.py"
HAND_EYE = ROOT / "data/calibration/handeye_0615/handeye_result.yaml"


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    episode_key: str
    source_episode: str
    date: str
    source_screen: str
    candidate_name: str
    method: str
    ba_iters: int | None
    strict_sync_offset_sec: float
    screen_ape_mm: float
    screen_rpe_mm: float
    screen_ape_deg: float
    screen_rpe_deg: float
    input_reference: str
    estimate_frame: str
    gt_json: Path
    episode_dir: Path
    calibration_json: Path
    imu_manifest: str
    inertial_coverage: float
    shadow_disagreement_mm: float
    shadow_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROBUST_ROOT / f"final_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--margin-mm", type=float, default=0.75)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: object, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).replace(".", "p")


def run_logged(cmd: list[str], log_path: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\n" + proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""),
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}; see {log_path}")


def read_summary(path: Path) -> dict[str, float]:
    return {row["metric"]: float(row["rmse"]) for row in read_rows(path)}


def write_pose_csv(path: Path, times_s: np.ndarray, positions: np.ndarray, rotations: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quats = Rotation.from_matrix(rotations).as_quat()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Timestamp_us", "X", "Y", "Z", "Quat_X", "Quat_Y", "Quat_Z", "Quat_W"])
        for time_s, pos, quat in zip(times_s, positions, quats):
            writer.writerow(
                [
                    str(int(round(float(time_s) * 1_000_000.0))),
                    f"{float(pos[0]):.9f}",
                    f"{float(pos[1]):.9f}",
                    f"{float(pos[2]):.9f}",
                    f"{float(quat[0]):.9f}",
                    f"{float(quat[1]):.9f}",
                    f"{float(quat[2]):.9f}",
                    f"{float(quat[3]):.9f}",
                ]
            )


def method_from_screen(source_screen: str, row: dict[str, str]) -> tuple[str, int | None]:
    name = row["candidate"]
    if source_screen == "wide_0008":
        return "stereo_ba10_wide_offset", None
    if name == "stereo":
        return "stereo_ba10", None
    manifest_path = row.get("imu_manifest", "")
    final_ba = 0
    if manifest_path:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            final_ba = int(manifest.get("resolved_parameter_overrides", {}).get("final_ba_iters", 0))
        except Exception:
            final_ba = 0
    if name == "stereo_imu":
        return f"stereo_inertial_ba{final_ba}", final_ba
    alpha = name.removeprefix("fusion_a")
    return f"fusion_ba{final_ba}_a{alpha}", final_ba


def source_date(source_episode: str) -> str:
    match = re.search(r"episode_(\d{8})_", source_episode)
    return match.group(1) if match else ""


def screen_candidate_records(opt: dict[str, Any]) -> tuple[list[CandidateRecord], dict[tuple[str, str, str], dict[str, str]]]:
    specs = [
        ("ba0_fast_screen", BA0_SCREEN / "candidate_summary.csv"),
        ("ba10_fast_screen", BA10_SCREEN / "candidate_summary.csv"),
        ("wide_0008", WIDE_0008_SCREEN / "candidate_summary.csv"),
        ("supplemental_0006", SUPPLEMENTAL_0006_SCREEN / "candidate_summary.csv"),
    ]
    row_lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    records: list[CandidateRecord] = []
    seen: dict[tuple[str, str, str, str], CandidateRecord] = {}
    all_seq_root: Path = opt["ALL_SEQ_ROOT"]
    for source_screen, path in specs:
        for row in read_rows(path):
            if row.get("status") != "ok":
                continue
            if source_screen == "wide_0008" and row["candidate"] != "stereo":
                continue
            row_lookup[(source_screen, row["episode"], row["candidate"])] = row
            method, ba_iters = method_from_screen(source_screen, row)
            episode_key = row["episode"]
            gt_json = (
                RECOVERED_0006_GT
                if episode_key == "episode_gripper_0006"
                else (all_seq_root / f"rm75_pose_traj_{int(episode_key[-4:])}.json").resolve()
            )
            episode_dir = (
                RECOVERED_0006_EPISODE
                if episode_key == "episode_gripper_0006"
                else (all_seq_root / episode_key).resolve()
            )
            input_reference = row.get("source", "")
            dedup_key = (
                episode_key,
                method,
                f"{fnum(row.get('best_offset_sec')):.12f}",
                input_reference if not row["candidate"].startswith("fusion_") else source_screen,
            )
            rec = CandidateRecord(
                candidate_id=safe_slug(f"{source_screen}__{method}__{row['candidate']}"),
                episode_key=episode_key,
                source_episode=row["source_episode"],
                date=source_date(row["source_episode"]),
                source_screen=source_screen,
                candidate_name=row["candidate"],
                method=method,
                ba_iters=ba_iters,
                strict_sync_offset_sec=fnum(row["best_offset_sec"]),
                screen_ape_mm=fnum(row["ape_mm"]),
                screen_rpe_mm=fnum(row["rpe_mm"]),
                screen_ape_deg=fnum(row["ape_deg"]),
                screen_rpe_deg=fnum(row["rpe_deg"]),
                input_reference=input_reference,
                estimate_frame="camera",
                gt_json=gt_json,
                episode_dir=episode_dir,
                calibration_json=(episode_dir / "calibration.json").resolve(),
                imu_manifest=row.get("imu_manifest", ""),
                inertial_coverage=fnum(row.get("inertial_coverage")),
                shadow_disagreement_mm=fnum(row.get("shadow_disagreement_mm")),
                shadow_scale=fnum(row.get("shadow_scale")),
            )
            old = seen.get(dedup_key)
            if old is None or (rec.screen_ape_mm, rec.screen_rpe_mm) < (old.screen_ape_mm, old.screen_rpe_mm):
                seen[dedup_key] = rec
    records.extend(seen.values())
    return records, row_lookup


def hybrid_formal_records(opt: dict[str, Any]) -> list[CandidateRecord]:
    all_seq_root: Path = opt["ALL_SEQ_ROOT"]
    records: list[CandidateRecord] = []
    for row in read_rows(HYBRID_FORMAL_ROOT / "final_22_results.csv"):
        episode_key = row["episode_key"]
        method = row["selected_method"]
        estimate_frame = "imu" if method.startswith("stereo_inertial") else "camera"
        gt_json = Path(row["ground_truth"]).resolve()
        episode_dir = (all_seq_root / episode_key).resolve()
        records.append(
            CandidateRecord(
                candidate_id=safe_slug(f"hybrid_formal__{method}"),
                episode_key=episode_key,
                source_episode=row["episode"],
                date=source_date(row["episode"]),
                source_screen="hybrid_formal",
                candidate_name=method,
                method=method,
                ba_iters=10 if method.endswith("ba10") else (0 if method.endswith("ba0") else None),
                strict_sync_offset_sec=fnum(row["best_offset_sec"]),
                screen_ape_mm=fnum(row["ape_translation_se3_mm"]),
                screen_rpe_mm=fnum(row["rpe_translation_1mm_mm"]),
                screen_ape_deg=fnum(row["ape_rotation_deg"]),
                screen_rpe_deg=fnum(row["rpe_rotation_1mm_deg"]),
                input_reference=row["input_csv"],
                estimate_frame=estimate_frame,
                gt_json=gt_json,
                episode_dir=episode_dir,
                calibration_json=(episode_dir / "calibration.json").resolve(),
                imu_manifest="",
                inertial_coverage=fnum(row.get("inertial_coverage")),
                shadow_disagreement_mm=fnum(row.get("shadow_disagreement_mm")),
                shadow_scale=fnum(row.get("shadow_scale")),
            )
        )
    return records


def load_candidate_pool(opt: dict[str, Any]) -> tuple[list[CandidateRecord], dict[tuple[str, str, str], dict[str, str]]]:
    screen_records, row_lookup = screen_candidate_records(opt)
    records = screen_records + hybrid_formal_records(opt)
    records.sort(key=lambda rec: (rec.episode_key, rec.screen_ape_mm, rec.screen_rpe_mm, rec.method, rec.source_screen))
    return records, row_lookup


def candidate_as_dict(rec: CandidateRecord) -> dict[str, Any]:
    out = rec.__dict__.copy()
    for key in ("gt_json", "episode_dir", "calibration_json"):
        out[key] = str(out[key])
    return out


def build_tcp_candidate(
    *,
    rec: CandidateRecord,
    row_lookup: dict[tuple[str, str, str], dict[str, str]],
    opt: dict[str, Any],
    fast: dict[str, Any],
    eval_api: dict[str, Any],
    smooth_args: SimpleNamespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_cls = opt["Candidate"]

    def source_to_candidate(row: dict[str, str], estimate_frame: str, name: str) -> Any:
        raw = fast["read_tum_trajectory"](Path(row["source"]).resolve())
        smoothed = opt["smooth_trajectory"](fast, raw, smooth_args)
        times, positions, rotations = opt["transform_to_tcp"](
            fast,
            eval_api,
            rec.episode_dir,
            estimate_frame,
            smoothed,
        )
        return candidate_cls(
            name=name,
            times_s=times,
            positions_tcp=positions,
            rotations_tcp=rotations,
            center_offset_sec=fnum(row.get("center_offset_sec"), 0.0),
            source=row["source"],
        )

    if rec.candidate_name in ("stereo", "stereo_imu"):
        row = row_lookup[(rec.source_screen, rec.episode_key, rec.candidate_name)]
        frame = "camera" if rec.candidate_name == "stereo" else "imu"
        candidate = source_to_candidate(row, frame, rec.candidate_name)
        return candidate.times_s, candidate.positions_tcp, candidate.rotations_tcp
    if not rec.candidate_name.startswith("fusion_a"):
        raise ValueError(f"cannot rebuild screen candidate {rec.candidate_name}")
    stereo_row = row_lookup[(rec.source_screen, rec.episode_key, "stereo")]
    imu_row = row_lookup[(rec.source_screen, rec.episode_key, "stereo_imu")]
    stereo = source_to_candidate(stereo_row, "camera", "stereo")
    inertial = source_to_candidate(imu_row, "imu", "stereo_imu")
    alpha = float(rec.candidate_name.removeprefix("fusion_a"))
    fused, _ = opt["make_fusion_candidate"](stereo, inertial, alpha)
    return fused.times_s, fused.positions_tcp, fused.rotations_tcp


def write_screen_candidate_camera_csv(
    *,
    rec: CandidateRecord,
    output_csv: Path,
    row_lookup: dict[tuple[str, str, str], dict[str, str]],
    opt: dict[str, Any],
    fast: dict[str, Any],
    eval_api: dict[str, Any],
    smooth_args: SimpleNamespace,
) -> None:
    times_s, positions_tcp, rotations_tcp = build_tcp_candidate(
        rec=rec,
        row_lookup=row_lookup,
        opt=opt,
        fast=fast,
        eval_api=eval_api,
        smooth_args=smooth_args,
    )
    t_tcp_camera = eval_api["load_tcp_camera_transform"](HAND_EYE)
    positions_camera: list[np.ndarray] = []
    rotations_camera: list[np.ndarray] = []
    for pos_tcp, rot_tcp in zip(positions_tcp, rotations_tcp):
        t_world_tcp = np.eye(4, dtype=float)
        t_world_tcp[:3, :3] = rot_tcp
        t_world_tcp[:3, 3] = pos_tcp
        t_world_camera = t_world_tcp @ t_tcp_camera
        positions_camera.append(t_world_camera[:3, 3].copy())
        rotations_camera.append(t_world_camera[:3, :3].copy())
    write_pose_csv(output_csv, times_s, np.asarray(positions_camera), np.asarray(rotations_camera))


def verify_candidate(
    *,
    rec: CandidateRecord,
    candidate_root: Path,
    row_lookup: dict[tuple[str, str, str], dict[str, str]],
    opt: dict[str, Any],
    fast: dict[str, Any],
    eval_api: dict[str, Any],
    smooth_args: SimpleNamespace,
    resume: bool,
) -> dict[str, Any]:
    out_dir = candidate_root / rec.episode_key / rec.candidate_id
    eval_dir = out_dir / "eval"
    summary_csv = eval_dir / "summary.csv"
    if resume and summary_csv.is_file():
        summary = read_summary(summary_csv)
    else:
        if rec.source_screen == "hybrid_formal":
            estimate_csv = Path(rec.input_reference).resolve()
            estimate_frame = rec.estimate_frame
        else:
            estimate_csv = out_dir / "candidate_camera.csv"
            estimate_frame = "camera"
            write_screen_candidate_camera_csv(
                rec=rec,
                output_csv=estimate_csv,
                row_lookup=row_lookup,
                opt=opt,
                fast=fast,
                eval_api=eval_api,
                smooth_args=smooth_args,
            )
        strict_csv = out_dir / "selected_strictsync.csv"
        run_logged(
            [
                sys.executable,
                str(RESAMPLER),
                "--estimate-csv",
                str(estimate_csv),
                "--gt-json",
                str(rec.gt_json),
                "--output-csv",
                str(strict_csv),
                "--time-offset-sec",
                str(rec.strict_sync_offset_sec),
                "--max-gap-sec",
                "0.05",
            ],
            out_dir / "resample.log",
        )
        run_logged(
            [
                sys.executable,
                str(EVALUATOR),
                "--estimate",
                str(strict_csv),
                "--ground-truth",
                str(rec.gt_json),
                "--output-dir",
                str(eval_dir),
                "--handeye-yaml",
                str(HAND_EYE),
                "--calibration-json",
                str(rec.calibration_json),
                "--camera-rig",
                "stereo_right",
                "--estimate-frame",
                estimate_frame,
                "--time-association",
                "evo",
                "--rpe-distance-m",
                "0.001",
                "--time-offset-sec",
                "0",
                "--t-max-diff-sec",
                "0.0001",
            ],
            out_dir / "evaluate.log",
        )
        summary = read_summary(summary_csv)
    strict_csv = out_dir / "selected_strictsync.csv"
    estimate_csv = Path(rec.input_reference).resolve() if rec.source_screen == "hybrid_formal" else out_dir / "candidate_camera.csv"
    return {
        **candidate_as_dict(rec),
        "status": "ok",
        "formal_ape_translation_se3_mm": summary["ape_translation_se3"],
        "formal_rpe_translation_eval_mm": summary["rpe_translation_5cm"],
        "formal_ape_rotation_se3_deg": summary["ape_rotation_se3"],
        "formal_rpe_rotation_eval_deg": summary["rpe_rotation_5cm"],
        "formal_ape_translation_sim3_mm": summary["ape_translation_sim3"],
        "estimate_csv_for_resample": str(estimate_csv),
        "selected_strictsync_csv": str(strict_csv),
        "eval_dir": str(eval_dir),
    }


def best_stereo_candidate(candidates: list[CandidateRecord]) -> CandidateRecord:
    stereo = [rec for rec in candidates if rec.method in ("stereo_ba10", "stereo_ba10_wide_offset")]
    if not stereo:
        raise RuntimeError("no stereo fallback candidate")
    return min(stereo, key=lambda rec: (rec.screen_ape_mm, rec.screen_rpe_mm, rec.strict_sync_offset_sec))


def diagnostics_by_episode_ba(candidates: list[CandidateRecord]) -> dict[tuple[str, int], CandidateRecord]:
    out: dict[tuple[str, int], CandidateRecord] = {}
    for rec in candidates:
        if rec.ba_iters is None:
            continue
        key = (rec.episode_key, rec.ba_iters)
        if key not in out and rec.source_screen != "hybrid_formal":
            out[key] = rec
    return out


def gate_pass(rec: CandidateRecord, policy: dict[str, Any]) -> tuple[bool, str]:
    if not math.isfinite(rec.inertial_coverage) or rec.inertial_coverage < policy["min_coverage"]:
        return False, "low_inertial_coverage"
    if not math.isfinite(rec.shadow_disagreement_mm):
        return False, "missing_shadow_metrics"
    if rec.shadow_disagreement_mm > policy["max_shadow_disagreement_mm"]:
        return False, "shadow_disagreement"
    if math.isfinite(rec.shadow_scale) and not (
        policy["extreme_scale_min"] <= rec.shadow_scale <= policy["extreme_scale_max"]
    ):
        return False, "extreme_shadow_scale"
    return True, "shadow_gate_pass"


def select_shadow_gate(candidates: list[CandidateRecord], policy: dict[str, Any]) -> tuple[CandidateRecord, str]:
    by_method = {rec.method: rec for rec in candidates if rec.source_screen != "hybrid_formal"}
    diag = diagnostics_by_episode_ba(candidates)
    episode_key = candidates[0].episode_key
    failed: list[str] = []
    for ba_iters in policy["ba_preference"]:
        probe = diag.get((episode_key, ba_iters))
        if probe is None:
            failed.append(f"ba{ba_iters}:unavailable")
            continue
        ok, reason = gate_pass(probe, policy)
        if not ok:
            failed.append(f"ba{ba_iters}:{reason}")
            continue
        fixed_fusion = f"fusion_ba{ba_iters}_a{policy['fixed_fusion_alpha']}"
        if fixed_fusion in by_method:
            return by_method[fixed_fusion], f"{reason}; fixed_alpha={policy['fixed_fusion_alpha']}; ba{ba_iters}"
        inertial = f"stereo_inertial_ba{ba_iters}"
        if inertial in by_method:
            return by_method[inertial], f"{reason}; no_fusion_candidate; ba{ba_iters}"
        failed.append(f"ba{ba_iters}:candidate_unavailable")
    fallback = best_stereo_candidate(candidates)
    return fallback, "fail_closed_to_stereo; " + "; ".join(failed)


def select_verification_set(
    by_episode: dict[str, list[CandidateRecord]],
    shadow_choices: dict[str, CandidateRecord],
    top_k: int,
    margin_mm: float,
) -> list[CandidateRecord]:
    selected: dict[tuple[str, str], CandidateRecord] = {}
    for episode_key, candidates in by_episode.items():
        ordered = sorted(candidates, key=lambda rec: (rec.screen_ape_mm, rec.screen_rpe_mm, rec.method))
        best_ape = ordered[0].screen_ape_mm
        for rec in ordered[:top_k]:
            selected[(rec.episode_key, rec.candidate_id)] = rec
        for rec in ordered:
            if rec.screen_ape_mm <= best_ape + margin_mm:
                selected[(rec.episode_key, rec.candidate_id)] = rec
        shadow_rec = shadow_choices[episode_key]
        selected[(shadow_rec.episode_key, shadow_rec.candidate_id)] = shadow_rec
    return sorted(selected.values(), key=lambda rec: (rec.episode_key, rec.screen_ape_mm, rec.candidate_id))


def stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean_ape_mm": float(np.mean(arr)),
        "median_ape_mm": float(np.median(arr)),
        "p90_ape_mm": float(np.percentile(arr, 90)),
        "max_ape_mm": float(np.max(arr)),
        "count_le_10mm": int(np.sum(arr <= 10.0)),
    }


def method_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[str(row[key])] = out.get(str(row[key]), 0) + 1
    return dict(sorted(out.items()))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def lodo_validation(candidates: list[CandidateRecord]) -> dict[str, Any]:
    by_episode: dict[str, list[CandidateRecord]] = {}
    for rec in candidates:
        if rec.source_screen != "hybrid_formal":
            by_episode.setdefault(rec.episode_key, []).append(rec)
    dates = sorted({records[0].date for records in by_episode.values()})
    param_grid = [
        {
            "min_coverage": cov,
            "max_shadow_disagreement_mm": disagreement,
            "extreme_scale_min": 0.70,
            "extreme_scale_max": 1.30,
            "fixed_fusion_alpha": alpha,
            "ba_preference": pref,
        }
        for cov in (0.25, 0.50, 0.75, 0.85, 0.90, 0.95)
        for disagreement in (10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0)
        for alpha in ("0.25", "0.50", "0.75")
        for pref in ([0, 10], [10, 0])
    ]
    folds: list[dict[str, Any]] = []
    for holdout in dates:
        train = {ep: recs for ep, recs in by_episode.items() if recs[0].date != holdout}
        test = {ep: recs for ep, recs in by_episode.items() if recs[0].date == holdout}

        def mean_for(policy: dict[str, Any], split: dict[str, list[CandidateRecord]]) -> float:
            values = [select_shadow_gate(recs, policy)[0].screen_ape_mm for recs in split.values()]
            return float(np.mean(values))

        best_policy = min(param_grid, key=lambda policy: mean_for(policy, train))
        test_rows = []
        for episode_key, recs in sorted(test.items()):
            rec, reason = select_shadow_gate(recs, best_policy)
            test_rows.append(
                {
                    "episode_key": episode_key,
                    "selected_method": rec.method,
                    "fast_ape_mm": rec.screen_ape_mm,
                    "reason": reason,
                }
            )
        folds.append(
            {
                "holdout_date": holdout,
                "selected_policy": best_policy,
                "train_mean_fast_ape_mm": mean_for(best_policy, train),
                "test_mean_fast_ape_mm": float(np.mean([row["fast_ape_mm"] for row in test_rows])),
                "test_rows": test_rows,
            }
        )
    all_test_values = [row["fast_ape_mm"] for fold in folds for row in fold["test_rows"]]
    return {
        "folds": folds,
        "pooled_lodo_fast": stats(all_test_values),
    }


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for path in (
        BA0_SCREEN / "candidate_summary.csv",
        BA10_SCREEN / "candidate_summary.csv",
        WIDE_0008_SCREEN / "candidate_summary.csv",
        SUPPLEMENTAL_0006_SCREEN / "candidate_summary.csv",
        RECOVERED_0006_GT,
        RECOVERED_0006_EPISODE / "calibration.json",
        HYBRID_FORMAL_ROOT / "final_22_results.csv",
        SELECTED_BA10_JSON,
        FINAL_OPTIMIZER,
        RESAMPLER,
        EVALUATOR,
        HAND_EYE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    opt = runpy.run_path(str(FINAL_OPTIMIZER), run_name="__rm75_final_merge__")
    fast = runpy.run_path(str(opt["FAST_SCAN_SCRIPT"]), run_name="__rm75_final_merge_fast__")
    eval_api = fast["load_eval_api"]()
    smooth_args = SimpleNamespace(position_window=21, position_poly=2, rotation_window=9)

    candidates, row_lookup = load_candidate_pool(opt)
    by_episode: dict[str, list[CandidateRecord]] = {}
    for rec in candidates:
        by_episode.setdefault(rec.episode_key, []).append(rec)
    if len(by_episode) != 22:
        raise RuntimeError(f"expected 22 episodes, found {len(by_episode)}")

    shadow_policy = {
        "min_coverage": 0.25,
        "max_shadow_disagreement_mm": 30.0,
        "extreme_scale_min": 0.70,
        "extreme_scale_max": 1.30,
        "fixed_fusion_alpha": "0.75",
        "ba_preference": [0, 10],
        "selection_inputs": "coverage + stereo/IMU shadow disagreement + extreme scale guard only; no APE/GT",
    }
    shadow_choices = {
        episode_key: select_shadow_gate(recs, shadow_policy)[0]
        for episode_key, recs in by_episode.items()
    }
    shadow_reasons = {
        episode_key: select_shadow_gate(recs, shadow_policy)[1]
        for episode_key, recs in by_episode.items()
    }
    verify_records = select_verification_set(by_episode, shadow_choices, args.top_k, args.margin_mm)
    candidate_pool_csv = output_root / "candidate_pool.csv"
    write_rows(candidate_pool_csv, [candidate_as_dict(rec) for rec in candidates])

    verification_rows: list[dict[str, Any]] = []
    verification_csv = output_root / "candidate_verification.csv"
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and verification_csv.is_file():
        for row in read_rows(verification_csv):
            completed[(row["episode_key"], row["candidate_id"])] = row
    for index, rec in enumerate(verify_records, start=1):
        key = (rec.episode_key, rec.candidate_id)
        if key in completed:
            print(f"[SKIP {index}/{len(verify_records)}] {rec.episode_key} {rec.method}", flush=True)
            verification_rows.append(completed[key])
            continue
        print(
            f"[VERIFY {index}/{len(verify_records)}] {rec.episode_key} {rec.method} "
            f"offset={rec.strict_sync_offset_sec * 1000.0:+.3f} ms screen_APE={rec.screen_ape_mm:.3f}",
            flush=True,
        )
        try:
            row = verify_candidate(
                rec=rec,
                candidate_root=output_root / "candidate_verification",
                row_lookup=row_lookup,
                opt=opt,
                fast=fast,
                eval_api=eval_api,
                smooth_args=smooth_args,
                resume=args.resume,
            )
        except Exception as exc:
            row = {**candidate_as_dict(rec), "status": "failed", "error": str(exc)}
        verification_rows.append(row)
        verification_rows.sort(key=lambda item: (item["episode_key"], item["candidate_id"]))
        write_rows(verification_csv, verification_rows)

    ok_by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in verification_rows:
        if row.get("status") == "ok":
            ok_by_episode.setdefault(str(row["episode_key"]), []).append(row)
    final_rows: list[dict[str, Any]] = []
    for episode_key in sorted(by_episode):
        ok_rows = ok_by_episode.get(episode_key, [])
        if not ok_rows:
            raise RuntimeError(f"no verified candidates for {episode_key}")
        oracle = min(
            ok_rows,
            key=lambda row: (
                float(row["formal_ape_translation_se3_mm"]),
                float(row["formal_rpe_translation_eval_mm"]),
                str(row["method"]),
            ),
        )
        shadow_rec = shadow_choices[episode_key]
        shadow = next(
            (
                row
                for row in ok_rows
                if row["candidate_id"] == shadow_rec.candidate_id and row["source_screen"] == shadow_rec.source_screen
            ),
            None,
        )
        shadow_reason = shadow_reasons[episode_key]
        if shadow is None or shadow.get("status") != "ok":
            fallback_rec = best_stereo_candidate(by_episode[episode_key])
            shadow = next(
                (
                    row
                    for row in ok_rows
                    if row["candidate_id"] == fallback_rec.candidate_id and row["source_screen"] == fallback_rec.source_screen
                ),
                None,
            )
            if shadow is None:
                raise RuntimeError(f"shadow fallback was not verified for {episode_key}")
            shadow_reason += "; verification_failed_fail_closed"
        final_rows.append(
            {
                "episode_key": episode_key,
                "source_episode": oracle["source_episode"],
                "date": oracle["date"],
                "oracle_method": oracle["method"],
                "oracle_candidate_source": oracle["source_screen"],
                "oracle_strict_sync_offset_sec": oracle["strict_sync_offset_sec"],
                "oracle_strict_sync_offset_ms": float(oracle["strict_sync_offset_sec"]) * 1000.0,
                "ape_translation_se3_mm": oracle["formal_ape_translation_se3_mm"],
                "rpe_translation_eval_mm": oracle["formal_rpe_translation_eval_mm"],
                "ape_rotation_se3_deg": oracle["formal_ape_rotation_se3_deg"],
                "rpe_rotation_eval_deg": oracle["formal_rpe_rotation_eval_deg"],
                "ape_translation_sim3_mm": oracle["formal_ape_translation_sim3_mm"],
                "screen_ape_mm": oracle["screen_ape_mm"],
                "screen_rpe_mm": oracle["screen_rpe_mm"],
                "input_reference": oracle["input_reference"],
                "selected_strictsync_csv": oracle["selected_strictsync_csv"],
                "eval_dir": oracle["eval_dir"],
                "ground_truth": oracle["gt_json"],
                "calibration_json": oracle["calibration_json"],
                "shadow_gate_method": shadow["method"],
                "shadow_gate_candidate_source": shadow["source_screen"],
                "shadow_gate_reason": shadow_reason,
                "shadow_gate_strict_sync_offset_sec": shadow["strict_sync_offset_sec"],
                "shadow_gate_ape_translation_se3_mm": shadow["formal_ape_translation_se3_mm"],
                "shadow_gate_rpe_translation_eval_mm": shadow["formal_rpe_translation_eval_mm"],
                "inertial_coverage": shadow["inertial_coverage"],
                "shadow_disagreement_mm": shadow["shadow_disagreement_mm"],
                "shadow_scale": shadow["shadow_scale"],
            }
        )

    final_csv = output_root / "final_22_results.csv"
    write_rows(final_csv, final_rows)
    best_method_map = {
        row["episode_key"]: {
            "source_episode": row["source_episode"],
            "method": row["oracle_method"],
            "candidate_source": row["oracle_candidate_source"],
            "strict_sync_offset_sec": float(row["oracle_strict_sync_offset_sec"]),
            "strict_sync_offset_ms": float(row["oracle_strict_sync_offset_ms"]),
            "ape_translation_se3_mm": float(row["ape_translation_se3_mm"]),
            "rpe_translation_eval_mm": float(row["rpe_translation_eval_mm"]),
            "selected_strictsync_csv": row["selected_strictsync_csv"],
            "eval_dir": row["eval_dir"],
        }
        for row in final_rows
    }
    (output_root / "best_method_map.json").write_text(
        json.dumps(best_method_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    offsets = {row["episode_key"]: float(row["oracle_strict_sync_offset_sec"]) for row in final_rows}
    (output_root / "best_strict_sync_offsets.json").write_text(
        json.dumps(offsets, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    oracle_values = [float(row["ape_translation_se3_mm"]) for row in final_rows]
    shadow_values = [float(row["shadow_gate_ape_translation_se3_mm"]) for row in final_rows]
    summary_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "episodes": len(final_rows),
        "oracle": stats(oracle_values),
        "shadow_gate_deployable": stats(shadow_values),
        "oracle_method_counts": method_counts(final_rows, "oracle_method"),
        "shadow_gate_method_counts": method_counts(final_rows, "shadow_gate_method"),
        "shadow_gate_policy": shadow_policy,
        "leave_one_date_validation_fast": lodo_validation(candidates),
        "source_summaries": {
            "historical_original": {"mean_ape_mm": 15.040, "count_le_10mm": 12},
            "ba0_fast_screen": load_json(BA0_SCREEN / "summary.json"),
            "ba10_fast_screen": load_json(BA10_SCREEN / "summary.json"),
            "wide_0008_screen": load_json(WIDE_0008_SCREEN / "summary.json"),
            "supplemental_0006_screen": load_json(SUPPLEMENTAL_0006_SCREEN / "summary.json"),
            "hybrid_formal": load_json(HYBRID_FORMAL_ROOT / "summary.json"),
        },
        "inputs": {
            "ba0_screen": str(BA0_SCREEN),
            "ba10_screen": str(BA10_SCREEN),
            "wide_0008_screen": str(WIDE_0008_SCREEN),
            "supplemental_0006_screen": str(SUPPLEMENTAL_0006_SCREEN),
            "recovered_0006_gt": str(RECOVERED_0006_GT),
            "hybrid_formal": str(HYBRID_FORMAL_ROOT),
            "selected_ba10_json": str(SELECTED_BA10_JSON),
        },
        "outputs": {
            "final_csv": str(final_csv),
            "candidate_pool_csv": str(candidate_pool_csv),
            "candidate_verification_csv": str(verification_csv),
            "best_method_map": str(output_root / "best_method_map.json"),
            "best_strict_sync_offsets": str(output_root / "best_strict_sync_offsets.json"),
            "report_md": str(output_root / "REPORT.md"),
        },
        "notes": [
            "Oracle selection is scored after formal evaluator verification using APE then RPE.",
            "Shadow-gate mode selection uses only coverage, shadow disagreement, and an extreme scale guard; APE/GT are used only afterward for reporting.",
            "Strict-sync offsets are evaluation alignment parameters and are not inputs to shadow-gate mode selection.",
            "0006 uses a recovered GT JSON derived from the preserved standard-evaluator ref_tcp.tum because its original GT JSON is absent from this workspace.",
        ],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# RM75 22-Episode Final Robust Result",
        "",
        "All selected candidates were verified with the standard TCP evaluator after independent per-episode strict-sync resampling.",
        "",
        "## Summary",
        "",
        f"- Oracle mean / median / P90 / max APE: {summary_payload['oracle']['mean_ape_mm']:.6f} / "
        f"{summary_payload['oracle']['median_ape_mm']:.6f} / {summary_payload['oracle']['p90_ape_mm']:.6f} / "
        f"{summary_payload['oracle']['max_ape_mm']:.6f} mm",
        f"- Oracle <= 10 mm: {summary_payload['oracle']['count_le_10mm']}/22",
        f"- Shadow-gate mean / median / P90 / max APE: {summary_payload['shadow_gate_deployable']['mean_ape_mm']:.6f} / "
        f"{summary_payload['shadow_gate_deployable']['median_ape_mm']:.6f} / "
        f"{summary_payload['shadow_gate_deployable']['p90_ape_mm']:.6f} / "
        f"{summary_payload['shadow_gate_deployable']['max_ape_mm']:.6f} mm",
        f"- Shadow-gate <= 10 mm: {summary_payload['shadow_gate_deployable']['count_le_10mm']}/22",
        f"- Final CSV: `{final_csv}`",
        f"- Method map: `{output_root / 'best_method_map.json'}`",
        f"- Offset map: `{output_root / 'best_strict_sync_offsets.json'}`",
        "",
        "## Shadow-Gate Policy",
        "",
        "- Inputs: inertial coverage, stereo/IMU shadow disagreement, and an extreme scale guard only.",
        "- Gate: coverage >= 0.25, disagreement <= 30 mm, scale either missing or inside [0.70, 1.30].",
        "- Trusted IMU action: fixed fusion alpha=0.75, preferring BA0 then BA10 when both are available; if fusion is unavailable, use the matching stereo-inertial trajectory.",
        "- Fail closed: missing shadow metrics, low coverage, high disagreement, or extreme scale falls back to pure stereo.",
        "",
        "## Per-Episode Results",
        "",
        "| episode | source | oracle method | offset ms | APE mm | RPE mm | shadow method | shadow APE mm |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in final_rows:
        report_lines.append(
            f"| {row['episode_key']} | {row['source_episode']} | {row['oracle_method']} | "
            f"{float(row['oracle_strict_sync_offset_ms']):+.3f} | "
            f"{float(row['ape_translation_se3_mm']):.6f} | "
            f"{float(row['rpe_translation_eval_mm']):.6f} | "
            f"{row['shadow_gate_method']} | {float(row['shadow_gate_ape_translation_se3_mm']):.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## Candidate Sources",
            "",
            "- BA0 fast screen: `fast_screen_20260722_144000`",
            "- BA10 fast screen: `fast_screen_20260722_154636`",
        "- 0008 wide-offset screen: `fast_screen_20260722_144824`",
        "- 0006 supplemental screen: `supplemental_0006/fast_screen_20260724_112616` (GT recovered from preserved evaluator reference)",
            "- Existing hybrid formal result: `rm75_22_hybrid_final_20260722`",
            "- Candidate audit: `candidate_pool.csv` and `candidate_verification.csv`",
        ]
    )
    (output_root / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary_payload["oracle"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(summary_payload["shadow_gate_deployable"], ensure_ascii=False, indent=2), flush=True)
    print(f"[OK] {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
