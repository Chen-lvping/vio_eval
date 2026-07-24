#!/usr/bin/env python3
"""Screen stereo, stereo-inertial, and fixed-weight fusion for all 22 RM75 episodes.

Every (episode, candidate) pair receives an independent coarse/fine/refined
strict-sync offset search.  The script only post-processes existing trajectories;
it does not rerun ORB-SLAM3.
"""

from __future__ import annotations

import argparse
import csv
import json
import runpy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[2]
ALL_SEQ_ROOT = ROOT / "data/gripper/gripper_all_seq"
STEREO_ROOT = ROOT / "data/evaluation/workbench/rm75_best_all_gt_20260716"
STEREO_AGGREGATE = STEREO_ROOT / "aggregate_summary.csv"
IMU_BATCH_ROOT = (
    ROOT
    / "data/evaluation/workbench/rm75_stereo_imu_shadow_gate_full_20260721"
    / "orbslam3_rm75_batch_eval_20260721_213748"
)
OFFSET_JSON = ROOT / "data/evaluation/config/rm75_gripper_22_strict_sync_offsets.json"
FAST_SCAN_SCRIPT = ROOT / "script/experiments/run_data_ok_shared_algo_sweep.py"
HAND_EYE = ROOT / "data/calibration/handeye_0615/handeye_result.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "data/evaluation/workbench/rm75_22_final_robust_20260722"


@dataclass(frozen=True)
class Candidate:
    name: str
    times_s: np.ndarray
    positions_tcp: np.ndarray
    rotations_tcp: np.ndarray
    center_offset_sec: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--coarse-span-ms", type=float, default=350.0)
    parser.add_argument("--coarse-step-ms", type=float, default=10.0)
    parser.add_argument("--fine-span-ms", type=float, default=12.0)
    parser.add_argument("--fine-step-ms", type=float, default=1.0)
    parser.add_argument("--refine-span-ms", type=float, default=1.2)
    parser.add_argument("--refine-step-ms", type=float, default=0.1)
    parser.add_argument("--max-gap-sec", type=float, default=0.05)
    parser.add_argument(
        "--scan-max-gt-samples",
        type=int,
        default=3000,
        help="Uniformly subsample GT during offset search; final metrics always use all matched samples.",
    )
    parser.add_argument("--fusion-weights", default="0.25,0.50,0.75")
    parser.add_argument("--position-window", type=int, default=21)
    parser.add_argument("--position-poly", type=int, default=2)
    parser.add_argument("--rotation-window", type=int, default=9)
    parser.add_argument(
        "--imu-manifest-json",
        type=Path,
        default=None,
        help="Optional episode-to-manifest mapping that replaces the BA0 stereo-IMU sources.",
    )
    parser.add_argument(
        "--fixed-offset-json",
        type=Path,
        default=None,
        help="Optional episode-to-offset mapping. When set, score every candidate at the same fixed offset without an offset search.",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        default=[],
        help="Optional unified names such as episode_gripper_0020.",
    )
    parser.add_argument(
        "--episode-dir-override",
        type=Path,
        default=None,
        help="Use a recovered episode directory for one explicitly selected episode.",
    )
    parser.add_argument(
        "--ground-truth-override",
        type=Path,
        default=None,
        help="Use a recovered GT JSON for one explicitly selected episode.",
    )
    return parser.parse_args()


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files:\n" + "\n".join(missing))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def valid_odd_window(requested: int, size: int, poly: int) -> int:
    if size <= poly + 1:
        return 0
    window = max(poly + 2, requested)
    if window % 2 == 0:
        window += 1
    max_window = size if size % 2 == 1 else size - 1
    window = min(window, max_window)
    return window if window > poly else 0


def smooth_positions(positions: np.ndarray, window: int, poly: int) -> np.ndarray:
    actual = valid_odd_window(window, len(positions), poly)
    if actual == 0:
        return positions.copy()
    return np.vstack(
        [savgol_filter(positions[:, axis], actual, poly, mode="interp") for axis in range(3)]
    ).T


def make_quats_continuous(quats: np.ndarray) -> np.ndarray:
    out = quats.copy()
    for index in range(1, len(out)):
        if float(np.dot(out[index - 1], out[index])) < 0.0:
            out[index] *= -1.0
    return out


def smooth_rotations(rotations: np.ndarray, window: int) -> np.ndarray:
    quats = make_quats_continuous(Rotation.from_matrix(rotations).as_quat())
    actual = valid_odd_window(window, len(quats), 0)
    if actual == 0:
        return Rotation.from_quat(quats).as_matrix()
    rots = Rotation.from_quat(quats)
    half = actual // 2
    smoothed: list[np.ndarray] = []
    for index in range(len(rots)):
        start = max(0, index - half)
        end = min(len(rots), index + half + 1)
        center = rots[index]
        relative = (center.inv() * rots[start:end]).as_rotvec()
        smoothed.append((center * Rotation.from_rotvec(relative.mean(axis=0))).as_matrix())
    return np.asarray(smoothed, dtype=float)


def smooth_trajectory(fast: dict[str, Any], trajectory: Any, args: argparse.Namespace) -> Any:
    cls = fast["SmoothedTrajectory"]
    return cls(
        trajectory.times_s.copy(),
        smooth_positions(trajectory.positions, args.position_window, args.position_poly),
        smooth_rotations(trajectory.rotations, args.rotation_window),
    )


def find_stereo_sources() -> dict[str, dict[str, Any]]:
    rows = read_csv_rows(STEREO_AGGREGATE)
    sources: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        unified = f"episode_gripper_{index:04d}"
        episode_name = row["episode"]
        episode_dirs = sorted(STEREO_ROOT.glob(f"orbslam3_rm75_best_batch_*/{episode_name}"))
        if len(episode_dirs) != 1:
            raise RuntimeError(f"expected one stereo result dir for {episode_name}, found {episode_dirs}")
        tum_paths = sorted(episode_dirs[0].glob("f_colleague_*_ffba10.txt"))
        if len(tum_paths) != 1:
            raise RuntimeError(f"expected one ffba10 TUM for {episode_name}, found {tum_paths}")
        resolved_episode = (ALL_SEQ_ROOT / unified).resolve()
        if resolved_episode.name != episode_name:
            raise RuntimeError(
                f"aggregate/unified ordering mismatch: {unified} -> {resolved_episode.name}, row={episode_name}"
            )
        sources[unified] = {
            "source_episode": episode_name,
            "trajectory": tum_paths[0].resolve(),
            "center_offset_sec": float(row["offset"]),
            "historical_ape_mm": float(row["ape_mm"]),
        }
    return sources


def load_imu_manifest(
    unified: str,
    overrides: dict[str, Path],
) -> tuple[Path, dict[str, Any]]:
    path = overrides.get(
        unified,
        IMU_BATCH_ROOT / f"eval_{unified}" / "orbslam3_tcp_eval_manifest.json",
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return path, manifest


def transform_to_tcp(
    fast: dict[str, Any],
    eval_api: dict[str, Any],
    episode_dir: Path,
    estimate_frame: str,
    trajectory: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return fast["transform_estimate_to_tcp"](
        eval_api,
        episode_dir / "calibration.json",
        "stereo_right",
        HAND_EYE,
        estimate_frame,
        trajectory,
    )


def interpolate_pose(
    source_times: np.ndarray,
    source_pos: np.ndarray,
    source_rot: np.ndarray,
    query_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.column_stack(
        [np.interp(query_times, source_times, source_pos[:, axis]) for axis in range(3)]
    )
    rotations = Slerp(source_times, Rotation.from_matrix(source_rot))(query_times).as_matrix()
    return positions, rotations


def rigid_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u_mat, _, vt_mat = np.linalg.svd(source_zero.T @ target_zero)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0.0:
        vt_mat[-1, :] *= -1.0
        rotation = vt_mat.T @ u_mat.T
    translation = target_center - rotation @ source_center
    aligned = (rotation @ source.T).T + translation
    rmse_mm = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))) * 1000.0)
    return rotation, translation, rmse_mm


def make_fusion_candidate(
    stereo: Candidate,
    inertial: Candidate,
    alpha: float,
) -> tuple[Candidate, dict[str, float]]:
    valid = (stereo.times_s >= inertial.times_s[0]) & (stereo.times_s <= inertial.times_s[-1])
    times = stereo.times_s[valid]
    stereo_pos = stereo.positions_tcp[valid]
    stereo_rot = stereo.rotations_tcp[valid]
    if len(times) < 30:
        raise RuntimeError(f"insufficient stereo/IMU overlap: {len(times)}")
    imu_pos, imu_rot = interpolate_pose(
        inertial.times_s,
        inertial.positions_tcp,
        inertial.rotations_tcp,
        times,
    )
    global_rot, global_trans, disagreement_mm = rigid_align(imu_pos, stereo_pos)
    imu_pos_aligned = (global_rot @ imu_pos.T).T + global_trans
    imu_rot_aligned = np.einsum("ij,njk->nik", global_rot, imu_rot)
    fused_pos = (1.0 - alpha) * stereo_pos + alpha * imu_pos_aligned
    relative = Rotation.from_matrix(stereo_rot).inv() * Rotation.from_matrix(imu_rot_aligned)
    fused_rot = (
        Rotation.from_matrix(stereo_rot) * Rotation.from_rotvec(alpha * relative.as_rotvec())
    ).as_matrix()
    return (
        Candidate(
            name=f"fusion_a{alpha:.2f}",
            times_s=times,
            positions_tcp=fused_pos,
            rotations_tcp=fused_rot,
            center_offset_sec=stereo.center_offset_sec,
            source=f"SE3-aligned stereo/IMU alpha={alpha:.2f}",
        ),
        {"fusion_alignment_disagreement_mm": disagreement_mm, "fusion_overlap_rows": int(len(times))},
    )


def scan_candidate(
    fast: dict[str, Any],
    helpers: dict[str, Any],
    gt: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate: Candidate,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    scan_gt = gt
    if args.scan_max_gt_samples > 0 and len(gt[0]) > args.scan_max_gt_samples:
        indices = np.linspace(0, len(gt[0]) - 1, args.scan_max_gt_samples, dtype=int)
        indices = np.unique(indices)
        scan_gt = (gt[0][indices], gt[1][indices], gt[2][indices])
    first_best, first_rows = fast["search_best_offset"](
        helpers,
        scan_gt[0],
        scan_gt[1],
        scan_gt[2],
        candidate.times_s,
        candidate.positions_tcp,
        candidate.rotations_tcp,
        candidate.center_offset_sec,
        args.coarse_span_ms,
        args.coarse_step_ms,
        args.fine_span_ms,
        args.fine_step_ms,
        args.max_gap_sec,
    )
    refined_best, refined_rows = fast["search_best_offset"](
        helpers,
        scan_gt[0],
        scan_gt[1],
        scan_gt[2],
        candidate.times_s,
        candidate.positions_tcp,
        candidate.rotations_tcp,
        first_best["offset_sec"],
        args.refine_span_ms,
        args.refine_step_ms,
        args.refine_step_ms,
        max(args.refine_step_ms / 5.0, 0.01),
        args.max_gap_sec,
    )
    merged: dict[int, dict[str, float]] = {}
    for row in first_rows + refined_rows:
        merged[int(round(float(row["offset_sec"]) * 1_000_000_000.0))] = row
    matched = fast["strict_sync_match"](
        gt[0],
        gt[1],
        gt[2],
        candidate.times_s,
        candidate.positions_tcp,
        candidate.rotations_tcp,
        refined_best["offset_sec"],
        args.max_gap_sec,
    )
    full_metrics = fast["evaluate_rmse"](
        helpers,
        matched[0],
        matched[1],
        matched[2],
        matched[3],
        matched[4],
    )
    full_best = {
        "offset_sec": float(refined_best["offset_sec"]),
        "matched_samples": float(len(matched[0])),
        "matched_duration_s": float(matched[0][-1] - matched[0][0]) if len(matched[0]) > 1 else 0.0,
        **full_metrics,
        "search_ape_mm": float(refined_best["ape_mm"]),
        "search_gt_samples": float(len(scan_gt[0])),
    }
    return full_best, [merged[key] for key in sorted(merged)]


def evaluate_candidate_at_fixed_offset(
    fast: dict[str, Any],
    helpers: dict[str, Any],
    gt: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate: Candidate,
    max_gap_sec: float,
) -> dict[str, float]:
    """Evaluate one trajectory at its prescribed, episode-level offset."""
    matched = fast["strict_sync_match"](
        gt[0], gt[1], gt[2],
        candidate.times_s, candidate.positions_tcp, candidate.rotations_tcp,
        candidate.center_offset_sec, max_gap_sec,
    )
    metrics = fast["evaluate_rmse"](helpers, matched[0], matched[1], matched[2], matched[3], matched[4])
    return {
        "offset_sec": float(candidate.center_offset_sec),
        "matched_samples": float(len(matched[0])),
        "matched_duration_s": float(matched[0][-1] - matched[0][0]) if len(matched[0]) > 1 else 0.0,
        **metrics,
    }


def robust_gate_candidate(
    rows: list[dict[str, Any]],
    coverage: float,
    shadow_mm: float,
    shadow_scale: float,
) -> str:
    names = {str(row["candidate"]) for row in rows if row.get("status") == "ok"}
    trusted_imu = (
        coverage >= 0.25
        and np.isfinite(shadow_mm)
        and shadow_mm <= 30.0
        and (not np.isfinite(shadow_scale) or 0.70 <= shadow_scale <= 1.30)
    )
    if trusted_imu and "fusion_a0.75" in names:
        return "fusion_a0.75"
    if trusted_imu and "stereo_imu" in names:
        return "stereo_imu"
    return "stereo"


def main() -> int:
    args = parse_args()
    require_files([STEREO_AGGREGATE, OFFSET_JSON, FAST_SCAN_SCRIPT, HAND_EYE])
    weights = [float(part.strip()) for part in args.fusion_weights.split(",") if part.strip()]
    if any(not 0.0 < value < 1.0 for value in weights):
        raise ValueError("fusion weights must be inside (0, 1)")

    fast = runpy.run_path(str(FAST_SCAN_SCRIPT), run_name="__rm75_22_final_optimizer__")
    eval_api = fast["load_eval_api"]()
    helpers = eval_api["load_helpers"]()
    stereo_sources = find_stereo_sources()
    imu_offsets = {
        str(key): float(value)
        for key, value in json.loads(OFFSET_JSON.read_text(encoding="utf-8")).items()
    }
    fixed_offsets: dict[str, float] = {}
    if args.fixed_offset_json is not None:
        fixed_path = args.fixed_offset_json.expanduser().resolve()
        fixed_offsets = {str(key): float(value) for key, value in json.loads(fixed_path.read_text(encoding="utf-8")).items()}
    imu_manifest_overrides: dict[str, Path] = {}
    if args.imu_manifest_json is not None:
        manifest_map_path = args.imu_manifest_json.expanduser().resolve()
        raw_manifest_map = json.loads(manifest_map_path.read_text(encoding="utf-8"))
        raw_manifest_map = raw_manifest_map.get("manifests", raw_manifest_map)
        for episode, raw_path in raw_manifest_map.items():
            path = Path(str(raw_path)).expanduser()
            if not path.is_absolute():
                path = (manifest_map_path.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"missing IMU manifest override for {episode}: {path}")
            imu_manifest_overrides[str(episode)] = path
    wanted = set(args.episodes or sorted(stereo_sources))
    unknown = sorted(wanted - set(stereo_sources))
    if unknown:
        raise ValueError(f"unknown episodes: {unknown}")
    missing_fixed = sorted(wanted - set(fixed_offsets)) if fixed_offsets else []
    if missing_fixed:
        raise ValueError(f"missing fixed offsets for: {missing_fixed}")
    if (args.episode_dir_override is not None or args.ground_truth_override is not None) and len(wanted) != 1:
        raise ValueError("episode/GT overrides require exactly one --episodes value")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root.expanduser().resolve() / f"fast_screen_{stamp}"
    scans_dir = out_dir / "offset_scans"
    scans_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for unified in sorted(wanted):
        source = stereo_sources[unified]
        episode_dir = (
            args.episode_dir_override.expanduser().resolve()
            if args.episode_dir_override is not None
            else (ALL_SEQ_ROOT / unified).resolve()
        )
        gt_path = (
            args.ground_truth_override.expanduser().resolve()
            if args.ground_truth_override is not None
            else (ALL_SEQ_ROOT / f"rm75_pose_traj_{int(unified[-4:])}.json").resolve()
        )
        gt = fast["load_ground_truth"](eval_api, gt_path)
        episode_offset = fixed_offsets.get(unified)

        stereo_raw = fast["read_tum_trajectory"](source["trajectory"])
        stereo_smoothed = smooth_trajectory(fast, stereo_raw, args)
        stereo_tcp = transform_to_tcp(fast, eval_api, episode_dir, "camera", stereo_smoothed)
        stereo_candidate = Candidate(
            name="stereo",
            times_s=stereo_tcp[0],
            positions_tcp=stereo_tcp[1],
            rotations_tcp=stereo_tcp[2],
            center_offset_sec=float(episode_offset if episode_offset is not None else source["center_offset_sec"]),
            source=str(source["trajectory"]),
        )

        imu_manifest_path, imu_manifest = load_imu_manifest(unified, imu_manifest_overrides)
        shadow = imu_manifest.get("stereo_shadow_consistency", {})
        inertial_path = Path(str(shadow.get("inertial_tum") or ""))
        if not inertial_path.is_file() and (
            imu_manifest.get("mode") == "stereo-inertial"
            and not bool(imu_manifest.get("fallback_to_stereo"))
        ):
            inertial_path = Path(str(imu_manifest.get("trajectory") or ""))
        coverage = float(imu_manifest.get("inertial_coverage") or 0.0)
        shadow_mm = float(shadow.get("translation_rmse_mm") or float("nan"))
        shadow_scale = float(shadow.get("sim3_scale_stereo_over_inertial") or float("nan"))

        candidates: list[tuple[Candidate, dict[str, Any]]] = [(stereo_candidate, {})]
        if inertial_path.is_file():
            try:
                imu_raw = fast["read_tum_trajectory"](inertial_path)
                imu_smoothed = smooth_trajectory(fast, imu_raw, args)
                imu_tcp = transform_to_tcp(fast, eval_api, episode_dir, "imu", imu_smoothed)
                imu_candidate = Candidate(
                    name="stereo_imu",
                    times_s=imu_tcp[0],
                    positions_tcp=imu_tcp[1],
                    rotations_tcp=imu_tcp[2],
                    center_offset_sec=float(episode_offset if episode_offset is not None else imu_offsets.get(unified, source["center_offset_sec"])),
                    source=str(inertial_path.resolve()),
                )
                candidates.append((imu_candidate, {}))
                if coverage >= 0.50:
                    for alpha in weights:
                        fused, metadata = make_fusion_candidate(stereo_candidate, imu_candidate, alpha)
                        candidates.append((fused, metadata))
            except Exception as exc:
                candidate_rows.append(
                    {
                        "episode": unified,
                        "source_episode": source["source_episode"],
                        "candidate": "stereo_imu",
                        "status": "failed_prepare",
                        "error": str(exc),
                        "inertial_coverage": coverage,
                        "shadow_disagreement_mm": shadow_mm,
                        "shadow_scale": shadow_scale,
                    }
                )

        episode_rows: list[dict[str, Any]] = []
        for candidate, metadata in candidates:
            print(f"[SCAN] {unified} {candidate.name}", flush=True)
            row: dict[str, Any] = {
                "episode": unified,
                "source_episode": source["source_episode"],
                "candidate": candidate.name,
                "status": "ok",
                "center_offset_sec": candidate.center_offset_sec,
                "source": candidate.source,
                "historical_stereo_ape_mm": source["historical_ape_mm"],
                "inertial_coverage": coverage,
                "shadow_disagreement_mm": shadow_mm,
                "shadow_scale": shadow_scale,
                "imu_manifest": str(imu_manifest_path),
                **metadata,
            }
            try:
                if fixed_offsets:
                    best = evaluate_candidate_at_fixed_offset(fast, helpers, gt, candidate, args.max_gap_sec)
                    scan_rows = []
                else:
                    best, scan_rows = scan_candidate(fast, helpers, gt, candidate, args)
                row.update(
                    {
                        "best_offset_sec": best["offset_sec"],
                        "best_offset_ms": best["offset_sec"] * 1000.0,
                        "ape_mm": best["ape_mm"],
                        "rpe_mm": best["rpe_mm"],
                        "ape_deg": best["ape_deg"],
                        "rpe_deg": best["rpe_deg"],
                        "matched_samples": int(best["matched_samples"]),
                        "matched_duration_s": best["matched_duration_s"],
                    }
                )
                if fixed_offsets:
                    row["scan_json"] = ""
                else:
                    scan_path = scans_dir / f"{unified}_{candidate.name}.json"
                    scan_path.write_text(
                        json.dumps({"best": best, "rows": scan_rows}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    row["scan_json"] = str(scan_path)
            except Exception as exc:
                row["status"] = "failed_scan"
                row["error"] = str(exc)
            candidate_rows.append(row)
            episode_rows.append(row)

        ok_rows = [row for row in episode_rows if row.get("status") == "ok"]
        if not ok_rows:
            final_rows.append(
                {
                    "episode": unified,
                    "source_episode": source["source_episode"],
                    "status": "failed",
                }
            )
            continue
        oracle = min(ok_rows, key=lambda row: (float(row["ape_mm"]), float(row["rpe_mm"])))
        gate_name = robust_gate_candidate(ok_rows, coverage, shadow_mm, shadow_scale)
        gate = next((row for row in ok_rows if row["candidate"] == gate_name), None)
        if gate is None:
            gate = next(row for row in ok_rows if row["candidate"] == "stereo")
        stereo_result = next(row for row in ok_rows if row["candidate"] == "stereo")
        final_rows.append(
            {
                "episode": unified,
                "source_episode": source["source_episode"],
                "status": "ok",
                "historical_ape_mm": source["historical_ape_mm"],
                "rescanned_stereo_ape_mm": stereo_result["ape_mm"],
                "rescanned_stereo_offset_sec": stereo_result["best_offset_sec"],
                "oracle_candidate": oracle["candidate"],
                "oracle_ape_mm": oracle["ape_mm"],
                "oracle_rpe_mm": oracle["rpe_mm"],
                "oracle_ape_deg": oracle["ape_deg"],
                "oracle_rpe_deg": oracle["rpe_deg"],
                "oracle_offset_sec": oracle["best_offset_sec"],
                "robust_gate_candidate": gate["candidate"],
                "robust_gate_ape_mm": gate["ape_mm"],
                "robust_gate_offset_sec": gate["best_offset_sec"],
                "inertial_coverage": coverage,
                "shadow_disagreement_mm": shadow_mm,
                "shadow_scale": shadow_scale,
            }
        )

    candidates_csv = out_dir / "candidate_summary.csv"
    final_csv = out_dir / "final_summary.csv"
    write_csv(candidates_csv, candidate_rows)
    write_csv(final_csv, final_rows)

    ok_final = [row for row in final_rows if row.get("status") == "ok"]
    oracle_values = [float(row["oracle_ape_mm"]) for row in ok_final]
    gate_values = [float(row["robust_gate_ape_mm"]) for row in ok_final]
    stereo_values = [float(row["rescanned_stereo_ape_mm"]) for row in ok_final]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "configuration": vars(args) | {
            "output_root": str(args.output_root),
            "stereo_aggregate": str(STEREO_AGGREGATE),
            "imu_batch_root": str(IMU_BATCH_ROOT),
            "offset_json": str(OFFSET_JSON),
            "imu_manifest_json": str(args.imu_manifest_json) if args.imu_manifest_json else "",
            "fixed_offset_json": str(args.fixed_offset_json) if args.fixed_offset_json else "",
        },
        "episodes_total": len(final_rows),
        "episodes_ok": len(ok_final),
        "oracle_mean_ape_mm": float(np.mean(oracle_values)) if oracle_values else float("nan"),
        "oracle_under_10mm": int(sum(value <= 10.0 for value in oracle_values)),
        "robust_gate_mean_ape_mm": float(np.mean(gate_values)) if gate_values else float("nan"),
        "robust_gate_under_10mm": int(sum(value <= 10.0 for value in gate_values)),
        "rescanned_stereo_mean_ape_mm": float(np.mean(stereo_values)) if stereo_values else float("nan"),
        "rescanned_stereo_under_10mm": int(sum(value <= 10.0 for value in stereo_values)),
        "candidate_summary_csv": str(candidates_csv),
        "final_summary_csv": str(final_csv),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    report = [
        "# RM75 22-Episode Robust Postprocess Screen",
        "",
        "- Timing: fixed episode-level offsets shared by every candidate." if fixed_offsets else "- Each episode/candidate uses an independent strict-sync offset scan.",
        f"- Smoothing: Savitzky-Golay {args.position_window}/{args.position_poly}, rotation {args.rotation_window}.",
        f"- Oracle mean APE: `{payload['oracle_mean_ape_mm']:.6f} mm` ({payload['oracle_under_10mm']}/22 <= 10 mm).",
        f"- GT-free robust-gate mean APE: `{payload['robust_gate_mean_ape_mm']:.6f} mm` ({payload['robust_gate_under_10mm']}/22 <= 10 mm).",
        f"- Rescanned stereo mean APE: `{payload['rescanned_stereo_mean_ape_mm']:.6f} mm` ({payload['rescanned_stereo_under_10mm']}/22 <= 10 mm).",
        "",
        "| unified | source | oracle | offset ms | APE mm | gate | gate APE mm |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in final_rows:
        if row.get("status") != "ok":
            report.append(f"| {row['episode']} | {row['source_episode']} | failed | | | | |")
            continue
        report.append(
            f"| {row['episode']} | {row['source_episode']} | {row['oracle_candidate']} | "
            f"{float(row['oracle_offset_sec']) * 1000.0:+.3f} | {float(row['oracle_ape_mm']):.6f} | "
            f"{row['robust_gate_candidate']} | {float(row['robust_gate_ape_mm']):.6f} |"
        )
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"[OK] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
