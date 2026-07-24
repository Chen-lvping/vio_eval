#!/usr/bin/env python3
"""Direct rich-state diagnosis for rerun RM75 / 0617 episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
TCP_EVAL_SCRIPT = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/rm75_rich_vins_state_alignment"
DEFAULT_HANDEYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"


@dataclass(frozen=True)
class EpisodeStateSpec:
    episode_id: str
    pose_dir: Path
    ground_truth: Path
    metrics_json: Path
    generated_config: Path


@dataclass
class EpisodeAligned:
    spec: EpisodeStateSpec
    time_offset_sec: float
    matched_times: np.ndarray
    matched_gt_pos: np.ndarray
    matched_gt_rot: np.ndarray
    aligned_est_pos: np.ndarray
    aligned_est_rot: np.ndarray
    translation_error_world_mm: np.ndarray
    err_norm_mm: np.ndarray
    speed_mps: np.ndarray
    ang_speed_deg_s: np.ndarray
    progress: np.ndarray


DEFAULT_EPISODES = [
    EpisodeStateSpec(
        episode_id="rm75_0004",
        pose_dir=REPO_ROOT / "data/gripper_data2/episode_20260618_0004/right",
        ground_truth=REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_4.json",
        metrics_json=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0004/metrics.json",
        generated_config=REPO_ROOT / "data/gripper_data2/episode_20260618_0004/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml",
    ),
    EpisodeStateSpec(
        episode_id="episode_20260617_0005",
        pose_dir=REPO_ROOT / "data/gripper_data_1/episode_20260617_0005/right",
        ground_truth=REPO_ROOT / "data/ground_truth/trajectory_samples0617/trajectory_sync_rawpose_005.json",
        metrics_json=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_0617_0005_0614mode_v2_local/metrics.json",
        generated_config=REPO_ROOT / "data/gripper_data_1/episode_20260617_0005/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml",
    ),
    EpisodeStateSpec(
        episode_id="episode_20260617_0006",
        pose_dir=REPO_ROOT / "data/gripper_data_1/episode_20260617_0006/right",
        ground_truth=REPO_ROOT / "data/ground_truth/trajectory_samples0617/trajectory_sync_rawpose_006.json",
        metrics_json=REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_0617_0006_0614mode_v2_local/metrics.json",
        generated_config=REPO_ROOT / "data/gripper_data_1/episode_20260617_0006/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml",
    ),
]


def load_tcp_eval() -> Dict[str, object]:
    return runpy.run_path(str(TCP_EVAL_SCRIPT), run_name="__rich_vins_state_alignment_direct__")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def load_state_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty")
    cols = {name: np.asarray([float(row[name]) for row in rows], dtype=float) for name in reader.fieldnames or []}
    return cols


def interp_series(query_t_us: np.ndarray, src_t_us: np.ndarray, values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return np.interp(query_t_us, src_t_us, values)
    out = np.empty((query_t_us.shape[0], values.shape[1]), dtype=float)
    for idx in range(values.shape[1]):
        out[:, idx] = np.interp(query_t_us, src_t_us, values[:, idx])
    return out


def vector_norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(values, dtype=float), axis=1)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or b.size < 3:
        return float("nan")
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def normalized_progress(gt_pos: np.ndarray) -> np.ndarray:
    step = np.zeros(gt_pos.shape[0], dtype=float)
    if gt_pos.shape[0] > 1:
        step[1:] = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)
    progress = np.cumsum(step)
    total = float(progress[-1]) if progress.size else 0.0
    if total > 1e-12:
        progress /= total
    return progress


def compute_motion_features(times: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> Dict[str, np.ndarray]:
    vel = np.gradient(pos, times, axis=0)
    speed = vector_norm(vel)
    ang_speed_rad_s = np.zeros(times.size, dtype=float)
    for idx in range(1, times.size):
        delta_t = max(float(times[idx] - times[idx - 1]), 1e-9)
        delta_rot = Rotation.from_matrix(rot[idx - 1].T @ rot[idx]).as_rotvec()
        ang_speed_rad_s[idx] = float(np.linalg.norm(delta_rot) / delta_t)
    return {
        "speed": speed,
        "ang_speed_deg_s": np.rad2deg(ang_speed_rad_s),
        "progress": normalized_progress(pos),
    }


def load_config_flags(path: Path) -> Dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    values: Dict[str, object] = {"estimate_extrinsic": None, "estimate_td": None, "td": None}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in values:
            continue
        value = value.strip()
        if value == "":
            continue
        try:
            if key in ("estimate_extrinsic", "estimate_td"):
                values[key] = int(float(value))
            else:
                values[key] = float(value)
        except ValueError:
            values[key] = value
    return values


def load_time_offset(metrics_json: Path) -> float:
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    if "time_offset_sec" not in payload:
        raise KeyError(f"{metrics_json} does not contain time_offset_sec")
    return float(payload["time_offset_sec"])


def align_episode(spec: EpisodeStateSpec, tcp_eval: Dict[str, object], helpers: Dict[str, object], handeye: np.ndarray) -> EpisodeAligned:
    time_offset_sec = load_time_offset(spec.metrics_json)
    gt_times, gt_pos, gt_rot = tcp_eval["load_robot_tcp_trajectory"](spec.ground_truth, helpers)
    est_times, est_pos_tcp, est_rot_tcp = tcp_eval["load_vio_tcp_trajectory"](spec.pose_dir / "pose_data.csv", helpers, handeye, "imu")
    assoc = tcp_eval["associate_by_nearest_time"](
        gt_times,
        gt_pos,
        gt_rot,
        est_times,
        est_pos_tcp,
        est_rot_tcp,
        time_offset_sec,
        0.01,
    )
    matched_times, matched_gt_pos, matched_gt_rot, matched_est_pos, matched_est_rot = assoc
    _se3, aligned_est_pos, aligned_est_rot, _, _ = helpers["evaluate_alignment"](
        "se3",
        False,
        matched_gt_pos,
        matched_gt_rot,
        matched_est_pos,
        matched_est_rot,
        matched_times,
        1.0,
        30,
    )
    translation_error_world_mm = (matched_gt_pos - aligned_est_pos) * 1000.0
    motion = compute_motion_features(matched_times, matched_gt_pos, matched_gt_rot)
    return EpisodeAligned(
        spec=spec,
        time_offset_sec=time_offset_sec,
        matched_times=np.asarray(matched_times, dtype=float),
        matched_gt_pos=np.asarray(matched_gt_pos, dtype=float),
        matched_gt_rot=np.asarray(matched_gt_rot, dtype=float),
        aligned_est_pos=np.asarray(aligned_est_pos, dtype=float),
        aligned_est_rot=np.asarray(aligned_est_rot, dtype=float),
        translation_error_world_mm=np.asarray(translation_error_world_mm, dtype=float),
        err_norm_mm=vector_norm(translation_error_world_mm),
        speed_mps=np.asarray(motion["speed"], dtype=float),
        ang_speed_deg_s=np.asarray(motion["ang_speed_deg_s"], dtype=float),
        progress=np.asarray(motion["progress"], dtype=float),
    )


def state_feature_table(state: Dict[str, np.ndarray], query_t_us: np.ndarray) -> Dict[str, np.ndarray]:
    src_t_us = state["Timestamp_us"]
    features = {
        "ba_norm": interp_series(query_t_us, src_t_us, vector_norm(np.column_stack([state["Ba_X"], state["Ba_Y"], state["Ba_Z"]]))),
        "bg_norm": interp_series(query_t_us, src_t_us, vector_norm(np.column_stack([state["Bg_X"], state["Bg_Y"], state["Bg_Z"]]))),
        "gyr_norm": interp_series(query_t_us, src_t_us, vector_norm(np.column_stack([state["Gyr_X"], state["Gyr_Y"], state["Gyr_Z"]]))),
        "td_ms": interp_series(query_t_us, src_t_us, state["Td"]) * 1000.0,
        "cam0_tx_mm": interp_series(query_t_us, src_t_us, state["Body_Cam0_Tx"]) * 1000.0,
        "cam0_ty_mm": interp_series(query_t_us, src_t_us, state["Body_Cam0_Ty"]) * 1000.0,
        "cam0_tz_mm": interp_series(query_t_us, src_t_us, state["Body_Cam0_Tz"]) * 1000.0,
        "cam1_tx_mm": interp_series(query_t_us, src_t_us, state["Body_Cam1_Tx"]) * 1000.0,
        "cam1_ty_mm": interp_series(query_t_us, src_t_us, state["Body_Cam1_Ty"]) * 1000.0,
        "cam1_tz_mm": interp_series(query_t_us, src_t_us, state["Body_Cam1_Tz"]) * 1000.0,
    }
    for cam in ("Cam0", "Cam1"):
        quat = np.column_stack(
            [
                interp_series(query_t_us, src_t_us, state[f"Body_{cam}_Quat_X"]),
                interp_series(query_t_us, src_t_us, state[f"Body_{cam}_Quat_Y"]),
                interp_series(query_t_us, src_t_us, state[f"Body_{cam}_Quat_Z"]),
                interp_series(query_t_us, src_t_us, state[f"Body_{cam}_Quat_W"]),
            ]
        )
        quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
        rot = Rotation.from_quat(quat)
        rel = rot[0].inv() * rot
        features[f"{cam.lower()}_rot_delta_deg"] = np.rad2deg(rel.magnitude())
    return features


def slice_rmse(values: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if int(np.sum(mask)) == 0:
        return float("nan")
    return rmse(values[mask])


def progress_bin_rows(ep: EpisodeAligned, feature_map: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for left, right in zip(np.linspace(0.0, 1.0, 6)[:-1], np.linspace(0.0, 1.0, 6)[1:]):
        mask = (ep.progress >= left) & (ep.progress < right if right < 1.0 else ep.progress <= right)
        rows.append(
            {
                "episode_id": ep.spec.episode_id,
                "progress_bin": f"{left:.1f}-{right:.1f}",
                "samples": int(np.sum(mask)),
                "err_rmse_mm": slice_rmse(ep.err_norm_mm, mask),
                "speed_mean_mps": float(np.mean(ep.speed_mps[mask])),
                "ba_norm_mean": float(np.mean(feature_map["ba_norm"][mask])),
                "bg_norm_mean": float(np.mean(feature_map["bg_norm"][mask])),
            }
        )
    return rows


def feature_rows(ep: EpisodeAligned, feature_map: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, values in feature_map.items():
        rows.append(
            {
                "episode_id": ep.spec.episode_id,
                "feature": name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "span": float(np.max(values) - np.min(values)),
                "corr_err_norm": safe_corr(values, ep.err_norm_mm),
                "corr_speed": safe_corr(values, ep.speed_mps),
                "corr_progress": safe_corr(values, ep.progress),
            }
        )
    return rows


def summarize_episode(ep: EpisodeAligned, feature_map: Dict[str, np.ndarray], config_flags: Dict[str, object]) -> Dict[str, object]:
    speed = ep.speed_mps
    slow = speed <= np.percentile(speed, 35.0)
    fast = speed >= np.percentile(speed, 65.0)
    stationary = speed < 0.02
    turning = ep.ang_speed_deg_s > 20.0
    mid = (ep.progress >= 0.4) & (ep.progress < 0.6)
    late = ep.progress >= 0.8
    early = ep.progress < 0.2

    return {
        "episode_id": ep.spec.episode_id,
        "matched_samples": int(ep.matched_times.size),
        "duration_s": float(ep.matched_times[-1] - ep.matched_times[0]) if ep.matched_times.size > 1 else 0.0,
        "time_offset_sec": float(ep.time_offset_sec),
        "estimate_extrinsic": config_flags["estimate_extrinsic"],
        "estimate_td": config_flags["estimate_td"],
        "config_td_ms": None if config_flags["td"] is None else float(config_flags["td"]) * 1000.0,
        "ape_rmse_mm": rmse(ep.err_norm_mm),
        "slow35_rmse_mm": slice_rmse(ep.err_norm_mm, slow),
        "fast35_rmse_mm": slice_rmse(ep.err_norm_mm, fast),
        "stationary_rmse_mm": slice_rmse(ep.err_norm_mm, stationary),
        "slow_turn_rmse_mm": slice_rmse(ep.err_norm_mm, slow & turning),
        "slow_straight_rmse_mm": slice_rmse(ep.err_norm_mm, slow & (~turning) & (~stationary)),
        "fast_turn_rmse_mm": slice_rmse(ep.err_norm_mm, fast & turning),
        "fast_straight_rmse_mm": slice_rmse(ep.err_norm_mm, fast & (~turning)),
        "early20_rmse_mm": slice_rmse(ep.err_norm_mm, early),
        "mid40_60_rmse_mm": slice_rmse(ep.err_norm_mm, mid),
        "late20_rmse_mm": slice_rmse(ep.err_norm_mm, late),
        "ba_span": float(np.max(feature_map["ba_norm"]) - np.min(feature_map["ba_norm"])),
        "bg_span": float(np.max(feature_map["bg_norm"]) - np.min(feature_map["bg_norm"])),
        "td_span_ms": float(np.max(feature_map["td_ms"]) - np.min(feature_map["td_ms"])),
        "cam0_t_span_mm": float(
            max(
                np.max(feature_map["cam0_tx_mm"]) - np.min(feature_map["cam0_tx_mm"]),
                np.max(feature_map["cam0_ty_mm"]) - np.min(feature_map["cam0_ty_mm"]),
                np.max(feature_map["cam0_tz_mm"]) - np.min(feature_map["cam0_tz_mm"]),
            )
        ),
        "cam1_t_span_mm": float(
            max(
                np.max(feature_map["cam1_tx_mm"]) - np.min(feature_map["cam1_tx_mm"]),
                np.max(feature_map["cam1_ty_mm"]) - np.min(feature_map["cam1_ty_mm"]),
                np.max(feature_map["cam1_tz_mm"]) - np.min(feature_map["cam1_tz_mm"]),
            )
        ),
        "cam0_rot_span_deg": float(np.max(feature_map["cam0_rot_delta_deg"]) - np.min(feature_map["cam0_rot_delta_deg"])),
        "cam1_rot_span_deg": float(np.max(feature_map["cam1_rot_delta_deg"]) - np.min(feature_map["cam1_rot_delta_deg"])),
        "corr_err_ba": safe_corr(feature_map["ba_norm"], ep.err_norm_mm),
        "corr_err_bg": safe_corr(feature_map["bg_norm"], ep.err_norm_mm),
        "corr_err_td": safe_corr(feature_map["td_ms"], ep.err_norm_mm),
    }


def build_report(summary_rows: Sequence[Mapping[str, object]], feature_rows_all: Sequence[Mapping[str, object]], progress_rows_all: Sequence[Mapping[str, object]]) -> str:
    by_id = {str(row["episode_id"]): row for row in summary_rows}
    p0004 = by_id["rm75_0004"]
    all_fixed_td = all((row["estimate_td"] == 0 and abs(float(row["td_span_ms"])) <= 1e-12) for row in summary_rows)
    all_fixed_ex = all(
        (
            row["estimate_extrinsic"] == 0
            and abs(float(row["cam0_t_span_mm"])) <= 1e-12
            and abs(float(row["cam1_t_span_mm"])) <= 1e-12
            and abs(float(row["cam0_rot_span_deg"])) <= 1e-12
            and abs(float(row["cam1_rot_span_deg"])) <= 1e-12
        )
        for row in summary_rows
    )

    progress_0004 = [row for row in progress_rows_all if row["episode_id"] == "rm75_0004"]
    feature_lookup = {(row["episode_id"], row["feature"]): row for row in feature_rows_all}
    ba_0004 = feature_lookup[("rm75_0004", "ba_norm")]
    bg_0004 = feature_lookup[("rm75_0004", "bg_norm")]

    lines = [
        "# Rich VINS State Alignment",
        "",
        "## Main Findings",
        "",
        (
            f"1. All three reruns use `estimate_td: 0` and `estimate_extrinsic: 0`; "
            f"`Td` stays fixed at about `{float(p0004['config_td_ms']):.4f} ms`, so this rerun does not show any online slow-varying time-offset state."
            if all_fixed_td
            else "1. `Td` is not fully fixed across all reruns, so time-delay drift remains testable from the exported state."
        ),
        (
            "2. The exported camera extrinsics are exactly constant in all three reruns, so this run also does not show any online extrinsic drift state."
            if all_fixed_ex
            else "2. The exported camera extrinsics do move in at least one rerun, so extrinsic drift remains a live hypothesis."
        ),
        (
            f"3. `rm75_0004` still shows strong low-speed / loop-phase concentration: slow35 RMSE `{float(p0004['slow35_rmse_mm']):.2f} mm`, "
            f"fast35 `{float(p0004['fast35_rmse_mm']):.2f} mm`, stationary `{float(p0004['stationary_rmse_mm']):.2f} mm`, "
            f"late20 `{float(p0004['late20_rmse_mm']):.2f} mm` vs mid40-60 `{float(p0004['mid40_60_rmse_mm']):.2f} mm`."
        ),
        (
            f"4. `rm75_0004` bias magnitude alone does not explain the late-loop relapse: `ba_norm` correlation is only `{float(ba_0004['corr_err_norm']):+.3f}`, "
            f"while late-loop error rises again even after `ba_norm` falls well below its early-loop peak."
        ),
        (
            f"5. `bg_norm` is nearly flat in `rm75_0004` with span `{float(bg_0004['span']):.6f}`, "
            "so the remaining error looks more like hidden history / conditioning effects than plain gyro-bias growth."
        ),
        "",
        "## Episode Summary",
        "",
        "| episode | APE mm | slow35 mm | fast35 mm | stationary mm | early20 mm | mid40-60 mm | late20 mm | ba span | bg span | td span ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['episode_id']} | {float(row['ape_rmse_mm']):.3f} | {float(row['slow35_rmse_mm']):.3f} | "
            f"{float(row['fast35_rmse_mm']):.3f} | {float(row['stationary_rmse_mm']):.3f} | "
            f"{float(row['early20_rmse_mm']):.3f} | {float(row['mid40_60_rmse_mm']):.3f} | {float(row['late20_rmse_mm']):.3f} | "
            f"{float(row['ba_span']):.6f} | {float(row['bg_span']):.6f} | {float(row['td_span_ms']):.6f} |"
        )

    lines += [
        "",
        "Interpretation:",
        "- `0005` and `0006` also have low-speed penalties, but `0004` has the clearest late-loop comeback after the middle section improved.",
        "- That shape is much more phase-dependent than a single fixed hand-eye or fixed translation patch would allow.",
        "",
        "## RM75 0004 Progress Bins",
        "",
        "| progress bin | samples | err RMSE mm | speed mps | ba norm mean | bg norm mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in progress_0004:
        lines.append(
            f"| {row['progress_bin']} | {row['samples']} | {float(row['err_rmse_mm']):.2f} | "
            f"{float(row['speed_mean_mps']):.4f} | {float(row['ba_norm_mean']):.4f} | {float(row['bg_norm_mean']):.5f} |"
        )

    lines += [
        "",
        "Interpretation:",
        "- Error is worst in the first and last fifth of the loop, but `ba_norm` only peaks in the first fifth.",
        "- The late-loop jump therefore cannot be explained by a monotonic bias blow-up.",
        "",
        "## What This Actually Proves",
        "",
        "- This rerun validates that the patched VINS now exports the internal states we need for diagnosis.",
        "- It does not validate slow-varying `td` or extrinsic drift, because the current config fixes both states by design.",
        "- The strongest surviving root-cause candidate is still hidden history-dependent drift in low-speed / revisit segments, potentially amplified by orientation-dependent calibration mismatch that the current fixed model cannot absorb.",
        "",
        "## Next Best Move",
        "",
        "- Rerun one branch with `estimate_td` enabled, or with a piecewise time-offset parameterization, because the present run cannot falsify that hypothesis.",
        "- In parallel, instrument feature-track age / revisit count / stationary duration inside the estimator, since the exported explicit states do not explain the remaining `0004` phase pattern.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tcp_eval = load_tcp_eval()
    helpers = tcp_eval["load_helpers"]()
    handeye = tcp_eval["load_tcp_left_camera_transform"](DEFAULT_HANDEYE)

    summary_rows: List[Dict[str, object]] = []
    feature_rows_all: List[Dict[str, object]] = []
    progress_rows_all: List[Dict[str, object]] = []

    for spec in DEFAULT_EPISODES:
        state_path = spec.pose_dir / "pose_state.csv"
        if not state_path.exists():
            raise FileNotFoundError(state_path)
        state = load_state_csv(state_path)
        ep = align_episode(spec, tcp_eval, helpers, handeye)
        query_t_us = ep.matched_times * 1e6
        feature_map = state_feature_table(state, query_t_us)
        config_flags = load_config_flags(spec.generated_config)

        summary_rows.append(summarize_episode(ep, feature_map, config_flags))
        feature_rows_all.extend(feature_rows(ep, feature_map))
        progress_rows_all.extend(progress_bin_rows(ep, feature_map))

    write_csv(
        output_dir / "episode_summary.csv",
        [
            "episode_id",
            "matched_samples",
            "duration_s",
            "time_offset_sec",
            "estimate_extrinsic",
            "estimate_td",
            "config_td_ms",
            "ape_rmse_mm",
            "slow35_rmse_mm",
            "fast35_rmse_mm",
            "stationary_rmse_mm",
            "slow_turn_rmse_mm",
            "slow_straight_rmse_mm",
            "fast_turn_rmse_mm",
            "fast_straight_rmse_mm",
            "early20_rmse_mm",
            "mid40_60_rmse_mm",
            "late20_rmse_mm",
            "ba_span",
            "bg_span",
            "td_span_ms",
            "cam0_t_span_mm",
            "cam1_t_span_mm",
            "cam0_rot_span_deg",
            "cam1_rot_span_deg",
            "corr_err_ba",
            "corr_err_bg",
            "corr_err_td",
        ],
        summary_rows,
    )
    write_csv(
        output_dir / "state_feature_stats.csv",
        [
            "episode_id",
            "feature",
            "mean",
            "std",
            "min",
            "max",
            "span",
            "corr_err_norm",
            "corr_speed",
            "corr_progress",
        ],
        feature_rows_all,
    )
    write_csv(
        output_dir / "progress_bin_stats.csv",
        [
            "episode_id",
            "progress_bin",
            "samples",
            "err_rmse_mm",
            "speed_mean_mps",
            "ba_norm_mean",
            "bg_norm_mean",
        ],
        progress_rows_all,
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "episodes": summary_rows,
                "feature_stats": feature_rows_all,
                "progress_bins": progress_rows_all,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text(build_report(summary_rows, feature_rows_all, progress_rows_all), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
