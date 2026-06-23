#!/usr/bin/env python3
"""Diagnose rm75_0004 VIO state availability, time chain behavior, and extrinsic-like signatures."""

from __future__ import annotations

import argparse
import csv
import json
import re
import runpy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG_SCRIPT = REPO_ROOT / "script/diagnose_rm75_state_generalization.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/workbench/rm75_0004_vio_state_chain_diagnosis"
EP_DIR = REPO_ROOT / "data/gripper_data2/episode_20260618_0004/right"


def load_diag_module() -> Dict[str, object]:
    return runpy.run_path(str(DIAG_SCRIPT), run_name="__rm75_0004_vio_state_chain__")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_header(path: Path) -> List[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return list(next(reader))


def parse_yaml_scalar(path: Path, key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}\s*:\s*(.+?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(1)
    return None


def parse_initial_gyro_bias(log_text: str) -> List[float] | None:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", log_text)
    match = re.search(r"gyroscope bias initial calibration\s+([^\n]+)", clean)
    if not match:
        return None
    parts = match.group(1).split()
    values = []
    for part in parts[:3]:
        try:
            values.append(float(part))
        except ValueError:
            return None
    return values if len(values) == 3 else None


def pose_state_availability() -> Dict[str, object]:
    csv_header = read_csv_header(EP_DIR / "pose_data.csv")
    frame_header = read_csv_header(EP_DIR / "vio_log/frame_level_optimized_pose.csv")
    exported_bias_fields = [field for field in csv_header if "bias" in field.lower()]
    exported_angvel_fields = [field for field in csv_header if field.lower() in {"wx", "wy", "wz", "gx", "gy", "gz"}]
    return {
        "pose_data_columns": csv_header,
        "frame_level_columns": frame_header,
        "bias_columns_present": exported_bias_fields,
        "angular_rate_columns_present": exported_angvel_fields,
        "has_per_frame_bias_state": bool(exported_bias_fields),
    }


def orientation_signature_rows(episode) -> List[Dict[str, object]]:
    tcp_z = np.asarray(episode.matched_gt_rot[:, :, 2], dtype=float)
    err_local_mm = np.einsum(
        "nij,nj->ni",
        np.transpose(np.asarray(episode.matched_gt_rot, dtype=float), (0, 2, 1)),
        np.asarray(episode.translation_error_world, dtype=float) * 1000.0,
    )
    rows = []
    for axis_name, axis_values in (
        ("tcp_z_x", tcp_z[:, 0]),
        ("tcp_z_y", tcp_z[:, 1]),
        ("tcp_z_z", tcp_z[:, 2]),
    ):
        low = np.percentile(axis_values, 20.0)
        high = np.percentile(axis_values, 80.0)
        low_mask = axis_values <= low
        high_mask = axis_values >= high
        rows.append(
            {
                "orientation_axis": axis_name,
                "low_threshold": float(low),
                "high_threshold": float(high),
                "low_mean_local_tx_mm": float(np.mean(err_local_mm[low_mask, 0])),
                "low_mean_local_ty_mm": float(np.mean(err_local_mm[low_mask, 1])),
                "high_mean_local_tx_mm": float(np.mean(err_local_mm[high_mask, 0])),
                "high_mean_local_ty_mm": float(np.mean(err_local_mm[high_mask, 1])),
                "delta_high_minus_low_tx_mm": float(np.mean(err_local_mm[high_mask, 0]) - np.mean(err_local_mm[low_mask, 0])),
                "delta_high_minus_low_ty_mm": float(np.mean(err_local_mm[high_mask, 1]) - np.mean(err_local_mm[low_mask, 1])),
            }
        )
    return rows


def build_report(payload: Mapping[str, object]) -> str:
    state = payload["state_availability"]
    cfg = payload["config"]
    chain = payload["time_chain_reference"]
    orient_rows = payload["orientation_signature"]
    strongest_axis = max(
        orient_rows,
        key=lambda row: max(abs(row["delta_high_minus_low_tx_mm"]), abs(row["delta_high_minus_low_ty_mm"])),
    )
    lines = [
        "# RM75 0004 VIO State / Chain / Extrinsic Diagnosis",
        "",
        "## Main Findings",
        "",
        "1. The exported `pose_data.csv` and `frame_level_optimized_pose.csv` do not contain per-frame IMU bias states, so we cannot directly prove bias drift from the saved pose streams alone.",
        f"2. The VINS run used fixed chain parameters: `estimate_td={cfg['estimate_td']}`, `td={cfg['td']}`, `estimate_extrinsic={cfg['estimate_extrinsic']}`. That means neither time delay nor camera extrinsic was allowed to adapt online in this episode.",
        f"3. Segment-specific best offsets still vary by up to `{chain['max_abs_segment_delta_ms']:.1f} ms`, so one constant time offset is not sufficient to explain the data.",
        f"4. The strongest attitude-linked translation signature appears on `{strongest_axis['orientation_axis']}` with local residual swing `dtx={strongest_axis['delta_high_minus_low_tx_mm']:+.2f} mm`, `dty={strongest_axis['delta_high_minus_low_ty_mm']:+.2f} mm`.",
        "",
        "## Raw State Availability",
        "",
        f"- `pose_data.csv` columns: `{', '.join(state['pose_data_columns'])}`",
        f"- `frame_level_optimized_pose.csv` columns: `{', '.join(state['frame_level_columns'])}`",
        f"- Per-frame bias columns present: `{state['bias_columns_present']}`",
        f"- Per-frame angular-rate columns present: `{state['angular_rate_columns_present']}`",
        f"- Initial gyroscope bias calibration from `vins_node.log`: `{payload['log_summary']['initial_gyro_bias']}`",
        f"- Quality report flagged big IMU acc bias: `{payload['quality_report']['big_imu_acc_bias_count']}` times",
        f"- Quality report flagged big IMU gyr bias: `{payload['quality_report']['big_imu_gyr_bias_count']}` times",
        "",
        "Interpretation:",
        "- We do have an initial gyro-bias estimate from the startup log, but no saved time series of bias states.",
        "- Because `estimate_td` and `estimate_extrinsic` are both disabled, any real slow drift in those quantities would be forced to leak into pose error instead of being absorbed online.",
        "",
        "## Time Chain Constraints",
        "",
        f"- Config `td`: `{cfg['td']}`",
        f"- Config `estimate_td`: `{cfg['estimate_td']}`",
        f"- Closed-loop drift diagnosis max segment offset deviation: `{chain['max_abs_segment_delta_ms']:.1f} ms`",
        f"- Held-out variable-offset result: `{chain['heldout_variable_offset_mm']:.3f} mm` vs baseline `{chain['baseline_mm']:.3f} mm`",
        "",
        "Interpretation:",
        "- The run assumed a fixed `td`, but the segment analysis says different parts of the trajectory prefer materially different offsets.",
        "- At the same time, offset-only correction barely changes held-out APE, so time chain variation is likely a secondary symptom rather than the whole story.",
        "",
        "## Attitude / Extrinsic-Like Signature",
        "",
        "| axis | low thr | high thr | low tx mm | low ty mm | high tx mm | high ty mm | dtx mm | dty mm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in orient_rows:
        lines.append(
            f"| {row['orientation_axis']} | {row['low_threshold']:.3f} | {row['high_threshold']:.3f} | "
            f"{row['low_mean_local_tx_mm']:.2f} | {row['low_mean_local_ty_mm']:.2f} | "
            f"{row['high_mean_local_tx_mm']:.2f} | {row['high_mean_local_ty_mm']:.2f} | "
            f"{row['delta_high_minus_low_tx_mm']:+.2f} | {row['delta_high_minus_low_ty_mm']:+.2f} |"
        )
    lines += [
        "",
        "Interpretation:",
        "- The translation residual changes with TCP orientation, especially in local X/Y, which is exactly the kind of signature a pose-dependent extrinsic error or calibration mismatch can create.",
        "- This is still indirect evidence. The exported logs do not include an online extrinsic state, so we can only say the residual pattern is compatible with extrinsic-like / attitude-related error.",
        "",
        "## Bottom Line",
        "",
        "- Direct bias-drift confirmation is blocked by missing per-frame bias-state exports.",
        "- Time chain evidence is real, but `time-offset-only` correction is too weak to be the main answer.",
        "- The strongest remaining root-cause candidates are: low-speed closed-loop history dependence, attitude-related calibration mismatch, or a combination of both.",
    ]
    return "\n".join(lines) + "\n"


def run_diagnosis(output_dir: Path) -> Dict[str, object]:
    diag = load_diag_module()
    tcp_eval = diag["load_tcp_eval"]()
    helpers = tcp_eval["load_helpers"]()
    handeye = tcp_eval["load_tcp_left_camera_transform"](diag["DEFAULT_HANDEYE"])
    episodes = [diag["analyze_episode"](spec, tcp_eval, helpers, handeye) for spec in diag["EPISODES"]]
    ep = next(episode for episode in episodes if episode.spec.episode_id == "rm75_0004")

    log_text = (EP_DIR / "vio_log/vins_node.log").read_text(encoding="utf-8", errors="replace")
    quality = json.loads((EP_DIR / "vio_log/vins_quality_report.json").read_text(encoding="utf-8"))
    cfg_path = EP_DIR / "vio_log/generated_config/StereoIMU-vinsfusion.yaml"
    chain_report = json.loads((REPO_ROOT / "data/evaluation/workbench/rm75_0004_closed_loop_drift_diagnosis/diagnosis.json").read_text(encoding="utf-8"))
    variable_offset = json.loads((REPO_ROOT / "data/evaluation/workbench/rm75_variable_offset_experiment/experiment.json").read_text(encoding="utf-8"))

    segment_offsets = chain_report["rm75_0004"]["segment_offsets"]
    payload = {
        "state_availability": pose_state_availability(),
        "log_summary": {
            "initial_gyro_bias": parse_initial_gyro_bias(log_text),
        },
        "quality_report": {
            "big_imu_acc_bias_count": int(quality["log_analysis"]["failures"]["big_imu_acc_bias"]["count"]),
            "big_imu_gyr_bias_count": int(quality["log_analysis"]["failures"]["big_imu_gyr_bias"]["count"]),
            "misalign_visual_imu_count": int(quality["log_analysis"]["failures"]["misalign_visual_imu"]["count"]),
        },
        "config": {
            "estimate_td": parse_yaml_scalar(cfg_path, "estimate_td"),
            "td": parse_yaml_scalar(cfg_path, "td"),
            "estimate_extrinsic": parse_yaml_scalar(cfg_path, "estimate_extrinsic"),
        },
        "time_chain_reference": {
            "max_abs_segment_delta_ms": float(max(abs(row["delta_vs_global_ms"]) for row in segment_offsets)),
            "baseline_mm": float(variable_offset["baseline"]["ape_translation_rmse_mm"]),
            "heldout_variable_offset_mm": float(variable_offset["heldout_variable_offset"]["ape_translation_rmse_mm"]),
        },
        "orientation_signature": orientation_signature_rows(ep),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnosis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(output_dir / "orientation_signature.csv", payload["orientation_signature"][0].keys(), payload["orientation_signature"])
    (output_dir / "REPORT.md").write_text(build_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_diagnosis(args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
