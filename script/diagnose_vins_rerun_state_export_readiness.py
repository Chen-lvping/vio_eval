#!/usr/bin/env python3
"""Audit VINS rerun readiness, richer-state export hooks, and independent acceptance baselines."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data/evaluation/workbench/rm75_vins_rerun_state_export_readiness"
VINS_WS = Path("/home/chenlvping/0_SLAM/vinsfusion_ws")
VINS_BINARY = VINS_WS / "devel/lib/vins/vins_offline_bag_runner"
VINS_VIS = VINS_WS / "src/VINS-Fusion/vins_estimator/src/utility/visualization.cpp"
VINS_VIS_H = VINS_WS / "src/VINS-Fusion/vins_estimator/src/utility/visualization.h"
VINS_EST = VINS_WS / "src/VINS-Fusion/vins_estimator/src/estimator/estimator.cpp"

RM75_0004_METRICS = REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_rm75_0004_vins_latest/metrics.json"
EP_0005_METRICS = REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_0617_0005_0614mode_v2_local/metrics.json"
EP_0006_METRICS = REPO_ROOT / "data/evaluation/workbench/evo_vio_tcp_0617_0006_0614mode_v2_local/metrics.json"
LOW_SPEED_DIAG = REPO_ROOT / "data/evaluation/workbench/rm75_0004_closed_loop_drift_diagnosis/diagnosis.json"
STATE_CHAIN_DIAG = REPO_ROOT / "data/evaluation/workbench/rm75_0004_vio_state_chain_diagnosis/diagnosis.json"
SPEED_SPLIT_REPORT = REPO_ROOT / "data/evaluation/workbench/rm75_candidate_speed_split_poly_experiment/REPORT.md"


def run_cmd(args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_first_line(path: Path, needle: str) -> int | None:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return idx
    return None


def missing_libs_from_ldd(binary: Path) -> List[str]:
    proc = run_cmd(["ldd", str(binary)])
    missing: List[str] = []
    for line in proc.stdout.splitlines():
        if "=> not found" not in line:
            continue
        name = line.split("=>", 1)[0].strip()
        if name and name not in missing:
            missing.append(name)
    return missing


def command_path(name: str) -> str | None:
    found = shutil.which(name)
    return found if found else None


def metric_rmse_mm(path: Path) -> float:
    metrics = load_json(path)
    return float(metrics["evo"]["ape_translation_se3"]["rmse"])


def extract_best_speed_split_delta(report_path: Path) -> float | None:
    text = report_path.read_text(encoding="utf-8")
    needle = "Held-out delta: `"
    if needle not in text:
        return None
    start = text.index(needle) + len(needle)
    end = text.index(" mm`", start)
    return float(text[start:end])


def build_payload() -> Dict[str, Any]:
    runtime = {
        "rosrun": command_path("rosrun"),
        "roscore": command_path("roscore"),
        "catkin_make": command_path("catkin_make"),
        "python3": command_path("python3"),
        "opt_ros_distros": sorted(p.name for p in Path("/opt/ros").iterdir() if p.is_dir()) if Path("/opt/ros").exists() else [],
    }
    missing_libs = missing_libs_from_ldd(VINS_BINARY)

    low_speed = load_json(LOW_SPEED_DIAG)
    rm75_0004 = low_speed["rm75_0004"]
    progress_bins = rm75_0004["progress_bins"]
    revisit_row = next(row for row in low_speed["revisit_comparison"] if row["episode_id"] == "rm75_0004")
    state_chain = load_json(STATE_CHAIN_DIAG)
    strongest_orientation = max(
        state_chain["orientation_signature"],
        key=lambda row: max(abs(row["delta_high_minus_low_tx_mm"]), abs(row["delta_high_minus_low_ty_mm"])),
    )

    payload = {
        "runtime": runtime,
        "binary_deps": {
            "binary": str(VINS_BINARY),
            "missing_libs": missing_libs,
            "missing_count": len(missing_libs),
        },
        "source_hooks": {
            "visualization_header": str(VINS_VIS_H),
            "visualization_cpp": str(VINS_VIS),
            "estimator_cpp": str(VINS_EST),
            "pose_state_csv_line": find_first_line(VINS_VIS, "pose_state.csv"),
            "frame_level_state_csv_line": find_first_line(VINS_VIS, "frame_level_optimized_state.csv"),
            "save_vio_pose_signature_line": find_first_line(VINS_VIS_H, "void saveVioPose("),
            "save_frame_pose_signature_line": find_first_line(VINS_VIS_H, "void saveFrameLevelOptimizedPose("),
            "save_vio_pose_call_line": find_first_line(VINS_EST, "saveVioPose(pose_P,"),
            "save_frame_pose_call_line": find_first_line(VINS_EST, "saveFrameLevelOptimizedPose(Ps[frame_count],"),
            "exports_bias_and_td_fields": "Ba_X" in VINS_VIS.read_text(encoding="utf-8") and "Td," in VINS_VIS.read_text(encoding="utf-8"),
        },
        "rm75_0004_low_speed_evidence": {
            "progress_bin_0_0_2_rmse_mm": float(progress_bins[0]["mean_err_rmse_mm"]),
            "progress_bin_0_8_1_0_rmse_mm": float(progress_bins[-1]["mean_err_rmse_mm"]),
            "slow35_best_offset_delta_ms": float(next(row["delta_vs_global_ms"] for row in rm75_0004["segment_offsets"] if row["segment"] == "slow35")),
            "revisit_mean_error_diff_mm": float(revisit_row["mean_error_diff_mm"]),
            "strongest_orientation_axis": strongest_orientation["orientation_axis"],
            "strongest_orientation_dtx_mm": float(strongest_orientation["delta_high_minus_low_tx_mm"]),
            "strongest_orientation_dty_mm": float(strongest_orientation["delta_high_minus_low_ty_mm"]),
        },
        "independent_acceptance_baselines": {
            "rm75_0004_mm": metric_rmse_mm(RM75_0004_METRICS),
            "episode_20260617_0005_mm": metric_rmse_mm(EP_0005_METRICS),
            "episode_20260617_0006_mm": metric_rmse_mm(EP_0006_METRICS),
            "speed_split_candidate_delta_mm": extract_best_speed_split_delta(SPEED_SPLIT_REPORT),
        },
    }
    return payload


def build_report(payload: Dict[str, Any]) -> str:
    runtime = payload["runtime"]
    deps = payload["binary_deps"]
    hooks = payload["source_hooks"]
    low_speed = payload["rm75_0004_low_speed_evidence"]
    accept = payload["independent_acceptance_baselines"]
    speed_split_delta_text = (
        f"{accept['speed_split_candidate_delta_mm']:+.3f} mm"
        if accept["speed_split_candidate_delta_mm"] is not None
        else "unavailable"
    )
    lines = [
        "# RM75 VINS Rerun / State Export Readiness",
        "",
        "## Main Findings",
        "",
        f"1. The current machine is not ready for an honest local rerun of offline VINS: `rosrun={runtime['rosrun']}`, `roscore={runtime['roscore']}`, `catkin_make={runtime['catkin_make']}`, and `ldd` still reports `{deps['missing_count']}` unresolved runtime libraries for `vins_offline_bag_runner`.",
        f"2. The algorithm already has the internal states we need. Bias, time-delay, and extrinsic states are present in the VINS source, and this round added richer CSV export hooks at `{hooks['pose_state_csv_line']}` and `{hooks['frame_level_state_csv_line']}` in `{hooks['visualization_cpp']}`.",
        f"3. The low-speed / closed-loop root-cause hypothesis still holds quantitatively: early-loop RMSE is `{low_speed['progress_bin_0_0_2_rmse_mm']:.2f} mm`, late-loop RMSE is `{low_speed['progress_bin_0_8_1_0_rmse_mm']:.2f} mm`, `slow35` prefers `+{low_speed['slow35_best_offset_delta_ms']:.1f} ms`, and revisit inconsistency stays at `{low_speed['revisit_mean_error_diff_mm']:.2f} mm`.",
        f"4. Independent acceptance is not yet below `10 mm`: `rm75_0004={accept['rm75_0004_mm']:.3f} mm`, `20260617_0005={accept['episode_20260617_0005_mm']:.3f} mm`, `20260617_0006={accept['episode_20260617_0006_mm']:.3f} mm`.",
        "",
        "## Rerun Blocker",
        "",
        f"- `/opt/ros` distros present: `{', '.join(runtime['opt_ros_distros']) if runtime['opt_ros_distros'] else 'none'}`",
        f"- `vins_offline_bag_runner` unresolved libs: `{', '.join(deps['missing_libs'][:12])}`",
        "- Interpretation: the existing RM75 outputs came from a valid offline batch pipeline, but the current shell does not have the ROS1 runtime those binaries were linked against.",
        "",
        "## State Export Hook",
        "",
        f"- Header hook: `{hooks['visualization_header']}` line `{hooks['save_vio_pose_signature_line']}`",
        f"- Export implementation: `{hooks['visualization_cpp']}` lines `{hooks['pose_state_csv_line']}` and `{hooks['frame_level_state_csv_line']}`",
        f"- Estimator call sites: `{hooks['estimator_cpp']}` lines `{hooks['save_vio_pose_call_line']}` and `{hooks['save_frame_pose_call_line']}`",
        "- New files written after a rebuild/rerun: `pose_state.csv`, `frame_level_optimized_state.csv`",
        "- New fields include base-link pose/velocity plus `Ba`, `Bg`, raw gyro, current `td`, and both camera extrinsics.",
        "",
        "## Low-Speed Closed-Loop Evidence",
        "",
        f"- Early/late loop RMSE split: `{low_speed['progress_bin_0_0_2_rmse_mm']:.2f} mm` vs `{low_speed['progress_bin_0_8_1_0_rmse_mm']:.2f} mm`.",
        f"- `slow35` segment best-offset deviation: `+{low_speed['slow35_best_offset_delta_ms']:.1f} ms`.",
        f"- Revisit inconsistency mean: `{low_speed['revisit_mean_error_diff_mm']:.2f} mm`.",
        f"- Strongest attitude-linked residual swing: `{low_speed['strongest_orientation_axis']}` with `dtx={low_speed['strongest_orientation_dtx_mm']:+.2f} mm`, `dty={low_speed['strongest_orientation_dty_mm']:+.2f} mm`.",
        "- Interpretation: this still looks more like low-speed historical drift plus attitude-related calibration mismatch than a single fixed hand-eye offset.",
        "",
        "## Independent Acceptance",
        "",
        f"- `episode_20260617_0005`: `{accept['episode_20260617_0005_mm']:.3f} mm`",
        f"- `episode_20260617_0006`: `{accept['episode_20260617_0006_mm']:.3f} mm`",
        f"- Best current held-out RM75 model-side gain on `0004`: `{speed_split_delta_text}`",
        "- Interpretation: we do have independent episodes to validate on, but none currently proves real `<10 mm` generalization.",
        "",
        "## Next Step",
        "",
        "1. Rebuild the patched VINS workspace in a ROS1-compatible runtime and rerun `rm75_0004`, `20260617_0005`, and `20260617_0006` so the new state CSVs are populated.",
        "2. Use the richer state CSVs to test whether low-speed error spikes align with `Bg/Ba` drift, `td` motion, or online extrinsic motion; reject any explanation that stays flat while the residual surges.",
        "3. Only after that state evidence is in hand should we decide whether to harden the model-side patch or change estimator-side calibration/time-offset handling.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    (OUT_DIR / "diagnosis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT_DIR / "REPORT.md").write_text(build_report(payload), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
