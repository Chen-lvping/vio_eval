#!/usr/bin/env python3
"""Run a controlled Tbc x IMU-noise ablation on episode_gripper_0020."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_SINGLE = ROOT / "script/run_orbslam3_tcp_eval.py"
COMPARE_SHADOW = ROOT / "script/diagnose/compare_orb_stereo_inertial_consistency.py"
EPISODE = ROOT / "data/gripper/gripper_all_seq/episode_gripper_0020"
GROUND_TRUTH = ROOT / "data/gripper/gripper_all_seq/rm75_pose_traj_20.json"
LOCAL_VINS = EPISODE / "right/vio_log/generated_config/StereoIMU-vinsfusion.yaml"
VALIDATED_VINS = (
    ROOT
    / "data/gripper/gripper_data2/episode_20260618_0004/right/vio_log/generated_config/"
    "StereoIMU-vinsfusion.yaml"
)
STEREO_REFERENCE = (
    ROOT
    / "data/evaluation/workbench/rm75_shadow_gate_local_calib_20260722/"
    "orbslam3_batch_runs_20260722_094429/"
    "episode_gripper_0020_stereo_right_stereo-inertial_low-texture_fastinit0_smooth_strictsync/"
    "stereo_shadow/f_shadow_episode_20260707_0003_ffba10.txt"
)
OFFSET_SEC = -0.084884090424

# ORB-SLAM3 units after the repository's orb_from_vins conversion.
LOCAL_NOISE = {
    "gyro_noise": 2.52093729e-05,
    "acc_noise": 2.247028284e-04,
    "gyro_walk": 8.27197327e-06,
    "acc_walk": 7.16249455e-05,
}
VALIDATED_NOISE = {
    "gyro_noise": 1.563132044e-03,
    "acc_noise": 9.378792266e-03,
    "gyro_walk": 5.0e-04,
    "acc_walk": 5.0e-03,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/evaluation/workbench/rm75_0020_tbc_noise_ablation_20260722",
    )
    parser.add_argument("--timeout-sec", type=int, default=480)
    return parser.parse_args()


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files:\n" + "\n".join(missing))


def read_summary(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["metric"]] = float(row["rmse"])
    return result


def run_variant(
    *,
    root: Path,
    label: str,
    tbc_profile: str,
    noise_profile: str,
    vins_config: Path,
    noise_override: dict[str, float] | None,
    timeout_sec: int,
) -> dict[str, object]:
    variant_root = root / label
    run_dir = variant_root / "run"
    eval_dir = variant_root / "eval"
    cmd = [
        sys.executable,
        str(RUN_SINGLE),
        "--episode-dir",
        str(EPISODE),
        "--ground-truth",
        str(GROUND_TRUTH),
        "--output-dir",
        str(run_dir),
        "--eval-dir",
        str(eval_dir),
        "--camera-rig",
        "stereo_right",
        "--mode",
        "stereo-inertial",
        "--feature-preset",
        "low-texture",
        "--vins-config",
        str(vins_config),
        "--vins-noise-mode",
        "orb_from_vins",
        "--nfeatures",
        "3000",
        "--ini-fast",
        "12",
        "--min-fast",
        "3",
        "--imu-fast-init",
        "0",
        "--min-inertial-coverage",
        "0.25",
        "--smooth-window",
        "5",
        "--smooth-passes",
        "1",
        "--strict-sync-offset-sec",
        str(OFFSET_SEC),
        "--final-ba-iters",
        "0",
        "--timeout-sec",
        str(timeout_sec),
        "--no-offline-deterministic",
        "--skip-viewer",
        "--force-export",
    ]
    if noise_override is not None:
        for option, key in (
            ("--gyro-noise", "gyro_noise"),
            ("--acc-noise", "acc_noise"),
            ("--gyro-walk", "gyro_walk"),
            ("--acc-walk", "acc_walk"),
        ):
            cmd.extend([option, str(noise_override[key])])

    print(f"[ABLATION] {label}: Tbc={tbc_profile}, noise={noise_profile}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

    manifest_path = eval_dir / "orbslam3_tcp_eval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = read_summary(eval_dir / "summary.csv")
    consistency_path = eval_dir / "stereo_inertial_consistency.json"
    consistency: dict[str, object] = {}
    if manifest.get("mode") == "stereo-inertial":
        compare_cmd = [
            sys.executable,
            str(COMPARE_SHADOW),
            "--inertial-tum",
            str(manifest["trajectory"]),
            "--stereo-tum",
            str(STEREO_REFERENCE),
            "--inertial-settings",
            str(manifest["settings_yaml"]),
            "--output-json",
            str(consistency_path),
        ]
        subprocess.run(compare_cmd, cwd=ROOT, check=True)
        consistency = json.loads(consistency_path.read_text(encoding="utf-8"))

    return {
        "variant": label,
        "tbc_profile": tbc_profile,
        "noise_profile": noise_profile,
        "status": "ok",
        "requested_mode": manifest.get("requested_mode", ""),
        "selected_mode": manifest.get("mode", ""),
        "fallback_to_stereo": manifest.get("fallback_to_stereo", False),
        "fallback_reason": manifest.get("fallback_reason", ""),
        "inertial_coverage": manifest.get("inertial_coverage", ""),
        "ape_translation_se3_rmse_mm": summary["ape_translation_se3"],
        "rpe_translation_5cm_rmse_mm": summary["rpe_translation_5cm"],
        "ape_rotation_se3_rmse_deg": summary["ape_rotation_se3"],
        "shadow_translation_rmse_mm": consistency.get("translation_rmse_mm", ""),
        "shadow_rotation_rmse_deg": consistency.get("rotation_rmse_deg", ""),
        "shadow_sim3_scale_stereo_over_inertial": consistency.get(
            "sim3_scale_stereo_over_inertial", ""
        ),
        "vins_config": str(vins_config),
        "noise_override": json.dumps(noise_override or {}, sort_keys=True),
        "settings_yaml": manifest.get("settings_yaml", ""),
        "manifest": str(manifest_path),
        "eval_dir": str(eval_dir),
        "consistency_json": str(consistency_path) if consistency else "",
    }


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    require_files([RUN_SINGLE, COMPARE_SHADOW, EPISODE / "calibration.json", GROUND_TRUTH, LOCAL_VINS, VALIDATED_VINS, STEREO_REFERENCE])

    variants = [
        ("A_local_tbc_local_noise", "local", "local", LOCAL_VINS, None),
        ("B_validated_tbc_local_noise", "validated", "local", VALIDATED_VINS, LOCAL_NOISE),
        ("C_local_tbc_validated_noise", "local", "validated", LOCAL_VINS, VALIDATED_NOISE),
        ("D_validated_tbc_validated_noise", "validated", "validated", VALIDATED_VINS, None),
    ]
    rows: list[dict[str, object]] = []
    for label, tbc_profile, noise_profile, vins_config, noise_override in variants:
        try:
            row = run_variant(
                root=output_root,
                label=label,
                tbc_profile=tbc_profile,
                noise_profile=noise_profile,
                vins_config=vins_config,
                noise_override=noise_override,
                timeout_sec=args.timeout_sec,
            )
        except Exception as exc:
            row = {
                "variant": label,
                "tbc_profile": tbc_profile,
                "noise_profile": noise_profile,
                "status": "failed",
                "error": str(exc),
            }
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    summary_csv = output_root / "ablation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "episode": str(EPISODE),
        "ground_truth": str(GROUND_TRUTH),
        "camera_rig": "stereo_right",
        "strict_sync_offset_sec": OFFSET_SEC,
        "stereo_reference": str(STEREO_REFERENCE),
        "stereo_reference_ape_mm": 6.982474,
        "local_noise_orb_units": LOCAL_NOISE,
        "validated_noise_orb_units": VALIDATED_NOISE,
        "rows": rows,
    }
    summary_json = output_root / "ablation_summary.json"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# RM75 episode_gripper_0020 Tbc x IMU Noise Ablation",
        "",
        f"- Offset: `{OFFSET_SEC:+.12f} s` (fixed, no scan)",
        "- ORB: stereo-inertial, 3000/12/3, fastInit=0, smooth w5/p1, final BA=0",
        "- Stereo reference APE: `6.982474 mm`",
        "",
        "| variant | Tbc | noise | status | mode | coverage | APE mm | RPE mm | shadow mm | scale |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def value(key: str, digits: int = 6) -> str:
            raw = row.get(key, "")
            if raw in ("", None):
                return "n/a"
            try:
                return f"{float(raw):.{digits}f}"
            except (TypeError, ValueError):
                return str(raw)

        report.append(
            f"| {row['variant']} | {row['tbc_profile']} | {row['noise_profile']} | {row['status']} | "
            f"{row.get('selected_mode', 'n/a')} | {value('inertial_coverage', 3)} | "
            f"{value('ape_translation_se3_rmse_mm')} | {value('rpe_translation_5cm_rmse_mm')} | "
            f"{value('shadow_translation_rmse_mm', 3)} | "
            f"{value('shadow_sim3_scale_stereo_over_inertial', 9)} |"
        )
    (output_root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"[OK] summary: {summary_csv}")
    print(f"[OK] report: {output_root / 'REPORT.md'}")
    return 0 if all(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
