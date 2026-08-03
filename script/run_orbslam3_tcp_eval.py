#!/usr/bin/env python3
"""Run ORB-SLAM3 on one Fanysense episode and evaluate it with the TCP pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODE = REPO_ROOT / "data/gripper_data2/episode_20260618_0004"
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data/ground_truth/rm75/rm75_pose_traj_4.json"
DEFAULT_HAND_EYE = REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/evaluation/workbench"
DEFAULT_ORB_ROOT = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_clean")
DEFAULT_PANGOLIN_LIB = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/Pangolin_v06/install/lib")
DEFAULT_EXPORTER = Path("/home/chenlvping/5_skill/lwm/vinsfusion_ws/scripts/fanysense_episode_to_orbslam3_euroc.py")
DEFAULT_CONFIG_GENERATOR = Path("/home/chenlvping/5_skill/lwm/vinsfusion_ws/scripts/gen_orbslam3_config_from_calib.py")
CONVERT_TUM = REPO_ROOT / "script/postprocess/convert_tum_to_pose_csv.py"
SMOOTH_TUM = REPO_ROOT / "script/postprocess/smooth_text_trajectory.py"
RESAMPLE_GT_TIMES = REPO_ROOT / "script/postprocess/resample_pose_csv_to_gt_timestamps.py"
TCP_EVAL = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
VIEWER = REPO_ROOT / "script/visualize_single_tcp_trajectory_3d.py"
SHADOW_COMPARE = REPO_ROOT / "script/diagnose/compare_orb_stereo_inertial_consistency.py"
DEFAULT_COLLEAGUE_ORB_ROOT = Path("/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_colleague_sync")
DEFAULT_TCP_STRICT_SYNC_OFFSET_SEC = -0.10488409042358399
DEFAULT_SMOOTH_WINDOW = 5
DEFAULT_SMOOTH_PASSES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--handeye-yaml", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--output-dir", type=Path, help="ORB-SLAM3 run/export directory")
    parser.add_argument("--eval-dir", type=Path, help="TCP evaluation output directory")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--camera-rig", choices=("stereo_left", "stereo_right"), default="stereo_right")
    parser.add_argument("--mode", choices=("stereo", "stereo-inertial"), default="stereo-inertial")
    parser.add_argument("--feature-preset", choices=("baseline", "low-texture", "aggressive"), default="low-texture")
    parser.add_argument("--orb-root", type=Path, default=DEFAULT_ORB_ROOT)
    parser.add_argument("--exporter", type=Path, default=DEFAULT_EXPORTER)
    parser.add_argument("--config-generator", type=Path, default=DEFAULT_CONFIG_GENERATOR)
    parser.add_argument("--extra-ld-path", action="append", default=[])
    parser.add_argument("--timeout-sec", type=int, default=360)
    parser.add_argument(
        "--offline-deterministic",
        dest="offline_deterministic",
        action="store_true",
        help="Wait for LocalMapping after each input frame to make offline results scheduling-independent.",
    )
    parser.add_argument(
        "--no-offline-deterministic",
        dest="offline_deterministic",
        action="store_false",
        help="Disable per-frame LocalMapping waits.",
    )
    parser.add_argument("--offline-wait-timeout-sec", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--reuse-trajectory", action="store_true")
    parser.add_argument(
        "--min-inertial-coverage",
        type=float,
        default=0.25,
        help="Fall back to stereo when the inertial trajectory covers less than this fraction of input frames.",
    )
    parser.add_argument(
        "--no-stereo-fallback",
        dest="allow_stereo_fallback",
        action="store_false",
        help="Keep the requested stereo-inertial result even when its coverage is poor; initialization failures remain errors.",
    )
    parser.add_argument(
        "--stereo-shadow-gate",
        action="store_true",
        help="Run the validated offline stereo branch and fall back when it disagrees with stereo-inertial.",
    )
    parser.add_argument("--stereo-shadow-max-disagreement-mm", type=float, default=15.0)
    parser.add_argument("--colleague-orb-root", type=Path, default=DEFAULT_COLLEAGUE_ORB_ROOT)
    parser.add_argument("--stereo-fallback-offset-json", type=Path, default=None)
    parser.add_argument("--trajectory-name", default="")
    parser.add_argument("--vins-config", type=Path, default=None)
    parser.add_argument(
        "--vins-noise-only",
        action="store_true",
        help="Use IMU noise from --vins-config while keeping camera/IMU extrinsics from the episode calibration.",
    )
    parser.add_argument("--match-vins-config", action="store_true", help="Auto-resolve StereoIMU-vinsfusion.yaml under the episode rig folder")
    parser.add_argument("--apply-vins-td", action="store_true", help="Shift exported camera timestamps using td from the matched VINS config")
    parser.add_argument("--camera-time-shift-sec", type=float, default=None, help="Explicit camera timestamp shift for the EuRoC export stage")
    parser.add_argument("--vins-noise-mode", choices=("copy", "orb_from_vins"), default="orb_from_vins")
    parser.add_argument("--invert-tbc", action="store_true", help="Regenerate ORB settings with inverse Tbc")
    parser.add_argument("--nfeatures", type=int)
    parser.add_argument("--ini-fast", type=int)
    parser.add_argument("--min-fast", type=int)
    parser.add_argument("--imu-fast-init", type=int, choices=(0, 1))
    parser.add_argument("--gyro-noise", type=float)
    parser.add_argument("--acc-noise", type=float)
    parser.add_argument("--gyro-walk", type=float)
    parser.add_argument("--acc-walk", type=float)
    parser.add_argument(
        "--final-ba-iters",
        type=int,
        default=0,
        help="If > 0, run one final offline global backend BA before trajectory export.",
    )
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--clahe-clip-limit", type=float, default=None)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=None)
    parser.add_argument(
        "--smooth-trajectory",
        dest="smooth_trajectory",
        action="store_true",
        help="Apply symmetric moving-average smoothing to the ORB trajectory before CSV conversion.",
    )
    parser.add_argument(
        "--no-smooth-trajectory",
        dest="smooth_trajectory",
        action="store_false",
        help="Disable trajectory smoothing and evaluate the raw ORB trajectory.",
    )
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    parser.add_argument("--smooth-passes", type=int, default=DEFAULT_SMOOTH_PASSES)
    parser.add_argument(
        "--strict-sync-to-gt",
        dest="strict_sync_to_gt",
        action="store_true",
        help="Apply a fixed estimate time offset, then resample the ORB pose CSV onto GT timestamps before evaluation.",
    )
    parser.add_argument(
        "--no-strict-sync-to-gt",
        dest="strict_sync_to_gt",
        action="store_false",
        help="Disable fixed-offset strict GT timestamp resampling and fall back to the original evo auto-offset flow.",
    )
    parser.add_argument(
        "--strict-sync-offset-sec",
        type=float,
        default=DEFAULT_TCP_STRICT_SYNC_OFFSET_SEC,
        help="Constant offset added to estimate timestamps before strict sync resampling.",
    )
    parser.add_argument(
        "--strict-sync-offset-json",
        type=Path,
        default=None,
        help="Optional JSON mapping episode dir names to strict-sync offset seconds. Explicit --strict-sync-offset-sec still wins.",
    )
    parser.add_argument(
        "--strict-sync-max-gap-sec",
        type=float,
        default=0.05,
        help="Maximum interpolation bracket gap allowed when resampling to GT timestamps.",
    )
    parser.add_argument(
        "--strict-sync-offset-scan-span-ms",
        type=float,
        default=0.0,
        help="If > 0, evaluate a symmetric scan around the resolved strict-sync offset center.",
    )
    parser.add_argument(
        "--strict-sync-offset-scan-step-ms",
        type=float,
        default=0.0,
        help="Step size in milliseconds for the strict-sync offset scan.",
    )
    parser.add_argument(
        "--strict-sync-offset-scan-score",
        choices=("ape", "rpe", "rotation", "composite"),
        default="composite",
        help="Priority used to choose the best strict-sync offset candidate.",
    )
    parser.add_argument(
        "--estimate-frame",
        choices=("auto", "camera", "imu"),
        default="auto",
        help="Frame to pass into the TCP evaluator. auto uses camera for stereo and imu for stereo-inertial.",
    )
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--viewer-max-points", type=int, default=3000)
    parser.add_argument("--rpe-distance-m", type=float, default=0.001, help="RPE delta distance in meters (default 0.001).")
    parser.set_defaults(smooth_trajectory=True)
    parser.set_defaults(strict_sync_to_gt=True)
    parser.set_defaults(offline_deterministic=False)
    return parser.parse_args()


def run_checked(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def run_logged(
    cmd: list[str],
    log_path: Path,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout_sec: int = 0,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        if env and env.get("LD_LIBRARY_PATH"):
            handle.write("LD_LIBRARY_PATH=" + env["LD_LIBRARY_PATH"] + "\n")
        handle.write("\n")
        handle.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec if timeout_sec > 0 else None,
                check=False,
            )
            return int(proc.returncode)
        except subprocess.TimeoutExpired:
            handle.write(f"\n[TIMEOUT] exceeded {timeout_sec} seconds\n")
            return 124


def add_option(cmd: list[str], option: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([option, str(value)])


def auto_vins_config_path(episode_dir: Path, camera_rig: str) -> Path:
    rig_side = "left" if camera_rig == "stereo_left" else "right"
    return episode_dir / rig_side / "vio_log/generated_config/StereoIMU-vinsfusion.yaml"


def resolve_vins_config(args: argparse.Namespace) -> Path | None:
    if args.vins_config is not None:
        return args.vins_config.expanduser().resolve()
    if args.match_vins_config:
        return auto_vins_config_path(args.episode_dir, args.camera_rig).resolve()
    return None


def default_run_dir(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    suffix = f"{args.episode_dir.name}_{args.camera_rig}_{args.mode}_{args.feature_preset}"
    if args.invert_tbc:
        suffix += "_invert_tbc"
    if args.imu_fast_init is not None:
        suffix += f"_fastinit{args.imu_fast_init}"
    if args.vins_config is not None or args.match_vins_config:
        suffix += "_vins_match"
    if args.vins_noise_only:
        suffix += "_vins_noise_only"
    if args.apply_vins_td or args.match_vins_config:
        suffix += "_vins_td"
    if args.nfeatures is not None:
        suffix += f"_nf{args.nfeatures}"
    if args.ini_fast is not None:
        suffix += f"_if{args.ini_fast}"
    if args.min_fast is not None:
        suffix += f"_mf{args.min_fast}"
    if args.clahe:
        suffix += "_clahe"
    if args.smooth_trajectory:
        suffix += f"_smooth_w{args.smooth_window}_p{args.smooth_passes}"
    if args.strict_sync_to_gt:
        suffix += "_strictsync"
    return args.output_root / f"orbslam3_fresh_runs_{stamp}" / suffix


def default_eval_dir(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    seq = args.episode_dir.name.rsplit("_", 1)[-1] if "_" in args.episode_dir.name else "unknown"
    suffix = f"evo_orb_tcp_rm75_{seq}_fresh_{args.mode.replace('-', '_')}_{args.feature_preset}"
    if args.invert_tbc:
        suffix += "_invert_tbc"
    if args.imu_fast_init is not None:
        suffix += f"_fastinit{args.imu_fast_init}"
    if args.vins_config is not None or args.match_vins_config:
        suffix += "_vins_match"
    if args.vins_noise_only:
        suffix += "_vins_noise_only"
    if args.apply_vins_td or args.match_vins_config:
        suffix += "_vins_td"
    if args.nfeatures is not None:
        suffix += f"_nf{args.nfeatures}"
    if args.ini_fast is not None:
        suffix += f"_if{args.ini_fast}"
    if args.min_fast is not None:
        suffix += f"_mf{args.min_fast}"
    if args.clahe:
        suffix += "_clahe"
    if args.smooth_trajectory:
        suffix += f"_smooth_w{args.smooth_window}_p{args.smooth_passes}"
    if args.strict_sync_to_gt:
        suffix += "_strictsync"
    return args.output_root / f"{suffix}_{stamp}"


def settings_path(output_dir: Path, args: argparse.Namespace) -> Path:
    return output_dir / f"orbslam3_{args.camera_rig}_{args.mode}.yaml"


def build_export_cmd(args: argparse.Namespace, output_dir: Path) -> list[str]:
    vins_config = resolve_vins_config(args)
    cmd = [
        sys.executable,
        str(args.exporter),
        str(args.episode_dir),
        str(output_dir),
        "--camera-rig",
        args.camera_rig,
        "--mode",
        args.mode,
        "--orb-root",
        str(args.orb_root),
        "--config-generator",
        str(args.config_generator),
        "--feature-preset",
        args.feature_preset,
        "--max-frames",
        str(args.max_frames),
    ]
    if vins_config is not None:
        cmd.extend(["--vins-config", str(vins_config), "--vins-noise-mode", args.vins_noise_mode])
        if args.vins_noise_only:
            cmd.append("--vins-noise-only")
    add_option(cmd, "--camera-time-shift-sec", args.camera_time_shift_sec)
    if args.apply_vins_td or args.match_vins_config:
        cmd.append("--apply-vins-td")
    if args.force_export:
        cmd.append("--force")
    for option, value in (
        ("--nfeatures", args.nfeatures),
        ("--ini-fast", args.ini_fast),
        ("--min-fast", args.min_fast),
        ("--imu-fast-init", args.imu_fast_init),
        ("--gyro-noise", args.gyro_noise),
        ("--acc-noise", args.acc_noise),
        ("--gyro-walk", args.gyro_walk),
        ("--acc-walk", args.acc_walk),
        ("--clahe-clip-limit", args.clahe_clip_limit),
        ("--clahe-tile-grid-size", args.clahe_tile_grid_size),
    ):
        add_option(cmd, option, value)
    if args.clahe:
        cmd.append("--clahe")
    return cmd


def generate_settings(args: argparse.Namespace, output_dir: Path) -> list[str]:
    vins_config = resolve_vins_config(args)
    cmd = [
        sys.executable,
        str(args.config_generator),
        str(args.episode_dir),
        str(settings_path(output_dir, args)),
        "--camera-rig",
        args.camera_rig,
        "--mode",
        args.mode,
        "--feature-preset",
        args.feature_preset,
    ]
    if vins_config is not None and args.mode != "stereo":
        cmd.extend(["--vins-config", str(vins_config), "--vins-noise-mode", args.vins_noise_mode])
        if args.vins_noise_only:
            cmd.append("--vins-noise-only")
    for option, value in (
        ("--nfeatures", args.nfeatures),
        ("--ini-fast", args.ini_fast),
        ("--min-fast", args.min_fast),
        ("--imu-fast-init", args.imu_fast_init),
        ("--gyro-noise", args.gyro_noise),
        ("--acc-noise", args.acc_noise),
        ("--gyro-walk", args.gyro_walk),
        ("--acc-walk", args.acc_walk),
    ):
        add_option(cmd, option, value)
    return cmd


def regenerate_settings_if_needed(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.invert_tbc:
        return
    vins_config = resolve_vins_config(args)
    cmd = [
        sys.executable,
        str(args.config_generator),
        str(args.episode_dir),
        str(settings_path(output_dir, args)),
        "--camera-rig",
        args.camera_rig,
        "--mode",
        args.mode,
        "--feature-preset",
        args.feature_preset,
        "--invert-tbc",
    ]
    if vins_config is not None:
        cmd.extend(["--vins-config", str(vins_config), "--vins-noise-mode", args.vins_noise_mode])
        if args.vins_noise_only:
            cmd.append("--vins-noise-only")
    for option, value in (
        ("--nfeatures", args.nfeatures),
        ("--ini-fast", args.ini_fast),
        ("--min-fast", args.min_fast),
        ("--imu-fast-init", args.imu_fast_init),
        ("--gyro-noise", args.gyro_noise),
        ("--acc-noise", args.acc_noise),
        ("--gyro-walk", args.gyro_walk),
        ("--acc-walk", args.acc_walk),
    ):
        add_option(cmd, option, value)
    run_checked(cmd)


def orb_executable(orb_root: Path, mode: str) -> Path:
    if mode == "stereo":
        return orb_root / "Examples/Stereo/stereo_euroc"
    return orb_root / "Examples/Stereo-Inertial/stereo_inertial_euroc"


def runtime_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    candidates: list[Path | str] = [
        DEFAULT_PANGOLIN_LIB,
        # An explicitly requested runtime library is an experiment contract,
        # not merely an additional search location.  Keep it ahead of the
        # mutable ORB root so a locked reproduction can select its exact ABI.
        *args.extra_ld_path,
        args.orb_root / "lib",
        args.orb_root / "Thirdparty/DBoW2/lib",
        args.orb_root / "Thirdparty/g2o/lib",
    ]
    paths = [str(path) for path in candidates if Path(path).is_dir()]
    if env.get("LD_LIBRARY_PATH"):
        paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(paths))
    env["ORB_SLAM3_ENABLE_VIEWER"] = "0"
    env["ORB_SLAM3_OFFLINE_WAIT_LOCAL_MAPPING"] = "1" if args.offline_deterministic else "0"
    env["ORB_SLAM3_OFFLINE_WAIT_TIMEOUT_SEC"] = str(float(args.offline_wait_timeout_sec))
    if int(args.final_ba_iters) > 0:
        env["ORB_SLAM3_FINAL_BA_ITERS"] = str(int(args.final_ba_iters))
    return env


def trajectory_name(args: argparse.Namespace) -> str:
    if args.trajectory_name:
        return args.trajectory_name
    return f"{args.episode_dir.name}_{args.camera_rig}_{args.mode}"


def has_rows(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return any(line.strip() and not line.startswith("#") for line in handle)


def find_trajectory(output_dir: Path, name: str) -> Path:
    candidates = [
        output_dir / f"f_{name}.txt",
        output_dir / "CameraTrajectory.txt",
        output_dir / f"kf_{name}.txt",
        output_dir / "KeyFrameTrajectory.txt",
    ]
    for path in candidates:
        if has_rows(path):
            return path
    raise FileNotFoundError("ORB-SLAM3 did not produce a non-empty trajectory")


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def read_summary(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["metric"] + "_rmse"] = float(row["rmse"])
    return result


def is_imu_init_failure(log_path: Path) -> bool:
    """Check if ORB-SLAM3 failed due to IMU initialization."""
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = [
        "not enough acceleration",
        "not enough startup motion",
        "not enough reliable depth for inertial",
        "Not enough motion for initializing",
        "not IMU meas",
    ]
    return any(p in text for p in patterns)


def check_required(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required file(s):\n" + "\n".join(missing))


def load_offset_overrides(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in payload.items()}


def resolve_strict_sync_offset_sec(args: argparse.Namespace) -> float | None:
    if not args.strict_sync_to_gt:
        return None
    explicit = getattr(args, "_strict_sync_offset_sec_explicit", False)
    if explicit:
        return args.strict_sync_offset_sec
    overrides = load_offset_overrides(args.strict_sync_offset_json)
    if args.episode_dir.name in overrides:
        return float(overrides[args.episode_dir.name])
    return args.strict_sync_offset_sec


def strict_sync_suffix(offset_sec: float) -> str:
    ms = int(round(float(offset_sec) * 1000.0))
    sign = "p" if ms >= 0 else "m"
    return f"strictsync_{sign}{abs(ms):03d}ms"


def apply_strict_sync_with_offset(args: argparse.Namespace, pose_csv: Path, offset_sec: float) -> Path:
    if offset_sec is None:
        raise ValueError("--strict-sync-offset-sec is required when --strict-sync-to-gt is enabled")
    out_csv = pose_csv.with_name(f"{pose_csv.stem}_{strict_sync_suffix(offset_sec)}.csv")
    run_checked(
        [
            sys.executable,
            str(RESAMPLE_GT_TIMES),
            "--estimate-csv",
            str(pose_csv),
            "--gt-json",
            str(args.ground_truth),
            "--output-csv",
            str(out_csv),
            "--time-offset-sec",
            str(offset_sec),
            "--max-gap-sec",
            str(args.strict_sync_max_gap_sec),
        ]
    )
    return out_csv


def apply_strict_sync(args: argparse.Namespace, pose_csv: Path) -> Path:
    if args.resolved_strict_sync_offset_sec is None:
        raise ValueError("--strict-sync-offset-sec is required when --strict-sync-to-gt is enabled")
    return apply_strict_sync_with_offset(args, pose_csv, args.resolved_strict_sync_offset_sec)


def build_eval_cmd(
    args: argparse.Namespace,
    estimate_csv: Path,
    eval_output_dir: Path,
    estimate_frame: str,
) -> list[str]:
    eval_cmd = [
        sys.executable,
        str(TCP_EVAL),
        "--estimate",
        str(estimate_csv),
        "--ground-truth",
        str(args.ground_truth),
        "--output-dir",
        str(eval_output_dir),
        "--handeye-yaml",
        str(args.handeye_yaml),
        "--calibration-json",
        str(args.episode_dir / "calibration.json"),
        "--camera-rig",
        args.camera_rig,
        "--estimate-frame",
        estimate_frame,
        "--time-association",
        "evo",
        "--rpe-distance-m",
        str(args.rpe_distance_m),
    ]
    if args.strict_sync_to_gt:
        eval_cmd.extend(["--time-offset-sec", "0.0", "--t-max-diff-sec", "0.0001"])
    else:
        eval_cmd.extend(
            [
                "--time-offset-auto",
                "--t-max-diff-sec",
                "0.01",
                "--time-offset-auto-span-sec",
                "2.0",
                "--time-offset-auto-step-sec",
                "0.01",
                "--time-offset-auto-score",
                "translation",
                "--time-offset-auto-min-samples",
                "50",
            ]
        )
    return eval_cmd


def score_summary(summary: dict[str, float], score_mode: str) -> tuple[float, ...]:
    ape = float(summary["ape_translation_se3_rmse"])
    rpe = float(summary["rpe_translation_5cm_rmse"])
    rot_ape = float(summary["ape_rotation_se3_rmse"])
    rot_rpe = float(summary["rpe_rotation_5cm_rmse"])
    if score_mode == "ape":
        return (ape, rpe, rot_ape, rot_rpe)
    if score_mode == "rpe":
        return (rpe, ape, rot_ape, rot_rpe)
    if score_mode == "rotation":
        return (rot_ape, rot_rpe, ape, rpe)
    return (ape, rpe, rot_ape, rot_rpe)


def strict_sync_scan_offsets(center_sec: float, span_ms: float, step_ms: float) -> list[float]:
    if span_ms <= 0.0:
        return [float(center_sec)]
    if step_ms <= 0.0:
        raise ValueError("--strict-sync-offset-scan-step-ms must be > 0 when scan span is enabled")
    span_sec = float(span_ms) / 1000.0
    step_sec = float(step_ms) / 1000.0
    steps_each_side = int(math.floor((span_sec / step_sec) + 1e-9))
    offsets = [center_sec + idx * step_sec for idx in range(-steps_each_side, steps_each_side + 1)]
    if not any(abs(value - center_sec) < 1e-12 for value in offsets):
        offsets.append(center_sec)
    unique_sorted = sorted({round(value, 12) for value in offsets})
    return [float(value) for value in unique_sorted]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_smooth_trajectory(args: argparse.Namespace, trajectory_path: Path) -> Path:
    if not args.smooth_trajectory:
        return trajectory_path
    out_path = trajectory_path.with_name(
        f"{trajectory_path.stem}_smooth_w{args.smooth_window}_p{args.smooth_passes}{trajectory_path.suffix}"
    )
    run_checked(
        [
            sys.executable,
            str(SMOOTH_TUM),
            "--input",
            str(trajectory_path),
            "--output",
            str(out_path),
            "--window",
            str(args.smooth_window),
            "--passes",
            str(args.smooth_passes),
        ]
    )
    return out_path


def run_colleague_stereo_shadow(
    args: argparse.Namespace,
    output_dir: Path,
    inertial_trajectory: Path,
    inertial_settings: Path,
) -> tuple[Path, Path, dict[str, object]]:
    """Run the validated offline stereo branch and compare it to the IMU result."""
    shadow_dir = output_dir / "stereo_shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    runner = args.colleague_orb_root / "scripts/run_fays_orbslam3_stereo_right.py"
    smoother = args.colleague_orb_root / "scripts/smooth_pose_csv.py"
    raw_csv = shadow_dir / "pose_raw.csv"
    smooth_csv = shadow_dir / "pose_smooth.csv"
    shadow_name = f"shadow_{args.episode_dir.resolve().name}_ffba10"
    shadow_tum = shadow_dir / f"f_{shadow_name}.txt"
    run_checked(
        [
            sys.executable,
            str(runner),
            "--offline-accurate",
            "--tracking-mode",
            "stereo",
            "--episode-dir",
            str(args.episode_dir),
            "--work-dir",
            str(shadow_dir),
            "--output-csv",
            str(raw_csv),
            "--trajectory-name",
            shadow_name,
            "--gba-iterations",
            "100",
            "--full-frame-ba-iterations",
            "10",
            "--orb-features",
            "1200",
            "--orb-init-fast",
            "20",
            "--orb-min-fast",
            "7",
            "--overwrite",
        ],
        cwd=args.colleague_orb_root,
    )
    run_checked(
        [
            sys.executable,
            str(smoother),
            "--input-csv",
            str(raw_csv),
            "--output-csv",
            str(smooth_csv),
            "--position-window",
            "21",
            "--position-poly",
            "2",
            "--rotation-window",
            "9",
        ],
        cwd=args.colleague_orb_root,
    )
    consistency_json = shadow_dir / "stereo_inertial_consistency.json"
    run_checked(
        [
            sys.executable,
            str(SHADOW_COMPARE),
            "--inertial-tum",
            str(inertial_trajectory),
            "--stereo-tum",
            str(shadow_tum),
            "--inertial-settings",
            str(inertial_settings),
            "--output-json",
            str(consistency_json),
        ]
    )
    consistency = json.loads(consistency_json.read_text(encoding="utf-8"))
    return shadow_tum, smooth_csv, consistency


def main() -> int:
    args = parse_args()
    requested_mode = args.mode
    requested_estimate_frame = args.estimate_frame
    argv = sys.argv[1:]
    args._strict_sync_offset_sec_explicit = "--strict-sync-offset-sec" in argv
    args.episode_dir = args.episode_dir.expanduser().resolve()
    args.ground_truth = args.ground_truth.expanduser().resolve()
    args.handeye_yaml = args.handeye_yaml.expanduser().resolve()
    args.orb_root = args.orb_root.expanduser().resolve()
    args.colleague_orb_root = args.colleague_orb_root.expanduser().resolve()
    args.exporter = args.exporter.expanduser().resolve()
    args.config_generator = args.config_generator.expanduser().resolve()
    if args.strict_sync_offset_json is not None:
        args.strict_sync_offset_json = args.strict_sync_offset_json.expanduser().resolve()
    if args.stereo_fallback_offset_json is not None:
        args.stereo_fallback_offset_json = args.stereo_fallback_offset_json.expanduser().resolve()
    args.resolved_strict_sync_offset_sec = resolve_strict_sync_offset_sec(args)
    if args.strict_sync_to_gt and args.resolved_strict_sync_offset_sec is None:
        raise ValueError("--strict-sync-offset-sec must be set when --strict-sync-to-gt is used")
    vins_config = resolve_vins_config(args)
    output_dir = (args.output_dir or default_run_dir(args)).expanduser().resolve()
    eval_dir = (args.eval_dir or default_eval_dir(args)).expanduser().resolve()

    required = [
        args.episode_dir / "calibration.json",
        args.ground_truth,
        args.handeye_yaml,
        args.exporter,
        args.config_generator,
        orb_executable(args.orb_root, args.mode),
        args.orb_root / "Vocabulary/ORBvoc.txt",
    ]
    if vins_config is not None:
        required.append(vins_config)
    if args.smooth_trajectory:
        required.append(SMOOTH_TUM)
    if args.strict_sync_to_gt:
        required.append(RESAMPLE_GT_TIMES)
    if args.strict_sync_offset_json is not None:
        required.append(args.strict_sync_offset_json)
    if args.stereo_shadow_gate:
        required.extend(
            [
                SHADOW_COMPARE,
                args.colleague_orb_root / "scripts/run_fays_orbslam3_stereo_right.py",
                args.colleague_orb_root / "scripts/smooth_pose_csv.py",
                args.colleague_orb_root / "Examples/Stereo/stereo_euroc_offline",
            ]
        )
    if args.stereo_fallback_offset_json is not None:
        required.append(args.stereo_fallback_offset_json)
    check_required(required)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    settings_yaml = settings_path(output_dir, args)
    export_cmd = build_export_cmd(args, output_dir)
    settings_cmd = generate_settings(args, output_dir)
    fallback_to_stereo = False
    fallback_reason = ""
    input_frame_rows = 0
    inertial_trajectory_rows: int | None = None
    inertial_coverage: float | None = None
    shadow_consistency: dict[str, object] = {}
    shadow_error = ""
    fallback_pose_csv: Path | None = None
    viewer_cmd: list[str] | None = None
    fell_back = False
    final_ba_executed = False

    run_checked(export_cmd)
    run_checked(settings_cmd)
    input_frame_rows = count_rows(output_dir / "times.txt")

    name = trajectory_name(args)
    executable = orb_executable(args.orb_root, args.mode)
    orb_cmd = [
        str(executable),
        str(args.orb_root / "Vocabulary/ORBvoc.txt"),
        str(settings_path(output_dir, args)),
        str(output_dir),
        str(output_dir / "times.txt"),
        name,
    ]
    trajectory_path = output_dir / f"f_{name}.txt"
    if args.reuse_trajectory and has_rows(trajectory_path):
        print(f"[SKIP] reusing trajectory {trajectory_path}")
    else:
        rc = run_logged(orb_cmd, output_dir / "orbslam3_native.log", cwd=output_dir, env=runtime_env(args), timeout_sec=args.timeout_sec)
        if (
            rc != 0
            and args.mode == "stereo-inertial"
            and args.allow_stereo_fallback
            and is_imu_init_failure(output_dir / "orbslam3_native.log")
        ):
            print(f"[WARN] stereo-inertial IMU init failed (rc={rc}), falling back to stereo-only mode")
            args.mode = "stereo"
            fallback_to_stereo = True
            fallback_reason = "imu_init_failure"
            settings_cmd = generate_settings(args, output_dir)
            run_checked(settings_cmd)
            name = trajectory_name(args)
            orb_cmd = [
                str(orb_executable(args.orb_root, args.mode)),
                str(args.orb_root / "Vocabulary/ORBvoc.txt"),
                str(settings_path(output_dir, args)),
                str(output_dir),
                str(output_dir / "times.txt"),
                name,
            ]
            trajectory_path = output_dir / f"f_{name}.txt"
            rc = run_logged(orb_cmd, output_dir / "orbslam3_native.log", cwd=output_dir, env=runtime_env(args), timeout_sec=args.timeout_sec)
            fell_back = True
        if rc != 0 and not has_rows(trajectory_path):
            raise RuntimeError(f"ORB-SLAM3 failed with code {rc}; see {output_dir / 'orbslam3_native.log'}")

        if int(args.final_ba_iters) > 0 and requested_mode == "stereo-inertial":
            native_log = (output_dir / "orbslam3_native.log").read_text(encoding="utf-8", errors="replace")
            final_ba_executed = f"[FINAL_INERTIAL_BA] completed iterations={int(args.final_ba_iters)}" in native_log
            if not final_ba_executed:
                raise RuntimeError(
                    "requested final inertial BA was not confirmed by ORB-SLAM3; "
                    f"see {output_dir / 'orbslam3_native.log'}"
                )

    trajectory_path = find_trajectory(output_dir, name)
    if not fell_back and args.mode == "stereo-inertial":
        inertial_trajectory_rows = count_rows(trajectory_path)
        inertial_coverage = inertial_trajectory_rows / max(input_frame_rows, 1)
    min_inertial_rows = max(20, int(math.ceil(args.min_inertial_coverage * input_frame_rows)))
    # A short but non-empty inertial trajectory is still a failed initialization.
    if (
        not fell_back
        and args.mode == "stereo-inertial"
        and args.allow_stereo_fallback
        and inertial_trajectory_rows is not None
        and inertial_trajectory_rows < min_inertial_rows
    ):
        print(
            f"[WARN] stereo-inertial coverage too low "
            f"({inertial_trajectory_rows}/{input_frame_rows}={inertial_coverage:.3f}, "
            f"required>={args.min_inertial_coverage:.3f}); falling back to stereo"
        )
        args.mode = "stereo"
        fallback_to_stereo = True
        fallback_reason = "low_inertial_trajectory_coverage"
        settings_cmd = generate_settings(args, output_dir)
        run_checked(settings_cmd)
        name = trajectory_name(args)
        orb_cmd = [
            str(orb_executable(args.orb_root, args.mode)),
            str(args.orb_root / "Vocabulary/ORBvoc.txt"),
            str(settings_path(output_dir, args)),
            str(output_dir),
            str(output_dir / "times.txt"),
            name,
        ]
        trajectory_path = output_dir / f"f_{name}.txt"
        rc = run_logged(orb_cmd, output_dir / "orbslam3_native.log", cwd=output_dir, env=runtime_env(args), timeout_sec=args.timeout_sec)
        if rc != 0 and not has_rows(trajectory_path):
            raise RuntimeError(f"ORB-SLAM3 (stereo) failed with code {rc}; see {output_dir / 'orbslam3_native.log'}")
        trajectory_path = find_trajectory(output_dir, name)
        fell_back = True
    if not fell_back and args.mode == "stereo-inertial" and args.stereo_shadow_gate:
        inertial_trajectory = find_trajectory(output_dir, name)
        inertial_settings = settings_path(output_dir, args)
        try:
            shadow_tum, shadow_smooth_csv, shadow_consistency = run_colleague_stereo_shadow(
                args,
                output_dir,
                inertial_trajectory,
                inertial_settings,
            )
            disagreement_mm = float(shadow_consistency["translation_rmse_mm"])
            if disagreement_mm > float(args.stereo_shadow_max_disagreement_mm):
                print(
                    f"[WARN] stereo/IMU disagreement {disagreement_mm:.3f} mm exceeds "
                    f"{args.stereo_shadow_max_disagreement_mm:.3f} mm; selecting offline stereo fallback"
                )
                args.mode = "stereo"
                fallback_to_stereo = True
                fallback_reason = "stereo_shadow_disagreement"
                trajectory_path = shadow_tum
                fallback_pose_csv = shadow_smooth_csv
                fell_back = True
                fallback_offsets = load_offset_overrides(args.stereo_fallback_offset_json)
                source_episode = args.episode_dir.resolve().name
                if source_episode in fallback_offsets:
                    args.resolved_strict_sync_offset_sec = float(fallback_offsets[source_episode])
            else:
                print(
                    f"[OK] stereo/IMU disagreement {disagreement_mm:.3f} mm is within "
                    f"{args.stereo_shadow_max_disagreement_mm:.3f} mm; keeping stereo-inertial"
                )
        except Exception as exc:
            shadow_error = str(exc)
            # A failed shadow branch cannot validate the IMU trajectory.  Fail closed
            # to an independently generated pure-stereo trajectory.
            print(f"[WARN] stereo shadow gate failed; selecting pure stereo fallback: {exc}")
            args.mode = "stereo"
            fallback_to_stereo = True
            fallback_reason = "stereo_shadow_gate_failure"
            settings_cmd = generate_settings(args, output_dir)
            run_checked(settings_cmd)
            name = trajectory_name(args)
            orb_cmd = [
                str(orb_executable(args.orb_root, args.mode)),
                str(args.orb_root / "Vocabulary/ORBvoc.txt"),
                str(settings_path(output_dir, args)),
                str(output_dir),
                str(output_dir / "times.txt"),
                name,
            ]
            trajectory_path = output_dir / f"f_{name}.txt"
            rc = run_logged(
                orb_cmd,
                output_dir / "orbslam3_native.log",
                cwd=output_dir,
                env=runtime_env(args),
                timeout_sec=args.timeout_sec,
            )
            if rc != 0 and not has_rows(trajectory_path):
                raise RuntimeError(f"ORB-SLAM3 (stereo fallback) failed with code {rc}; see {output_dir / 'orbslam3_native.log'}")
            trajectory_path = find_trajectory(output_dir, name)
            fell_back = True
    if not fell_back:
        trajectory_path = find_trajectory(output_dir, name)
    if fallback_pose_csv is not None:
        eval_trajectory_path = trajectory_path
        pose_csv = fallback_pose_csv
    else:
        eval_trajectory_path = maybe_smooth_trajectory(args, trajectory_path)
        pose_csv_name = "orb_pose_data_imu.csv" if args.mode == "stereo-inertial" else "orb_pose_data_camera.csv"
        if args.smooth_trajectory:
            pose_csv_name = pose_csv_name.replace(".csv", f"_smooth_w{args.smooth_window}_p{args.smooth_passes}.csv")
        pose_csv = output_dir / pose_csv_name
        run_checked([sys.executable, str(CONVERT_TUM), str(eval_trajectory_path), str(pose_csv)])
    estimate_frame = args.estimate_frame
    if estimate_frame == "auto":
        estimate_frame = "imu" if args.mode == "stereo-inertial" else "camera"

    scan_enabled = bool(args.strict_sync_to_gt and float(args.strict_sync_offset_scan_span_ms) > 0.0)
    scan_rows: list[dict[str, object]] = []
    best_scan_row: dict[str, object] | None = None

    if scan_enabled:
        if args.resolved_strict_sync_offset_sec is None:
            raise ValueError("--strict-sync-offset-sec must resolve before running an offset scan")
        candidate_offsets = strict_sync_scan_offsets(
            args.resolved_strict_sync_offset_sec,
            float(args.strict_sync_offset_scan_span_ms),
            float(args.strict_sync_offset_scan_step_ms),
        )
        scan_root = eval_dir / "strict_sync_offset_scan"
        scan_root.mkdir(parents=True, exist_ok=True)
        for candidate_offset in candidate_offsets:
            candidate_csv = apply_strict_sync_with_offset(args, pose_csv, candidate_offset)
            candidate_eval_dir = scan_root / strict_sync_suffix(candidate_offset)
            candidate_eval_dir.mkdir(parents=True, exist_ok=True)
            candidate_eval_cmd = build_eval_cmd(args, candidate_csv, candidate_eval_dir, estimate_frame)
            run_checked(candidate_eval_cmd)
            candidate_summary = read_summary(candidate_eval_dir / "summary.csv")
            row = {
                "offset_sec": candidate_offset,
                "offset_ms": candidate_offset * 1000.0,
                "estimate_csv": str(candidate_csv),
                "eval_dir": str(candidate_eval_dir),
                "ape_translation_se3_rmse": candidate_summary["ape_translation_se3_rmse"],
                "rpe_translation_5cm_rmse": candidate_summary["rpe_translation_5cm_rmse"],
                "ape_rotation_se3_rmse": candidate_summary["ape_rotation_se3_rmse"],
                "rpe_rotation_5cm_rmse": candidate_summary["rpe_rotation_5cm_rmse"],
            }
            scan_rows.append(row)
            if best_scan_row is None or score_summary(candidate_summary, args.strict_sync_offset_scan_score) < score_summary(best_scan_row, args.strict_sync_offset_scan_score):
                best_scan_row = row
        if best_scan_row is None:
            raise RuntimeError("strict-sync offset scan produced no candidates")
        args.resolved_strict_sync_offset_sec = float(best_scan_row["offset_sec"])
        estimate_csv = Path(str(best_scan_row["estimate_csv"]))
        eval_cmd = build_eval_cmd(args, estimate_csv, eval_dir, estimate_frame)
        run_checked(eval_cmd)
        scan_csv = eval_dir / "strict_sync_offset_scan.csv"
        write_csv(
            scan_csv,
            [
                "offset_sec",
                "offset_ms",
                "ape_translation_se3_rmse",
                "rpe_translation_5cm_rmse",
                "ape_rotation_se3_rmse",
                "rpe_rotation_5cm_rmse",
                "estimate_csv",
                "eval_dir",
            ],
            scan_rows,
        )
    else:
        estimate_csv = apply_strict_sync(args, pose_csv) if args.strict_sync_to_gt else pose_csv
        eval_cmd = build_eval_cmd(args, estimate_csv, eval_dir, estimate_frame)
        run_checked(eval_cmd)

    if not args.skip_viewer:
        viewer_cmd = [
            sys.executable,
            str(VIEWER),
            "--eval-dir",
            str(eval_dir),
            "--output-dir",
            str(eval_dir),
            "--max-points",
            str(args.viewer_max_points),
        ]
        run_checked(viewer_cmd)

    summary = read_summary(eval_dir / "summary.csv")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cli_args": argv,
        "episode_dir": str(args.episode_dir),
        "ground_truth": str(args.ground_truth),
        "orb_root": str(args.orb_root),
        "output_dir": str(output_dir),
        "eval_dir": str(eval_dir),
        "requested_mode": requested_mode,
        "mode": args.mode,
        "camera_rig": args.camera_rig,
        "feature_preset": args.feature_preset,
        "offline_deterministic": bool(args.offline_deterministic),
        "offline_wait_timeout_sec": args.offline_wait_timeout_sec,
        "invert_tbc": bool(args.invert_tbc),
        "imu_fast_init": args.imu_fast_init,
        "final_ba_iters": args.final_ba_iters,
        "final_inertial_ba_executed": final_ba_executed,
        "vins_config": str(vins_config) if vins_config is not None else "",
        "vins_noise_only": bool(args.vins_noise_only),
        "match_vins_config": bool(args.match_vins_config),
        "apply_vins_td": bool(args.apply_vins_td or args.match_vins_config),
        "camera_time_shift_sec": args.camera_time_shift_sec,
        "vins_noise_mode": args.vins_noise_mode,
        "clahe": bool(args.clahe),
        "clahe_clip_limit": args.clahe_clip_limit,
        "clahe_tile_grid_size": args.clahe_tile_grid_size,
        "settings_yaml": str(settings_yaml),
        "orb_native_log": str(output_dir / "orbslam3_native.log"),
        "export_manifest": str(output_dir / "export_manifest.json") if (output_dir / "export_manifest.json").is_file() else "",
        "fallback_to_stereo": fallback_to_stereo,
        "fallback_reason": fallback_reason,
        "input_frame_rows": input_frame_rows,
        "inertial_trajectory_rows": inertial_trajectory_rows,
        "inertial_coverage": inertial_coverage,
        "min_inertial_coverage": args.min_inertial_coverage,
        "stereo_shadow_gate": bool(args.stereo_shadow_gate),
        "stereo_shadow_max_disagreement_mm": args.stereo_shadow_max_disagreement_mm,
        "stereo_shadow_consistency": shadow_consistency,
        "stereo_shadow_error": shadow_error,
        "stereo_fallback_offset_json": str(args.stereo_fallback_offset_json) if args.stereo_fallback_offset_json else "",
        "trajectory": str(trajectory_path),
        "trajectory_rows": count_rows(trajectory_path),
        "eval_trajectory": str(eval_trajectory_path),
        "pose_csv": str(pose_csv),
        "estimate_csv_for_eval": str(estimate_csv),
        "requested_estimate_frame": requested_estimate_frame,
        "estimate_frame": estimate_frame,
        "smooth_trajectory": bool(args.smooth_trajectory),
        "smooth_window": args.smooth_window,
        "smooth_passes": args.smooth_passes,
        "strict_sync_to_gt": bool(args.strict_sync_to_gt),
        "strict_sync_offset_sec": args.resolved_strict_sync_offset_sec,
        "strict_sync_offset_json": str(args.strict_sync_offset_json) if args.strict_sync_offset_json else "",
        "strict_sync_max_gap_sec": args.strict_sync_max_gap_sec,
        "strict_sync_offset_scan_enabled": scan_enabled,
        "strict_sync_offset_scan_span_ms": args.strict_sync_offset_scan_span_ms if scan_enabled else 0.0,
        "strict_sync_offset_scan_step_ms": args.strict_sync_offset_scan_step_ms if scan_enabled else 0.0,
        "strict_sync_offset_scan_score": args.strict_sync_offset_scan_score if scan_enabled else "",
        "strict_sync_offset_scan_csv": str(eval_dir / "strict_sync_offset_scan.csv") if scan_enabled else "",
        "strict_sync_offset_scan_best": best_scan_row or {},
        "resolved_parameter_overrides": {
            "nfeatures": args.nfeatures,
            "ini_fast": args.ini_fast,
            "min_fast": args.min_fast,
            "gyro_noise": args.gyro_noise,
            "acc_noise": args.acc_noise,
            "gyro_walk": args.gyro_walk,
            "acc_walk": args.acc_walk,
            "final_ba_iters": args.final_ba_iters,
        },
        "commands": {
            "export": export_cmd,
            "generate_settings": settings_cmd,
            "orb_run": orb_cmd,
            "tcp_eval": eval_cmd,
            "viewer": viewer_cmd or [],
        },
        "summary_rmse": summary,
    }
    (eval_dir / "orbslam3_tcp_eval_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] ORB-SLAM3 TCP evaluation finished: {eval_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
