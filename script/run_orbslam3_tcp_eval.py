#!/usr/bin/env python3
"""Run ORB-SLAM3 on one Fanysense episode and evaluate it with the TCP pipeline."""

from __future__ import annotations

import argparse
import csv
import json
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
CONVERT_TUM = REPO_ROOT / "script/convert_tum_to_pose_csv.py"
SMOOTH_TUM = REPO_ROOT / "script/smooth_text_trajectory.py"
RESAMPLE_GT_TIMES = REPO_ROOT / "script/resample_pose_csv_to_gt_timestamps.py"
TCP_EVAL = REPO_ROOT / "script/evaluate_vio_tcp_camera_evo.py"
VIEWER = REPO_ROOT / "script/visualize_single_tcp_trajectory_3d.py"
DEFAULT_TCP_STRICT_SYNC_OFFSET_SEC = -0.10488409042358399
DEFAULT_SMOOTH_WINDOW = 7
DEFAULT_SMOOTH_PASSES = 2


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
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--reuse-trajectory", action="store_true")
    parser.add_argument("--trajectory-name", default="")
    parser.add_argument("--vins-config", type=Path, default=None)
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
        "--estimate-frame",
        choices=("auto", "camera", "imu"),
        default="auto",
        help="Frame to pass into the TCP evaluator. auto uses camera for stereo and imu for stereo-inertial.",
    )
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--viewer-max-points", type=int, default=3000)
    parser.set_defaults(smooth_trajectory=True)
    parser.set_defaults(strict_sync_to_gt=True)
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
    if args.apply_vins_td or args.match_vins_config:
        suffix += "_vins_td"
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
    if args.apply_vins_td or args.match_vins_config:
        suffix += "_vins_td"
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
        args.orb_root / "lib",
        args.orb_root / "Thirdparty/DBoW2/lib",
        args.orb_root / "Thirdparty/g2o/lib",
    ]
    candidates.extend(args.extra_ld_path)
    paths = [str(path) for path in candidates if Path(path).is_dir()]
    if env.get("LD_LIBRARY_PATH"):
        paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(paths))
    env["ORB_SLAM3_ENABLE_VIEWER"] = "0"
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


def apply_strict_sync(args: argparse.Namespace, pose_csv: Path) -> Path:
    if args.resolved_strict_sync_offset_sec is None:
        raise ValueError("--strict-sync-offset-sec is required when --strict-sync-to-gt is enabled")
    out_csv = pose_csv.with_name(f"{pose_csv.stem}_{strict_sync_suffix(args.resolved_strict_sync_offset_sec)}.csv")
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
            str(args.resolved_strict_sync_offset_sec),
            "--max-gap-sec",
            str(args.strict_sync_max_gap_sec),
        ]
    )
    return out_csv


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


def main() -> int:
    args = parse_args()
    argv = sys.argv[1:]
    args._strict_sync_offset_sec_explicit = "--strict-sync-offset-sec" in argv
    args.episode_dir = args.episode_dir.expanduser().resolve()
    args.ground_truth = args.ground_truth.expanduser().resolve()
    args.handeye_yaml = args.handeye_yaml.expanduser().resolve()
    args.orb_root = args.orb_root.expanduser().resolve()
    args.exporter = args.exporter.expanduser().resolve()
    args.config_generator = args.config_generator.expanduser().resolve()
    if args.strict_sync_offset_json is not None:
        args.strict_sync_offset_json = args.strict_sync_offset_json.expanduser().resolve()
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
    check_required(required)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    run_checked(build_export_cmd(args, output_dir))
    regenerate_settings_if_needed(args, output_dir)

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
        if rc != 0:
            raise RuntimeError(f"ORB-SLAM3 failed with code {rc}; see {output_dir / 'orbslam3_native.log'}")

    trajectory_path = find_trajectory(output_dir, name)
    eval_trajectory_path = maybe_smooth_trajectory(args, trajectory_path)
    pose_csv_name = "orb_pose_data_imu.csv" if args.mode == "stereo-inertial" else "orb_pose_data_camera.csv"
    if args.smooth_trajectory:
        pose_csv_name = pose_csv_name.replace(".csv", f"_smooth_w{args.smooth_window}_p{args.smooth_passes}.csv")
    pose_csv = output_dir / pose_csv_name
    run_checked([sys.executable, str(CONVERT_TUM), str(eval_trajectory_path), str(pose_csv)])
    estimate_csv = apply_strict_sync(args, pose_csv) if args.strict_sync_to_gt else pose_csv

    estimate_frame = args.estimate_frame
    if estimate_frame == "auto":
        estimate_frame = "imu" if args.mode == "stereo-inertial" else "camera"

    eval_cmd = [
        sys.executable,
        str(TCP_EVAL),
        "--estimate",
        str(estimate_csv),
        "--ground-truth",
        str(args.ground_truth),
        "--output-dir",
        str(eval_dir),
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
        "0.05",
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
    run_checked(eval_cmd)

    if not args.skip_viewer:
        run_checked(
            [
                sys.executable,
                str(VIEWER),
                "--eval-dir",
                str(eval_dir),
                "--output-dir",
                str(eval_dir),
                "--max-points",
                str(args.viewer_max_points),
            ]
        )

    summary = read_summary(eval_dir / "summary.csv")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "episode_dir": str(args.episode_dir),
        "ground_truth": str(args.ground_truth),
        "orb_root": str(args.orb_root),
        "output_dir": str(output_dir),
        "eval_dir": str(eval_dir),
        "mode": args.mode,
        "camera_rig": args.camera_rig,
        "feature_preset": args.feature_preset,
        "invert_tbc": bool(args.invert_tbc),
        "imu_fast_init": args.imu_fast_init,
        "vins_config": str(vins_config) if vins_config is not None else "",
        "match_vins_config": bool(args.match_vins_config),
        "apply_vins_td": bool(args.apply_vins_td or args.match_vins_config),
        "camera_time_shift_sec": args.camera_time_shift_sec,
        "vins_noise_mode": args.vins_noise_mode,
        "trajectory": str(trajectory_path),
        "trajectory_rows": count_rows(trajectory_path),
        "eval_trajectory": str(eval_trajectory_path),
        "pose_csv": str(pose_csv),
        "estimate_csv_for_eval": str(estimate_csv),
        "estimate_frame": estimate_frame,
        "smooth_trajectory": bool(args.smooth_trajectory),
        "smooth_window": args.smooth_window,
        "smooth_passes": args.smooth_passes,
        "strict_sync_to_gt": bool(args.strict_sync_to_gt),
        "strict_sync_offset_sec": args.resolved_strict_sync_offset_sec,
        "strict_sync_offset_json": str(args.strict_sync_offset_json) if args.strict_sync_offset_json else "",
        "strict_sync_max_gap_sec": args.strict_sync_max_gap_sec,
        "summary_rmse": summary,
    }
    (eval_dir / "orbslam3_tcp_eval_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] ORB-SLAM3 TCP evaluation finished: {eval_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
