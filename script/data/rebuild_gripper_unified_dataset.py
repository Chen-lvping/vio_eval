#!/usr/bin/env python3
"""Build one sequential gripper dataset root with episode and GT symlinks."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
GRIPPER_ROOT = DATA_ROOT / "gripper"
GT_ROOT = DATA_ROOT / "ground_truth"
DEFAULT_OUTPUT_ROOT = GRIPPER_ROOT / "gripper_all_seq"
DEFAULT_VINS_CONFIG_GENERATOR = Path("/home/chenlvping/0_SLAM/vinsfusion_ws/scripts/gen_vins_config_from_calib.py")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    episode_root: Path
    gt_root: Path | None


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("gripper_data2", GRIPPER_ROOT / "gripper_data2", GT_ROOT / "rm75"),
    SourceSpec("gripper_data_6_23", GRIPPER_ROOT / "gripper_data_6_23", GT_ROOT / "rm75_6_23"),
    SourceSpec("gripper_data_6_24", GRIPPER_ROOT / "gripper_data_6_24", GT_ROOT / "rm75_6_24"),
    SourceSpec("gripper_data_6_25", GRIPPER_ROOT / "gripper_data_6_25", GT_ROOT / "rm75_6_25"),
    SourceSpec("0707", GRIPPER_ROOT / "0707", GT_ROOT / "rm75_0707_seq"),
)

OMITTED_NO_GT_DATASETS: tuple[str, ...] = (
    "0704",
    "data_6_30",
    "episode_C1-2-data",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episode-prefix", default="episode_gripper")
    parser.add_argument("--force", action="store_true", help="Remove and rebuild output root if it already exists.")
    parser.add_argument(
        "--vins-config-generator",
        type=Path,
        default=DEFAULT_VINS_CONFIG_GENERATOR,
        help="Generator used to create missing StereoIMU-vinsfusion.yaml from calibration.json.",
    )
    parser.add_argument(
        "--skip-generate-missing-vins-config",
        action="store_true",
        help="Do not create missing local VINS generated_config files before linking episodes.",
    )
    return parser.parse_args()


def episode_suffix(episode_dir: Path) -> int:
    suffix = episode_dir.name.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"cannot infer numeric suffix from {episode_dir}")
    return int(suffix)


def resolve_gt_path(spec: SourceSpec, episode_dir: Path) -> Path:
    suffix = episode_suffix(episode_dir)
    candidates: list[Path] = []
    if spec.gt_root is not None:
        candidates.append(spec.gt_root / f"rm75_pose_traj_{suffix}.json")
        candidates.append(spec.gt_root / f"rm75_pose_traj{suffix:02d}.json")
    candidates.append(spec.episode_root / f"rm75_pose_traj_{suffix}.json")
    candidates.append(spec.episode_root / f"rm75_pose_traj{suffix:02d}.json")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"missing GT for {episode_dir}")


def ensure_missing_vins_config(episode_dir: Path, generator: Path) -> str:
    config_dir = episode_dir / "right" / "vio_log" / "generated_config"
    config_path = config_dir / "StereoIMU-vinsfusion.yaml"
    if config_path.is_file():
        return "present"
    calib_path = episode_dir / "calibration.json"
    if not calib_path.is_file():
        return "missing_calibration"
    config_dir.mkdir(parents=True, exist_ok=True)
    generate_vins_config_from_calibration(calib_path, config_dir, generator)
    return "generated" if config_path.is_file() else "failed"


def _nested(data: dict, *keys: str):
    for key in keys:
        data = data[key]
    return data


def _fmt_matrix4x4(matrix: np.ndarray) -> str:
    parts = [f"{x:.15e}" for x in matrix.reshape(-1).tolist()]
    lines = []
    for i in range(4):
        row = ", ".join(parts[i * 4 : (i + 1) * 4])
        prefix = "          " if i > 0 else "   data: ["
        suffix = "," if i < 3 else "]"
        lines.append(f"{prefix}{row}{suffix}")
    return "\n".join(lines)


def _format_vector(values: list[float]) -> str:
    return "[" + ", ".join(f"{float(v):.15e}" for v in values) + "]"


def _write_cam_equidistant(path: Path, intrinsics: dict[str, float], dist: list[float], width: int, height: int) -> None:
    content = f"""%YAML:1.0
---
model_type: KANNALA_BRANDT
camera_name: camera
image_width: {width}
image_height: {height}
projection_parameters:
   k2: {dist[0]:.15e}
   k3: {dist[1]:.15e}
   k4: {dist[2]:.15e}
   k5: {dist[3]:.15e}
   mu: {intrinsics['fx']:.15e}
   mv: {intrinsics['fy']:.15e}
   u0: {intrinsics['ppx']:.15e}
   v0: {intrinsics['ppy']:.15e}
"""
    path.write_text(content, encoding="utf-8")


def _write_stereo_imu_yaml(path: Path, body_t_cam0: np.ndarray, body_t_cam1: np.ndarray, imu_params: dict[str, float], td: float, width: int, height: int) -> None:
    content = f"""%YAML:1.0

#common parameters
#support: 1 imu 1 cam; 1 imu 2 cam: 2 cam;
# Configuration generated from calibration.json
imu: 1
num_of_cam: 2

imu_topic: "/fays/atrak/imu"
image0_topic: "/fays/atrak/cam0"
image1_topic: "/fays/atrak/cam1"
output_path: "~/output/"

cam0_calib: "cam0_equidistant.yaml"
cam1_calib: "cam1_equidistant.yaml"
image_width: {width}
image_height: {height}

# base_to_imu order: [rz, ry, rx, tx, ty, tz], rad + meter
base_to_imu: [0, 0, 0, 0, 0, 0]

# Extrinsic parameter between IMU and Camera.
# body_T_cam = T_ic (camera to IMU), rows correspond to cam0_to_imu0_T_ic / cam1_to_imu0_T_ic
estimate_extrinsic: 0

body_T_cam0: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
{_fmt_matrix4x4(body_t_cam0)}

body_T_cam1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
{_fmt_matrix4x4(body_t_cam1)}

#Multiple thread support
multiple_thread: 0

#feature traker paprameters
max_cnt: 120
min_dist: 20
freq: 10
F_threshold: 1.0
show_track: 1
flow_back: 1
equalize: 1

#optimization parameters
max_solver_time: 0.04
max_num_iterations: 8
keyframe_parallax: 10.0

#imu parameters
acc_n: {imu_params['acc_n']:.15e}
gyr_n: {imu_params['gyr_n']:.15e}
acc_w: {imu_params['acc_w']:.15e}
gyr_w: {imu_params['gyr_w']:.15e}
g_norm: 9.81

#unsynchronization parameters
estimate_td: 0
td: {td:.15e}

#loop closure parameters
load_previous_pose_graph: 0
pose_graph_save_path: "~/output/pose_graph/"
save_image: 0
save_vio_pose: 1
vio_pose_save_path: "~/output/vio_pose/"
"""
    path.write_text(content, encoding="utf-8")


def _extract_calib_from_old_schema(calib: dict, camera_rig: str) -> tuple[np.ndarray, np.ndarray, dict[str, float], float, dict[str, object], dict[str, object]]:
    rig = "stereo_left" if camera_rig == "stereo_left" else "stereo_right"
    images = calib["observation"]["images"][rig]
    imu = calib["observation"]["imu"][f"imu_{rig.split('_', 1)[-1]}"]
    extr = images["extrinsics"]
    body_t_cam0 = np.array(extr["T_ic_cam0_to_imu0"], dtype=np.float64)
    body_t_cam1 = np.array(extr["T_ic_cam1_to_imu0"], dtype=np.float64)
    td = float(extr.get("timeshift_cam0_to_imu0_sec", extr["timeshift_cam0_to_imu0"]))
    cam0 = images["cam0"]["intrinsics"]["640x400"]
    cam1 = images["cam1"]["intrinsics"]["640x400"]
    cam0_dist = images["cam0"]["distortion_coeffs"]
    cam1_dist = images["cam1"]["distortion_coeffs"]
    imu_params = {
        "acc_n": float(imu["accelerometer"]["noise_density_discrete"]),
        "gyr_n": float(imu["gyroscope"]["noise_density_discrete"]),
        "acc_w": float(imu["accelerometer"]["random_walk"]),
        "gyr_w": float(imu["gyroscope"]["random_walk"]),
    }
    return body_t_cam0, body_t_cam1, imu_params, td, {"intrinsics": cam0, "dist": cam0_dist}, {"intrinsics": cam1, "dist": cam1_dist}


def _extract_calib_from_new_schema(calib: dict) -> tuple[np.ndarray, np.ndarray, dict[str, float], float, dict[str, object], dict[str, object]]:
    rig = calib["observation"]["images"]["stereo_right"]
    extr = rig["extrinsics"]
    body_t_cam0 = np.array(extr["T_ic_cam0_to_imu0"], dtype=np.float64)
    body_t_cam1 = np.array(extr["T_ic_cam1_to_imu0"], dtype=np.float64)
    td = float(extr.get("timeshift_cam0_to_imu0_sec", extr["timeshift_cam0_to_imu0"]))
    cam0 = rig["cam0"]["intrinsics"]["640x400"]
    cam1 = rig["cam1"]["intrinsics"]["640x400"]
    cam0_dist = rig["cam0"]["distortion_coeffs"]
    cam1_dist = rig["cam1"]["distortion_coeffs"]
    imu = calib["observation"]["imu"]["imu_right"]
    imu_params = {
        "acc_n": float(imu["accelerometer"]["noise_density_discrete"]),
        "gyr_n": float(imu["gyroscope"]["noise_density_discrete"]),
        "acc_w": float(imu["accelerometer"]["random_walk"]),
        "gyr_w": float(imu["gyroscope"]["random_walk"]),
    }
    return body_t_cam0, body_t_cam1, imu_params, td, {"intrinsics": cam0, "dist": cam0_dist}, {"intrinsics": cam1, "dist": cam1_dist}


def generate_vins_config_from_calibration(calib_path: Path, output_dir: Path, generator: Path) -> None:
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if "fays_imu_bundle" in calib.get("calibration_info", {}):
        # Existing legacy generator path for older datasets.
        cmd = [sys.executable, str(generator), str(calib_path), "-o", str(output_dir)]
        subprocess.run(cmd, check=True)
        return

    if "stereo_right" in calib.get("observation", {}).get("images", {}):
        body_t_cam0, body_t_cam1, imu_params, td, cam0, cam1 = _extract_calib_from_new_schema(calib)
    elif "stereo_left" in calib.get("observation", {}).get("images", {}):
        body_t_cam0, body_t_cam1, imu_params, td, cam0, cam1 = _extract_calib_from_old_schema(calib, "stereo_left")
    else:
        raise KeyError(f"unsupported calibration schema in {calib_path}")

    _write_cam_equidistant(output_dir / "cam0_equidistant.yaml", cam0["intrinsics"], cam0["dist"], 640, 400)
    _write_cam_equidistant(output_dir / "cam1_equidistant.yaml", cam1["intrinsics"], cam1["dist"], 640, 400)
    _write_stereo_imu_yaml(output_dir / "StereoIMU-vinsfusion.yaml", body_t_cam0, body_t_cam1, imu_params, td, 640, 400)


def symlink_force(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target)


def build_readme(output_root: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Unified Gripper Dataset",
        "",
        "This directory is a sequential, GT-aligned index over the current gripper episodes that have matching RM75 ground truth.",
        "",
        "Kept here:",
        "- sequential episode symlinks: `episode_gripper_0001`, `episode_gripper_0002`, ...",
        "- matching GT symlinks: `rm75_pose_traj_1.json`, `rm75_pose_traj_2.json`, ...",
        "- mapping files: `dataset_manifest.json` and `dataset_index.csv`",
        "",
        "Current scripts can use this dataset directly by passing the same directory to both `--episode-root` and `--gt-root`.",
        "",
        f"Total indexed episodes with GT: {len(rows)}",
        "",
        "Datasets omitted because no paired GT is currently available:",
    ]
    for name in OMITTED_NO_GT_DATASETS:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "Example:",
            "```bash",
            "python3 script/run_orbslam3_rm75_batch_eval.py \\",
            "  --episode-root data/gripper/gripper_all_seq \\",
            "  --gt-root data/gripper/gripper_all_seq \\",
            "  --episode-pattern 'episode_gripper_*'",
            "```",
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    generator = args.vins_config_generator.expanduser().resolve()

    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"{output_root} already exists; pass --force to rebuild it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    seq = 1
    for spec in SOURCES:
        episodes = sorted(path for path in spec.episode_root.glob("episode_*") if path.is_dir())
        for episode_dir in episodes:
            gt_path = resolve_gt_path(spec, episode_dir)
            vins_status = "skipped"
            if not args.skip_generate_missing_vins_config:
                vins_status = ensure_missing_vins_config(episode_dir.resolve(), generator)

            episode_link_name = f"{args.episode_prefix}_{seq:04d}"
            gt_link_name = f"rm75_pose_traj_{seq}.json"
            symlink_force(episode_dir.resolve(), output_root / episode_link_name)
            symlink_force(gt_path, output_root / gt_link_name)

            rows.append(
                {
                    "seq": seq,
                    "episode_link": episode_link_name,
                    "ground_truth_link": gt_link_name,
                    "source_dataset": spec.name,
                    "source_episode": episode_dir.name,
                    "source_ground_truth": gt_path.name,
                    "source_episode_path": str(episode_dir.resolve()),
                    "source_ground_truth_path": str(gt_path),
                    "local_vins_config_status": vins_status,
                }
            )
            seq += 1

    manifest = {
        "dataset_name": output_root.name,
        "episode_prefix": args.episode_prefix,
        "total_indexed_episodes": len(rows),
        "same_root_works_for_episode_and_gt": True,
        "sources": [
            {
                "name": spec.name,
                "episode_root": str(spec.episode_root.resolve()),
                "gt_root": str(spec.gt_root.resolve()) if spec.gt_root is not None else None,
            }
            for spec in SOURCES
        ],
        "omitted_no_gt_datasets": list(OMITTED_NO_GT_DATASETS),
        "entries": rows,
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (output_root / "dataset_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    build_readme(output_root, rows)

    print(f"[OK] unified dataset root: {output_root}")
    print(f"[OK] indexed episodes with GT: {len(rows)}")
    print(f"[OK] manifest: {output_root / 'dataset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
