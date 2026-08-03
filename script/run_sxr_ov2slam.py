#!/usr/bin/env python3
"""Run OV²SLAM on an SXR Ego recording with native fisheye stereo rectification."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import av
import cv2

from check_episode_data import CheckResult, inspect_mcap, resolve_data_dir
from run_sxr_orb_stereo_baseline import ROOT, head_start_offset_ns, inv_se3, read_records, write_reference


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--docker-image", default="vio-eval-ov2slam:noetic")
    parser.add_argument("--timeout-sec", type=int, default=2400)
    parser.add_argument("--rate", type=float, default=60.0)
    parser.add_argument("--nmaxdist", type=int, default=35)
    parser.add_argument("--full-ba", action="store_true")
    return parser.parse_args(argv)


def matrix_yaml(name: str, matrix: np.ndarray) -> str:
    values = ", ".join(f"{float(value):.12g}" for value in matrix.reshape(-1))
    return f"{name}: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: d\n   data: [{values}]"


def write_config(path: Path, calibration: dict, nmaxdist: int, full_ba: bool) -> None:
    if nmaxdist < 12:
        raise ValueError("nmaxdist must be >= 12")
    rgb = calibration["observation"]["images"]["rgb"]
    cam0, cam1 = rgb["cam0"], rgb["cam1"]
    intr0, intr1 = cam0["intrinsics"]["2328x1748"], cam1["intrinsics"]["2328x1748"]
    # ``T_ic_imu0_cam*`` is camera -> IMU despite the recorder's abbreviated
    # field name.  With body=cam0, OV²SLAM needs the right camera -> cam0 map.
    t_i_c0 = np.asarray(rgb["extrinsics"]["T_ic_imu0_cam0"], dtype=np.float64)
    t_i_c1 = np.asarray(rgb["extrinsics"]["T_ic_imu0_cam1"], dtype=np.float64)
    t_c0_c1 = t_i_c1 @ inv_se3(t_i_c0)
    # JSON factory rotations are rounded to six decimals. Sophus validates SO(3)
    # strictly, so project the composed transform back to the closest proper
    # rotation rather than feeding it a numerically non-orthogonal matrix.
    u, _singular_values, vt = np.linalg.svd(t_c0_c1[:3, :3])
    t_c0_c1[:3, :3] = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    lines = ["%YAML:1.0", "---", "Camera.topic_left: /cam0/image_raw", "Camera.topic_right: /cam1/image_raw",
             "Camera.model_left: fisheye", "Camera.model_right: fisheye", "Camera.left_nwidth: 2328", "Camera.left_nheight: 1748",
             "Camera.right_nwidth: 2328", "Camera.right_nheight: 1748"]
    for suffix, intr, dist in (("l", intr0, cam0["distortion_coeffs"][:4]), ("r", intr1, cam1["distortion_coeffs"][:4])):
        lines += [f"Camera.fx{suffix}: {intr['fx']:.12g}", f"Camera.fy{suffix}: {intr['fy']:.12g}",
                  f"Camera.cx{suffix}: {intr['ppx']:.12g}", f"Camera.cy{suffix}: {intr['ppy']:.12g}",
                  f"Camera.k1{suffix}: {dist[0]:.12g}", f"Camera.k2{suffix}: {dist[1]:.12g}",
                  f"Camera.p1{suffix}: {dist[2]:.12g}", f"Camera.p2{suffix}: {dist[3]:.12g}"]
    lines += [matrix_yaml("body_T_cam0", np.eye(4)), matrix_yaml("body_T_cam1", t_c0_c1),
              "debug: 0", "log_timings: 0", "mono: 0", "stereo: 1", "force_realtime: 0", "slam_mode: 1",
              "buse_loop_closer: 0", "bdo_stereo_rect: 1", "alpha: 0.0", "bdo_undist: 0", "finit_parallax: 20.0",
              "use_shi_tomasi: 0", "use_fast: 0", "use_brief: 1", "use_singlescale_detector: 1", f"nmaxdist: {nmaxdist}",
              "nfast_th: 10", "dmaxquality: 0.001", "use_clahe: 1", "fclahe_val: 3.0", "do_klt: 1", "klt_use_prior: 1",
              "btrack_keyframetoframe: 0", "nklt_win_size: 9", "nklt_pyr_lvl: 3", "nmax_iter: 30", "fmax_px_precision: 0.01",
              "fmax_fbklt_dist: 0.5", "nklt_err: 30.0", "bdo_track_localmap: 1", "fmax_desc_dist: 0.2", "fmax_proj_pxdist: 2.0",
              "doepipolar: 1", "dop3p: 0", "bdo_random: 0", "nransac_iter: 100", "fransac_err: 3.0", "fmax_reproj_err: 3.0",
              "buse_inv_depth: 1", "robust_mono_th: 5.9915", "robust_stereo_th: 7.8147", "use_sparse_schur: 1", "use_dogleg: 0",
              "use_subspace_dogleg: 0", "use_nonmonotic_step: 0", "apply_l2_after_robust: 1", "nmin_covscore: 25",
              "fkf_filtering_ratio: 0.95", f"do_full_ba: {1 if full_ba else 0}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_stereo_jpegs(video: Path, aligned: list[tuple[int, int]], destination: Path) -> int:
    """Use host PyAV, which is proven to decode the recorder's MP4 fully."""
    wanted = {index for index, _timestamp in aligned}
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    container = av.open(str(video))
    try:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index not in wanted:
                continue
            image = frame.to_ndarray(format="bgr24")
            if image.shape[:2] != (1748, 4656):
                raise ValueError(f"unexpected video size {image.shape[1]}x{image.shape[0]}")
            for suffix, half in (("left", image[:, :2328]), ("right", image[:, 2328:])):
                ok, encoded = cv2.imencode(".jpg", half, [cv2.IMWRITE_JPEG_QUALITY, 100])
                if not ok:
                    raise RuntimeError(f"cannot JPEG encode source frame {index}")
                encoded.tofile(str(destination / f"{index}_{suffix}.jpg"))
            written += 1
    finally:
        container.close()
    if written != len(aligned):
        raise RuntimeError(f"exported {written} stereo pairs, expected {len(aligned)}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _root, episode = resolve_data_dir(args.episode_dir)
    episode = episode.resolve()
    output = args.output_dir.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((episode / "calibration.json").read_text(encoding="utf-8"))
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    frames, _imu_times = read_records(episode / "sensor.mcap")
    head, _hand, _channels, _schemas = inspect_mcap(episode / "sensor.mcap", CheckResult())
    if len(frames) < 3 or len(head.poses) < 3:
        raise ValueError("episode lacks RGB metadata or head_pose")
    offset = head_start_offset_ns(metadata)
    head_start = min(item[0] for item in head.poses)
    aligned = [(index, head_start + middle - offset) for index, middle in frames]
    (output / "frame_timestamps_ns.txt").write_text("\n".join(f"{i} {t}" for i, t in aligned) + "\n", encoding="utf-8")
    write_reference(output / "head_pose_reference.tum", head.poses)
    config = output / "ov2slam_sxr_fisheye_rectified.yaml"
    write_config(config, calibration, args.nmaxdist, args.full_ba)
    subprocess.run(["docker", "image", "inspect", args.docker_image], check=True, stdout=subprocess.DEVNULL)
    repo = ROOT.resolve()
    out_in = "/workspace/vio_eval/" + str(output.relative_to(repo))
    image_dir = Path(tempfile.mkdtemp(prefix=f"sxr_ov2slam_{episode.name}_"))
    exported_pairs = export_stereo_jpegs(episode / "rgb.mp4", aligned, image_dir)
    lines = ["set -euo pipefail", "source /opt/ros/noetic/setup.bash", "SRC=/workspace/vio_eval/third_party/ov2slam", "WS=/workspace/vio_eval/third_party/ov2slam_ws",
             "cd $SRC", "cmake -S Thirdparty/obindex2 -B Thirdparty/obindex2/build -DCMAKE_BUILD_TYPE=Release", "cmake --build Thirdparty/obindex2/build -j$(nproc)",
             "cmake -S Thirdparty/ibow_lcd -B Thirdparty/ibow_lcd/build -DCMAKE_BUILD_TYPE=Release", "cmake --build Thirdparty/ibow_lcd/build -j$(nproc)",
             "cmake -S Thirdparty/Sophus -B Thirdparty/Sophus/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$SRC/Thirdparty/Sophus/install", "cmake --build Thirdparty/Sophus/build -j$(nproc) --target install",
             "mkdir -p $WS/src", "ln -sfn $SRC $WS/src/ov2slam", "cd $WS", "catkin_make -DCMAKE_BUILD_TYPE=Release", "source $WS/devel/setup.bash",
             f"OUT={out_in}", "roscore > $OUT/roscore.log 2>&1 & ROS_PID=$!", "sleep 2", "cd $OUT",
             "rosrun ov2slam ov2slam_node $OUT/ov2slam_sxr_fisheye_rectified.yaml > $OUT/ov2slam.log 2>&1 & NODE_PID=$!", "sleep 2",
             f"python3 /workspace/vio_eval/script/sxr_ov2slam_publish.py --image-dir /sxr_images --timestamps $OUT/frame_timestamps_ns.txt --rate {args.rate}",
             f"deadline=$((SECONDS+{args.timeout_sec}))", "while kill -0 $NODE_PID 2>/dev/null; do if [ $SECONDS -gt $deadline ]; then kill $NODE_PID; wait $NODE_PID || true; exit 124; fi; sleep 1; done", "wait $NODE_PID", "kill $ROS_PID 2>/dev/null || true"]
    command = ["docker", "run", "--rm", "--network", "host", "-v", f"{repo}:/workspace/vio_eval", "-v", f"{image_dir}:/sxr_images:ro", "-w", "/workspace/vio_eval", args.docker_image, "bash", "-lc", "\n".join(lines)]
    (output / "docker_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    try:
        with (output / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout_sec + 900)
    finally:
        shutil.rmtree(image_dir)
    estimate = output / "ov2slam_traj.txt"
    if not estimate.is_file():
        raise RuntimeError(f"OV²SLAM ended without {estimate}; inspect {output / 'ov2slam.log'}")
    from run_sxr_orb_stereo_baseline import build_viewer, evaluate
    evaluate(output / "head_pose_reference.tum", estimate, output / "evaluation")
    build_viewer(output / "head_pose_reference.tum", estimate, output / "evaluation/trajectory_viewer")
    count = sum(1 for line in estimate.open(encoding="utf-8") if line.strip())
    (output / "run_summary.txt").write_text(f"episode={episode}\nmode=OV2SLAM stereo fisheye rectified (no IMU)\nimage_pairs={len(aligned)}\nexported_pairs={exported_pairs}\ntrajectory_poses={count}\nnmaxdist={args.nmaxdist}\nreference=device head_pose (not independent ground truth)\n", encoding="utf-8")
    print(f"[OK] {episode.name}: {count}/{len(aligned)} poses -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
