#!/usr/bin/env python3
"""Run ORB-SLAM3 stereo on a local SXR DatasetRecorder CSV episode.

The adapter uses the factory KB4 calibration and selected-stream camera timestamps
shifted onto the IMU/device clock. ``head_pose.csv`` is read only after SLAM
finishes, as a device reference for SE(3)-aligned APE/RPE evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import av
import cv2
import numpy as np

from run_sxr_csv_basalt import (
    camera_time_offset_sec,
    inverse,
    load_stereo_camera1_from_camera0,
)
from run_sxr_csv_vinsfusion import write_reference
from run_sxr_orb_stereo_baseline import ROOT, build_viewer, evaluate, matrix_yaml


DEFAULT_ORB_ROOT = ROOT / "third_party/ORB_SLAM3_rm75_gyro_init"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stream", choices=("rgb", "tracking", "ctrl"), default="rgb")
    parser.add_argument("--orb-root", type=Path, default=DEFAULT_ORB_ROOT)
    parser.add_argument("--image-scale", type=float, default=0.5)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--nfeatures", type=int, default=3500)
    parser.add_argument("--ini-th-fast", type=int, default=10)
    parser.add_argument("--min-th-fast", type=int, default=3)
    parser.add_argument("--jpeg-qscale", type=int, default=4)
    parser.add_argument("--clahe", action="store_true", help="Apply grayscale CLAHE before JPEG export (experimental low-texture preset).")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid", type=int, default=8)
    parser.add_argument(
        "--robust-offline",
        action="store_true",
        help="Use the EGO offline robustness preset: 15-frame keyframe cap, 300 close points, no keyframe culling.",
    )
    parser.add_argument("--max-keyframe-interval-frames", type=int, default=0)
    parser.add_argument("--close-point-creation-limit", type=int, default=0)
    parser.add_argument("--disable-keyframe-culling", action="store_true")
    parser.add_argument(
        "--stereo-camera1-from-camera0-json",
        type=Path,
        default=None,
        help=(
            "Optional independently measured Camera1-from-Camera0 stereo "
            "calibration. The ORB YAML receives its inverse (Camera1 <- Camera2)."
        ),
    )
    parser.add_argument(
        "--invert-stereo-transform",
        action="store_true",
        help="Diagnostic only: use the rejected right-from-left transform.",
    )
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--keep-intermediate", action="store_true")
    return parser.parse_args(argv)


def camera_pose(camera: dict[str, Any]) -> np.ndarray:
    """Return raw factory pose (RGB stores IMU-to-camera)."""
    q = np.asarray(camera["extrinsics"]["rotation"], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = ((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
                          (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
                          (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)))
    transform[:3, 3] = np.asarray(camera["extrinsics"]["position"], dtype=np.float64)
    return transform


def write_settings(
    path: Path,
    image: dict[str, Any],
    scale: float,
    nfeatures: int,
    ini_fast: int,
    min_fast: int,
    invert_stereo: bool = False,
    camera1_from_camera0: np.ndarray | None = None,
    max_keyframe_interval_frames: int = 0,
    close_point_creation_limit: int = 0,
    disable_keyframe_culling: bool = False,
) -> dict[str, float]:
    cam0, cam1 = image["cameras"]
    if min_fast > ini_fast or min_fast < 1 or nfeatures < 500:
        raise ValueError("invalid ORB extractor settings")
    width, height = int(cam0["width"]), int(cam0["height"])
    # ORB-SLAM3's Stereo.T_c1_c2 is Camera1 <- Camera2, while the optional
    # calibration payload is explicitly Camera1-from-Camera0 (right <- left).
    # Therefore the external calibration must be inverted before writing it.
    t_c0_i, t_c1_i = camera_pose(cam0), camera_pose(cam1)
    t_c1_c0 = inverse(t_c0_i @ inverse(t_c1_i))
    if camera1_from_camera0 is not None:
        t_c1_c0 = inverse(camera1_from_camera0)
    if invert_stereo:
        t_c1_c0 = inverse(t_c1_c0)
    system_lines: list[str] = []
    if max_keyframe_interval_frames > 0:
        system_lines.append(f"System.MaxKeyFrameIntervalFrames: {max_keyframe_interval_frames}")
    if close_point_creation_limit > 0:
        system_lines.append(f"System.ClosePointCreationLimit: {close_point_creation_limit}")
    if disable_keyframe_culling:
        system_lines.append("System.DisableKeyFrameCulling: 1")
    lines = ["%YAML:1.0", "", 'File.version: "1.0"', "", 'Camera.type: "KannalaBrandt8"', ""]
    for number, camera in ((1, cam0), (2, cam1)):
        intr = camera["intrinsics"]
        distortion = intr["radialDistortion"][:4]
        lines.extend([
            f"Camera{number}.fx: {float(intr['focalX']) * scale:.12g}",
            f"Camera{number}.fy: {float(intr['focalY']) * scale:.12g}",
            f"Camera{number}.cx: {float(intr['centerX']) * scale:.12g}",
            f"Camera{number}.cy: {float(intr['centerY']) * scale:.12g}",
            *[f"Camera{number}.k{index + 1}: {float(value):.12g}" for index, value in enumerate(distortion)], "",
        ])
    lines.extend(matrix_yaml("Stereo.T_c1_c2", t_c1_c0))
    lines.extend([
        "", "Camera1.overlappingBegin: 0", f"Camera1.overlappingEnd: {round(width * scale) - 1}",
        "Camera2.overlappingBegin: 0", f"Camera2.overlappingEnd: {round(width * scale) - 1}",
        f"Camera.width: {round(width * scale)}", f"Camera.height: {round(height * scale)}", "Camera.fps: 60", "Camera.RGB: 0",
        "Stereo.ThDepth: 40.0", "", f"ORBextractor.nFeatures: {nfeatures}", "ORBextractor.scaleFactor: 1.2",
        "ORBextractor.nLevels: 8", f"ORBextractor.iniThFAST: {ini_fast}", f"ORBextractor.minThFAST: {min_fast}", "",
        *system_lines, "" if system_lines else "",
        "Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1.0", "Viewer.GraphLineWidth: 0.9", "Viewer.PointSize: 2.0",
        "Viewer.CameraSize: 0.08", "Viewer.CameraLineWidth: 3.0", "Viewer.ViewpointX: 0.0", "Viewer.ViewpointY: -0.7", "Viewer.ViewpointZ: -3.5", "Viewer.ViewpointF: 500.0",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"stereo_baseline_m": float(np.linalg.norm(t_c1_c0[:3, 3]))}


def load_frames(episode: Path, imu: dict[str, Any], stream: str, maximum: int) -> tuple[list[tuple[int, int]], float]:
    offset = camera_time_offset_sec(imu, stream, None)
    with (episode / f"{stream}_metainfo.csv").open(newline="", encoding="utf-8") as handle:
        frames = [(int(row["frame_index"]), int(row["mid_exposure_utc_ns"]) + round(offset * 1e9)) for row in csv.DictReader(handle)]
    if maximum:
        frames = frames[:maximum]
    if len(frames) < 3:
        raise ValueError("fewer than three RGB pairs")
    return frames, offset


def export_images(
    temp: Path,
    episode: Path,
    stream: str,
    frames: list[tuple[int, int]],
    image: dict[str, Any],
    scale: float,
    qscale: int,
    clahe_enabled: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: int = 8,
) -> str:
    raw0, raw1 = temp / "raw0", temp / "raw1"
    cam0, cam1 = temp / "mav0/cam0/data", temp / "mav0/cam1/data"
    raw0.mkdir(parents=True); raw1.mkdir(parents=True); cam0.mkdir(parents=True); cam1.mkdir(parents=True)
    width, height = int(image["cameras"][0]["width"]), int(image["cameras"][0]["height"])
    out_width, out_height = round(width * scale), round(height * scale)
    if clahe_enabled:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_grid, clahe_tile_grid))
        wanted = dict(frames)
        written: list[int] = []
        container = av.open(str(episode / f"{stream}.mp4"))
        try:
            for index, frame in enumerate(container.decode(container.streams.video[0])):
                timestamp = wanted.get(index)
                if timestamp is None:
                    continue
                stereo = frame.to_ndarray(format="bgr24")
                if stereo.shape[:2] != (height, width * 2):
                    raise ValueError(f"unexpected {stream} frame shape {stereo.shape[1]}x{stereo.shape[0]}")
                for destination, half in ((cam0, stereo[:, :width]), (cam1, stereo[:, width:])):
                    if scale != 1.0:
                        half = cv2.resize(half, (out_width, out_height), interpolation=cv2.INTER_AREA)
                    enhanced = clahe.apply(cv2.cvtColor(half, cv2.COLOR_BGR2GRAY))
                    ok, encoded = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 100 - qscale * 5])
                    if not ok:
                        raise RuntimeError(f"failed to JPEG encode {stream} frame {index}")
                    encoded.tofile(str(destination / f"{timestamp}.png"))
                written.append(timestamp)
        finally:
            container.close()
        if written != [timestamp for _index, timestamp in frames]:
            raise RuntimeError(f"exported {len(written)}/{len(frames)} requested stereo pairs")
        (temp / "times.txt").write_text("\n".join(str(stamp) for _index, stamp in frames) + "\n", encoding="utf-8")
        return "pyav_clahe"
    filters = (f"[0:v]crop={width}:{height}:0:0,scale={out_width}:{out_height}:flags=area[left];"
               f"[0:v]crop={width}:{height}:{width}:0,scale={out_width}:{out_height}:flags=area[right]")
    last_index = frames[-1][0] + 1
    command = ["ffmpeg", "-y", "-v", "error", "-threads", "1", "-filter_threads", "1", "-i", str(episode / f"{stream}.mp4"),
               "-filter_complex", filters, "-map", "[left]", "-frames:v", str(last_index), "-q:v", str(qscale), str(raw0 / "%06d.jpg"),
               "-map", "[right]", "-frames:v", str(last_index), "-q:v", str(qscale), str(raw1 / "%06d.jpg")]
    subprocess.run(command, check=True, timeout=900)
    missing = [index for index, _stamp in frames if not (raw0 / f"{index + 1:06d}.jpg").is_file()]
    if missing:
        raise RuntimeError(f"FFmpeg did not export {len(missing)} requested stereo frames")
    for index, stamp in frames:
        (raw0 / f"{index + 1:06d}.jpg").rename(cam0 / f"{stamp}.png")
        (raw1 / f"{index + 1:06d}.jpg").rename(cam1 / f"{stamp}.png")
    (temp / "times.txt").write_text("\n".join(str(stamp) for _index, stamp in frames) + "\n", encoding="utf-8")
    return "ffmpeg"


def convert_euroc(source: Path, destination: Path) -> int:
    """Convert only ORB poses that represent an active tracked state.

    This offline ORB fork emits an all-zero translation with an identity
    quaternion for every frame after tracking is lost.  Those are sentinel
    records, not estimated poses, and including them makes both coverage and
    SE(3)-aligned APE artificially optimistic.
    """
    count = 0
    last_timestamp: float | None = None
    with source.open(encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for line in src:
            fields = line.split()
            if len(fields) == 8:
                timestamp = float(fields[0])
                pose = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
                if np.allclose(pose[:3], 0.0, atol=1e-12) and np.allclose(
                    pose[3:], (0.0, 0.0, 0.0, 1.0), atol=1e-12
                ):
                    continue
                # ORB-SLAM3 writes the previous pose repeatedly while lost.
                # A repeated source timestamp is not an estimated new frame
                # and must not inflate coverage or bias APE toward zero.
                if last_timestamp is not None and timestamp <= last_timestamp:
                    continue
                dst.write(f"{timestamp * 1e-9:.9f} {' '.join(fields[1:])}\n")
                last_timestamp = timestamp
                count += 1
    return count


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.robust_offline:
        if args.max_keyframe_interval_frames == 0:
            args.max_keyframe_interval_frames = 15
        if args.close_point_creation_limit == 0:
            args.close_point_creation_limit = 300
        args.disable_keyframe_culling = True
    if not 0.2 <= args.image_scale <= 1.0 or args.max_pairs < 0 or not 2 <= args.jpeg_qscale <= 20:
        raise ValueError("invalid image scale, max pairs, or JPEG qscale")
    if args.clahe_clip_limit <= 0 or args.clahe_tile_grid < 2:
        raise ValueError("CLAHE clip limit must be positive and tile grid at least 2")
    if args.max_keyframe_interval_frames < 0 or args.close_point_creation_limit < 0:
        raise ValueError("keyframe and close-point limits must be non-negative")
    episode, output = args.episode_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    required = (f"{args.stream}.mp4", f"{args.stream}_metainfo.csv", "head_pose.csv", "imu_calibration.json", f"camera_params_{args.stream}.json")
    missing = [name for name in required if not (episode / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {episode}")
    output.mkdir(parents=True, exist_ok=True)
    image = json.loads((episode / f"camera_params_{args.stream}.json").read_text(encoding="utf-8"))
    imu = json.loads((episode / "imu_calibration.json").read_text(encoding="utf-8"))
    frames, offset = load_frames(episode, imu, args.stream, args.max_pairs)
    settings = output / "orbslam3_sxr_csv_stereo.yaml"
    stereo_camera1_from_camera0 = load_stereo_camera1_from_camera0(args.stereo_camera1_from_camera0_json)
    geometry = write_settings(
        settings,
        image,
        args.image_scale,
        args.nfeatures,
        args.ini_th_fast,
        args.min_th_fast,
        args.invert_stereo_transform,
        stereo_camera1_from_camera0,
        args.max_keyframe_interval_frames,
        args.close_point_creation_limit,
        args.disable_keyframe_culling,
    )
    reference = output / "head_pose_reference.tum"; write_reference(episode / "head_pose.csv", reference)
    orb_root = args.orb_root.expanduser().resolve()
    binary = orb_root / "Examples/Stereo/stereo_euroc_offline"
    vocabulary = orb_root / "Vocabulary/ORBvoc.txt"
    if not binary.is_file() or not vocabulary.is_file():
        raise FileNotFoundError("ORB-SLAM3 binary or vocabulary missing")
    temporary = tempfile.TemporaryDirectory(prefix=f"sxr_csv_orb_{episode.name}_", dir="/tmp")
    temp = output / "orb_export" if args.keep_intermediate else Path(temporary.name)
    try:
        image_export = export_images(
            temp, episode, args.stream, frames, image, args.image_scale, args.jpeg_qscale,
            args.clahe, args.clahe_clip_limit, args.clahe_tile_grid,
        )
        command = [
            str(binary), str(vocabulary), str(settings), str(temp), str(temp / "times.txt"),
            "sxr_orb_stereo", "100", "0", "10", "10",
        ]
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = str(orb_root / "lib") + ":" + environment.get("LD_LIBRARY_PATH", "")
        with (output / "orbslam3.log").open("w", encoding="utf-8") as handle:
            handle.write("COMMAND: " + " ".join(command) + "\n")
            handle.write(f"RGB_FACTORY_TIME_OFFSET_SEC: {offset:.9f}\n\n")
            status = subprocess.run(command, cwd=output, env=environment, stdout=handle, stderr=subprocess.STDOUT, timeout=args.timeout_sec, check=False).returncode
        if status:
            raise RuntimeError(f"ORB-SLAM3 exited {status}; see {output / 'orbslam3.log'}")
    finally:
        temporary.cleanup()
    estimate = output / "orb_stereo_trajectory.tum"
    count = convert_euroc(output / "f_sxr_orb_stereo.txt", estimate)
    if count < 3:
        raise RuntimeError(f"ORB output only has {count} poses")
    evaluate(reference, estimate, output / "evaluation")
    build_viewer(reference, estimate, output / "evaluation/trajectory_viewer")
    summary = {"algorithm": "ORB-SLAM3 stereo", "episode": str(episode), "stream": args.stream, "image_pairs": len(frames), "trajectory_cam0_poses": count,
               "trajectory_coverage": count / len(frames), "image_scale": args.image_scale, "jpeg_qscale": args.jpeg_qscale,
               "factory_camera_time_offset_sec": offset, "factory_pose_convention": "imu-to-camera", "stereo_transform": "T_left_to_right",
               "invert_stereo_transform_diagnostic": args.invert_stereo_transform,
               "robust_offline": args.robust_offline,
               "clahe": args.clahe,
               "clahe_clip_limit": args.clahe_clip_limit if args.clahe else None,
               "clahe_tile_grid": args.clahe_tile_grid if args.clahe else None,
               "image_export": image_export,
               "max_keyframe_interval_frames": args.max_keyframe_interval_frames,
               "close_point_creation_limit": args.close_point_creation_limit,
               "disable_keyframe_culling": args.disable_keyframe_culling,
               "stereo_camera1_from_camera0_source": "factory" if args.stereo_camera1_from_camera0_json is None else str(args.stereo_camera1_from_camera0_json.expanduser().resolve()),
               "right_kb4_distortion": "Camera2.k1..k4", "head_pose_note": "Device-provided reference, not independently verified ground truth.", **geometry}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {episode.name}: {count}/{len(frames)} poses -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
