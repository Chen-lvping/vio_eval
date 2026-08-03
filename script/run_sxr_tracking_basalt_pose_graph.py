#!/usr/bin/env python3
"""Add verified stereo loop edges to a Basalt SXR tracking trajectory.

The device reference trajectory is deliberately unavailable until the final,
scale-gated evaluation. Loop candidates are formed from rectified stereo
features, triangulated in the source keyframe, and verified with PnP-RANSAC.
The accepted relative-pose constraints are then optimized jointly with Basalt
odometry in an SE(3) pose graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import av
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from run_sxr_csv_basalt import (
    camera_time_offset_sec,
    corrected_imu,
    factory_t_imu_camera,
    inverse,
    load_camera_from_imu_rotation,
    load_frames,
)
from run_sxr_csv_vinsfusion import write_reference
from run_sxr_orb_stereo_baseline import ROOT, build_viewer, evaluate


DEFAULT_ROTATION = ROOT / (
    "data/evaluation/workbench/"
    "sxr_csv_20260730_205558_tracking_aprilgrid_preflight_v1/"
    "cam_imu_timing_extrinsic_validation.json"
)


@dataclass
class Keyframe:
    frame_index: int
    trajectory_index: int
    left_points: np.ndarray
    left_descriptors: np.ndarray
    right_points: np.ndarray
    right_descriptors: np.ndarray
    signature: np.ndarray


@dataclass
class LoopEdge:
    source: int
    target: int
    source_frame: int
    target_frame: int
    transform_target_from_source: np.ndarray
    inliers: int
    matches: int
    reprojection_rmse_px: float
    appearance_similarity: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--baseline-trajectory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--keyframe-stride", type=int, default=30)
    parser.add_argument("--minimum-separation", type=int, default=6)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--maximum-loops", type=int, default=3)
    parser.add_argument("--minimum-inliers", type=int, default=45)
    parser.add_argument("--nfeatures", type=int, default=2200)
    parser.add_argument("--camera-from-imu-rotation-json", type=Path, default=DEFAULT_ROTATION)
    return parser.parse_args(argv)


def load_tum(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    stamps: list[float] = []
    poses: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 8:
            continue
        stamp, tx, ty, tz, qx, qy, qz, qw = (float(value) for value in values)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
        transform[:3, 3] = (tx, ty, tz)
        stamps.append(stamp)
        poses.append(transform)
    if len(poses) < 3:
        raise ValueError(f"{path} has fewer than three valid poses")
    return np.asarray(stamps), poses


def camera_matrix(camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = camera["intrinsics"]
    matrix = np.array(
        (
            (float(intrinsics["focalX"]), 0.0, float(intrinsics["centerX"])),
            (0.0, float(intrinsics["focalY"]), float(intrinsics["centerY"])),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return matrix, np.asarray(intrinsics["radialDistortion"][:4], dtype=np.float64)


def make_rectifier(cameras: list[dict[str, Any]], t_imu_cam0: np.ndarray, t_imu_cam1: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    camera0, camera1 = cameras
    width, height = int(camera0["width"]), int(camera0["height"])
    k0, d0 = camera_matrix(camera0)
    k1, d1 = camera_matrix(camera1)
    cam1_from_cam0 = inverse(t_imu_cam1) @ t_imu_cam0
    r0, r1, p0, p1, _q = cv2.fisheye.stereoRectify(
        k0, d0, k1, d1, (width, height), cam1_from_cam0[:3, :3], cam1_from_cam0[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=(width, height), balance=0.0, fov_scale=1.0,
    )
    maps0 = cv2.fisheye.initUndistortRectifyMap(k0, d0, r0, p0[:, :3], (width, height), cv2.CV_32FC1)
    maps1 = cv2.fisheye.initUndistortRectifyMap(k1, d1, r1, p1[:, :3], (width, height), cv2.CV_32FC1)
    return maps0, maps1, p0, p1, p0[:, :3]


def descriptor_signature(descriptors: np.ndarray) -> np.ndarray:
    signature = np.bincount(descriptors[:, 0], minlength=256).astype(np.float64)
    norm = float(np.linalg.norm(signature))
    return signature / norm if norm else signature


def extract_keyframes(
    episode: Path,
    selected: dict[int, tuple[int, int]],
    maps0: tuple[np.ndarray, np.ndarray],
    maps1: tuple[np.ndarray, np.ndarray],
    trajectory_for_frame: dict[int, int],
    nfeatures: int,
) -> list[Keyframe]:
    detector = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=8)
    output: list[Keyframe] = []
    container = av.open(str(episode / "tracking.mp4"))
    try:
        for frame_number, frame in enumerate(container.decode(container.streams.video[0])):
            selected_item = selected.get(frame_number)
            if selected_item is None:
                continue
            image = frame.to_ndarray(format="bgr24")
            width = image.shape[1] // 2
            left = cv2.cvtColor(image[:, :width], cv2.COLOR_BGR2GRAY)
            right = cv2.cvtColor(image[:, width:], cv2.COLOR_BGR2GRAY)
            left = cv2.remap(left, maps0[0], maps0[1], cv2.INTER_LINEAR)
            right = cv2.remap(right, maps1[0], maps1[1], cv2.INTER_LINEAR)
            left_points, left_descriptors = detector.detectAndCompute(left, None)
            right_points, right_descriptors = detector.detectAndCompute(right, None)
            if left_descriptors is None or right_descriptors is None or len(left_points) < 80 or len(right_points) < 80:
                continue
            output.append(
                Keyframe(
                    frame_index=frame_number,
                    trajectory_index=trajectory_for_frame[frame_number],
                    left_points=np.asarray([point.pt for point in left_points], dtype=np.float64),
                    left_descriptors=left_descriptors,
                    right_points=np.asarray([point.pt for point in right_points], dtype=np.float64),
                    right_descriptors=right_descriptors,
                    signature=descriptor_signature(left_descriptors),
                )
            )
    finally:
        container.close()
    if len(output) < 4:
        raise RuntimeError(f"only {len(output)} usable rectified keyframes")
    return output


def ratio_matches(source: np.ndarray, target: np.ndarray, ratio: float = 0.75) -> list[cv2.DMatch]:
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(source, target, k=2)
    return [first for first, second in pairs if first.distance < ratio * second.distance]


def verify_loop(source: Keyframe, target: Keyframe, projection0: np.ndarray, projection1: np.ndarray, camera: np.ndarray, similarity: float, minimum_inliers: int) -> LoopEdge | None:
    stereo = ratio_matches(source.left_descriptors, source.right_descriptors)
    temporal = ratio_matches(source.left_descriptors, target.left_descriptors)
    right_lookup = {
        match.queryIdx: match.trainIdx
        for match in stereo
        if abs(source.left_points[match.queryIdx, 1] - source.right_points[match.trainIdx, 1]) < 2.0
        and 1.0 < source.left_points[match.queryIdx, 0] - source.right_points[match.trainIdx, 0] < 160.0
    }
    temporal_lookup = {match.queryIdx: match.trainIdx for match in temporal}
    indices = sorted(set(right_lookup).intersection(temporal_lookup))
    if len(indices) < minimum_inliers:
        return None
    source_points = source.left_points[indices]
    right_points = source.right_points[[right_lookup[index] for index in indices]]
    target_points = target.left_points[[temporal_lookup[index] for index in indices]]
    homogeneous = cv2.triangulatePoints(projection0, projection1, source_points.T, right_points.T)
    points3d = (homogeneous[:3] / homogeneous[3]).T
    keep = np.isfinite(points3d).all(axis=1) & (points3d[:, 2] > 0.08) & (points3d[:, 2] < 30.0)
    if int(keep.sum()) < minimum_inliers:
        return None
    ok, rvec, translation, inliers = cv2.solvePnPRansac(
        points3d[keep], target_points[keep], camera, None, iterationsCount=1500,
        reprojectionError=2.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < minimum_inliers:
        return None
    inlier_points3d = points3d[keep][inliers[:, 0]]
    inlier_points2d = target_points[keep][inliers[:, 0]]
    projected, _ = cv2.projectPoints(inlier_points3d, rvec, translation, camera, None)
    rmse = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - inlier_points2d) ** 2, axis=1))))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3], _ = cv2.Rodrigues(rvec)
    transform[:3, 3] = translation[:, 0]
    return LoopEdge(-1, -1, source.frame_index, target.frame_index, transform, len(inliers), len(indices), rmse, similarity)


def find_loops(keyframes: list[Keyframe], p0: np.ndarray, p1: np.ndarray, camera: np.ndarray, separation: int, candidate_count: int, maximum_loops: int, minimum_inliers: int) -> tuple[list[LoopEdge], dict[str, int]]:
    loops: list[LoopEdge] = []
    report = {"eligible_pairs": 0, "appearance_candidates": 0, "geometrically_verified": 0}
    for target_index in range(separation, len(keyframes)):
        target = keyframes[target_index]
        similarities = np.asarray([float(np.dot(keyframes[index].signature, target.signature)) for index in range(target_index - separation)])
        if similarities.size == 0:
            continue
        report["eligible_pairs"] += int(similarities.size)
        for source_index in np.argsort(similarities)[-min(candidate_count, similarities.size):][::-1]:
            report["appearance_candidates"] += 1
            edge = verify_loop(keyframes[int(source_index)], target, p0, p1, camera, float(similarities[source_index]), minimum_inliers)
            if edge is None:
                continue
            edge.source, edge.target = int(source_index), target_index
            loops.append(edge)
            report["geometrically_verified"] += 1
    loops.sort(key=lambda edge: (edge.inliers, -edge.reprojection_rmse_px, edge.appearance_similarity), reverse=True)
    selected: list[LoopEdge] = []
    for edge in loops:
        if all(abs(edge.source - existing.source) > 1 or abs(edge.target - existing.target) > 1 for existing in selected):
            selected.append(edge)
        if len(selected) >= maximum_loops:
            break
    return selected, report


def pose_graph(initial: list[np.ndarray], loops: list[LoopEdge]) -> list[np.ndarray]:
    odometry = [inverse(initial[index + 1]) @ initial[index] for index in range(len(initial) - 1)]

    def transforms(values: np.ndarray) -> list[np.ndarray]:
        output = [initial[0]]
        for index in range(1, len(initial)):
            delta = values[(index - 1) * 6:index * 6]
            correction = np.eye(4, dtype=np.float64)
            correction[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix()
            correction[:3, 3] = delta[3:]
            output.append(correction @ initial[index])
        return output

    def edge_error(measurement: np.ndarray, source: np.ndarray, target: np.ndarray, translation_sigma: float, rotation_sigma: float) -> np.ndarray:
        residual = inverse(measurement) @ (inverse(target) @ source)
        return np.concatenate((Rotation.from_matrix(residual[:3, :3]).as_rotvec() / rotation_sigma, residual[:3, 3] / translation_sigma))

    def residual(values: np.ndarray) -> np.ndarray:
        current = transforms(values)
        parts = [edge_error(odometry[index], current[index], current[index + 1], 0.02, math.radians(2.0)) for index in range(len(odometry))]
        parts.extend(edge_error(edge.transform_target_from_source, current[edge.source], current[edge.target], 0.01, math.radians(1.0)) for edge in loops)
        return np.concatenate(parts)

    result = least_squares(residual, np.zeros((len(initial) - 1) * 6), method="trf", loss="huber", f_scale=1.0, max_nfev=150)
    if not result.success:
        raise RuntimeError(f"pose graph optimization failed: {result.message}")
    return transforms(result.x)


def write_corrected_trajectory(stamps: np.ndarray, poses_imu: list[np.ndarray], keyframe_trajectory_indices: list[int], initial_camera: list[np.ndarray], optimized_camera: list[np.ndarray], t_imu_cam0: np.ndarray, destination: Path) -> None:
    corrections = [optimized @ inverse(initial) for initial, optimized in zip(initial_camera, optimized_camera)]
    with destination.open("w", encoding="utf-8") as handle:
        for trajectory_index, (stamp, pose_imu) in enumerate(zip(stamps, poses_imu)):
            right = min(np.searchsorted(keyframe_trajectory_indices, trajectory_index), len(keyframe_trajectory_indices) - 1)
            left = max(0, right - 1)
            if left == right:
                fraction = 0.0
            else:
                denominator = keyframe_trajectory_indices[right] - keyframe_trajectory_indices[left]
                fraction = (trajectory_index - keyframe_trajectory_indices[left]) / denominator
            rotation = Slerp((0.0, 1.0), Rotation.from_matrix((corrections[left][:3, :3], corrections[right][:3, :3])))(fraction).as_matrix()
            correction = np.eye(4, dtype=np.float64)
            correction[:3, :3] = rotation
            correction[:3, 3] = (1.0 - fraction) * corrections[left][:3, 3] + fraction * corrections[right][:3, 3]
            corrected_camera = correction @ pose_imu @ t_imu_cam0
            corrected_imu = corrected_camera @ inverse(t_imu_cam0)
            quaternion = Rotation.from_matrix(corrected_imu[:3, :3]).as_quat()
            handle.write(f"{stamp:.9f} {corrected_imu[0, 3]:.9f} {corrected_imu[1, 3]:.9f} {corrected_imu[2, 3]:.9f} {quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} {quaternion[3]:.9f}\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.keyframe_stride < 1 or args.minimum_separation < 2 or args.candidate_count < 1 or args.maximum_loops < 1:
        raise ValueError("invalid loop-detection settings")
    episode, output = args.episode_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    trajectory = args.baseline_trajectory.expanduser().resolve()
    required = ("tracking.mp4", "tracking_metainfo.csv", "camera_params_tracking.json", "imu_calibration.json", "accel.csv", "gyro.csv", "head_pose.csv")
    missing = [name for name in required if not (episode / name).is_file()]
    if missing or not trajectory.is_file():
        raise FileNotFoundError(f"missing episode files {missing} or trajectory {trajectory}")
    output.mkdir(parents=True, exist_ok=True)
    stamps, poses_imu = load_tum(trajectory)
    calibration = json.loads((episode / "imu_calibration.json").read_text(encoding="utf-8"))
    image = json.loads((episode / "camera_params_tracking.json").read_text(encoding="utf-8"))
    samples = corrected_imu(episode, calibration)
    offset_ns = round(camera_time_offset_sec(calibration, "tracking", None) * 1e9)
    frames = load_frames(episode, "tracking", samples, 1, 0, offset_ns)
    if len(frames) != len(poses_imu):
        raise RuntimeError(f"Basalt trajectory has {len(poses_imu)} poses but tracking export has {len(frames)} frames")
    camera_rotation = load_camera_from_imu_rotation(args.camera_from_imu_rotation_json)
    t_imu_cam0 = factory_t_imu_camera(image["cameras"][0], "imu-to-camera", "wxyz")
    t_imu_cam1 = factory_t_imu_camera(image["cameras"][1], "imu-to-camera", "wxyz")
    if camera_rotation is not None:
        stereo = inverse(t_imu_cam0) @ t_imu_cam1
        t_imu_cam0[:3, :3] = camera_rotation.T
        t_imu_cam1 = t_imu_cam0 @ stereo
    maps0, maps1, p0, p1, camera = make_rectifier(image["cameras"], t_imu_cam0, t_imu_cam1)
    selected_frames = frames[::args.keyframe_stride]
    if frames[-1] != selected_frames[-1]:
        selected_frames.append(frames[-1])
    selected = {frame_index: (frame_index, stamp) for frame_index, stamp in selected_frames}
    trajectory_for_frame = {frame_index: index for index, (frame_index, _stamp) in enumerate(frames)}
    keyframes = extract_keyframes(episode, selected, maps0, maps1, trajectory_for_frame, args.nfeatures)
    loops, search_report = find_loops(keyframes, p0, p1, camera, args.minimum_separation, args.candidate_count, args.maximum_loops, args.minimum_inliers)
    if not loops:
        (output / "loop_search_report.json").write_text(
            json.dumps({"keyframes": len(keyframes), **search_report}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("no geometrically verified non-adjacent stereo loop was detected")
    initial_camera = [poses_imu[keyframe.trajectory_index] @ t_imu_cam0 for keyframe in keyframes]
    optimized_camera = pose_graph(initial_camera, loops)
    corrected = output / "basalt_tracking_pose_graph_imu.tum"
    write_corrected_trajectory(stamps, poses_imu, [keyframe.trajectory_index for keyframe in keyframes], initial_camera, optimized_camera, t_imu_cam0, corrected)
    reference = output / "head_pose_reference.tum"
    write_reference(episode / "head_pose.csv", reference)
    evaluate(reference, corrected, output / "evaluation")
    build_viewer(reference, corrected, output / "evaluation/trajectory_viewer", "imu")
    provenance = {
        "algorithm": "Basalt stereo-inertial plus verified stereo pose graph",
        "baseline_trajectory": str(trajectory),
        "episode": str(episode),
        "stream": "tracking",
        "keyframe_stride": args.keyframe_stride,
        "keyframes": len(keyframes),
        "loop_search": search_report,
        "camera_timestamp_offset_ns": offset_ns,
        "camera_from_imu_rotation": str(args.camera_from_imu_rotation_json),
        "loops": [
            {
                "source_frame": edge.source_frame,
                "target_frame": edge.target_frame,
                "inliers": edge.inliers,
                "joined_matches": edge.matches,
                "reprojection_rmse_px": edge.reprojection_rmse_px,
                "appearance_similarity": edge.appearance_similarity,
                "translation_m": float(np.linalg.norm(edge.transform_target_from_source[:3, 3])),
            }
            for edge in loops
        ],
        "head_pose_note": "Not used for loop detection, pose graph optimization, or model selection; used only for the final scale-gated evaluation.",
    }
    (output / "pose_graph_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
