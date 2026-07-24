#!/usr/bin/env python3
"""Build a portfolio-style local showcase package for the RM75 mainline report."""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
import shutil
import subprocess
import textwrap
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

import build_mainline_accuracy_report as report


REPO_ROOT = report.REPO_ROOT
WORKBENCH = report.WORKBENCH
SHOWCASE_ROOT = REPO_ROOT / "data" / "evaluation" / "showcase"
DEFAULT_PACKAGE_DIR = SHOWCASE_ROOT / "rm75_mainline_showcase_202606"
DEFAULT_ZIP_PATH = SHOWCASE_ROOT / "rm75_mainline_showcase_202606.zip"
VISUAL_REF_DIRNAME = "visual_refs"
SHOWCASE_FPS = 24
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
LEFT_WIDTH = 720
RIGHT_WIDTH = CANVAS_WIDTH - LEFT_WIDTH
LEFT_VIDEO_SIZE = (LEFT_WIDTH, CANVAS_HEIGHT)
OVERLAY_TARGET_CENTER = (0.5, 0.5)

SIX18_DATA_ROOT = REPO_ROOT / "data" / "gripper_data2"
SIX24_DATA_ROOT = REPO_ROOT / "data" / "gripper_data_6_24"
SIX24_BATCH_DIR = WORKBENCH / "orbslam3_rm75_batch_eval_20260630_103032"


@dataclass
class PreviewMeta:
    video: Path
    poster: Path
    eval_dir_name: str
    source_video_name: str
    source_duration_s: float
    trajectory_duration_s: float
    video_clip_start_s: float
    trajectory_clip_start_s: float
    clip_duration_s: float
    cam_first_s: float
    matched_first_s: float
    preview_start_delta_ms: float


@dataclass
class ProjectionContext:
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    t_gripper_to_cam: np.ndarray


def rel(path: Path, root: Path) -> str:
    return html.escape(path.relative_to(root).as_posix())


def fmt_seconds(seconds: float) -> str:
    total = max(0.0, float(seconds))
    minutes = int(total // 60)
    rem = total - minutes * 60
    return f"{minutes:02d}:{rem:05.2f}"


def fmt_signed_seconds(seconds: float) -> str:
    value = float(seconds)
    if value < 0:
        return f"-{fmt_seconds(abs(value))}"
    return fmt_seconds(value)


def fmt_trajectory_status(trajectory_t: float, duration: float) -> str:
    if trajectory_t < 0:
        return f"traj waits {fmt_seconds(-trajectory_t)}"
    if trajectory_t > duration:
        return f"traj complete +{fmt_seconds(trajectory_t - duration)}"
    return f"traj t = {fmt_seconds(trajectory_t)} / {fmt_seconds(duration)}"


def fmt_epoch(ts: float) -> str:
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def episode_data_dir(row: report.EpisodeResult) -> Path:
    if row.dataset_key == "6_18":
        return SIX18_DATA_ROOT / row.episode
    if row.dataset_key == "6_24":
        return SIX24_DATA_ROOT / row.episode
    raise KeyError(f"Unsupported dataset key: {row.dataset_key}")


def copy_evidence_dirs(package_dir: Path) -> list[report.DatasetSummary]:
    datasets = copy.deepcopy([report.load_6_18_dataset(), report.load_6_24_dataset()])

    for ds in datasets:
        if ds.key == "6_18":
            for row in ds.rows:
                src_dir = row.source_path.parent
                dst_dir = package_dir / src_dir.name
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                row.source_path = dst_dir / "summary.csv"
                row.viewer_path = dst_dir / "index.html"
            continue

        if ds.key == "6_24":
            dst_batch = package_dir / SIX24_BATCH_DIR.name
            shutil.copytree(SIX24_BATCH_DIR, dst_batch, dirs_exist_ok=True)
            for row in ds.rows:
                row.source_path = dst_batch / row.source_path.relative_to(SIX24_BATCH_DIR)
                row.viewer_path = dst_batch / row.viewer_path.relative_to(SIX24_BATCH_DIR)
            continue

        raise KeyError(f"Unsupported dataset key: {ds.key}")

    return datasets


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ffprobe_value(path: Path, entry: str) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            entry,
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def load_manifest_timing(manifest_path: Path, episode_dir: Path) -> tuple[Path | None, float, float]:
    manifest = json.loads(manifest_path.read_text())
    gt_path = manifest.get("ground_truth")
    manifest_episode_dir = Path(manifest["episode_dir"]) if manifest.get("episode_dir") else None
    matched_first = float(next(manifest_path.parent.joinpath("gt_tcp_matched.tum").open()).split()[0])

    if gt_path:
        gt = json.loads(Path(gt_path).read_text())
        gt_first = float(gt.get("start_time_s") or gt.get("capture_started_host_s"))
        meta = json.loads((episode_dir / "metadata.json").read_text())
        cam_meta = next(item for item in meta["video_details"] if item["name"] == "stereo_right.mkv")
        mkv_start = ffprobe_value(episode_dir / "stereo_right.mkv", "format=start_time")
        cam_first = gt_first + float(cam_meta["start_offset_us"]) / 1e6 + mkv_start
        return manifest_episode_dir, cam_first, matched_first

    timestamp_audit = manifest.get("timestamp_audit") or {}
    cam_first = timestamp_audit.get("export_cam_first_s")
    if cam_first is None:
        raise KeyError(f"Unable to determine stereo_right start time for {manifest_path}")
    return manifest_episode_dir, float(cam_first), matched_first


def load_sync_points(viewer_path: Path) -> tuple[list[dict[str, object]], float]:
    payload = json.loads(viewer_path.read_text())
    algo = payload["algorithms"][0]
    points: list[dict[str, object]] = []
    for item in algo["points"]:
        points.append(
            {
                "t": float(item["t"]),
                "gt": tuple(float(v) for v in item["gt"]),
                "gt_r": tuple(tuple(float(cell) for cell in row) for row in item["gt_r"]),
                "se3": tuple(float(v) for v in item["se3"]),
                "se3_r": tuple(tuple(float(cell) for cell in row) for row in item["se3_r"]),
                "err_mm": float(item["se3_error_m"]) * 1000.0,
            }
        )
    return points, float(algo["duration_s"])


def manifest_flag_value(manifest: dict, flag: str) -> str | None:
    commands = manifest.get("commands") or {}
    argv = commands.get("tcp_eval") or []
    for idx, token in enumerate(argv[:-1]):
        if token == flag:
            return str(argv[idx + 1])
    return None


def parse_resolution_key(value: str) -> tuple[int, int] | None:
    if "x" not in value:
        return None
    left, right = value.lower().split("x", 1)
    if not left.isdigit() or not right.isdigit():
        return None
    return int(left), int(right)


def load_projection_context(episode_dir: Path, manifest_path: Path) -> ProjectionContext:
    manifest = json.loads(manifest_path.read_text())
    camera_rig = str(manifest.get("camera_rig") or "stereo_right")
    calibration_path = manifest.get("calibration_json") or manifest_flag_value(manifest, "--calibration-json")
    handeye_path = manifest.get("handeye_yaml") or manifest_flag_value(manifest, "--handeye-yaml")
    calibration_file = Path(calibration_path) if calibration_path else episode_dir / "calibration.json"
    handeye_file = Path(handeye_path) if handeye_path else REPO_ROOT / "data/calibration/handeye_0615/handeye_result.yaml"

    calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    rig = calibration["observation"]["images"][camera_rig]
    cam0 = rig["cam0"]
    intrinsics_by_res = cam0.get("intrinsics") or {}
    if not intrinsics_by_res:
        raise KeyError(f"{calibration_file}: missing intrinsics for {camera_rig}.cam0")
    if "640x400" in intrinsics_by_res:
        res_key = "640x400"
    else:
        res_key = sorted(intrinsics_by_res)[0]
    intr = intrinsics_by_res[res_key]
    parsed_res = parse_resolution_key(res_key)
    if parsed_res is not None:
        image_width, image_height = parsed_res
    else:
        shape = rig.get("shape") or [400, 640]
        image_width = int(shape[1])
        image_height = int(shape[0] // 2)

    camera_matrix = np.array(
        [
            [float(intr["fx"]), 0.0, float(intr["ppx"])],
            [0.0, float(intr["fy"]), float(intr["ppy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion_coeffs = np.asarray(cam0.get("distortion_coeffs") or [0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(-1, 1)
    if distortion_coeffs.size < 4:
        distortion_coeffs = np.pad(distortion_coeffs.reshape(-1), (0, 4 - distortion_coeffs.size)).reshape(-1, 1)
    elif distortion_coeffs.size > 4:
        distortion_coeffs = distortion_coeffs[:4].reshape(-1, 1)

    handeye = yaml.safe_load(handeye_file.read_text(encoding="utf-8"))
    t_gripper_to_cam = np.asarray(handeye["result"]["T_gripper_to_cam"], dtype=np.float64)
    if t_gripper_to_cam.shape != (4, 4):
        raise ValueError(f"{handeye_file}: result.T_gripper_to_cam must be 4x4")

    return ProjectionContext(
        image_width=image_width,
        image_height=image_height,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        t_gripper_to_cam=t_gripper_to_cam,
    )


def validate_episode_binding(
    episode_dir: Path,
    manifest_path: Path,
    viewer_path: Path,
) -> None:
    manifest = json.loads(manifest_path.read_text())
    viewer = json.loads(viewer_path.read_text())
    manifest_episode = Path(manifest["episode_dir"]).name if manifest.get("episode_dir") else None
    require(
        manifest_episode == episode_dir.name,
        f"Manifest episode mismatch: expected {episode_dir.name}, got {manifest_episode} in {manifest_path}",
    )
    subtitle = str(viewer.get("subtitle") or "")
    require(
        episode_dir.name in subtitle,
        f"Viewer subtitle mismatch: expected {episode_dir.name} in {viewer_path}",
    )
    inputs = viewer.get("inputs") or {}
    eval_dir = Path(inputs.get("eval_dir", ""))
    gt_tum = Path(inputs.get("gt_tum", ""))
    estimate_tum = Path(inputs.get("estimate_tum", ""))
    require(
        eval_dir.name == manifest_path.parent.name,
        f"Viewer eval_dir mismatch for {episode_dir.name}: {eval_dir} vs {manifest_path.parent}",
    )
    require(
        gt_tum.parent.name == manifest_path.parent.name and estimate_tum.parent.name == manifest_path.parent.name,
        f"Viewer tum binding mismatch for {episode_dir.name}: gt={gt_tum} est={estimate_tum}",
    )


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def infer_projection_yaw(points: list[dict[str, object]]) -> float:
    gt = np.asarray([point["gt"] for point in points], dtype=float)
    if gt.shape[0] < 2:
        return math.radians(-42.0)
    delta = gt[-1] - gt[0]
    planar = delta[:2]
    if float(np.linalg.norm(planar)) < 1e-6:
        centered = gt[:, :2] - gt[:, :2].mean(axis=0, keepdims=True)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        planar = eigvecs[:, int(np.argmax(eigvals))]
    return -math.atan2(float(planar[1]), float(planar[0]))


def project_point(point: tuple[float, float, float], yaw: float) -> tuple[float, float]:
    x, y, z = point
    pitch = math.radians(24.0)
    xr = math.cos(yaw) * x - math.sin(yaw) * y
    yr = math.sin(yaw) * x + math.cos(yaw) * y
    zr = z
    yr2 = math.cos(pitch) * yr - math.sin(pitch) * zr
    zr2 = math.sin(pitch) * yr + math.cos(pitch) * zr
    return xr, -zr2 + yr2 * 0.18


def find_active_index(points: list[dict[str, object]], t_value: float) -> int:
    if t_value <= 0:
        return 0
    for idx, point in enumerate(points):
        if float(point["t"]) >= t_value:
            return idx
    return len(points) - 1


def pose_matrix(position: tuple[float, float, float], rotation: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rotation, dtype=np.float64)
    out[:3, 3] = np.asarray(position, dtype=np.float64)
    return out


def project_world_points_to_camera(
    world_points: list[tuple[float, float, float]],
    world_camera: np.ndarray,
    context: ProjectionContext,
) -> tuple[np.ndarray, np.ndarray]:
    if not world_points:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    t_camera_world = np.linalg.inv(world_camera)
    xyz = np.asarray(world_points, dtype=np.float64)
    cam_xyz = (t_camera_world[:3, :3] @ xyz.T).T + t_camera_world[:3, 3]
    valid = cam_xyz[:, 2] > 1e-3
    projected = np.full((xyz.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        pts = cam_xyz[valid].reshape(-1, 1, 3)
        uv, _ = cv2.fisheye.projectPoints(
            pts,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            context.camera_matrix,
            context.distortion_coeffs,
        )
        projected[valid] = uv.reshape(-1, 2).astype(np.float32)
    return projected, valid


def display_coords_from_raw(
    uv: np.ndarray,
    raw_size: tuple[int, int],
    display_size: tuple[int, int],
) -> np.ndarray:
    raw_w, raw_h = raw_size
    display_w, display_h = display_size
    scale = min(display_w / raw_w, display_h / raw_h)
    pad_x = (display_w - raw_w * scale) * 0.5
    pad_y = (display_h - raw_h * scale) * 0.5
    out = np.empty_like(uv, dtype=np.float32)
    out[:, 0] = uv[:, 0] * scale + pad_x
    out[:, 1] = uv[:, 1] * scale + pad_y
    return out


def recenter_overlay_points(
    gt_uv: np.ndarray,
    gt_valid: np.ndarray,
    est_uv: np.ndarray,
    est_valid: np.ndarray,
    display_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    display_w, display_h = display_size
    target = np.array([display_w * OVERLAY_TARGET_CENTER[0], display_h * OVERLAY_TARGET_CENTER[1]], dtype=np.float32)

    anchors: list[np.ndarray] = []
    if gt_uv.shape[0] and gt_valid.shape[0] and bool(gt_valid[-1]) and np.isfinite(gt_uv[-1]).all():
        anchors.append(gt_uv[-1])
    if est_uv.shape[0] and est_valid.shape[0] and bool(est_valid[-1]) and np.isfinite(est_uv[-1]).all():
        anchors.append(est_uv[-1])
    if not anchors:
        visible_points = []
        if np.any(gt_valid):
            visible_points.append(gt_uv[gt_valid])
        if np.any(est_valid):
            visible_points.append(est_uv[est_valid])
        if visible_points:
            anchors.append(np.vstack(visible_points).mean(axis=0))
    if not anchors:
        return gt_uv, est_uv

    anchor = np.vstack(anchors).mean(axis=0)
    shift = target - anchor
    return gt_uv + shift, est_uv + shift


def compute_overlay_shift(
    gt_uv: np.ndarray,
    gt_valid: np.ndarray,
    est_uv: np.ndarray,
    est_valid: np.ndarray,
    display_size: tuple[int, int],
) -> np.ndarray:
    shifted_gt, shifted_est = recenter_overlay_points(gt_uv, gt_valid, est_uv, est_valid, display_size)
    for raw, shifted, valid in ((gt_uv, shifted_gt, gt_valid), (est_uv, shifted_est, est_valid)):
        for idx, ok in enumerate(valid):
            if ok and np.isfinite(raw[idx]).all() and np.isfinite(shifted[idx]).all():
                return shifted[idx] - raw[idx]
    return np.zeros((2,), dtype=np.float32)


def draw_arrowhead(
    image: np.ndarray,
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
    color: tuple[int, int, int],
    size: int,
    thickness: int,
    outline_color: tuple[int, int, int] | None = None,
) -> None:
    dx = float(end_xy[0] - start_xy[0])
    dy = float(end_xy[1] - start_xy[1])
    norm = math.hypot(dx, dy)
    if norm < 1.0:
        return
    ux = dx / norm
    uy = dy / norm
    px = -uy
    py = ux
    tip = np.array(end_xy, dtype=np.float32)
    base = tip - np.array([ux, uy], dtype=np.float32) * size
    left = base + np.array([px, py], dtype=np.float32) * (size * 0.48)
    right = base - np.array([px, py], dtype=np.float32) * (size * 0.48)
    triangle = np.round(np.vstack([tip, left, right])).astype(np.int32)
    if outline_color is not None:
        cv2.fillConvexPoly(image, triangle, outline_color, cv2.LINE_AA)
        cv2.polylines(image, [triangle], True, outline_color, thickness + 2, cv2.LINE_AA)
    cv2.fillConvexPoly(image, triangle, color, cv2.LINE_AA)
    cv2.polylines(image, [triangle], True, color, thickness, cv2.LINE_AA)


def draw_projected_path(
    image: np.ndarray,
    points_xy: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int],
    width: int,
    *,
    outline_color: tuple[int, int, int] | None = None,
    arrow_size: int = 0,
) -> None:
    segment: list[tuple[int, int]] = []
    for idx, ok in enumerate(valid):
        if ok and np.isfinite(points_xy[idx]).all():
            segment.append((int(round(points_xy[idx, 0])), int(round(points_xy[idx, 1]))))
            continue
        if len(segment) >= 2:
            segment_arr = np.asarray(segment, dtype=np.int32)
            if outline_color is not None:
                cv2.polylines(image, [segment_arr], False, outline_color, width + 4, cv2.LINE_AA)
            cv2.polylines(image, [segment_arr], False, color, width, cv2.LINE_AA)
            if arrow_size > 0:
                draw_arrowhead(image, tuple(segment_arr[-2]), tuple(segment_arr[-1]), color, arrow_size, max(1, width - 1), outline_color)
        segment = []
    if len(segment) >= 2:
        segment_arr = np.asarray(segment, dtype=np.int32)
        if outline_color is not None:
            cv2.polylines(image, [segment_arr], False, outline_color, width + 4, cv2.LINE_AA)
        cv2.polylines(image, [segment_arr], False, color, width, cv2.LINE_AA)
        if arrow_size > 0:
            draw_arrowhead(image, tuple(segment_arr[-2]), tuple(segment_arr[-1]), color, arrow_size, max(1, width - 1), outline_color)


def draw_camera_overlay_panel(
    frame: np.ndarray,
    points: list[dict[str, object]],
    active_index: int,
    context: ProjectionContext,
    episode_name: str,
    video_elapsed_s: float,
    playback_duration_s: float,
    trajectory_t_s: float,
) -> np.ndarray:
    display_h, display_w = frame.shape[:2]
    current = points[active_index]
    trajectory_duration = float(points[-1]["t"]) if points else 0.0
    t_world_camera = pose_matrix(current["gt"], current["gt_r"]) @ context.t_gripper_to_cam
    current_t = float(current["t"])
    out = frame.copy()
    if 0.0 <= trajectory_t_s <= trajectory_duration:
        history_window_s = 4.2
        future_window_s = 0.55
        selected = [
            point
            for point in points
            if current_t - history_window_s <= float(point["t"]) <= current_t + future_window_s
        ]
        if len(selected) < 2:
            selected = points[max(0, active_index - 24) : min(len(points), active_index + 12)]

        gt_world = [point["gt"] for point in selected]
        est_world = [point["se3"] for point in selected]
        gt_uv_raw, gt_valid = project_world_points_to_camera(gt_world, t_world_camera, context)
        est_uv_raw, est_valid = project_world_points_to_camera(est_world, t_world_camera, context)
        raw_size = (context.image_width, context.image_height)
        display_size = (display_w, display_h)
        gt_uv = display_coords_from_raw(gt_uv_raw, raw_size, display_size)
        est_uv = display_coords_from_raw(est_uv_raw, raw_size, display_size)
        overlay_shift = compute_overlay_shift(gt_uv, gt_valid, est_uv, est_valid, display_size)
        gt_uv, est_uv = recenter_overlay_points(gt_uv, gt_valid, est_uv, est_valid, display_size)

        tint = out.copy()
        draw_projected_path(tint, gt_uv, gt_valid, (72, 220, 136), 10, outline_color=(6, 10, 16), arrow_size=18)
        draw_projected_path(tint, est_uv, est_valid, (82, 162, 255), 10, outline_color=(6, 10, 16), arrow_size=18)
        cv2.addWeighted(tint, 0.78, out, 0.22, 0.0, out)
        draw_projected_path(out, gt_uv, gt_valid, (124, 255, 172), 4, outline_color=(8, 12, 18), arrow_size=14)
        draw_projected_path(out, est_uv, est_valid, (102, 190, 255), 4, outline_color=(8, 12, 18), arrow_size=14)

        gt_now_raw, gt_now_valid = project_world_points_to_camera([current["gt"]], t_world_camera, context)
        est_now_raw, est_now_valid = project_world_points_to_camera([current["se3"]], t_world_camera, context)
        for uv_raw, ok, color in [
            (gt_now_raw, gt_now_valid[0], (72, 220, 136)),
            (est_now_raw, est_now_valid[0], (82, 162, 255)),
        ]:
            if not ok or not np.isfinite(uv_raw[0]).all():
                continue
            uv = display_coords_from_raw(uv_raw, raw_size, display_size)[0] + overlay_shift
            center = (int(round(uv[0])), int(round(uv[1])))
            cv2.circle(out, center, 12, (8, 12, 18), -1, cv2.LINE_AA)
            cv2.circle(out, center, 8, (248, 250, 252), -1, cv2.LINE_AA)
            cv2.circle(out, center, 5, color, -1, cv2.LINE_AA)

    cv2.rectangle(out, (18, 18), (428, 118), (10, 16, 24), thickness=-1)
    cv2.rectangle(out, (18, 18), (428, 118), (52, 72, 94), thickness=1)
    overlay_lines = [
        f"{episode_name}   stereo_right/cam0",
        f"video {fmt_seconds(video_elapsed_s)} / {fmt_seconds(playback_duration_s)}",
        f"{fmt_trajectory_status(trajectory_t_s, trajectory_duration)}   APE {float(current['err_mm']):.2f} mm",
    ]
    y = 44
    for idx, text in enumerate(overlay_lines):
        cv2.putText(
            out,
            text,
            (32, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60 if idx == 0 else 0.54,
            (240, 246, 252) if idx == 0 else (188, 202, 216),
            1,
            cv2.LINE_AA,
        )
        y += 24
    cv2.rectangle(out, (30, display_h - 62), (332, display_h - 20), (10, 16, 24), thickness=-1)
    cv2.rectangle(out, (30, display_h - 62), (332, display_h - 20), (52, 72, 94), thickness=1)
    cv2.circle(out, (48, display_h - 41), 5, (72, 220, 136), -1, cv2.LINE_AA)
    cv2.putText(out, "Robot TCP GT", (62, display_h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (196, 208, 220), 1, cv2.LINE_AA)
    cv2.circle(out, (188, display_h - 41), 5, (82, 162, 255), -1, cv2.LINE_AA)
    cv2.putText(out, "VIO TCP SE(3)", (202, display_h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (196, 208, 220), 1, cv2.LINE_AA)
    return out


def draw_trajectory_panel(
    points: list[dict[str, object]],
    panel_size: tuple[int, int],
    active_index: int,
    trajectory_duration: float,
    video_elapsed_s: float,
    trajectory_t_s: float,
    playback_duration_s: float,
    episode: str,
) -> Image.Image:
    width, height = panel_size
    image = Image.new("RGB", (width, height), (12, 18, 26))
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    body_font = load_font(18)
    small_font = load_font(15)
    tiny_font = load_font(13)
    projection_yaw = infer_projection_yaw(points)
    full_gt = [project_point(point["gt"], projection_yaw) for point in points]
    full_est = [project_point(point["se3"], projection_yaw) for point in points]
    xs = [pt[0] for pt in full_gt + full_est]
    ys = [pt[1] for pt in full_gt + full_est]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    pad = 30
    plot_left = 26
    plot_top = 110
    plot_right = width - 26
    plot_bottom = height - 84
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top
    scale = min((plot_w - 2 * pad) / span_x, (plot_h - 2 * pad) / span_y)

    def to_px(pt: tuple[float, float]) -> tuple[float, float]:
        px = plot_left + pad + (pt[0] - min_x) * scale
        py = plot_top + pad + (pt[1] - min_y) * scale
        py = plot_bottom - (py - plot_top)
        return px, py

    draw.rounded_rectangle((12, 14, width - 12, height - 14), radius=24, fill=(18, 26, 37), outline=(34, 46, 62))
    draw.text((28, 28), "SE(3)-Aligned TCP Trajectory", fill=(235, 240, 246), font=title_font)
    current = points[active_index]
    hud = (
        f"{episode[-4:]}   video t = {fmt_seconds(video_elapsed_s)} / {fmt_seconds(playback_duration_s)}"
        f"   {fmt_trajectory_status(trajectory_t_s, trajectory_duration)}"
        f"   current APE = {float(current['err_mm']):.2f} mm"
    )
    draw.text((28, 64), hud, fill=(150, 166, 183), font=body_font)
    draw.text(
        (28, 88),
        "stereo_right/cam0 playback with high-contrast hand-eye image overlay recentered for presentation; overview auto-orients to the motion direction",
        fill=(116, 137, 156),
        font=tiny_font,
    )
    draw.rounded_rectangle((plot_left, plot_top, plot_right, plot_bottom), radius=18, fill=(10, 15, 22), outline=(38, 50, 66))
    for step in range(1, 4):
        x = plot_left + step * plot_w / 4
        y = plot_top + step * plot_h / 4
        draw.line((x, plot_top + 10, x, plot_bottom - 10), fill=(32, 43, 56), width=1)
        draw.line((plot_left + 10, y, plot_right - 10, y), fill=(32, 43, 56), width=1)

    gt_px = [to_px(pt) for pt in full_gt]
    est_px = [to_px(pt) for pt in full_est]
    hist_gt = gt_px[: active_index + 1]
    hist_est = est_px[: active_index + 1]
    if len(gt_px) >= 2:
        draw.line(gt_px, fill=(56, 92, 66), width=2)
        draw.line(est_px, fill=(60, 89, 123), width=2)
    if len(hist_gt) >= 2:
        draw.line(hist_gt, fill=(68, 208, 127), width=4)
        draw.line(hist_est, fill=(78, 161, 255), width=4)
    for px, py, color in [
        (*hist_gt[-1], (68, 208, 127)),
        (*hist_est[-1], (78, 161, 255)),
    ]:
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=color, outline=(248, 250, 252), width=2)

    legend_y = height - 54
    draw.ellipse((30, legend_y, 42, legend_y + 12), fill=(68, 208, 127))
    draw.text((50, legend_y - 4), "Robot TCP GT", fill=(188, 199, 212), font=small_font)
    draw.ellipse((190, legend_y, 202, legend_y + 12), fill=(78, 161, 255))
    draw.text((210, legend_y - 4), "VIO TCP (SE3 aligned)", fill=(188, 199, 212), font=small_font)
    return image


def draw_video_overlay(
    canvas: np.ndarray,
    episode_name: str,
    video_elapsed_s: float,
    playback_duration_s: float,
    source_video_t_s: float,
    trajectory_t_s: float,
) -> None:
    x0, y0, x1, y1 = 24, 24, 388, 104
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (10, 16, 24), thickness=-1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (52, 72, 94), thickness=1)
    lines = [
        episode_name,
        f"full video  {fmt_seconds(video_elapsed_s)} / {fmt_seconds(playback_duration_s)}",
        f"cam@{fmt_seconds(source_video_t_s)}  traj@{fmt_signed_seconds(trajectory_t_s)}",
    ]
    y = 49
    for idx, text in enumerate(lines):
        font_scale = 0.66 if idx == 0 else 0.58
        color = (240, 246, 252) if idx == 0 else (188, 202, 216)
        cv2.putText(canvas, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        y += 24


def compute_preview_meta(
    episode_dir: Path,
    manifest_path: Path,
    viewer_path: Path,
    mp4_path: Path,
    jpg_path: Path,
) -> PreviewMeta:
    validate_episode_binding(episode_dir, manifest_path, viewer_path)
    _, total_duration = load_sync_points(viewer_path)
    manifest_episode_dir, cam_first_s, matched_first_s = load_manifest_timing(manifest_path, episode_dir)
    if manifest_episode_dir is not None and manifest_episode_dir.resolve() != episode_dir.resolve():
        raise RuntimeError(
            f"Episode mismatch: manifest points to {manifest_episode_dir.name}, but video source is {episode_dir.name}"
        )
    source_video = episode_dir / "stereo_right.mkv"
    source_duration = ffprobe_value(source_video, "format=duration")
    video_start_s = 0.0
    clip_start_t = cam_first_s - matched_first_s
    clip_duration = source_duration
    if clip_duration <= 1.0:
        raise RuntimeError(f"Clip duration too short for {episode_dir}")
    preview_start_delta_ms = ((cam_first_s + video_start_s) - (matched_first_s + clip_start_t)) * 1000.0
    return PreviewMeta(
        video=mp4_path,
        poster=jpg_path,
        eval_dir_name=manifest_path.parent.name,
        source_video_name=source_video.name,
        source_duration_s=source_duration,
        trajectory_duration_s=total_duration,
        video_clip_start_s=video_start_s,
        trajectory_clip_start_s=clip_start_t,
        clip_duration_s=clip_duration,
        cam_first_s=cam_first_s,
        matched_first_s=matched_first_s,
        preview_start_delta_ms=preview_start_delta_ms,
    )


def compose_sync_video(
    episode_dir: Path,
    manifest_path: Path,
    viewer_path: Path,
    mp4_path: Path,
    jpg_path: Path,
) -> PreviewMeta:
    points, total_duration = load_sync_points(viewer_path)
    meta = compute_preview_meta(episode_dir, manifest_path, viewer_path, mp4_path, jpg_path)
    source_video = episode_dir / meta.source_video_name
    projection_context = load_projection_context(episode_dir, manifest_path)

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    temp_left = mp4_path.with_suffix(".left.tmp.mp4")
    temp_canvas = mp4_path.with_suffix(".canvas.tmp.mp4")
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{meta.video_clip_start_s:.3f}",
            "-t",
            f"{meta.clip_duration_s:.3f}",
            "-i",
            str(source_video),
            "-vf",
            (
                "crop=iw:ih/2:0:0,"
                f"fps={SHOWCASE_FPS},"
                f"scale={LEFT_VIDEO_SIZE[0]}:{LEFT_VIDEO_SIZE[1]}:force_original_aspect_ratio=decrease,"
                f"pad={LEFT_VIDEO_SIZE[0]}:{LEFT_VIDEO_SIZE[1]}:(ow-iw)/2:(oh-ih)/2:black"
            ),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "30",
            str(temp_left),
        ]
    )

    cap = cv2.VideoCapture(str(temp_left))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(temp_canvas),
        cv2.VideoWriter_fourcc(*"mp4v"),
        SHOWCASE_FPS,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
    )
    first_canvas: np.ndarray | None = None
    for frame_idx in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break
        video_elapsed_s = frame_idx / SHOWCASE_FPS
        source_video_t_s = meta.video_clip_start_s + video_elapsed_s
        t_value = meta.trajectory_clip_start_s + video_elapsed_s
        active_index = find_active_index(points, t_value)
        frame = draw_camera_overlay_panel(
            frame,
            points,
            active_index,
            projection_context,
            episode_dir.name,
            video_elapsed_s,
            meta.clip_duration_s,
            t_value,
        )
        panel = np.array(
            draw_trajectory_panel(
                points,
                (RIGHT_WIDTH, CANVAS_HEIGHT),
                active_index,
                total_duration,
                video_elapsed_s,
                t_value,
                meta.clip_duration_s,
                episode_dir.name,
            )
        )
        canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        canvas[:, :LEFT_WIDTH, :] = frame
        canvas[:, LEFT_WIDTH:, :] = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        draw_video_overlay(
            canvas,
            episode_dir.name,
            video_elapsed_s,
            meta.clip_duration_s,
            source_video_t_s,
            t_value,
        )
        writer.write(canvas)
        if first_canvas is None:
            first_canvas = canvas.copy()
    cap.release()
    writer.release()

    if first_canvas is not None:
        cv2.imwrite(str(jpg_path), first_canvas)

    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_canvas),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ]
    )
    temp_left.unlink(missing_ok=True)
    temp_canvas.unlink(missing_ok=True)
    return meta


def build_previews(datasets: list[report.DatasetSummary], package_dir: Path, force: bool) -> dict[str, PreviewMeta]:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required to build visual reference previews.")

    visual_ref_dir = package_dir / VISUAL_REF_DIRNAME
    visual_ref_dir.mkdir(parents=True, exist_ok=True)
    preview_index: dict[str, PreviewMeta] = {}
    keep_names: set[str] = set()
    for ds in datasets:
        for row in ds.rows:
            base_name = f"{ds.key}_{row.episode}"
            keep_names.add(f"{base_name}_preview.mp4")
            keep_names.add(f"{base_name}_poster.jpg")
            mp4_path = visual_ref_dir / f"{base_name}_preview.mp4"
            jpg_path = visual_ref_dir / f"{base_name}_poster.jpg"
            if force or not mp4_path.exists() or not jpg_path.exists():
                preview = compose_sync_video(
                    episode_data_dir(row),
                    row.source_path.parent / "orbslam3_tcp_eval_manifest.json",
                    row.source_path.parent / "viewer_data_3d.json",
                    mp4_path,
                    jpg_path,
                )
            else:
                preview = compute_preview_meta(
                    episode_data_dir(row),
                    row.source_path.parent / "orbslam3_tcp_eval_manifest.json",
                    row.source_path.parent / "viewer_data_3d.json",
                    mp4_path,
                    jpg_path,
                )
            preview_index[row.episode] = preview
    for path in visual_ref_dir.iterdir():
        if path.is_file() and path.name not in keep_names:
            path.unlink()
    return preview_index


def render_metric_card(title: str, value: str, note: str) -> str:
    return (
        "<article class='metric-card'>"
        f"<span>{html.escape(title)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        f"<em>{html.escape(note)}</em>"
        "</article>"
    )


def render_episode_card(
    package_dir: Path,
    row: report.EpisodeResult,
    preview: PreviewMeta,
) -> str:
    offset_ms = row.strict_sync_offset_sec * 1000.0 if row.strict_sync_offset_sec is not None else None
    trajectory_anchor_s = preview.matched_first_s - preview.cam_first_s
    if trajectory_anchor_s >= 0:
        timeline_anchor = f"matched trajectory t=00:00.00 begins at video {fmt_seconds(trajectory_anchor_s)}"
    else:
        timeline_anchor = f"stereo_right starts {fmt_seconds(-trajectory_anchor_s)} after matched trajectory"
    chips = [
        f"<span>APE {row.ape_mm:.3f} mm</span>",
        f"<span>RPE {row.rpe_mm:.3f} mm</span>" if row.rpe_mm is not None else "",
        f"<span>Rot {row.ape_rot_deg:.3f} deg</span>" if row.ape_rot_deg is not None else "",
        f"<span>Offset {offset_ms:+.1f} ms</span>" if offset_ms is not None else "",
    ]
    facts = [
        ("Evidence Pair", f"{row.episode} -> {preview.eval_dir_name}"),
        (
            "Original Durations",
            f"{preview.source_video_name} {fmt_seconds(preview.source_duration_s)} | matched traj {fmt_seconds(preview.trajectory_duration_s)}",
        ),
        (
            "Full Playback",
            f"rendered from stereo_right/cam0 {fmt_seconds(preview.video_clip_start_s)} to {fmt_seconds(preview.video_clip_start_s + preview.clip_duration_s)}",
        ),
        (
            "Timeline Anchor",
            timeline_anchor,
        ),
        (
            "Aligned Start",
            (
                f"video {fmt_epoch(preview.cam_first_s + preview.video_clip_start_s)}"
                f" | traj {fmt_epoch(preview.matched_first_s + preview.trajectory_clip_start_s)}"
                f" | delta {preview.preview_start_delta_ms:+.1f} ms"
            ),
        ),
    ]
    return f"""
      <article class="episode-card">
        <div class="episode-head">
          <div>
            <div class="episode-kicker">{html.escape(row.dataset_label)}</div>
            <h3>{html.escape(row.episode)}</h3>
          </div>
          <div class="episode-ape">{row.ape_mm:.3f}<span>mm</span></div>
        </div>
        <div class="media-label">Full-length stereo_right/cam0 with hand-eye overlay + synchronized SE(3) trajectory</div>
        <video class="episode-video" controls muted preload="none" playsinline poster="{rel(preview.poster, package_dir)}">
          <source src="{rel(preview.video, package_dir)}" type="video/mp4">
        </video>
        <div class="episode-chip-row">
          {''.join(chips)}
        </div>
        <div class="episode-facts">
          {''.join(f'<div><strong>{html.escape(label)}</strong><span>{html.escape(value)}</span></div>' for label, value in facts)}
        </div>
        <div class="episode-actions">
          <a href="{rel(row.viewer_path, package_dir)}">Open full 3D viewer</a>
          <a href="{rel(row.source_path, package_dir)}">summary csv</a>
        </div>
      </article>
    """


def render_dataset_section(
    package_dir: Path,
    ds: report.DatasetSummary,
    preview_index: dict[str, PreviewMeta],
) -> str:
    best = min(ds.rows, key=lambda row: row.ape_mm)
    cards = "".join(render_episode_card(package_dir, row, preview_index[row.episode]) for row in ds.rows)
    return f"""
    <section class="dataset-section">
      <div class="dataset-topline">
        <div>
          <div class="eyebrow">{html.escape(ds.label)}</div>
          <h2>{html.escape(ds.label)}</h2>
          <p>All episodes below use the same validated ORB-SLAM3 mainline and satisfy the curated APE <= {report.SHOWCASE_APE_MAX_MM:.0f} mm gate. Each card plays the full original `stereo_right/cam0` duration: left overlays GT and SE(3)-aligned TCP onto the calibrated camera image using the hand-eye chain, right keeps the synchronized trajectory overview on the same wall-clock timeline.</p>
        </div>
        <div class="dataset-badge">
          <span>Mean APE</span>
          <strong>{ds.mean:.3f} mm</strong>
          <em>Best {best.episode[-4:]} at {best.ape_mm:.3f} mm</em>
        </div>
      </div>
      <div class="episode-grid">
        {cards}
      </div>
    </section>
    """


def build_showcase_html(
    package_dir: Path,
    datasets: list[report.DatasetSummary],
    preview_index: dict[str, PreviewMeta],
) -> str:
    all_rows = [row for ds in datasets for row in ds.rows]
    best = min(all_rows, key=lambda row: row.ape_mm)
    worst = max(all_rows, key=lambda row: row.ape_mm)
    combined_mean = sum(row.ape_mm for row in all_rows) / len(all_rows)
    delta = datasets[1].mean - datasets[0].mean
    included_count = len(all_rows)

    sections = "".join(render_dataset_section(package_dir, ds, preview_index) for ds in datasets)
    report_link = rel(package_dir / "report.html", package_dir)
    best_viewer = rel(best.viewer_path, package_dir)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RM75 Mainline Showcase</title>
  <style>
    :root {{
      --bg: #f1f5f9;
      --ink: #102033;
      --muted: #5f6f80;
      --line: rgba(16,32,51,0.10);
      --panel: rgba(255,255,255,0.90);
      --hero: #0f172a;
      --teal: #0f766e;
      --amber: #c2410c;
      --shadow: 0 28px 70px rgba(15,23,42,0.14);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; }}
    body {{
      color: var(--ink);
      font: 15px/1.6 "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.12), transparent 24%),
        radial-gradient(circle at top right, rgba(194,65,12,0.10), transparent 28%),
        linear-gradient(180deg, #fbfdff 0%, #edf2f7 100%);
    }}
    a {{ color: inherit; text-decoration: none; }}
    .shell {{
      width: min(1380px, calc(100vw - 36px));
      margin: 18px auto 40px;
    }}
    .hero {{
      position: relative;
      overflow: hidden;
      border-radius: 34px;
      padding: 34px 36px 30px;
      background:
        linear-gradient(145deg, rgba(15,23,42,0.98), rgba(15,23,42,0.90)),
        linear-gradient(135deg, rgba(15,118,110,0.55), rgba(194,65,12,0.35));
      color: white;
      box-shadow: 0 38px 90px rgba(15,23,42,0.24);
    }}
    .hero::after {{
      content: "";
      position: absolute;
      right: -90px;
      bottom: -110px;
      width: 360px;
      height: 360px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(255,255,255,0.12), transparent 70%);
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(255,255,255,0.72);
    }}
    h1 {{
      margin: 12px 0 0;
      max-width: 11ch;
      font-size: clamp(36px, 5.8vw, 74px);
      line-height: 0.96;
      letter-spacing: -0.06em;
    }}
    .lede {{
      margin: 18px 0 0;
      max-width: 78ch;
      color: rgba(255,255,255,0.82);
      font-size: 16px;
    }}
    .hero-actions {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 24px;
    }}
    .hero-actions a {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 12px 16px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.10);
      color: white;
      backdrop-filter: blur(8px);
    }}
    .hero-actions a.primary {{
      background: white;
      color: var(--hero);
      font-weight: 600;
    }}
    .hero-metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 26px;
      position: relative;
      z-index: 1;
    }}
    .metric-card {{
      padding: 16px 18px;
      border-radius: 22px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.08);
      backdrop-filter: blur(6px);
    }}
    .metric-card span {{
      display: block;
      color: rgba(255,255,255,0.70);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 12px;
    }}
    .metric-card strong {{
      display: block;
      margin-top: 10px;
      font-size: 30px;
      line-height: 1;
      letter-spacing: -0.05em;
    }}
    .metric-card em {{
      display: block;
      margin-top: 8px;
      color: rgba(255,255,255,0.75);
      font-style: normal;
      font-size: 13px;
    }}
    .overview {{
      margin-top: 18px;
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
    }}
    .panel {{
      border-radius: 28px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
      padding: 24px;
    }}
    .panel h2 {{
      margin: 0;
      font-size: 24px;
      letter-spacing: -0.04em;
    }}
    .panel p {{
      margin: 8px 0 0;
      color: var(--muted);
    }}
    .key-list {{
      margin: 16px 0 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 12px;
    }}
    .key-list li {{
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(15,23,42,0.03);
      border: 1px solid rgba(16,32,51,0.06);
    }}
    .dataset-section {{
      margin-top: 18px;
      padding: 24px;
      border-radius: 30px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .dataset-topline {{
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 18px;
    }}
    .dataset-topline h2 {{
      margin: 4px 0 0;
      font-size: 28px;
      letter-spacing: -0.04em;
    }}
    .dataset-topline p {{
      margin: 8px 0 0;
      max-width: 72ch;
      color: var(--muted);
    }}
    .dataset-badge {{
      min-width: 260px;
      padding: 16px 18px;
      border-radius: 22px;
      background: linear-gradient(180deg, rgba(15,118,110,0.10), rgba(255,255,255,0.98));
      border: 1px solid rgba(15,118,110,0.14);
    }}
    .dataset-badge span {{
      display: block;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.09em;
      font-size: 12px;
    }}
    .dataset-badge strong {{
      display: block;
      margin-top: 8px;
      font-size: 34px;
      color: var(--teal);
      letter-spacing: -0.05em;
    }}
    .dataset-badge em {{
      display: block;
      margin-top: 8px;
      color: var(--muted);
      font-style: normal;
    }}
    .episode-grid {{
      margin-top: 20px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .episode-card {{
      padding: 16px;
      border-radius: 24px;
      background: linear-gradient(180deg, rgba(248,250,252,0.95), rgba(241,245,249,0.98));
      border: 1px solid rgba(16,32,51,0.08);
    }}
    .episode-head {{
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 12px;
    }}
    .episode-kicker {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--muted);
    }}
    .episode-head h3 {{
      margin: 4px 0 0;
      font-size: 22px;
      letter-spacing: -0.03em;
    }}
    .episode-ape {{
      font-size: 28px;
      line-height: 1;
      letter-spacing: -0.05em;
      color: var(--amber);
      white-space: nowrap;
    }}
    .episode-ape span {{
      margin-left: 4px;
      font-size: 12px;
      letter-spacing: 0.08em;
      color: var(--muted);
      text-transform: uppercase;
    }}
    .episode-video {{
      display: block;
      width: 100%;
      margin-top: 10px;
      border-radius: 18px;
      background: #0f172a;
      border: 1px solid rgba(16,32,51,0.08);
    }}
    .media-label {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .episode-chip-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .episode-chip-row span {{
      padding: 8px 10px;
      border-radius: 999px;
      background: white;
      border: 1px solid rgba(16,32,51,0.08);
      color: var(--muted);
      font-size: 12px;
    }}
    .episode-facts {{
      display: grid;
      gap: 8px;
      margin-top: 14px;
    }}
    .episode-facts div {{
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(16,32,51,0.06);
    }}
    .episode-facts strong {{
      display: block;
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .episode-facts span {{
      display: block;
      margin-top: 4px;
      font-size: 13px;
      color: var(--ink);
      line-height: 1.45;
      word-break: break-word;
    }}
    .episode-actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .episode-actions a {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 10px 12px;
      border-radius: 999px;
      background: white;
      border: 1px solid rgba(16,32,51,0.10);
    }}
    .footer {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 13px;
      text-align: center;
    }}
    @media (max-width: 1120px) {{
      .hero-metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .overview {{ grid-template-columns: 1fr; }}
      .dataset-topline {{ flex-direction: column; }}
      .dataset-badge {{ min-width: 0; width: 100%; }}
    }}
    @media (max-width: 820px) {{
      .shell {{ width: min(100vw - 20px, 1380px); margin-top: 10px; }}
      .hero {{ padding: 24px 22px; border-radius: 24px; }}
      .hero-metrics {{ grid-template-columns: 1fr; }}
      .episode-grid {{ grid-template-columns: 1fr; }}
      .panel, .dataset-section {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Portfolio Showcase</div>
      <h1>RM75 Visual-Inertial Mainline</h1>
      <p class="lede">
        A presentation-ready package for a fixed ORB-SLAM3 RM75 evaluation mainline. It combines formal accuracy reporting,
        3D trajectory viewers, and synchronized single-canvas evidence so the quality claim is supported by both
        quantitative metrics and direct visual traces from the original episodes. This curated package keeps only
        episodes whose TCP translation APE RMSE is at or below {report.SHOWCASE_APE_MAX_MM:.0f} mm.
      </p>
      <div class="hero-actions">
        <a class="primary" href="{report_link}">Open formal report</a>
        <a href="{best_viewer}">Open best 3D viewer</a>
      </div>
      <div class="hero-metrics">
        {render_metric_card("Combined mean APE", f"{combined_mean:.3f} mm", f"{included_count} retained episodes under one mainline")}
        {render_metric_card("Best episode", f"{best.ape_mm:.3f} mm", f"{best.episode} on {best.dataset_label}")}
        {render_metric_card("Worst episode", f"{worst.ape_mm:.3f} mm", f"{worst.episode} on {worst.dataset_label}")}
        {render_metric_card("Cross-session delta", f"{abs(delta):.3f} mm", "Difference between 2026-06-18 and 2026-06-24 means")}
      </div>
    </section>

    <section class="overview">
      <article class="panel">
        <h2>What This Package Shows</h2>
        <p>The same production-intent evaluation configuration is reused across two independent recording sessions. The report and viewers establish the metric result, while the per-episode video panels expose the underlying visual scene and motion quality with centered, high-contrast trajectory overlays and motion-aligned overview views.</p>
        <ul class="key-list">
          <li>Formal HTML accuracy report with per-dataset APE distribution and embedded 3D TCP trajectory viewers.</li>
          <li>One full-length synchronized playback per episode, with calibrated `stereo_right/cam0` image overlay on the left and a motion-aligned SE(3) trajectory overview on the right.</li>
          <li>Portable folder layout plus zip archive for offline review, internal sharing, or personal portfolio presentation.</li>
        </ul>
      </article>
      <article class="panel">
        <h2>Evaluation Scope</h2>
        <p>Included here are only the currently most stable mainline results that satisfy the showcase gate of APE <= {report.SHOWCASE_APE_MAX_MM:.0f} mm: 2026-06-18 `gripper_data2` keeps episode `0004`, and 2026-06-24 `gripper_data_6_24` keeps episodes `0001`, `0003`, `0004`, `0005`, and `0006`.</p>
        <ul class="key-list">
          <li>Primary metric: TCP translation APE RMSE in millimeters.</li>
          <li>Supporting metrics: translation RPE, rotation APE, and strict-sync offset.</li>
          <li>Excluded for clarity: prototype branches, CLAHE experiments, and unstable 6_25 variants.</li>
        </ul>
      </article>
    </section>

    {sections}

    <div class="footer">
      Package generated from curated outputs in <code>data/evaluation/workbench</code> and original episode media under
      <code>data/gripper_data2</code> and <code>data/gripper_data_6_24</code>.
    </div>
  </div>
</body>
</html>
"""


def write_readme(package_dir: Path, zip_path: Path) -> None:
    content = textwrap.dedent(
        f"""
        # RM75 Mainline Showcase

        This package is a self-contained presentation bundle for the current stable RM75 ORB-SLAM3 mainline.
        It includes only episodes whose TCP translation APE RMSE is at or below {report.SHOWCASE_APE_MAX_MM:.0f} mm.

        ## Open order

        1. `index.html` for the portfolio-style overview
        2. `report.html` for the formal accuracy report
        3. Any linked `index.html` inside the copied evaluation directories for full 3D trajectory inspection

        ## Included content

        - `visual_refs/`: full-length synchronized video previews and posters for each included episode
        - `orbslam3_rm75_batch_eval_20260630_103032/`: copied 6_24 batch evaluation evidence
        - `eval_episode_20260618_*`: copied 6_18 per-episode evaluation evidence

        ## Archive

        - Zip package: `{zip_path.name}`
        """
    ).strip() + "\n"
    (package_dir / "README.md").write_text(content, encoding="utf-8")


def zip_package(package_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package_dir.rglob("*")):
            archive.write(path, path.relative_to(package_dir.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--force-previews", action="store_true")
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    package_dir.mkdir(parents=True, exist_ok=True)

    datasets = copy_evidence_dirs(package_dir)
    preview_index = build_previews(datasets, package_dir, force=args.force_previews)

    report_path = package_dir / "report.html"
    original_workbench = report.WORKBENCH
    report.WORKBENCH = package_dir
    try:
        report_html = report.build_html(datasets, report_path)
    finally:
        report.WORKBENCH = original_workbench
    report_path.write_text(report_html, encoding="utf-8")

    showcase_html = build_showcase_html(package_dir, datasets, preview_index)
    (package_dir / "index.html").write_text(showcase_html, encoding="utf-8")
    write_readme(package_dir, args.zip_path.resolve())
    zip_package(package_dir, args.zip_path.resolve())

    print(package_dir)
    print(args.zip_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
