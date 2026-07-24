#!/usr/bin/env python3
"""Find and export a contiguous RM75 showcase window whose cropped APE SE3 RMSE stays below a target."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import runpy
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
SHOWCASE_ROOT = REPO_ROOT / "data" / "evaluation" / "showcase" / "rm75_mainline_showcase_202606"
COMPARISON_VIEWER = REPO_ROOT / "script" / "visualize" / "visualize_tcp_trajectory_comparison.py"

SHOWCASE_FPS = 24
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
LEFT_WIDTH = 720
RIGHT_WIDTH = CANVAS_WIDTH - LEFT_WIDTH
LEFT_VIDEO_SIZE = (LEFT_WIDTH, CANVAS_HEIGHT)


@dataclass
class EvalContext:
    eval_dir: Path
    viewer_path: Path
    manifest_path: Path
    gt_tum: Path
    est_tum: Path
    episode_dir: Path
    source_video: Path
    source_duration_s: float
    trajectory_first_rel_s: float
    trajectory_last_rel_s: float
    video_anchor_rel_s: float


@dataclass
class WindowCandidate:
    context: EvalContext
    start_s: float
    end_s: float
    duration_s: float
    heuristic_rmse_mm: float
    peak_error_mm: float
    exact_rmse_mm: float | None = None
    sample_count: int | None = None


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


def load_manifest_timing(manifest_path: Path, episode_dir: Path) -> tuple[float, float]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matched_first = float(next(manifest_path.parent.joinpath("gt_tcp_matched.tum").open()).split()[0])
    gt_path = manifest.get("ground_truth")
    if gt_path:
        gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
        gt_first = float(gt.get("start_time_s") or gt.get("capture_started_host_s"))
        meta = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        cam_meta = next(item for item in meta["video_details"] if item["name"] == "cam_right.mkv")
        mkv_start = ffprobe_value(episode_dir / "cam_right.mkv", "format=start_time")
        cam_first = gt_first + float(cam_meta["start_offset_us"]) / 1e6 + mkv_start
        return cam_first, matched_first
    timestamp_audit = manifest.get("timestamp_audit") or {}
    cam_first = timestamp_audit.get("export_cam_first_s")
    if cam_first is None:
        raise KeyError(f"Unable to determine cam_right start time for {manifest_path}")
    return float(cam_first), matched_first


def rel_seconds(timestamp_s: np.ndarray) -> np.ndarray:
    return timestamp_s - float(timestamp_s[0])


def fmt_seconds(seconds: float) -> str:
    total = max(0.0, float(seconds))
    minutes = int(total // 60)
    rem = total - minutes * 60
    return f"{minutes:02d}:{rem:05.2f}"


def fmt_epoch(ts: float) -> str:
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def load_comparison_module() -> dict[str, object]:
    return runpy.run_path(str(COMPARISON_VIEWER), run_name="__export_rm75_ape_window__")


def read_tum_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(part) for part in line.split()])
    return rows


def write_tum_rows(path: Path, rows: Iterable[list[float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f"{row[0]:.9f} {row[1]:.9f} {row[2]:.9f} {row[3]:.9f} "
                f"{row[4]:.9f} {row[5]:.9f} {row[6]:.9f} {row[7]:.9f}\n"
            )


def infer_eval_context(eval_dir: Path) -> EvalContext:
    viewer_path = eval_dir / "viewer_data_3d.json"
    manifest_path = eval_dir / "orbslam3_tcp_eval_manifest.json"
    gt_tum = eval_dir / "gt_tcp_matched.tum"
    est_tum = eval_dir / "vio_tcp_matched.tum"
    if not viewer_path.is_file():
        raise FileNotFoundError(f"missing viewer json: {viewer_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing eval manifest: {manifest_path}")
    if not gt_tum.is_file() or not est_tum.is_file():
        raise FileNotFoundError(f"missing matched TUM files in {eval_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_dir = Path(manifest["episode_dir"]).resolve()
    source_video = episode_dir / "cam_right.mkv"
    source_duration_s = ffprobe_value(source_video, "format=duration")
    cam_first_s, matched_first_s = load_manifest_timing(manifest_path, episode_dir)

    comparison = load_comparison_module()
    helpers = comparison["load_helpers"]()
    gt_t, _, _ = comparison["read_tum"](gt_tum, helpers)
    rel_t = rel_seconds(gt_t)

    return EvalContext(
        eval_dir=eval_dir,
        viewer_path=viewer_path,
        manifest_path=manifest_path,
        gt_tum=gt_tum,
        est_tum=est_tum,
        episode_dir=episode_dir,
        source_video=source_video,
        source_duration_s=source_duration_s,
        trajectory_first_rel_s=float(rel_t[0]),
        trajectory_last_rel_s=float(rel_t[-1]),
        video_anchor_rel_s=matched_first_s - cam_first_s,
    )


def discover_eval_dirs(showcase_root: Path) -> list[Path]:
    found: list[Path] = []
    for manifest in sorted(showcase_root.rglob("orbslam3_tcp_eval_manifest.json")):
        found.append(manifest.parent)
    return found


def feasible_rel_window(context: EvalContext) -> tuple[float, float]:
    trajectory_rel_at_video_zero = -context.video_anchor_rel_s
    feasible_start = max(context.trajectory_first_rel_s, trajectory_rel_at_video_zero)
    feasible_end = min(context.trajectory_last_rel_s, trajectory_rel_at_video_zero + context.source_duration_s)
    return feasible_start, feasible_end


def find_heuristic_window(context: EvalContext, threshold_mm: float) -> WindowCandidate | None:
    payload = json.loads(context.viewer_path.read_text(encoding="utf-8"))
    points = payload["algorithms"][0]["points"]
    if not points:
        return None

    feasible_start, feasible_end = feasible_rel_window(context)
    times = np.asarray([float(p["t"]) for p in points], dtype=float)
    errs = np.asarray([float(p["se3_error_m"]) * 1000.0 for p in points], dtype=float)
    mask = (times >= feasible_start) & (times <= feasible_end)
    if not np.any(mask):
        return None

    times = times[mask]
    errs = errs[mask]
    prefix = np.zeros(errs.shape[0] + 1, dtype=float)
    prefix[1:] = np.cumsum(errs * errs)

    best: tuple[float, int, int, float, float] | None = None
    j = 0
    for i in range(errs.shape[0]):
        if j < i:
            j = i
        while j < errs.shape[0]:
            count = j - i + 1
            rmse = math.sqrt(float(prefix[j + 1] - prefix[i]) / count)
            if rmse <= threshold_mm:
                duration = float(times[j] - times[i])
                peak = float(np.max(errs[i : j + 1]))
                candidate = (duration, i, j, rmse, peak)
                if best is None or candidate[0] > best[0] or (
                    math.isclose(candidate[0], best[0]) and candidate[3] < best[3]
                ):
                    best = candidate
                j += 1
            else:
                break
    if best is None:
        return None

    duration, i, j, rmse, peak = best
    return WindowCandidate(
        context=context,
        start_s=float(times[i]),
        end_s=float(times[j]),
        duration_s=duration,
        heuristic_rmse_mm=rmse,
        peak_error_mm=peak,
    )


def validate_exact_ape(candidate: WindowCandidate) -> WindowCandidate:
    comparison = load_comparison_module()
    helpers = comparison["load_helpers"]()
    read_tum = comparison["read_tum"]
    align_positions = comparison["align_positions"]

    gt_t, gt_pos, gt_rot = read_tum(candidate.context.gt_tum, helpers)
    est_t, est_pos, est_rot = read_tum(candidate.context.est_tum, helpers)
    rel_t = rel_seconds(gt_t)
    mask = (rel_t >= candidate.start_s) & (rel_t <= candidate.end_s)
    if int(np.count_nonzero(mask)) < 3:
        candidate.exact_rmse_mm = math.inf
        candidate.sample_count = int(np.count_nonzero(mask))
        return candidate
    _, _, se3_err = align_positions(
        helpers,
        gt_pos[mask],
        gt_rot[mask],
        est_pos[mask],
        est_rot[mask],
        gt_t[mask],
        False,
    )
    candidate.exact_rmse_mm = math.sqrt(float(np.mean(se3_err**2))) * 1000.0
    candidate.sample_count = int(np.count_nonzero(mask))
    return candidate


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
    yr2 = math.cos(pitch) * yr - math.sin(pitch) * z
    zr2 = math.sin(pitch) * yr + math.cos(pitch) * z
    return xr, -zr2 + yr2 * 0.18


def find_active_index(points: list[dict[str, object]], t_value: float) -> int:
    if t_value <= 0:
        return 0
    for idx, point in enumerate(points):
        if float(point["t"]) >= t_value:
            return idx
    return len(points) - 1


def draw_trajectory_panel(
    points: list[dict[str, object]],
    panel_size: tuple[int, int],
    active_index: int,
    duration_s: float,
    playback_t_s: float,
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
    full_gt = [project_point(tuple(float(v) for v in point["gt"]), projection_yaw) for point in points]
    full_est = [project_point(tuple(float(v) for v in point["se3"]), projection_yaw) for point in points]
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
    draw.text((28, 28), "Cropped SE(3)-Aligned TCP Trajectory", fill=(235, 240, 246), font=title_font)
    current = points[active_index]
    hud = (
        f"{episode[-4:]}   playback {fmt_seconds(playback_t_s)} / {fmt_seconds(duration_s)}"
        f"   current APE = {float(current['se3_error_m']) * 1000.0:.2f} mm"
    )
    draw.text((28, 64), hud, fill=(150, 166, 183), font=body_font)
    draw.text(
        (28, 88),
        "window metric is recomputed on this cropped segment; trajectory view follows the segment motion trend",
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
    playback_t_s: float,
    duration_s: float,
    source_t_s: float,
    ape_rmse_mm: float,
) -> None:
    x0, y0, x1, y1 = 24, 24, 430, 108
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (10, 16, 24), thickness=-1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (52, 72, 94), thickness=1)
    lines = [
        episode_name,
        f"window playback  {fmt_seconds(playback_t_s)} / {fmt_seconds(duration_s)}",
        f"cam@{fmt_seconds(source_t_s)}  rmse {ape_rmse_mm:.3f} mm",
    ]
    y = 49
    for idx, text in enumerate(lines):
        font_scale = 0.66 if idx == 0 else 0.58
        color = (240, 246, 252) if idx == 0 else (188, 202, 216)
        cv2.putText(canvas, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        y += 24


def render_window_preview(
    candidate: WindowCandidate,
    output_video: Path,
    output_poster: Path,
    payload: dict[str, object],
) -> None:
    points = payload["algorithms"][0]["points"]
    duration_s = float(payload["algorithms"][0]["duration_s"])
    episode_name = candidate.context.episode_dir.name
    video_start_s = candidate.context.video_anchor_rel_s + candidate.start_s
    output_video.parent.mkdir(parents=True, exist_ok=True)
    temp_left = output_video.with_suffix(".left.tmp.mp4")
    temp_canvas = output_video.with_suffix(".canvas.tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{video_start_s:.3f}",
            "-t",
            f"{duration_s:.3f}",
            "-i",
            str(candidate.context.source_video),
            "-vf",
            (
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
        ],
        check=True,
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
        playback_t_s = frame_idx / SHOWCASE_FPS
        active_index = find_active_index(points, playback_t_s)
        panel = np.array(
            draw_trajectory_panel(
                points,
                (RIGHT_WIDTH, CANVAS_HEIGHT),
                active_index,
                duration_s,
                playback_t_s,
                episode_name,
            )
        )
        canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        canvas[:, :LEFT_WIDTH, :] = frame
        canvas[:, LEFT_WIDTH:, :] = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        draw_video_overlay(
            canvas,
            episode_name,
            playback_t_s,
            duration_s,
            video_start_s + playback_t_s,
            float(candidate.exact_rmse_mm or candidate.heuristic_rmse_mm),
        )
        writer.write(canvas)
        if first_canvas is None:
            first_canvas = canvas.copy()
    cap.release()
    writer.release()

    if first_canvas is not None:
        cv2.imwrite(str(output_poster), first_canvas)

    subprocess.run(
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
            str(output_video),
        ],
        check=True,
    )
    temp_left.unlink(missing_ok=True)
    temp_canvas.unlink(missing_ok=True)


def infer_episode_label(eval_dir: Path) -> str:
    match = re.search(r"(episode_\d{8}_\d{4})", str(eval_dir))
    return match.group(1) if match else eval_dir.name


def write_window_viewer(output_dir: Path, candidate: WindowCandidate, payload: dict[str, object]) -> Path:
    comparison = load_comparison_module()
    html = comparison["HTML"]
    title = infer_episode_label(candidate.context.eval_dir)
    html = html.replace("<title>TCP trajectory comparison</title>", f"<title>{title} APE window viewer</title>")
    html = html.replace(
        "TCP trajectory comparison: robot GT vs VINS / DynaVINS",
        f"{title} cropped APE window viewer",
    )
    html = html.replace(
        "TCP comparison. VIO chain: T_world_tcp = T_world_imu @ inv(T_left_camera_imu) @ inv(T_tcp_left_camera).",
        f"{title}: cropped window with recomputed SE(3) APE under threshold.",
    )
    html = html.replace(
        "<span><i class=\"dot\" style=\"background:var(--dynavins)\"></i>DynaVINS-derived TCP</span>\n",
        "",
    )
    path = output_dir / "index.html"
    path.write_text(html.replace("__VIEWER_DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    return path


def export_candidate(candidate: WindowCandidate, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = load_comparison_module()
    helpers = comparison["load_helpers"]()
    read_tum = comparison["read_tum"]
    align_positions = comparison["align_positions"]
    gt_t, gt_pos, gt_rot = read_tum(candidate.context.gt_tum, helpers)
    est_t, est_pos, est_rot = read_tum(candidate.context.est_tum, helpers)
    rel_t = rel_seconds(gt_t)
    mask = (rel_t >= candidate.start_s) & (rel_t <= candidate.end_s)

    gt_rows = read_tum_rows(candidate.context.gt_tum)
    est_rows = read_tum_rows(candidate.context.est_tum)
    gt_rows = [row for row, keep in zip(gt_rows, mask) if keep]
    est_rows = [row for row, keep in zip(est_rows, mask) if keep]
    gt_window_tum = output_dir / "gt_tcp_matched.tum"
    est_window_tum = output_dir / "vio_tcp_matched.tum"
    write_tum_rows(gt_window_tum, gt_rows)
    write_tum_rows(est_window_tum, est_rows)

    gt_t_w = gt_t[mask]
    gt_pos_w = gt_pos[mask]
    gt_rot_w = gt_rot[mask]
    est_t_w = est_t[mask]
    est_pos_w = est_pos[mask]
    est_rot_w = est_rot[mask]
    se3_pos, se3_rot, se3_errors = align_positions(helpers, gt_pos_w, gt_rot_w, est_pos_w, est_rot_w, gt_t_w, False)
    sim3_pos, sim3_rot, sim3_errors = align_positions(helpers, gt_pos_w, gt_rot_w, est_pos_w, est_rot_w, gt_t_w, True)
    raw_errors = np.linalg.norm(est_pos_w - gt_pos_w, axis=1)
    duration_s = float(gt_t_w[-1] - gt_t_w[0]) if gt_t_w.size >= 2 else 0.0

    points = []
    t0 = float(gt_t_w[0])
    for i in range(gt_t_w.shape[0]):
        points.append(
            {
                "t": float(gt_t_w[i] - t0),
                "gt": gt_pos_w[i].tolist(),
                "gt_r": gt_rot_w[i].tolist(),
                "raw": est_pos_w[i].tolist(),
                "raw_r": est_rot_w[i].tolist(),
                "se3": se3_pos[i].tolist(),
                "se3_r": se3_rot[i].tolist(),
                "sim3": sim3_pos[i].tolist(),
                "sim3_r": sim3_rot[i].tolist(),
                "raw_error_m": float(raw_errors[i]),
                "se3_error_m": float(se3_errors[i]),
                "sim3_error_m": float(sim3_errors[i]),
            }
        )

    metrics = {
        "ape_translation_se3_rmse_mm": math.sqrt(float(np.mean(se3_errors**2))) * 1000.0,
        "ape_translation_sim3_rmse_mm": math.sqrt(float(np.mean(sim3_errors**2))) * 1000.0,
    }
    payload = {
        "subtitle": (
            f"{candidate.context.episode_dir.name}: cropped APE window "
            f"{fmt_seconds(candidate.start_s)} -> {fmt_seconds(candidate.end_s)}"
        ),
        "inputs": {
            "source_eval_dir": str(candidate.context.eval_dir),
            "source_episode_dir": str(candidate.context.episode_dir),
            "source_video": str(candidate.context.source_video),
            "gt_tum": str(gt_window_tum),
            "estimate_tum": str(est_window_tum),
            "threshold_mm": 4.0,
            "window_start_s": candidate.start_s,
            "window_end_s": candidate.end_s,
            "window_duration_s": candidate.duration_s,
            "frame": "robot TCP",
            "association_source": "matched-window",
        },
        "algorithms": [
            {
                "name": "VIO TCP",
                "color": "#4ea1ff",
                "eval_dir": str(output_dir),
                "sample_count": int(gt_t_w.shape[0]),
                "duration_s": duration_s,
                "metrics": metrics,
                "points": points,
            }
        ],
    }
    viewer_json = output_dir / "viewer_data_3d.json"
    viewer_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_window_viewer(output_dir, candidate, payload)

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "rmse", "unit"])
        writer.writeheader()
        writer.writerow({"metric": "ape_translation_se3", "rmse": f"{metrics['ape_translation_se3_rmse_mm']:.6f}", "unit": "mm"})
        writer.writerow({"metric": "ape_translation_sim3", "rmse": f"{metrics['ape_translation_sim3_rmse_mm']:.6f}", "unit": "mm"})

    preview_mp4 = output_dir / "window_preview.mp4"
    poster_jpg = output_dir / "window_poster.jpg"
    render_window_preview(candidate, preview_mp4, poster_jpg, payload)

    manifest = {
        "source_eval_dir": str(candidate.context.eval_dir),
        "source_episode_dir": str(candidate.context.episode_dir),
        "source_video": str(candidate.context.source_video),
        "threshold_mm": 4.0,
        "window_start_s": candidate.start_s,
        "window_end_s": candidate.end_s,
        "window_duration_s": candidate.duration_s,
        "video_window_start_s": candidate.context.video_anchor_rel_s + candidate.start_s,
        "video_window_end_s": candidate.context.video_anchor_rel_s + candidate.end_s,
        "heuristic_window_rmse_mm": candidate.heuristic_rmse_mm,
        "exact_window_rmse_mm": candidate.exact_rmse_mm,
        "peak_error_mm_on_full_alignment_curve": candidate.peak_error_mm,
        "sample_count": candidate.sample_count,
        "gt_tum": str(gt_window_tum),
        "estimate_tum": str(est_window_tum),
        "viewer_json": str(viewer_json),
        "viewer_html": str(output_dir / "index.html"),
        "preview_mp4": str(preview_mp4),
        "poster_jpg": str(poster_jpg),
    }
    (output_dir / "window_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showcase-root", type=Path, default=SHOWCASE_ROOT)
    parser.add_argument("--eval-dir", type=Path, default=None, help="optional single eval dir to search")
    parser.add_argument("--threshold-mm", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=8, help="validate the top K heuristic windows exactly")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SHOWCASE_ROOT / "cropped_ape_window_under_4mm",
        help="directory to place the exported cropped package",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.eval_dir:
        eval_dirs = [args.eval_dir.expanduser().resolve()]
    else:
        eval_dirs = discover_eval_dirs(args.showcase_root.expanduser().resolve())
    contexts = [infer_eval_context(path) for path in eval_dirs]
    heuristic: list[WindowCandidate] = []
    for context in contexts:
        candidate = find_heuristic_window(context, args.threshold_mm)
        if candidate is not None:
            heuristic.append(candidate)
    if not heuristic:
        raise SystemExit(f"No heuristic APE window under {args.threshold_mm:.3f} mm found.")

    heuristic.sort(key=lambda item: (-item.duration_s, item.heuristic_rmse_mm))
    validated: list[WindowCandidate] = []
    for candidate in heuristic[: max(1, args.top_k)]:
        validated.append(validate_exact_ape(candidate))
    validated = [item for item in validated if item.exact_rmse_mm is not None and item.exact_rmse_mm <= args.threshold_mm]
    if not validated:
        raise SystemExit(f"Heuristic windows were found, but none stayed under {args.threshold_mm:.3f} mm after exact re-evaluation.")

    best = sorted(validated, key=lambda item: (-item.duration_s, float(item.exact_rmse_mm)))[0]
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    manifest = export_candidate(best, args.output_dir)

    print(f"Selected eval: {best.context.eval_dir}")
    print(f"Episode: {best.context.episode_dir.name}")
    print(f"Window: {best.start_s:.3f}s -> {best.end_s:.3f}s ({best.duration_s:.3f}s)")
    print(f"Exact cropped APE SE3 RMSE: {best.exact_rmse_mm:.3f} mm")
    print(f"Output dir: {args.output_dir.resolve()}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
