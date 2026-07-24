#!/usr/bin/env python3
"""Build a packaged RM75 showcase video from existing evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPISODE_DIR = REPO_ROOT / "data/gripper_data_6_24/episode_20260624_0007"
DEFAULT_EVAL_DIR = (
    REPO_ROOT
    / "data/evaluation/workbench/evo_orb_tcp_rm75_0007_fresh_stereo_inertial_low-texture_fastinit0_vins_match_smooth_w7_p2_strictsync_20260624"
)
DEFAULT_OUTPUT_DIR = Path.home() / "Desktop" / "rm75_showcase_20260624_0007"


BG_TOP = np.array([15, 18, 28], dtype=np.float32)
BG_BOTTOM = np.array([7, 10, 16], dtype=np.float32)
PANEL_BG = (20, 24, 34)
PANEL_BORDER = (70, 78, 100)
TEXT_MAIN = (232, 238, 247)
TEXT_MUTED = (155, 165, 188)
ACCENT_CYAN = (214, 168, 42)
ACCENT_ORANGE = (55, 138, 240)
ACCENT_GREEN = (112, 220, 130)
ACCENT_RED = (94, 98, 255)


@dataclass
class VideoReader:
    path: Path
    cap: cv2.VideoCapture
    fps: float
    frame_count: int
    duration_s: float
    current_index: int = -1
    current_frame: np.ndarray | None = None

    @classmethod
    def open(cls, path: Path) -> "VideoReader":
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        duration_s = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        return cls(path=path, cap=cap, fps=fps, frame_count=frame_count, duration_s=duration_s)

    def frame_at(self, t_sec: float) -> np.ndarray:
        if self.frame_count <= 0:
            raise RuntimeError(f"video has no frames: {self.path}")
        frame_index = int(round(max(0.0, min(self.duration_s, t_sec)) * self.fps))
        frame_index = max(0, min(self.frame_count - 1, frame_index))
        if self.current_frame is not None and frame_index == self.current_index:
            return self.current_frame

        if self.current_index < 0 or frame_index < self.current_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to decode frame {frame_index} from {self.path}")
            self.current_index = frame_index
            self.current_frame = frame
            return frame

        while self.current_index < frame_index:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to decode frame {frame_index} from {self.path}")
            self.current_index += 1
            self.current_frame = frame
        if self.current_frame is None:
            raise RuntimeError(f"failed to cache frame {frame_index} from {self.path}")
        return self.current_frame

    def close(self) -> None:
        self.cap.release()


@dataclass
class TrajectoryScene:
    times: np.ndarray
    gt_2d: np.ndarray
    est_2d: np.ndarray
    gt_3d: np.ndarray
    est_3d: np.ndarray
    est_err_mm: np.ndarray
    metrics: dict[str, float]
    duration_s: float

    @classmethod
    def from_viewer_json(cls, path: Path) -> "TrajectoryScene":
        payload = json.loads(path.read_text(encoding="utf-8"))
        algorithm = payload["algorithms"][0]
        points = algorithm["points"]
        if not points:
            raise RuntimeError(f"no trajectory points in {path}")
        times = np.asarray([float(item["t"]) for item in points], dtype=np.float32)
        gt_3d = np.asarray([item["gt"] for item in points], dtype=np.float32)
        est_3d = np.asarray([item["se3"] for item in points], dtype=np.float32)
        est_err_mm = np.asarray([float(item["se3_error_m"]) * 1000.0 for item in points], dtype=np.float32)

        combined = np.vstack([gt_3d, est_3d])
        center = combined.mean(axis=0, keepdims=True)
        normalized = combined - center

        yaw = math.radians(34.0)
        pitch = math.radians(-26.0)
        rot_z = np.asarray(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        rot_x = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(pitch), -math.sin(pitch)],
                [0.0, math.sin(pitch), math.cos(pitch)],
            ],
            dtype=np.float32,
        )
        rotated = normalized @ rot_z.T @ rot_x.T
        gt_rot = rotated[: len(gt_3d)]
        est_rot = rotated[len(gt_3d) :]
        proj_all = np.vstack([gt_rot[:, [0, 1]], est_rot[:, [0, 1]]])

        lo = proj_all.min(axis=0)
        hi = proj_all.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)

        def to_2d(values: np.ndarray) -> np.ndarray:
            uv = (values[:, [0, 1]] - lo) / span
            uv[:, 1] = 1.0 - uv[:, 1]
            return uv.astype(np.float32)

        return cls(
            times=times,
            gt_2d=to_2d(gt_rot),
            est_2d=to_2d(est_rot),
            gt_3d=gt_3d,
            est_3d=est_3d,
            est_err_mm=est_err_mm,
            metrics={key: float(value) for key, value in algorithm["metrics"].items()},
            duration_s=float(algorithm["duration_s"]),
        )

    def index_at(self, t_sec: float) -> int:
        idx = int(np.searchsorted(self.times, t_sec, side="right") - 1)
        return max(0, min(len(self.times) - 1, idx))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--duration-sec", type=float, default=0.0, help="Override video duration; default uses shared shortest duration.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_summary(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["metric"]] = row
    return result


def load_media_offsets(episode_dir: Path) -> dict[str, float]:
    metadata_path = episode_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    details = {item["name"]: item for item in metadata.get("video_details", [])}
    required = ("cam_right.mkv", "stereo_right.mkv", "tcam_right_l.mkv", "tcam_right_r.mkv")
    missing = [name for name in required if name not in details]
    if missing:
        raise KeyError(f"{metadata_path}: missing video offset(s): {', '.join(missing)}")
    return {name: float(details[name].get("start_offset_us", 0.0)) / 1e6 for name in required}


def common_media_timeline(
    scene_duration_s: float,
    cam_reader: VideoReader,
    stereo_reader: VideoReader,
    touch_l_reader: VideoReader,
    touch_r_reader: VideoReader,
    offsets: dict[str, float],
) -> tuple[float, float]:
    """Return the cam_right-relative interval available in every display stream."""
    cam_offset = offsets["cam_right.mkv"]
    stereo_start = offsets["stereo_right.mkv"] - cam_offset
    touch_l_start = offsets["tcam_right_l.mkv"] - cam_offset
    touch_r_start = offsets["tcam_right_r.mkv"] - cam_offset
    start_s = max(0.0, stereo_start, touch_l_start, touch_r_start)
    end_s = min(
        scene_duration_s,
        cam_reader.duration_s,
        stereo_start + stereo_reader.duration_s,
        touch_l_start + touch_l_reader.duration_s,
        touch_r_start + touch_r_reader.duration_s,
    )
    if end_s <= start_s:
        raise RuntimeError(f"no common media interval after timestamp alignment: start={start_s:.3f}, end={end_s:.3f}")
    return start_s, end_s


def gradient_background(width: int, height: int) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    base = BG_TOP * (1.0 - y) + BG_BOTTOM * y
    glow = np.zeros((height, width, 3), dtype=np.float32)
    glow[..., 0] = 18.0 * np.exp(-((x[..., 0] - 0.22) ** 2) / 0.03)
    glow[..., 1] = 10.0 * np.exp(-((x[..., 0] - 0.72) ** 2) / 0.08)
    glow[..., 2] = 26.0 * np.exp(-((x[..., 0] - 0.58) ** 2) / 0.06)
    out = np.clip(base + glow, 0, 255)
    return out.astype(np.uint8)


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.7,
    color: tuple[int, int, int] = TEXT_MAIN,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def fit_image(frame: np.ndarray, width: int, height: int, bg_color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def draw_panel(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    content: np.ndarray,
) -> None:
    x, y, w, h = rect
    radius_bg = PANEL_BG
    panel = canvas[y : y + h, x : x + w]
    panel[:] = radius_bg
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), PANEL_BORDER, 1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (w - 1, 44), (26, 31, 43), -1, cv2.LINE_AA)
    draw_text(panel, title, (18, 28), scale=0.74, thickness=2)
    draw_text(panel, subtitle, (18, 42), scale=0.47, color=TEXT_MUTED)
    content_h = h - 54
    content = fit_image(content, w - 18, content_h - 10, (10, 12, 16))
    panel[50 : 50 + content.shape[0], 9 : 9 + content.shape[1]] = content


def render_trajectory_panel(scene: TrajectoryScene, size: tuple[int, int], t_sec: float, episode_name: str) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), (12, 16, 24), dtype=np.uint8)
    idx = scene.index_at(t_sec)

    draw_area = (26, 28, width - 52, height - 90)
    ax_x, ax_y, ax_w, ax_h = draw_area
    cv2.rectangle(panel, (ax_x, ax_y), (ax_x + ax_w, ax_y + ax_h), (26, 31, 43), -1, cv2.LINE_AA)
    cv2.rectangle(panel, (ax_x, ax_y), (ax_x + ax_w, ax_y + ax_h), (48, 56, 74), 1, cv2.LINE_AA)

    def to_px(points_uv: np.ndarray) -> np.ndarray:
        xs = ax_x + 22 + points_uv[:, 0] * (ax_w - 44)
        ys = ax_y + 18 + points_uv[:, 1] * (ax_h - 36)
        return np.round(np.stack([xs, ys], axis=1)).astype(np.int32)

    gt_px = to_px(scene.gt_2d)
    est_px = to_px(scene.est_2d)

    for poly, color in (
        (gt_px, (86, 198, 255)),
        (est_px, (82, 142, 255)),
    ):
        cv2.polylines(panel, [poly], False, tuple(int(c * 0.35) for c in color), 1, cv2.LINE_AA)

    cv2.polylines(panel, [gt_px[: idx + 1]], False, (92, 214, 255), 2, cv2.LINE_AA)
    cv2.polylines(panel, [est_px[: idx + 1]], False, (70, 150, 255), 3, cv2.LINE_AA)
    cv2.circle(panel, tuple(gt_px[idx]), 6, (92, 214, 255), -1, cv2.LINE_AA)
    cv2.circle(panel, tuple(est_px[idx]), 7, (70, 150, 255), -1, cv2.LINE_AA)

    metric_x = ax_x + 18
    metric_y = height - 42
    draw_text(panel, f"{episode_name}  |  aligned TCP trajectory", (metric_x, metric_y), scale=0.58, color=TEXT_MUTED)

    badges = [(f"APE {scene.metrics['ape_translation_se3_rmse_mm']:.2f} mm", ACCENT_ORANGE)]
    rpe_mm = scene.metrics.get("rpe_translation_5cm_rmse_mm", float("nan"))
    if math.isfinite(rpe_mm):
        badges.append((f"RPE {rpe_mm:.2f} mm", ACCENT_GREEN))
    badges.append((f"Rot {scene.metrics['ape_rotation_se3_rmse_deg']:.2f} deg", ACCENT_CYAN))
    bx = ax_x + 18
    by = ax_y + 34
    for text, color in badges:
        size_text, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(panel, (bx - 10, by - 22), (bx + size_text[0] + 12, by + 8), (22, 28, 40), -1, cv2.LINE_AA)
        cv2.rectangle(panel, (bx - 10, by - 22), (bx + size_text[0] + 12, by + 8), color, 1, cv2.LINE_AA)
        draw_text(panel, text, (bx, by), scale=0.52, color=TEXT_MAIN)
        bx += size_text[0] + 34

    current_err = float(scene.est_err_mm[idx])
    draw_text(panel, f"Current aligned error: {current_err:.2f} mm", (ax_x + 18, ax_y + ax_h - 16), scale=0.58, color=TEXT_MAIN)
    return panel


def render_monocular_panel(frame: np.ndarray, size: tuple[int, int], t_sec: float) -> np.ndarray:
    width, height = size
    panel = fit_image(frame, width, height, (8, 8, 10))
    overlay = panel.copy()
    cv2.rectangle(overlay, (18, height - 58), (320, height - 18), (14, 18, 26), -1, cv2.LINE_AA)
    panel = cv2.addWeighted(overlay, 0.78, panel, 0.22, 0.0)
    draw_text(panel, f"Monocular camera  |  t = {t_sec:05.2f}s", (32, height - 32), scale=0.66, thickness=2)
    return panel


def render_stereo_orb_panel(stereo_frame: np.ndarray, size: tuple[int, int], t_sec: float, orb: cv2.ORB) -> np.ndarray:
    height, width = stereo_frame.shape[:2]
    split = height // 2
    cam0 = stereo_frame[:split]
    cam1 = stereo_frame[split:]
    gray0 = cv2.cvtColor(cam0, cv2.COLOR_BGR2GRAY) if cam0.ndim == 3 else cam0
    gray1 = cv2.cvtColor(cam1, cv2.COLOR_BGR2GRAY) if cam1.ndim == 3 else cam1
    keypoints0, descriptors0 = orb.detectAndCompute(gray0, None)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    matches = []
    if descriptors0 is not None and descriptors1 is not None:
        knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors0, descriptors1, k=2)
        matches = [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        matches.sort(key=lambda match: match.distance)
    match_view = cv2.drawMatches(
        cam0, keypoints0, cam1, keypoints1, matches[:120], None,
        matchColor=(80, 220, 130), singlePointColor=(125, 170, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    panel = fit_image(match_view, size[0], size[1], (8, 8, 10))
    overlay = panel.copy()
    cv2.rectangle(overlay, (18, 18), (420, 108), (8, 12, 18), -1, cv2.LINE_AA)
    panel = cv2.addWeighted(overlay, 0.78, panel, 0.22, 0.0)
    draw_text(panel, "STEREO ORB", (34, 46), scale=0.74, color=ACCENT_GREEN, thickness=2)
    draw_text(panel, f"Cam0: {len(keypoints0):03d}  Cam1: {len(keypoints1):03d}  Matches: {len(matches):03d}", (34, 72), scale=0.52, color=TEXT_MAIN)
    draw_text(panel, f"Stereo-right time: {t_sec:05.2f}s", (34, 96), scale=0.54, color=TEXT_MUTED)
    return panel


def render_tactile_panel(left: np.ndarray, right: np.ndarray, size: tuple[int, int], t_sec: float) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), (8, 8, 10), dtype=np.uint8)
    cell_w = (width - 36) // 2
    cell_h = height - 64
    left_fit = fit_image(left, cell_w, cell_h, (6, 6, 6))
    right_fit = fit_image(right, cell_w, cell_h, (6, 6, 6))
    y0 = 22
    panel[y0 : y0 + left_fit.shape[0], 12 : 12 + left_fit.shape[1]] = left_fit
    panel[y0 : y0 + right_fit.shape[0], width - 12 - right_fit.shape[1] : width - 12] = right_fit
    draw_text(panel, "Touch L", (18, height - 22), scale=0.56, color=TEXT_MAIN)
    draw_text(panel, "Touch R", (width - 116, height - 22), scale=0.56, color=TEXT_MAIN)
    draw_text(panel, f"Synchronized tactile view  |  cam t = {t_sec:05.2f}s", (18, 18), scale=0.54, color=TEXT_MUTED)
    return panel


def start_ffmpeg(output_path: Path, width: int, height: int, fps: int, log_path: Path) -> tuple[subprocess.Popen[bytes], object]:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_handle)
    return proc, log_handle


def write_bundle_index(path: Path, title: str, video_name: str, viewer_rel: str) -> None:
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #091018;
      --card: rgba(22, 28, 40, 0.92);
      --border: rgba(134, 156, 192, 0.32);
      --text: #e9eef7;
      --muted: #9aa8c1;
      --accent: #4aa5ff;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 18% 12%, rgba(70, 126, 255, 0.18), transparent 32%),
        radial-gradient(circle at 78% 24%, rgba(77, 225, 167, 0.14), transparent 26%),
        linear-gradient(180deg, #0e1623 0%, var(--bg) 100%);
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 40px 24px 60px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.1;
    }}
    p {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.6;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 20px;
      backdrop-filter: blur(8px);
      box-shadow: 0 30px 80px rgba(0, 0, 0, 0.35);
    }}
    video {{
      width: 100%;
      border-radius: 14px;
      display: block;
      background: #000;
    }}
    .links {{
      margin-top: 16px;
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
    }}
    a {{
      color: white;
      background: linear-gradient(135deg, var(--accent), #4de0bc);
      text-decoration: none;
      padding: 10px 16px;
      border-radius: 999px;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{title}</h1>
    <p>Showcase bundle with the stitched four-panel video and the original TCP trajectory viewer.</p>
    <div class="card">
      <video controls autoplay loop muted playsinline src="{video_name}"></video>
      <div class="links">
        <a href="{video_name}">Download video</a>
        <a href="{viewer_rel}">Open trajectory viewer</a>
      </div>
    </div>
  </div>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def write_bundle_readme(path: Path, episode_dir: Path, eval_dir: Path, output_video: Path) -> None:
    lines = [
        "# RM75 Showcase Bundle",
        "",
        f"- Episode: `{episode_dir}`",
        f"- Evaluation viewer: `{eval_dir}`",
        f"- Showcase video: `{output_video.name}`",
        "- `media_sync.json` records the per-stream offsets applied to tactile playback.",
        "",
        "Open `index.html` for the packaged presentation page, or play the MP4 directly.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.episode_dir = args.episode_dir.expanduser().resolve()
    args.eval_dir = args.eval_dir.expanduser().resolve()
    args.output_dir = ensure_dir(args.output_dir.expanduser().resolve())

    camera_video = args.episode_dir / "cam_right.mkv"
    stereo_video = args.episode_dir / "stereo_right.mkv"
    tactile_left_video = args.episode_dir / "tcam_right_l.mkv"
    tactile_right_video = args.episode_dir / "tcam_right_r.mkv"
    viewer_json = args.eval_dir / "viewer_data_3d.json"
    summary_csv = args.eval_dir / "summary.csv"

    required = [camera_video, stereo_video, tactile_left_video, tactile_right_video, viewer_json, summary_csv]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))

    scene = TrajectoryScene.from_viewer_json(viewer_json)
    summary = load_summary(summary_csv)

    cam_reader = VideoReader.open(camera_video)
    stereo_reader = VideoReader.open(stereo_video)
    touch_l_reader = VideoReader.open(tactile_left_video)
    touch_r_reader = VideoReader.open(tactile_right_video)
    offsets = load_media_offsets(args.episode_dir)
    timeline_start_s, timeline_end_s = common_media_timeline(
        scene.duration_s, cam_reader, stereo_reader, touch_l_reader, touch_r_reader, offsets
    )
    duration_s = timeline_end_s - timeline_start_s
    if args.duration_sec > 0.0:
        duration_s = min(duration_s, float(args.duration_sec))
    if duration_s <= 0.0:
        raise RuntimeError("resolved duration is not positive")

    width = int(args.width)
    height = int(args.height)
    fps = int(args.fps)
    total_frames = max(1, int(round(duration_s * fps)))

    bg = gradient_background(width, height)
    orb = cv2.ORB_create(nfeatures=900, fastThreshold=10)

    output_video = args.output_dir / "rm75_showcase.mp4"
    temp_output_video = args.output_dir / "rm75_showcase.rendering.mp4"
    ffmpeg_log = args.output_dir / "ffmpeg_encode.log"
    if temp_output_video.exists():
        temp_output_video.unlink()
    if output_video.exists():
        output_video.unlink()
    media_sync = {
        "time_base": "cam_right.mkv local time",
        "offsets_from_metadata_s": offsets,
        "display_interval_cam_right_s": {
            "start": timeline_start_s,
            "end": timeline_start_s + duration_s,
            "duration": duration_s,
        },
        "tactile_frame_time": "t_cam_right + offset(cam_right) - offset(tactile)",
        "stereo_frame_time": "t_cam_right + offset(cam_right) - offset(stereo_right)",
    }
    (args.output_dir / "media_sync.json").write_text(json.dumps(media_sync, indent=2), encoding="utf-8")
    ffmpeg, ffmpeg_log_handle = start_ffmpeg(temp_output_video, width, height, fps, ffmpeg_log)
    if ffmpeg.stdin is None:
        raise RuntimeError("failed to open ffmpeg stdin")

    header_h = 74
    margin = 28
    gap = 18
    panel_w = (width - 2 * margin - gap) // 2
    panel_h = (height - header_h - 2 * margin - gap) // 2
    rects = {
        "traj": (margin, header_h, panel_w, panel_h),
        "cam": (margin + panel_w + gap, header_h, panel_w, panel_h),
        "orb": (margin, header_h + panel_h + gap, panel_w, panel_h),
        "touch": (margin + panel_w + gap, header_h + panel_h + gap, panel_w, panel_h),
    }

    cover_written = False
    episode_name = args.episode_dir.name
    ape_text = float(summary["ape_translation_se3"]["rmse"])
    rpe_text = float(summary["rpe_translation_5cm"]["rmse"])
    metric_text = f"APE {ape_text:.2f} mm"
    if math.isfinite(rpe_text):
        metric_text += f"  |  RPE {rpe_text:.2f} mm"

    try:
        for frame_idx in range(total_frames):
            t_sec = timeline_start_s + frame_idx / fps
            frame = bg.copy()

            draw_text(frame, f"RM75 Showcase  |  {episode_name}", (30, 38), scale=1.0, thickness=2)
            draw_text(
                frame,
                f"ORB-SLAM3 stereo-inertial  |  {metric_text}  |  duration {duration_s:.2f} s",
                (30, 64),
                scale=0.58,
                color=TEXT_MUTED,
            )

            cam_frame = cam_reader.frame_at(t_sec)
            stereo_t_sec = t_sec + offsets["cam_right.mkv"] - offsets["stereo_right.mkv"]
            stereo_frame = stereo_reader.frame_at(stereo_t_sec)
            touch_l_frame = touch_l_reader.frame_at(
                t_sec + offsets["cam_right.mkv"] - offsets["tcam_right_l.mkv"]
            )
            touch_r_frame = touch_r_reader.frame_at(
                t_sec + offsets["cam_right.mkv"] - offsets["tcam_right_r.mkv"]
            )

            traj_panel = render_trajectory_panel(scene, (panel_w - 18, panel_h - 60), t_sec, episode_name)
            mono_panel = render_monocular_panel(cam_frame, (panel_w - 18, panel_h - 60), t_sec)
            orb_panel = render_stereo_orb_panel(stereo_frame, (panel_w - 18, panel_h - 60), stereo_t_sec, orb)
            tactile_panel = render_tactile_panel(touch_l_frame, touch_r_frame, (panel_w - 18, panel_h - 60), t_sec)

            draw_panel(frame, rects["traj"], "Trajectory Visualization", "GT vs ORB aligned TCP path", traj_panel)
            draw_panel(frame, rects["cam"], "Monocular Video", "Primary camera playback", mono_panel)
            draw_panel(frame, rects["orb"], "Stereo ORB Feature Matching", "Stereo-right Cam0/Cam1 feature correspondence", orb_panel)
            draw_panel(frame, rects["touch"], "Tactile Video", "Right gripper tactile pair", tactile_panel)

            ffmpeg.stdin.write(frame.tobytes())
            if frame_idx and frame_idx % max(1, fps * 5) == 0:
                progress = 100.0 * frame_idx / total_frames
                print(f"[progress] {frame_idx}/{total_frames} frames ({progress:.1f}%)", flush=True)

            if not cover_written and frame_idx == min(10, total_frames - 1):
                cv2.imwrite(str(args.output_dir / "cover.png"), frame)
                cover_written = True
    finally:
        ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        ffmpeg_log_handle.close()
        cam_reader.close()
        stereo_reader.close()
        touch_l_reader.close()
        touch_r_reader.close()
        if return_code != 0:
            ffmpeg_log_text = ffmpeg_log.read_text(encoding="utf-8", errors="ignore") if ffmpeg_log.exists() else ""
            raise RuntimeError(f"ffmpeg encode failed with code {return_code}\n{ffmpeg_log_text}")

    temp_output_video.replace(output_video)

    viewer_copy = args.output_dir / "trajectory_viewer"
    if viewer_copy.exists():
        shutil.rmtree(viewer_copy)
    shutil.copytree(args.eval_dir, viewer_copy)

    write_bundle_index(args.output_dir / "index.html", f"RM75 Showcase {episode_name}", output_video.name, "trajectory_viewer/index.html")
    write_bundle_readme(args.output_dir / "README.md", args.episode_dir, args.eval_dir, output_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
