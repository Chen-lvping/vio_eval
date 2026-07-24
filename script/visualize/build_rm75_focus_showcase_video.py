#!/usr/bin/env python3
"""Build a clean two-panel RM75 showcase: trajectory and ORB feature points."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from build_rm75_showcase_video import (
    ACCENT_GREEN,
    BG_BOTTOM,
    BG_TOP,
    PANEL_BORDER,
    TEXT_MAIN,
    TEXT_MUTED,
    TrajectoryScene,
    VideoReader,
    draw_panel,
    draw_text,
    fit_image,
    gradient_background,
    load_summary,
    render_trajectory_panel,
    start_ffmpeg,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def render_keypoint_panel(frame: np.ndarray, size: tuple[int, int], t_sec: float, orb: cv2.ORB) -> np.ndarray:
    width, height = size
    background = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    background = cv2.GaussianBlur(background, (0, 0), 18)
    panel = cv2.addWeighted(background, 0.28, np.full_like(background, (8, 10, 14)), 0.72, 0.0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints = sorted(orb.detect(gray, None), key=lambda point: point.response, reverse=True)[:450]
    source_h, source_w = frame.shape[:2]
    scale = min((width - 56) / source_w, (height - 46) / source_h)
    shown_w, shown_h = max(1, int(round(source_w * scale))), max(1, int(round(source_h * scale)))
    x0, y0 = (width - shown_w) // 2, (height - shown_h) // 2
    shown = cv2.resize(frame, (shown_w, shown_h), interpolation=cv2.INTER_AREA)
    panel[y0 : y0 + shown_h, x0 : x0 + shown_w] = shown
    for keypoint in keypoints:
        x = x0 + int(round(keypoint.pt[0] * scale))
        y = y0 + int(round(keypoint.pt[1] * scale))
        radius = max(2, min(5, int(round(keypoint.size * scale * 0.08))))
        cv2.circle(panel, (x, y), radius, ACCENT_GREEN, 1, cv2.LINE_AA)

    overlay = panel.copy()
    cv2.rectangle(overlay, (22, 20), (430, 112), (8, 12, 18), -1, cv2.LINE_AA)
    panel = cv2.addWeighted(overlay, 0.80, panel, 0.20, 0.0)
    draw_text(panel, "ORB FEATURE POINTS", (40, 50), scale=0.78, color=ACCENT_GREEN, thickness=2)
    draw_text(panel, f"Detected keypoints: {len(keypoints):03d}", (40, 78), scale=0.58, color=TEXT_MAIN)
    draw_text(panel, f"cam_right time: {t_sec:05.2f}s", (40, 103), scale=0.54, color=TEXT_MUTED)
    return panel


def ffprobe_start_s(video_path: Path) -> float:
    output = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        text=True,
    ).strip()
    return float(output or 0.0)


def resolve_camera_trajectory_alignment(episode_dir: Path, eval_dir: Path, camera_video: Path) -> tuple[float, float, dict[str, float]]:
    """Map local cam_right seconds to the viewer trajectory's relative seconds."""
    raw_eval_dir = eval_dir.parent
    evaluation = json.loads((raw_eval_dir / "summary.json").read_text(encoding="utf-8"))
    robot_gt = json.loads(Path(evaluation["inputs"]["robot_json"]).read_text(encoding="utf-8"))
    gt_start_s = float(robot_gt.get("start_time_s") or robot_gt["capture_started_host_s"])
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    detail = next(item for item in metadata["video_details"] if item["name"] == camera_video.name)
    camera_first_s = gt_start_s + float(detail["start_offset_us"]) / 1e6 + ffprobe_start_s(camera_video)
    matched_path = raw_eval_dir / "prior_viewer_input/gt_tcp_matched.tum"
    matched_first_s = float(next(line for line in matched_path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")).split()[0])
    camera_at_trajectory_zero_s = matched_first_s - camera_first_s
    if camera_at_trajectory_zero_s >= 0.0:
        camera_start_s, trajectory_start_s = camera_at_trajectory_zero_s, 0.0
    else:
        camera_start_s, trajectory_start_s = 0.0, -camera_at_trajectory_zero_s
    return camera_start_s, trajectory_start_s, {
        "camera_first_host_s": camera_first_s,
        "matched_trajectory_first_host_s": matched_first_s,
        "camera_at_trajectory_zero_s": camera_at_trajectory_zero_s,
    }


def write_index(path: Path, episode: str, video_name: str) -> None:
    path.write_text(
        f"""<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>RM75 Focus Showcase {episode}</title><style>body{{margin:0;background:#10131a;color:#edf1f6;font:16px/1.5 system-ui,sans-serif}}main{{max-width:1440px;margin:32px auto;padding:0 20px}}video{{width:100%;display:block;background:#000;border:1px solid #354154}}a{{color:#8cdbc1}}</style>
<main><h1>RM75 Focus Showcase | {episode}</h1><p>Trajectory comparison and live ORB feature points.</p><video controls autoplay loop muted playsinline src='{video_name}'></video><p><a href='{video_name}'>Download video</a> | <a href='trajectory_viewer/index.html'>Open trajectory viewer</a></p></main></html>""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    args = parser.parse_args()

    episode_dir = args.episode_dir.expanduser().resolve()
    eval_dir = args.eval_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_video = episode_dir / "cam_right.mkv"
    viewer_json = eval_dir / "viewer_data_3d.json"
    summary_csv = eval_dir / "summary.csv"
    missing = [str(path) for path in (camera_video, viewer_json, summary_csv) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))

    scene = TrajectoryScene.from_viewer_json(viewer_json)
    summary = load_summary(summary_csv)
    camera = VideoReader.open(camera_video)
    camera_start_s, trajectory_start_s, timing = resolve_camera_trajectory_alignment(episode_dir, eval_dir, camera_video)
    duration_s = min(scene.duration_s - trajectory_start_s, camera.duration_s - camera_start_s)
    if args.duration_sec > 0:
        duration_s = min(duration_s, args.duration_sec)
    if duration_s <= 0:
        raise RuntimeError("resolved duration is not positive")

    width, height, fps = int(args.width), int(args.height), int(args.fps)
    total_frames = max(1, int(round(duration_s * fps)))
    header_h, margin, gap = 76, 30, 22
    content_h = height - header_h - 2 * margin
    feature_w = int((width - 2 * margin - gap) * 0.59)
    trajectory_w = width - 2 * margin - gap - feature_w
    feature_rect = (margin, header_h + margin, feature_w, content_h)
    trajectory_rect = (margin + feature_w + gap, header_h + margin, trajectory_w, content_h)
    background = gradient_background(width, height)
    orb = cv2.ORB_create(nfeatures=1200, fastThreshold=10)
    ape = float(summary["ape_translation_se3"]["rmse"])
    rotation = float(summary["ape_rotation_se3"]["rmse"])
    output_video = output_dir / "rm75_showcase.mp4"
    temporary_video = output_dir / "rm75_showcase.rendering.mp4"
    if temporary_video.exists():
        temporary_video.unlink()
    if output_video.exists():
        output_video.unlink()
    ffmpeg, log_handle = start_ffmpeg(temporary_video, width, height, fps, output_dir / "ffmpeg_encode.log")
    if ffmpeg.stdin is None:
        raise RuntimeError("failed to open ffmpeg stdin")

    try:
        for frame_idx in range(total_frames):
            elapsed_s = frame_idx / fps
            camera_t_sec = camera_start_s + elapsed_s
            trajectory_t_sec = trajectory_start_s + elapsed_s
            canvas = background.copy()
            draw_text(canvas, f"RM75 Focus Showcase  |  {episode_dir.name}", (32, 39), scale=1.0, thickness=2)
            draw_text(canvas, f"ORB-SLAM3 stereo-inertial  |  APE {ape:.2f} mm  |  Rot {rotation:.2f} deg  |  duration {duration_s:.2f} s", (32, 65), scale=0.58, color=TEXT_MUTED)
            keypoints = render_keypoint_panel(camera.frame_at(camera_t_sec), (feature_rect[2] - 18, feature_rect[3] - 58), camera_t_sec, orb)
            trajectory = render_trajectory_panel(scene, (trajectory_rect[2] - 18, trajectory_rect[3] - 58), trajectory_t_sec, episode_dir.name)
            draw_panel(canvas, feature_rect, "ORB Feature Process", "cam_right feature points from the aligned current frame", keypoints)
            draw_panel(canvas, trajectory_rect, "Trajectory Visualization", "GT vs ORB aligned TCP path", trajectory)
            ffmpeg.stdin.write(canvas.tobytes())
            if frame_idx and frame_idx % max(1, fps * 5) == 0:
                print(f"[progress] {frame_idx}/{total_frames} frames ({100.0 * frame_idx / total_frames:.1f}%)", flush=True)
            if frame_idx == min(10, total_frames - 1):
                cv2.imwrite(str(output_dir / "cover.png"), canvas)
    finally:
        ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        log_handle.close()
        camera.close()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg encode failed with code {return_code}")

    temporary_video.replace(output_video)
    viewer_copy = output_dir / "trajectory_viewer"
    if viewer_copy.exists():
        shutil.rmtree(viewer_copy)
    shutil.copytree(eval_dir, viewer_copy)
    write_index(output_dir / "index.html", episode_dir.name, output_video.name)
    (output_dir / "presentation.json").write_text(json.dumps({
        "layout": "trajectory_and_keypoints",
        "camera": str(camera_video),
        "camera_start_s": camera_start_s,
        "trajectory_start_s": trajectory_start_s,
        "timing": timing,
    }, indent=2), encoding="utf-8")
    print(f"[DONE] showcase={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
