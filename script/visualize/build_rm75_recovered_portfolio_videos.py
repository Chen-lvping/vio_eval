#!/usr/bin/env python3
"""Render 18 recovered RM75 GT-vs-ORB trajectories as portfolio videos."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIEWERS = ROOT / "data/evaluation/workbench/rm75_recovered_baseline_viewers_20260731"
DEFAULT_TABLE = ROOT / "data/evaluation/workbench/rm75_22_final_robust_20260722/full_stereo_vs_stereo_imu_comparison_20260724.md"
DEFAULT_OUTPUT = ROOT / "data/evaluation/workbench/rm75_ape_collection_portfolio_20260731"

BG = (20, 22, 24)
PANEL = (28, 31, 33)
GRID = (50, 55, 57)
TEXT = (238, 240, 239)
MUTED = (165, 171, 170)
GT = (112, 211, 91)
ORB = (236, 165, 69)
ERROR = (86, 104, 239)
ACCENT = (68, 186, 231)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer-root", type=Path, default=DEFAULT_VIEWERS)
    parser.add_argument("--comparison-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def historical_stereo_values(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) >= 2 and fields[0].startswith("main/"):
            try:
                values[fields[0]] = float(fields[1])
            except ValueError:
                pass
    return values


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.6,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def trajectory_projection(gt: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([gt, estimate])
    centered = combined - combined.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes[:2].T
    gt_2d = projected[: len(gt)]
    est_2d = projected[len(gt) :]
    direction = gt_2d[-1] - gt_2d[0]
    if direction[0] < 0:
        gt_2d[:, 0] *= -1
        est_2d[:, 0] *= -1
    return gt_2d, est_2d


def fit_points(points: np.ndarray, rect: tuple[int, int, int, int], bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    x, y, width, height = rect
    lo, hi = bounds
    span = np.maximum(hi - lo, 1e-9)
    scale = min((width - 56) / span[0], (height - 56) / span[1])
    center = (lo + hi) * 0.5
    result = np.empty_like(points)
    result[:, 0] = x + width * 0.5 + (points[:, 0] - center[0]) * scale
    result[:, 1] = y + height * 0.5 - (points[:, 1] - center[1]) * scale
    return np.round(result).astype(np.int32)


def draw_grid(image: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x, y, width, height = rect
    cv2.rectangle(image, (x, y), (x + width, y + height), PANEL, -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), GRID, 1, cv2.LINE_AA)
    for index in range(1, 5):
        xx = x + width * index // 5
        yy = y + height * index // 5
        cv2.line(image, (xx, y), (xx, y + height), GRID, 1, cv2.LINE_AA)
        cv2.line(image, (x, yy), (x + width, yy), GRID, 1, cv2.LINE_AA)


def draw_progress_curve(
    image: np.ndarray,
    errors_mm: np.ndarray,
    index: int,
    rect: tuple[int, int, int, int],
) -> None:
    x, y, width, height = rect
    draw_grid(image, rect)
    max_error = max(1.0, float(np.nanpercentile(errors_mm, 99)) * 1.12)
    px = x + 22 + np.arange(len(errors_mm)) * (width - 44) / max(1, len(errors_mm) - 1)
    py = y + height - 24 - np.clip(errors_mm, 0, max_error) * (height - 48) / max_error
    curve = np.round(np.column_stack([px, py])).astype(np.int32)
    cv2.polylines(image, [curve], False, tuple(int(channel * 0.45) for channel in ERROR), 1, cv2.LINE_AA)
    cv2.polylines(image, [curve[: index + 1]], False, ERROR, 2, cv2.LINE_AA)
    cv2.circle(image, tuple(curve[index]), 5, ERROR, -1, cv2.LINE_AA)
    draw_text(image, f"{max_error:.1f} mm", (x + 8, y + 20), scale=0.42, color=MUTED)
    draw_text(image, "0", (x + 8, y + height - 8), scale=0.42, color=MUTED)


def open_encoder(path: Path, width: int, height: int, fps: int) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("failed to open ffmpeg input")
    return process


def render_episode(
    row: dict[str, str],
    historical_mm: float,
    viewer_root: Path,
    output: Path,
    width: int,
    height: int,
    fps: int,
    duration_sec: float,
    ordinal: int,
    total: int,
) -> dict[str, object]:
    episode_slug = row["episode"].replace("/", "_")
    viewer_json = viewer_root / episode_slug / "viewer/viewer_data_3d.json"
    payload = json.loads(viewer_json.read_text(encoding="utf-8"))
    algorithm = payload["algorithms"][0]
    points = algorithm["points"]
    gt = np.asarray([point["gt"] for point in points], dtype=np.float64)
    estimate = np.asarray([point["se3"] for point in points], dtype=np.float64)
    errors_mm = np.asarray([float(point["se3_error_m"]) * 1000.0 for point in points], dtype=np.float64)
    gt_2d, estimate_2d = trajectory_projection(gt, estimate)
    combined = np.vstack([gt_2d, estimate_2d])
    lo = combined.min(axis=0)
    hi = combined.max(axis=0)
    padding = np.maximum((hi - lo) * 0.08, 1e-6)
    bounds = lo - padding, hi + padding

    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = open_encoder(output, width, height, fps)
    total_frames = max(1, int(round(duration_sec * fps)))
    plot_rect = (36, 118, int(width * 0.66), height - 170)
    error_rect = (plot_rect[0] + plot_rect[2] + 24, 300, width - plot_rect[0] - plot_rect[2] - 60, 180)
    gt_pixels = fit_points(gt_2d, plot_rect, bounds)
    estimate_pixels = fit_points(estimate_2d, plot_rect, bounds)
    cover: np.ndarray | None = None
    try:
        for frame_number in range(total_frames):
            progress = frame_number / max(1, total_frames - 1)
            eased = 1.0 - (1.0 - progress) ** 2.2
            index = min(len(points) - 1, int(round(eased * (len(points) - 1))))
            image = np.full((height, width, 3), BG, dtype=np.uint8)
            draw_text(image, f"RM75 Stereo SLAM Evaluation  |  Episode {ordinal:02d}/{total:02d}", (36, 44), scale=0.92, thickness=2)
            draw_text(image, "Robot TCP ground truth vs ORB-SLAM3 trajectory", (36, 76), scale=0.55, color=MUTED)
            draw_text(image, "TCP frame  |  SE(3) alignment  |  translation APE", (36, 101), scale=0.50, color=MUTED)

            draw_grid(image, plot_rect)
            cv2.polylines(image, [gt_pixels], False, tuple(int(channel * 0.34) for channel in GT), 1, cv2.LINE_AA)
            cv2.polylines(image, [estimate_pixels], False, tuple(int(channel * 0.34) for channel in ORB), 1, cv2.LINE_AA)
            cv2.polylines(image, [gt_pixels[: index + 1]], False, GT, 3, cv2.LINE_AA)
            cv2.polylines(image, [estimate_pixels[: index + 1]], False, ORB, 3, cv2.LINE_AA)
            cv2.line(image, tuple(gt_pixels[index]), tuple(estimate_pixels[index]), ERROR, 1, cv2.LINE_AA)
            cv2.circle(image, tuple(gt_pixels[index]), 6, GT, -1, cv2.LINE_AA)
            cv2.circle(image, tuple(estimate_pixels[index]), 6, ORB, -1, cv2.LINE_AA)
            draw_text(image, "Robot TCP GT", (plot_rect[0] + 18, plot_rect[1] + 28), scale=0.52, color=GT, thickness=2)
            draw_text(image, "ORB result", (plot_rect[0] + 168, plot_rect[1] + 28), scale=0.52, color=ORB, thickness=2)

            info_x = error_rect[0]
            draw_text(image, "APE RESULT", (info_x, 142), scale=0.60, color=ACCENT, thickness=2)
            recovered_mm = float(row["ape_translation_se3_rmse_mm"])
            draw_text(image, f"APE SE(3)          {recovered_mm:7.3f} mm", (info_x, 181), scale=0.62, thickness=2)
            draw_text(image, f"Matched samples    {row['pairs']}", (info_x, 218), scale=0.55, color=MUTED)
            draw_text(image, "TRANSLATION ERROR", (info_x, error_rect[1] - 14), scale=0.55, color=ACCENT, thickness=2)
            draw_progress_curve(image, errors_mm, index, error_rect)
            draw_text(image, f"Current {errors_mm[index]:.2f} mm", (info_x, error_rect[1] + error_rect[3] + 28), scale=0.56, color=ERROR, thickness=2)
            draw_text(image, f"Episode {ordinal:02d} of {total:02d}", (info_x, error_rect[1] + error_rect[3] + 64), scale=0.54, color=ACCENT, thickness=2)
            draw_text(image, f"Progress {100.0 * progress:5.1f}%", (width - 180, height - 22), scale=0.48, color=MUTED)
            encoder.stdin.write(image.tobytes())
            if frame_number == min(fps, total_frames - 1):
                cover = image.copy()
    finally:
        encoder.stdin.close()
        return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {row['episode']} with code {return_code}")
    cover_path = output.with_suffix(".png")
    cv2.imwrite(str(cover_path), cover if cover is not None else image)
    return {
        "sequence": f"Episode {ordinal:02d}", "source_episode": row["episode"], "source_tag": row["tag"],
        "historical_stereo_mm": historical_mm, "ape_translation_se3_rmse_mm": float(row["ape_translation_se3_rmse_mm"]),
        "pairs": int(row["pairs"]), "fresh_input_status": row["fresh_input_status"],
        "video": output.name, "cover": cover_path.name,
    }


def concatenate_videos(output_root: Path, entries: list[dict[str, object]]) -> Path:
    concat = output_root / "concat.txt"
    concat.write_text("".join(f"file '{entry['video']}'\n" for entry in entries), encoding="utf-8")
    combined = output_root / "rm75_18_episode_ape_portfolio.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(combined)],
        cwd=output_root,
        check=True,
    )
    return combined


def write_index(output_root: Path, entries: list[dict[str, object]], combined: Path) -> None:
    rows = []
    for entry in entries:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(entry['sequence']))}</td>"
            f"<td>{float(entry['ape_translation_se3_rmse_mm']):.3f} mm</td>"
            f"<td><a href='{html.escape(str(entry['video']))}'>MP4</a></td>"
            "</tr>"
        )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RM75 18-Episode APE Portfolio</title><style>
body{{margin:0;background:#f4f6f5;color:#182226;font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}header{{padding:22px max(18px,calc((100vw - 1120px)/2));background:#fff;border-bottom:1px solid #d9e1df}}h1{{margin:0 0 5px;font-size:25px;letter-spacing:0}}p{{margin:0;color:#687674}}main{{max-width:1120px;margin:0 auto;padding:22px 18px 36px}}video{{display:block;width:100%;aspect-ratio:16/9;background:#141618}}table{{width:100%;margin-top:20px;border-collapse:collapse;background:#fff;border:1px solid #d9e1df}}th,td{{padding:10px 12px;border-bottom:1px solid #d9e1df;text-align:left}}th{{background:#edf2f0;color:#687674;font-size:12px}}a{{color:#086a91;text-decoration:none}}a:hover{{text-decoration:underline}}
</style></head><body><header><h1>RM75 18-Episode APE Portfolio</h1><p>Robot TCP ground truth and ORB-SLAM3 trajectory evaluation.</p></header><main><video controls preload="metadata" poster="{entries[0]['cover']}" src="{combined.name}"></video><table><thead><tr><th>Episode</th><th>APE SE(3)</th><th>Video</th></tr></thead><tbody>{''.join(rows)}</tbody></table></main></body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    args = parse_args()
    viewer_root = args.viewer_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_rows(viewer_root / "summary.csv")
    historical = historical_stereo_values(args.comparison_table.expanduser().resolve())
    if len(rows) != 18:
        raise RuntimeError(f"expected 18 recovered baselines, found {len(rows)}")
    entries = []
    for ordinal, row in enumerate(rows, start=1):
        episode = row["episode"]
        if episode not in historical:
            raise KeyError(f"missing historical table value for {episode}")
        print(f"[RENDER {ordinal}/18] {episode}", flush=True)
        entries.append(
            render_episode(
                row, historical[episode], viewer_root, output_root / f"episode_{ordinal:02d}.mp4",
                args.width, args.height, args.fps, args.duration_sec, ordinal, len(rows),
            )
        )
    combined = concatenate_videos(output_root, entries)
    public_fields = ["sequence", "ape_translation_se3_rmse_mm", "pairs", "video", "cover"]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    internal_fields = list(entries[0])
    with (output_root / "source_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=internal_fields)
        writer.writeheader()
        writer.writerows(entries)
    public_entries = [{key: entry[key] for key in public_fields} for entry in entries]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "RM75 18-episode GT-vs-ORB APE video portfolio",
        "render": {"width": args.width, "height": args.height, "fps": args.fps, "duration_sec": args.duration_sec},
        "combined_video": combined.name,
        "coordinate_frame": "robot TCP",
        "alignment": "SE(3)",
        "entries": public_entries,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_root / "README.txt").write_text(
        "RM75 18-Episode APE Portfolio\n\nExtract the ZIP and open index.html.\n"
        "The complete page and all 18 videos work offline.\n",
        encoding="utf-8",
    )
    write_index(output_root, entries, combined)
    print(f"[DONE] portfolio={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
