#!/usr/bin/env python3
"""Build an 18-episode RM75 portfolio with timestamp-synchronized camera proof."""

from __future__ import annotations

import argparse
import bisect
import csv
import html
import json
import os
import struct
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from build_rm75_recovered_portfolio_videos import (
    ACCENT,
    BG,
    ERROR,
    GT,
    GRID,
    MUTED,
    ORB,
    PANEL,
    TEXT,
    draw_grid,
    draw_progress_curve,
    draw_text,
    fit_points,
    open_encoder,
    read_rows,
    trajectory_projection,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIEWERS = ROOT / "data/evaluation/workbench/rm75_recovered_baseline_viewers_20260731"
DEFAULT_AUDIT = ROOT / "data/evaluation/workbench/rm75_historical_input_audit_20260731/input_audit.csv"
DEFAULT_OUTPUT = ROOT / "data/evaluation/workbench/rm75_18_episode_camera_trajectory_portfolio_20260731"
DEFAULT_PACKAGE = ROOT / "data/evaluation/packages/rm75_18_episode_camera_trajectory_portfolio_20260731.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer-root", type=Path, default=DEFAULT_VIEWERS)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="Keep only audit-ready episodes with synchronized camera evidence.",
    )
    parser.add_argument(
        "--max-ape-mm",
        type=float,
        default=None,
        help="Keep entries whose reported SE(3) translation APE is at most this value.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def iter_mcap_messages(path: Path) -> Iterable[tuple[Any, Any, Any]]:
    try:
        from mcap.reader import make_reader
        from mcap.stream_reader import StreamReader
    except ImportError as exc:
        raise RuntimeError("Python package 'mcap' is required") from exc

    with path.open("rb") as handle:
        reader = make_reader(handle)
        yielded = 0
        for item in reader.iter_messages():
            yielded += 1
            yield item
        if yielded:
            return

    with path.open("rb") as handle:
        schemas: dict[int, Any] = {}
        channels: dict[int, Any] = {}
        for record in StreamReader(handle).records:
            name = type(record).__name__
            if name == "Schema":
                schemas[record.id] = record
            elif name == "Channel":
                channels[record.id] = record
            elif name == "Message":
                channel = channels.get(record.channel_id)
                if channel is not None:
                    yield schemas.get(channel.schema_id), channel, record


def load_camera_timestamps(path: Path, topic: str = "c") -> tuple[np.ndarray, np.ndarray]:
    frame_to_time: dict[int, int] = {}
    first_log_ns: int | None = None
    base_publish_ns: int | None = None
    for _schema, channel, message in iter_mcap_messages(path):
        if channel.topic != topic or len(message.data) < 4:
            continue
        log_ns = int(message.log_time)
        publish_ns = int(message.publish_time)
        if first_log_ns is None:
            first_log_ns = log_ns
            base_publish_ns = publish_ns
        frame_index = struct.unpack("<I", message.data[:4])[0]
        frame_to_time[frame_index] = int(base_publish_ns + (log_ns - first_log_ns))
    if not frame_to_time:
        raise RuntimeError(f"no camera timestamps on topic {topic!r}: {path}")
    ordered = sorted(frame_to_time.items(), key=lambda item: item[1])
    frame_indices = np.asarray([item[0] for item in ordered], dtype=np.int64)
    timestamps_s = np.asarray([item[1] * 1e-9 for item in ordered], dtype=np.float64)
    if np.any(np.diff(timestamps_s) <= 0):
        raise RuntimeError(f"camera timestamps are not strictly increasing: {path}")
    return frame_indices, timestamps_s


def read_tum_timestamps(path: Path) -> np.ndarray:
    timestamps = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                timestamps.append(float(stripped.split()[0]))
    if not timestamps:
        raise RuntimeError(f"empty TUM trajectory: {path}")
    return np.asarray(timestamps, dtype=np.float64)


def nearest_camera_indices(
    trajectory_times: np.ndarray,
    frame_indices: np.ndarray,
    camera_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(camera_times, trajectory_times, side="left")
    positions = np.clip(positions, 0, len(camera_times) - 1)
    previous = np.maximum(positions - 1, 0)
    use_previous = np.abs(camera_times[previous] - trajectory_times) <= np.abs(
        camera_times[positions] - trajectory_times
    )
    positions = np.where(use_previous, previous, positions)
    deltas_ms = np.abs(camera_times[positions] - trajectory_times) * 1000.0
    return frame_indices[positions], deltas_ms


def letterbox(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = rect
    canvas = np.full((height, width, 3), PANEL, dtype=np.uint8)
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    offset_x = (width - resized_w) // 2
    offset_y = (height - resized_h) // 2
    canvas[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
    return canvas


def draw_camera_panel(
    image: np.ndarray,
    frame: np.ndarray | None,
    rect: tuple[int, int, int, int],
    synchronized: bool,
) -> None:
    x, y, width, height = rect
    cv2.rectangle(image, (x, y), (x + width, y + height), PANEL, -1)
    if frame is not None:
        image[y : y + height, x : x + width] = letterbox(frame, rect)
        cv2.rectangle(image, (x, y), (x + width, y + height), GRID, 1, cv2.LINE_AA)
        cv2.rectangle(image, (x + 14, y + 14), (x + 238, y + 44), BG, -1)
        draw_text(image, "SYNCHRONIZED STEREO CAMERA", (x + 23, y + 35), scale=0.43, color=ACCENT, thickness=2)
    else:
        cv2.rectangle(image, (x, y), (x + width, y + height), GRID, 1, cv2.LINE_AA)
        for step in range(1, 5):
            xx = x + width * step // 5
            yy = y + height * step // 5
            cv2.line(image, (xx, y), (xx, y + height), GRID, 1, cv2.LINE_AA)
            cv2.line(image, (x, yy), (x + width, yy), GRID, 1, cv2.LINE_AA)
        label = "Trajectory-only evaluation" if not synchronized else "Camera frame unavailable"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.64, 2)[0]
        draw_text(
            image,
            label,
            (x + (width - text_size[0]) // 2, y + height // 2),
            scale=0.64,
            color=MUTED,
            thickness=2,
        )


def load_episode_payload(row: dict[str, str], viewer_root: Path) -> tuple[dict[str, Any], np.ndarray]:
    episode_slug = row["episode"].replace("/", "_")
    viewer_json = viewer_root / episode_slug / "viewer/viewer_data_3d.json"
    payload = json.loads(viewer_json.read_text(encoding="utf-8"))
    algorithm = payload["algorithms"][0]
    full_trajectory_times = read_tum_timestamps(Path(row["source_dir"]) / "gt_tcp_matched.tum")
    # Viewer generation caps long trajectories at 3000 points but preserves
    # each retained point's time relative to the first matched TUM sample.
    trajectory_times = full_trajectory_times[0] + np.asarray(
        [float(point["t"]) for point in algorithm["points"]], dtype=np.float64
    )
    if trajectory_times[-1] > full_trajectory_times[-1] + 1e-3:
        raise RuntimeError(f"viewer time range exceeds matched trajectory for {row['episode']}")
    return algorithm, trajectory_times


def render_episode(
    row: dict[str, str],
    audit: dict[str, str],
    viewer_root: Path,
    output: Path,
    width: int,
    height: int,
    fps: int,
    duration_sec: float,
    ordinal: int,
    total: int,
) -> dict[str, object]:
    algorithm, trajectory_times = load_episode_payload(row, viewer_root)
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

    camera_ready = audit["status"] == "ready"
    video_path = Path(audit["raw_dataset"]) / "stereo_right.mkv"
    cap: cv2.VideoCapture | None = None
    camera_indices: np.ndarray | None = None
    camera_deltas_ms: np.ndarray | None = None
    if camera_ready:
        frame_indices, camera_times = load_camera_timestamps(Path(audit["raw_dataset"]) / "fays_data_right.mcap")
        camera_indices, camera_deltas_ms = nearest_camera_indices(trajectory_times, frame_indices, camera_times)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source camera video: {video_path}")

    camera_rect = (36, 118, 470, height - 160)
    plot_rect = (534, 118, width - 570, 350)
    error_rect = (534, 526, 392, 140)
    gt_pixels = fit_points(gt_2d, plot_rect, bounds)
    estimate_pixels = fit_points(estimate_2d, plot_rect, bounds)
    total_frames = max(1, int(round(duration_sec * fps)))

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.mp4")
    encoder = open_encoder(partial, width, height, fps)
    current_video_index = -1
    current_camera_frame: np.ndarray | None = None
    cover: np.ndarray | None = None
    image = np.full((height, width, 3), BG, dtype=np.uint8)
    try:
        for frame_number in range(total_frames):
            progress = frame_number / max(1, total_frames - 1)
            eased = 1.0 - (1.0 - progress) ** 2.2
            index = min(len(points) - 1, int(round(eased * (len(points) - 1))))

            if cap is not None and camera_indices is not None:
                target_video_index = int(camera_indices[index])
                if target_video_index < current_video_index:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_video_index)
                    current_video_index = target_video_index - 1
                while current_video_index < target_video_index:
                    ok, decoded = cap.read()
                    if not ok:
                        raise RuntimeError(
                            f"camera decode ended at frame {current_video_index}, target {target_video_index}: {video_path}"
                        )
                    current_video_index += 1
                    current_camera_frame = decoded

            image = np.full((height, width, 3), BG, dtype=np.uint8)
            draw_text(
                image,
                f"RM75 Stereo SLAM Evaluation  |  Episode {ordinal:02d}/{total:02d}",
                (36, 44),
                scale=0.92,
                thickness=2,
            )
            subtitle = "Synchronized camera evidence and TCP trajectory" if camera_ready else "Robot TCP ground truth and ORB-SLAM3 trajectory"
            draw_text(image, subtitle, (36, 76), scale=0.55, color=MUTED)
            draw_text(image, "TCP frame  |  SE(3) alignment  |  translation APE", (36, 101), scale=0.50, color=MUTED)

            draw_camera_panel(image, current_camera_frame, camera_rect, camera_ready)

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

            metric_x = 952
            draw_text(image, "APE RESULT", (metric_x, 505), scale=0.52, color=ACCENT, thickness=2)
            recovered_mm = float(row["ape_translation_se3_rmse_mm"])
            draw_text(image, f"SE(3)  {recovered_mm:.3f} mm", (metric_x, 542), scale=0.67, thickness=2)
            draw_text(image, f"Matched samples  {row['pairs']}", (metric_x, 576), scale=0.49, color=MUTED)
            draw_text(image, f"Current error  {errors_mm[index]:.2f} mm", (metric_x, 608), scale=0.49, color=ERROR, thickness=2)
            if camera_deltas_ms is not None:
                draw_text(image, "Camera sync  timestamp matched", (metric_x, 640), scale=0.43, color=ACCENT)
                draw_text(image, f"Nearest-frame delta  {camera_deltas_ms[index]:.2f} ms", (metric_x, 665), scale=0.41, color=MUTED)
            else:
                draw_text(image, "Trajectory-only evaluation", (metric_x, 646), scale=0.45, color=MUTED)

            draw_text(image, "TRANSLATION ERROR", (error_rect[0], error_rect[1] - 14), scale=0.50, color=ACCENT, thickness=2)
            draw_progress_curve(image, errors_mm, index, error_rect)
            draw_text(image, f"Progress {100.0 * progress:5.1f}%", (width - 180, height - 22), scale=0.48, color=MUTED)
            encoder.stdin.write(image.tobytes())
            if frame_number == min(fps, total_frames - 1):
                cover = image.copy()
    finally:
        if cap is not None:
            cap.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {row['episode']} with code {return_code}")
    os.replace(partial, output)
    cover_path = output.with_suffix(".png")
    cv2.imwrite(str(cover_path), cover if cover is not None else image)
    return {
        "sequence": f"Episode {ordinal:02d}",
        "source_episode": row["episode"],
        "source_tag": row["tag"],
        "ape_translation_se3_rmse_mm": float(row["ape_translation_se3_rmse_mm"]),
        "pairs": int(row["pairs"]),
        "evidence": "camera_and_trajectory" if camera_ready else "trajectory_only",
        "max_camera_delta_ms": float(np.max(camera_deltas_ms)) if camera_deltas_ms is not None else "",
        "raw_dataset": audit["raw_dataset"] if camera_ready else "",
        "video": output.name,
        "cover": cover_path.name,
    }


def concatenate_videos(output_root: Path, entries: list[dict[str, object]]) -> Path:
    concat = output_root / "concat.txt"
    concat.write_text("".join(f"file '{entry['video']}'\n" for entry in entries), encoding="utf-8")
    combined = output_root / f"rm75_{len(entries)}_episode_camera_trajectory_portfolio.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(combined),
        ],
        cwd=output_root,
        check=True,
    )
    return combined


def write_index(output_root: Path, entries: list[dict[str, object]], combined: Path) -> None:
    total = len(entries)
    rows = []
    for entry in entries:
        evidence = "Camera + trajectory" if entry["evidence"] == "camera_and_trajectory" else "Trajectory only"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(entry['sequence']))}</td>"
            f"<td>{float(entry['ape_translation_se3_rmse_mm']):.3f} mm</td>"
            f"<td>{evidence}</td>"
            f"<td><a href='{html.escape(str(entry['video']))}'>MP4</a></td>"
            "</tr>"
        )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RM75 Camera and Trajectory Portfolio</title><style>
body{{margin:0;background:#f4f6f5;color:#182226;font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}header{{padding:22px max(18px,calc((100vw - 1120px)/2));background:#fff;border-bottom:1px solid #d9e1df}}h1{{margin:0 0 5px;font-size:25px;letter-spacing:0}}p{{margin:0;color:#687674}}main{{max-width:1120px;margin:0 auto;padding:22px 18px 36px}}video{{display:block;width:100%;aspect-ratio:16/9;background:#141618}}table{{width:100%;margin-top:20px;border-collapse:collapse;background:#fff;border:1px solid #d9e1df}}th,td{{padding:10px 12px;border-bottom:1px solid #d9e1df;text-align:left}}th{{background:#edf2f0;color:#687674;font-size:12px}}a{{color:#086a91;text-decoration:none}}a:hover{{text-decoration:underline}}
</style></head><body><header><h1>RM75 {total}-Episode Camera and Trajectory Portfolio</h1><p>Timestamp-synchronized stereo imagery, Robot TCP ground truth and ORB-SLAM3 trajectory evaluation.</p></header><main><video controls preload="metadata" poster="{entries[0]['cover']}" src="{combined.name}"></video><table><thead><tr><th>Episode</th><th>APE SE(3)</th><th>Evidence</th><th>Video</th></tr></thead><tbody>{''.join(rows)}</tbody></table></main></body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def write_package(package: Path, output_root: Path, entries: list[dict[str, object]], combined: Path) -> None:
    package.parent.mkdir(parents=True, exist_ok=True)
    public_files = [
        output_root / "index.html",
        output_root / "README.txt",
        output_root / "summary.csv",
        output_root / "manifest.json",
        combined,
    ]
    for entry in entries:
        public_files.extend([output_root / str(entry["video"]), output_root / str(entry["cover"])])
    partial = package.with_name(package.name + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in public_files:
            archive.write(path, arcname=f"rm75_camera_trajectory_portfolio/{path.name}")
    os.replace(partial, package)


def load_existing_entry(path: Path, row: dict[str, str], audit: dict[str, str], ordinal: int) -> dict[str, object]:
    return {
        "sequence": f"Episode {ordinal:02d}",
        "source_episode": row["episode"],
        "source_tag": row["tag"],
        "ape_translation_se3_rmse_mm": float(row["ape_translation_se3_rmse_mm"]),
        "pairs": int(row["pairs"]),
        "evidence": "camera_and_trajectory" if audit["status"] == "ready" else "trajectory_only",
        "max_camera_delta_ms": "",
        "raw_dataset": audit["raw_dataset"] if audit["status"] == "ready" else "",
        "video": path.name,
        "cover": path.with_suffix(".png").name,
    }


def validate_inputs(
    rows: list[dict[str, str]],
    audits: dict[str, dict[str, str]],
    viewer_root: Path,
    *,
    camera_only: bool,
) -> None:
    expected_total = len(rows)
    if len(rows) != expected_total:
        raise RuntimeError(f"expected {expected_total} portfolio entries, found {len(rows)}")
    ready_count = 0
    for row in rows:
        if row["episode"] not in audits:
            raise KeyError(f"audit entry missing for {row['episode']}")
        audit = audits[row["episode"]]
        load_episode_payload(row, viewer_root)
        if audit["status"] == "ready":
            ready_count += 1
            raw = Path(audit["raw_dataset"])
            for required in (raw / "stereo_right.mkv", raw / "fays_data_right.mcap"):
                if not required.is_file():
                    raise FileNotFoundError(required)
    expected_ready = expected_total if camera_only else min(12, expected_total)
    if ready_count != expected_ready:
        raise RuntimeError(f"expected {expected_ready} camera-ready entries, found {ready_count}")


def main() -> int:
    args = parse_args()
    viewer_root = args.viewer_root.expanduser().resolve()
    audit_csv = args.audit_csv.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    package = args.package.expanduser().resolve()
    rows = read_rows(viewer_root / "summary.csv")
    audits = {row["episode"]: row for row in read_rows(audit_csv)}
    if args.camera_only:
        rows = [row for row in rows if audits[row["episode"]]["status"] == "ready"]
    if args.max_ape_mm is not None:
        rows = [row for row in rows if float(row["ape_translation_se3_rmse_mm"]) <= args.max_ape_mm]
    validate_inputs(rows, audits, viewer_root, camera_only=args.camera_only)
    trajectory_only_count = sum(audits[row["episode"]]["status"] != "ready" for row in rows)
    print(
        f"[VALID] episodes={len(rows)} camera_and_trajectory={len(rows) - trajectory_only_count} "
        f"trajectory_only={trajectory_only_count}",
        flush=True,
    )
    if args.validate_only:
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for ordinal, row in enumerate(rows, start=1):
        output = output_root / f"episode_{ordinal:02d}.mp4"
        cover = output.with_suffix(".png")
        if not args.no_resume and output.is_file() and cover.is_file():
            print(f"[RESUME {ordinal:02d}/{len(rows):02d}] {output.name}", flush=True)
            entries.append(load_existing_entry(output, row, audits[row["episode"]], ordinal))
            continue
        evidence = "camera+trajectory" if audits[row["episode"]]["status"] == "ready" else "trajectory-only"
        print(f"[RENDER {ordinal:02d}/{len(rows):02d}] {evidence}", flush=True)
        entries.append(
            render_episode(
                row,
                audits[row["episode"]],
                viewer_root,
                output,
                args.width,
                args.height,
                args.fps,
                args.duration_sec,
                ordinal,
                len(rows),
            )
        )

    combined = concatenate_videos(output_root, entries)
    public_fields = ["sequence", "ape_translation_se3_rmse_mm", "pairs", "evidence", "video", "cover"]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    with (output_root / "source_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    public_entries = [{key: entry[key] for key in public_fields} for entry in entries]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": f"RM75 {len(entries)}-episode synchronized camera and GT-vs-ORB trajectory portfolio",
        "render": {"width": args.width, "height": args.height, "fps": args.fps, "duration_sec": args.duration_sec},
        "combined_video": combined.name,
        "coordinate_frame": "robot TCP",
        "alignment": "SE(3)",
        "camera_synchronization": "nearest MCAP camera timestamp for each matched trajectory timestamp",
        "camera_and_trajectory_count": sum(entry["evidence"] == "camera_and_trajectory" for entry in entries),
        "trajectory_only_count": sum(entry["evidence"] == "trajectory_only" for entry in entries),
        "entries": public_entries,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_root / "README.txt").write_text(
        f"RM75 {len(entries)}-Episode Camera and Trajectory Portfolio\n\n"
        "Extract the ZIP and open index.html. The page and all videos work offline.\n"
        "Camera-backed episodes use the nearest original MCAP camera timestamp for each trajectory sample.\n",
        encoding="utf-8",
    )
    write_index(output_root, entries, combined)
    write_package(package, output_root, entries, combined)
    print(f"[DONE] portfolio={output_root}", flush=True)
    print(f"[DONE] package={package}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
