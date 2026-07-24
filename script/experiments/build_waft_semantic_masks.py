#!/usr/bin/env python3
"""Build WAFT-Stereo confidence masks aligned to the EuRoC export image space."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


DEFAULT_WAFT_ROOT = Path("/home/chenlvping/1_DM_work/depth_eval/external/WAFT-Stereo")
DEFAULT_WAFT_CONFIG = DEFAULT_WAFT_ROOT / "configs/SynLarge/DAv2L-5.yaml"
DEFAULT_WAFT_CHECKPOINT = DEFAULT_WAFT_ROOT / "ckpts/SynLarge/DAv2L-5.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True, help="Existing EuRoC export directory with mav0/cam*/data")
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--camera-rig", choices=("stereo_left", "stereo_right"), default="stereo_right")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--crop-size", default="480,640")
    parser.add_argument("--factors", default="0.5,1.0")
    parser.add_argument("--uncertainty-quantile", type=float, default=0.85, help="Mask pixels above this per-frame uncertainty quantile")
    parser.add_argument("--min-disp-px", type=float, default=0.5)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--dilate-px", type=int, default=5)
    parser.add_argument("--debug-every", type=int, default=100, help="Save debug overlays every N processed frames; 0 disables")
    parser.add_argument("--waft-root", type=Path, default=DEFAULT_WAFT_ROOT)
    parser.add_argument("--waft-config", type=Path, default=DEFAULT_WAFT_CONFIG)
    parser.add_argument("--waft-checkpoint", type=Path, default=DEFAULT_WAFT_CHECKPOINT)
    return parser.parse_args()


def parse_pair(text: str) -> tuple[int, int]:
    parts = [item.strip() for item in str(text).split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"expected two comma-separated integers, got {text!r}")
    return int(parts[0]), int(parts[1])


def parse_factor_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("at least one inference factor is required")
    return values


def merge_peft_modules(model: torch.nn.Module) -> None:
    if PeftModel is None:
        return
    for _, module in model.named_modules():
        if isinstance(module, PeftModel):
            module.merge_and_unload()


def setup_waft(waft_root: Path, config_path: Path):
    sys.path.insert(0, str(waft_root))
    from algorithms.waft import WAFT  # noqa: WPS433
    from bridgedepth.config import get_cfg  # noqa: WPS433

    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.freeze()
    return WAFT, cfg


def load_waft_model(waft_root: Path, config_path: Path, checkpoint_path: Path):
    waft_root = waft_root.resolve()
    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    if not waft_root.is_dir():
        raise FileNotFoundError(f"WAFT root not found: {waft_root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"WAFT config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"WAFT checkpoint not found: {checkpoint_path}")

    old_cwd = Path.cwd()
    os.chdir(waft_root)
    try:
        waft_cls, cfg = setup_waft(waft_root, config_path)
        model = waft_cls(cfg).eval().cuda()
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        weights = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        load_info = model.load_state_dict(weights, strict=False)
        merge_peft_modules(model)
    finally:
        os.chdir(old_cwd)
    return model, cfg, load_info


def load_rig_stream(calibration_path: Path, camera_rig: str) -> dict:
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    images = payload.get("observation", {}).get("images", {})
    if camera_rig not in images:
        raise KeyError(f"{camera_rig} missing in {calibration_path}")
    stream = images[camera_rig]
    if stream.get("distortion_model") != "equidistant":
        raise ValueError(f"only equidistant fisheye distortion is supported, got {stream.get('distortion_model')!r}")
    return stream


def intrinsics_for_size(cam_info: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = cam_info.get("intrinsics") or {}
    key = f"{width}x{height}"
    if key not in intrinsics:
        raise KeyError(f"intrinsics for {key} missing; available={sorted(intrinsics)}")
    intr = intrinsics[key]
    matrix = np.array(
        [
            [float(intr["fx"]), 0.0, float(intr["ppx"])],
            [0.0, float(intr["fy"]), float(intr["ppy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    coeffs = cam_info.get("distortion_coeffs") or []
    if len(coeffs) < 4:
        raise ValueError("fisheye distortion requires four coefficients")
    distortion = np.asarray(coeffs[:4], dtype=np.float64)
    return matrix, distortion


def invert_se3(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def stereo_transform(stream: dict) -> tuple[np.ndarray, np.ndarray]:
    extrinsics = stream.get("extrinsics") or {}
    t_i_c0 = np.asarray(extrinsics["T_ic_cam0_to_imu0"], dtype=np.float64)
    t_i_c1 = np.asarray(extrinsics["T_ic_cam1_to_imu0"], dtype=np.float64)
    if t_i_c0.shape != (4, 4) or t_i_c1.shape != (4, 4):
        raise ValueError("camera extrinsics must be 4x4 matrices")
    t_lr = invert_se3(t_i_c0) @ t_i_c1
    return t_lr[:3, :3], t_lr[:3, 3]


def compute_baseline_from_projection(p2: np.ndarray, fallback_norm: float) -> float:
    baselines = []
    if abs(float(p2[0, 0])) > 1e-9:
        baselines.append(abs(float(p2[0, 3] / p2[0, 0])))
    if abs(float(p2[1, 1])) > 1e-9:
        baselines.append(abs(float(p2[1, 3] / p2[1, 1])))
    baselines = [value for value in baselines if value > 1e-9]
    return max(baselines) if baselines else fallback_norm


def build_inverse_rectify_map(k: np.ndarray, d: np.ndarray, r: np.ndarray, p: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    raw_points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    rectified = cv2.fisheye.undistortPoints(raw_points, k, d, R=r, P=p[:, :3])
    rectified = rectified.reshape(height, width, 2).astype(np.float32)
    return rectified[..., 0], rectified[..., 1]


def load_export_pairs(export_dir: Path) -> list[tuple[str, Path, Path]]:
    left_dir = export_dir / "mav0" / "cam0" / "data"
    right_dir = export_dir / "mav0" / "cam1" / "data"
    if not left_dir.is_dir() or not right_dir.is_dir():
        raise FileNotFoundError(f"missing cam0/cam1 data folders under {export_dir}")

    left_map = {path.stem: path for path in left_dir.glob("*.png")}
    right_map = {path.stem: path for path in right_dir.glob("*.png")}
    common = sorted(set(left_map) & set(right_map), key=lambda item: int(item))
    if not common:
        raise RuntimeError(f"no paired PNG files found in {left_dir} and {right_dir}")
    return [(stem, left_map[stem], right_map[stem]) for stem in common]


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def waft_infer(
    model,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    crop_size: tuple[int, int],
    factors: list[float],
    autocast_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    sample = {
        "img1": torch.from_numpy(left_rgb).cuda().float().permute(2, 0, 1).unsqueeze(0),
        "img2": torch.from_numpy(right_rgb).cuda().float().permute(2, 0, 1).unsqueeze(0),
    }
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
            if len(factors) > 1:
                results = model.heirarchical_inference(sample, size=list(crop_size), factor_list=factors)
            else:
                results = model.inference(sample, size=list(crop_size), factor=factors[0])
    disp = results["disp_pred"][0].detach().float().cpu().numpy()
    info = results["delta_info_preds"][-1][:, :2].softmax(dim=1)[:, 0]
    heatmap = info[0].detach().float().cpu().numpy()
    return disp, heatmap


def compute_depth(disp: np.ndarray, fx: float, baseline: float, max_depth_m: float, min_disp_px: float) -> np.ndarray:
    depth = np.full(disp.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disp) & (disp > min_disp_px)
    depth[valid] = fx * baseline / disp[valid]
    depth[(depth <= 0.02) | (depth > max_depth_m)] = np.nan
    return depth


def dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def right_mask_from_left(left_mask: np.ndarray, disp: np.ndarray) -> np.ndarray:
    height, width = left_mask.shape
    ys, xs = np.nonzero(left_mask)
    if len(xs) == 0:
        return np.zeros_like(left_mask, dtype=bool)
    right_mask = np.zeros_like(left_mask, dtype=bool)
    disp_values = disp[ys, xs]
    valid = np.isfinite(disp_values)
    shifted_x = np.rint(xs.astype(np.float32) - np.where(valid, disp_values, 0.0)).astype(np.int32)
    keep = (shifted_x >= 0) & (shifted_x < width)
    right_mask[ys[keep], shifted_x[keep]] = True
    return right_mask


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    overlay[mask] = np.array([255, 64, 64], dtype=np.uint8)
    return cv2.addWeighted(image, 0.6, overlay, 0.4, 0.0)


def write_debug_artifacts(
    debug_dir: Path,
    stem: str,
    left_rect_rgb: np.ndarray,
    right_rect_rgb: np.ndarray,
    heatmap: np.ndarray,
    left_mask_rect: np.ndarray,
    right_mask_rect: np.ndarray,
    left_raw_rgb: np.ndarray,
    right_raw_rgb: np.ndarray,
    left_mask_raw: np.ndarray,
    right_mask_raw: np.ndarray,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    heatmap_u8 = np.clip(heatmap * 255.0, 0.0, 255.0).astype(np.uint8)
    heatmap_rgb = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

    panels = [
        np.concatenate([left_rect_rgb, heatmap_rgb], axis=1),
        np.concatenate([overlay_mask(left_rect_rgb, left_mask_rect), overlay_mask(right_rect_rgb, right_mask_rect)], axis=1),
        np.concatenate([overlay_mask(left_raw_rgb, left_mask_raw), overlay_mask(right_raw_rgb, right_mask_raw)], axis=1),
    ]
    mosaic = np.concatenate(panels, axis=0)
    cv2.imwrite(str(debug_dir / f"{stem}.png"), cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this WAFT experiment")

    args.episode_dir = args.episode_dir.expanduser().resolve()
    args.export_dir = args.export_dir.expanduser().resolve()
    args.mask_dir = args.mask_dir.expanduser().resolve()
    args.waft_root = args.waft_root.expanduser().resolve()
    args.waft_config = args.waft_config.expanduser().resolve()
    args.waft_checkpoint = args.waft_checkpoint.expanduser().resolve()
    crop_size = parse_pair(args.crop_size)
    factors = parse_factor_list(args.factors)

    pairs = load_export_pairs(args.export_dir)
    start_index = max(0, int(args.start_index))
    frame_step = max(1, int(args.frame_step))
    pairs = pairs[start_index::frame_step]
    if args.max_frames > 0:
        pairs = pairs[: args.max_frames]
    if not pairs:
        raise ValueError("no frames selected after applying start/max/step filters")

    first_left = cv2.imread(str(pairs[0][1]), cv2.IMREAD_UNCHANGED)
    first_right = cv2.imread(str(pairs[0][2]), cv2.IMREAD_UNCHANGED)
    if first_left is None or first_right is None:
        raise RuntimeError("failed to read the first stereo pair from export")
    if first_left.shape[:2] != first_right.shape[:2]:
        raise ValueError("cam0/cam1 export image sizes do not match")

    height, width = first_left.shape[:2]
    stream = load_rig_stream(args.episode_dir / "calibration.json", args.camera_rig)
    k0, d0 = intrinsics_for_size(stream["cam0"], width, height)
    k1, d1 = intrinsics_for_size(stream["cam1"], width, height)
    r_lr, t_lr = stereo_transform(stream)
    size = (width, height)
    r0, r1, p0, p1, _ = cv2.fisheye.stereoRectify(
        k0,
        d0,
        k1,
        d1,
        size,
        r_lr,
        t_lr,
        cv2.CALIB_ZERO_DISPARITY,
        size,
        0.0,
        1.0,
    )
    rect_map0_x, rect_map0_y = cv2.fisheye.initUndistortRectifyMap(k0, d0, r0, p0[:, :3], size, cv2.CV_32FC1)
    rect_map1_x, rect_map1_y = cv2.fisheye.initUndistortRectifyMap(k1, d1, r1, p1[:, :3], size, cv2.CV_32FC1)
    raw_to_rect0_x, raw_to_rect0_y = build_inverse_rectify_map(k0, d0, r0, p0, size)
    raw_to_rect1_x, raw_to_rect1_y = build_inverse_rectify_map(k1, d1, r1, p1, size)
    rect_k = p0[:3, :3]
    baseline = compute_baseline_from_projection(p1, float(np.linalg.norm(t_lr)))

    model, cfg, load_info = load_waft_model(args.waft_root, args.waft_config, args.waft_checkpoint)
    autocast_dtype = torch.bfloat16

    left_mask_dir = args.mask_dir / "left"
    right_mask_dir = args.mask_dir / "right"
    debug_dir = args.mask_dir / "_debug"
    left_mask_dir.mkdir(parents=True, exist_ok=True)
    right_mask_dir.mkdir(parents=True, exist_ok=True)

    frame_reports = []
    run_start = time.time()

    for loop_index, (stem, left_path, right_path) in enumerate(pairs):
        left_raw = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
        right_raw = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
        if left_raw is None or right_raw is None:
            raise RuntimeError(f"failed to read stereo pair {left_path} / {right_path}")

        left_rect = cv2.remap(left_raw, rect_map0_x, rect_map0_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        right_rect = cv2.remap(right_raw, rect_map1_x, rect_map1_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        left_rect_rgb = to_rgb(left_rect)
        right_rect_rgb = to_rgb(right_rect)

        infer_start = time.time()
        disp, heatmap = waft_infer(model, left_rect_rgb, right_rect_rgb, crop_size, factors, autocast_dtype)
        infer_seconds = time.time() - infer_start

        depth = compute_depth(disp, float(rect_k[0, 0]), baseline, float(args.max_depth_m), float(args.min_disp_px))
        invalid = ~np.isfinite(depth)
        finite_heatmap = np.isfinite(heatmap)
        if not np.any(finite_heatmap):
            raise RuntimeError(f"WAFT produced no finite heatmap values for {stem}")
        threshold = float(np.quantile(heatmap[finite_heatmap], float(args.uncertainty_quantile)))
        uncertain = heatmap >= threshold
        left_mask_rect = dilate_mask(invalid | uncertain, int(args.dilate_px))
        right_mask_rect = dilate_mask(right_mask_from_left(left_mask_rect, disp), int(args.dilate_px))

        left_mask_raw = cv2.remap(
            (left_mask_rect.astype(np.uint8) * 255),
            raw_to_rect0_x,
            raw_to_rect0_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        right_mask_raw = cv2.remap(
            (right_mask_rect.astype(np.uint8) * 255),
            raw_to_rect1_x,
            raw_to_rect1_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0

        cv2.imwrite(str(left_mask_dir / f"{stem}.png"), left_mask_raw.astype(np.uint8) * 255)
        cv2.imwrite(str(right_mask_dir / f"{stem}.png"), right_mask_raw.astype(np.uint8) * 255)

        if args.debug_every > 0 and (loop_index == 0 or (loop_index + 1) % int(args.debug_every) == 0):
            write_debug_artifacts(
                debug_dir,
                stem,
                left_rect_rgb,
                right_rect_rgb,
                heatmap,
                left_mask_rect,
                right_mask_rect,
                to_rgb(left_raw),
                to_rgb(right_raw),
                left_mask_raw,
                right_mask_raw,
            )

        frame_report = {
            "stem": stem,
            "inference_seconds": infer_seconds,
            "uncertainty_threshold": threshold,
            "invalid_ratio_rect": float(invalid.mean()),
            "left_mask_ratio_rect": float(left_mask_rect.mean()),
            "right_mask_ratio_rect": float(right_mask_rect.mean()),
            "left_mask_ratio_raw": float(left_mask_raw.mean()),
            "right_mask_ratio_raw": float(right_mask_raw.mean()),
            "disp_p05_p50_p95_px": [float(value) for value in np.percentile(disp[np.isfinite(disp)], [5, 50, 95])]
            if np.isfinite(disp).any()
            else None,
            "depth_p05_p50_p95_m": [float(value) for value in np.percentile(depth[np.isfinite(depth)], [5, 50, 95])]
            if np.isfinite(depth).any()
            else None,
        }
        frame_reports.append(frame_report)

        if loop_index == 0 or (loop_index + 1) % 25 == 0 or loop_index == len(pairs) - 1:
            elapsed = time.time() - run_start
            print(
                json.dumps(
                    {
                        "processed_frames": loop_index + 1,
                        "total_frames": len(pairs),
                        "elapsed_seconds": elapsed,
                        "avg_seconds_per_frame": elapsed / (loop_index + 1),
                        "last_stem": stem,
                        "last_left_mask_ratio_raw": frame_report["left_mask_ratio_raw"],
                    }
                ),
                flush=True,
            )

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episode_dir": str(args.episode_dir),
        "export_dir": str(args.export_dir),
        "mask_dir": str(args.mask_dir),
        "camera_rig": args.camera_rig,
        "selected_frames": len(pairs),
        "start_index": start_index,
        "frame_step": frame_step,
        "crop_size": list(crop_size),
        "factors": factors,
        "uncertainty_quantile": float(args.uncertainty_quantile),
        "min_disp_px": float(args.min_disp_px),
        "max_depth_m": float(args.max_depth_m),
        "dilate_px": int(args.dilate_px),
        "rectified_fx": float(rect_k[0, 0]),
        "baseline_m": float(baseline),
        "waft_root": str(args.waft_root),
        "waft_config": str(args.waft_config),
        "waft_checkpoint": str(args.waft_checkpoint),
        "waft_missing_keys": list(load_info.missing_keys),
        "waft_unexpected_keys": list(load_info.unexpected_keys),
        "runtime_seconds": time.time() - run_start,
        "mean_inference_seconds": float(np.mean([item["inference_seconds"] for item in frame_reports])),
        "mean_left_mask_ratio_raw": float(np.mean([item["left_mask_ratio_raw"] for item in frame_reports])),
        "mean_right_mask_ratio_raw": float(np.mean([item["right_mask_ratio_raw"] for item in frame_reports])),
        "frame_reports": frame_reports,
    }
    report_path = args.mask_dir / "waft_mask_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
