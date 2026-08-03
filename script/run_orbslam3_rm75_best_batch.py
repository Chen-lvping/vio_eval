#!/usr/bin/env python3
"""Run the locked 2026-07-16 colleague ORB-SLAM3 RM75 stereo baseline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "data/evaluation/config/rm75_historical_stereo_baseline_20260716.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path)
    parser.add_argument("--gt-root", type=Path)
    parser.add_argument("--episode-pattern", default="episode_*")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/evaluation/workbench")
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--skip-viewer", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Verify all fixed inputs and exit without running an episode.")
    parser.add_argument("--dry-run", action="store_true", help="Verify inputs and print commands without writing results.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    completed = subprocess.run(
        ["sha256sum", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.split(maxsplit=1)[0].lower()


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_locked_file(path: Path, expected_sha256: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"locked {label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise RuntimeError(
            f"locked {label} checksum changed: {path}\n"
            f"expected={expected_sha256.lower()}\nactual={actual}"
        )
    return actual


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def canonical_active_calibration_sha256(calibration: dict[str, Any]) -> str:
    observation = require_object(calibration.get("observation"), "calibration.observation")
    images = require_object(observation.get("images"), "calibration.observation.images")
    imus = require_object(observation.get("imu"), "calibration.observation.imu")
    active = {
        "stereo_right": images["stereo_right"],
        "imu_right": imus["imu_right"],
    }
    payload = json.dumps(active, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def load_and_verify_profile(profile_path: Path) -> dict[str, Any]:
    profile_path = profile_path.expanduser().resolve()
    verify_locked_file(profile_path, sha256_file(profile_path), "profile")
    profile = require_object(json.loads(profile_path.read_text(encoding="utf-8")), "profile")
    if profile.get("profile_id") != "rm75-historical-stereo-20260716":
        raise ValueError(f"unsupported profile_id: {profile.get('profile_id')!r}")

    algorithm = require_object(profile.get("algorithm"), "profile.algorithm")
    if algorithm.get("tracking_mode") != "stereo" or algorithm.get("imu_fusion") is not False:
        raise ValueError("historical baseline must be explicit pure stereo with imu_fusion=false")
    if algorithm.get("allow_fallback") is not False:
        raise ValueError("historical baseline must have allow_fallback=false")

    calibration_lock = require_object(profile.get("calibration_lock"), "profile.calibration_lock")
    calibration_path = resolve_repo_path(str(calibration_lock["path"]))
    calibration_sha = verify_locked_file(calibration_path, str(calibration_lock["sha256"]), "calibration")
    calibration = require_object(json.loads(calibration_path.read_text(encoding="utf-8")), "calibration")

    contract = require_object(profile.get("dataset_contract"), "profile.dataset_contract")
    active_sha = canonical_active_calibration_sha256(calibration)
    expected_active_sha = str(contract["active_stereo_right_plus_imu_right_canonical_sha256"])
    if active_sha != expected_active_sha:
        raise RuntimeError(f"active stereo_right/imu_right calibration changed: expected={expected_active_sha}, actual={active_sha}")
    images = calibration["observation"]["images"]
    if images["tcam_right_l"].get("serial") != contract.get("left_camera_serial"):
        raise RuntimeError("unified calibration left stereo_right serial does not match the profile")
    if images["tcam_right_r"].get("serial") != contract.get("right_camera_serial"):
        raise RuntimeError("unified calibration right stereo_right serial does not match the profile")

    offset_lock = require_object(profile.get("offset_lock"), "profile.offset_lock")
    offset_path = resolve_repo_path(str(offset_lock["path"]))
    offset_sha = verify_locked_file(offset_path, str(offset_lock["sha256"]), "offset JSON")
    offsets = require_object(json.loads(offset_path.read_text(encoding="utf-8")), "offset JSON")
    expected_count = int(contract["episode_count"])
    if len(offsets) != expected_count:
        raise RuntimeError(f"offset JSON must contain exactly {expected_count} clean episodes, got {len(offsets)}")
    for episode, value in offsets.items():
        if not episode.startswith("episode_") or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid offset entry: {episode!r} -> {value!r}")

    runtime_lock = require_object(profile.get("runtime_lock"), "profile.runtime_lock")
    orb_root = Path(str(runtime_lock["root"])).expanduser().resolve()
    locked_runtime: dict[str, Path] = {}
    for name in ("executable", "orb_library", "dbow2_library", "g2o_library", "runner", "smoother", "vocabulary"):
        item = require_object(runtime_lock.get(name), f"profile.runtime_lock.{name}")
        path = (orb_root / str(item["path"])).resolve()
        verify_locked_file(path, str(item["sha256"]), f"runtime {name}")
        locked_runtime[name] = path

    print(f"[VERIFY] profile={profile['profile_id']} sha256={sha256_file(profile_path)}")
    print(f"[VERIFY] pure stereo, imu_fusion=false, camera_rig={contract['camera_rig']}")
    print(f"[VERIFY] calibration={calibration_path} sha256={calibration_sha}")
    print(f"[VERIFY] offsets={offset_path} sha256={offset_sha} episodes={len(offsets)}")
    print(f"[VERIFY] runtime={orb_root}")
    return {
        "profile": profile,
        "profile_path": profile_path,
        "profile_sha256": sha256_file(profile_path),
        "algorithm": algorithm,
        "contract": contract,
        "calibration_path": calibration_path,
        "calibration_sha256": calibration_sha,
        "offset_path": offset_path,
        "offset_sha256": offset_sha,
        "offsets": offsets,
        "orb_root": orb_root,
        "runtime": locked_runtime,
    }


def run(cmd: list[object], cwd: Path | None = None, dry_run: bool = False) -> None:
    normalized = [str(item) for item in cmd]
    print("[RUN]", " ".join(normalized), flush=True)
    if not dry_run:
        subprocess.run(normalized, cwd=str(cwd) if cwd else None, check=True)


def resolve_ground_truth(gt_root: Path, episode_tag: str) -> Path:
    suffix = episode_tag.rsplit("_", 1)[-1]
    candidates = [
        gt_root / f"rm75_pose_traj_{int(suffix)}.json",
        gt_root / f"rm75_pose_traj{int(suffix):02d}.json",
        gt_root / f"rm75_pose_traj{int(suffix)}.json",
    ]
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise FileNotFoundError("ground truth not found; checked:\n" + "\n".join(str(path) for path in candidates))
    return found


def main() -> int:
    args = parse_args()
    locked = load_and_verify_profile(args.config)
    if args.verify_only:
        return 0
    if args.episode_root is None:
        raise SystemExit("--episode-root is required unless --verify-only is used")

    episode_root = args.episode_root.expanduser().resolve()
    if not episode_root.is_dir():
        raise FileNotFoundError(f"episode root not found: {episode_root}")
    episodes = sorted(path for path in episode_root.glob(args.episode_pattern) if path.is_dir())
    if not episodes:
        raise FileNotFoundError(f"no episodes match {args.episode_pattern!r} under {episode_root}")
    missing_offsets = [episode.name for episode in episodes if episode.name not in locked["offsets"]]
    if missing_offsets:
        raise RuntimeError("fixed historical offsets are missing for:\n" + "\n".join(missing_offsets))

    algorithm = locked["algorithm"]
    runtime = locked["runtime"]
    offsets = locked["offsets"]
    gt_root = args.gt_root.expanduser().resolve() if args.gt_root is not None else None
    output_root = args.output_root.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch = output_root / f"orbslam3_rm75_historical_stereo_20260716_{stamp}"
    if not args.dry_run:
        batch.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    for episode in episodes:
        tag = episode.name
        offset = float(offsets[tag])
        work = batch / tag
        raw = work / "pose_raw.csv"
        smooth = work / "pose_smooth.csv"
        strict = work / "pose_smooth_strictsync.csv"
        eval_dir = work / "eval"
        name = f"colleague_{tag}_ffba{algorithm['full_frame_visual_ba_iterations']}"
        if not args.dry_run:
            work.mkdir(parents=True, exist_ok=False)

        run([
            sys.executable, runtime["runner"],
            "--offline-accurate",
            "--tracking-mode", algorithm["tracking_mode"],
            "--episode-dir", episode,
            "--calibration-json", locked["calibration_path"],
            "--camera-key", locked["contract"]["camera_rig"],
            "--orbslam-binary", runtime["executable"],
            "--vocabulary", runtime["vocabulary"],
            "--work-dir", work,
            "--output-csv", raw,
            "--eval-dir", work / "eval_raw",
            "--trajectory-name", name,
            "--gba-iterations", algorithm["gba_iterations"],
            "--full-frame-ba-iterations", algorithm["full_frame_visual_ba_iterations"],
            "--orb-features", algorithm["orb_features"],
            "--orb-scale-factor", algorithm["orb_scale_factor"],
            "--orb-levels", algorithm["orb_levels"],
            "--orb-init-fast", algorithm["orb_ini_th_fast"],
            "--orb-min-fast", algorithm["orb_min_th_fast"],
            "--overwrite",
        ], cwd=locked["orb_root"], dry_run=args.dry_run)
        run([
            sys.executable, runtime["smoother"],
            "--input-csv", raw,
            "--output-csv", smooth,
            "--position-window", algorithm["position_smoothing_window"],
            "--position-poly", algorithm["position_smoothing_poly"],
            "--rotation-window", algorithm["rotation_smoothing_window"],
        ], cwd=locked["orb_root"], dry_run=args.dry_run)
        run([
            sys.executable, runtime["smoother"],
            "--input-csv", raw,
            "--output-csv", strict,
            "--position-window", algorithm["position_smoothing_window"],
            "--position-poly", algorithm["position_smoothing_poly"],
            "--rotation-window", algorithm["rotation_smoothing_window"],
            "--timestamp-offset-sec", offset,
        ], cwd=locked["orb_root"], dry_run=args.dry_run)

        row: dict[str, Any] = {
            "episode": tag,
            "status": "dry_run" if args.dry_run else "trajectory_ok",
            "strict_sync_offset_sec": offset,
            "trajectory_csv": str(strict),
        }
        if gt_root is not None:
            gt = resolve_ground_truth(gt_root, tag)
            run([
                sys.executable, runtime["runner"],
                "--evaluate-only",
                "--tracking-mode", algorithm["tracking_mode"],
                "--eval-pose-frame", "cam0",
                "--episode-dir", episode,
                "--calibration-json", locked["calibration_path"],
                "--camera-key", locked["contract"]["camera_rig"],
                "--orbslam-binary", runtime["executable"],
                "--vocabulary", runtime["vocabulary"],
                "--work-dir", work,
                "--output-csv", strict,
                "--eval-dir", eval_dir,
                "--robot-json", gt,
                "--trajectory-name", name,
                "--t-max-diff", 0.01,
            ], cwd=locked["orb_root"], dry_run=args.dry_run)
            if not args.dry_run:
                summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
                se3 = summary["results"]["se3"]["all"]
                row.update({
                    "status": "ok",
                    "ape_translation_se3_rmse_mm": se3["trans_part"]["rmse"] * 1000,
                    "ape_rotation_se3_rmse_deg": se3["angle_deg"]["rmse"],
                    "pairs": se3["pairs"],
                    "pass_10mm": se3["pass_10mm_rmse"],
                })
                if not args.skip_viewer:
                    viewer = ROOT / "script/visualize/visualize_trajectory_pair.py"
                    run([
                        sys.executable, viewer,
                        "--ref", eval_dir / "ref_tcp.tum",
                        "--est", eval_dir / "est_tcp.tum",
                        "--output-dir", eval_dir / "viewer3d",
                        "--ref-name", "TCP ground truth",
                        "--est-name", "historical colleague ORB stereo",
                        "--title", tag,
                    ])
        rows.append(row)
        print(f"[DONE] {tag} offset={offset:+.6f}", flush=True)

    if args.dry_run:
        print(f"[DRY-RUN] verified {len(rows)} episode(s); no output was written")
        return 0

    fields = sorted({key for row in rows for key in row})
    with (batch / "batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "profile": str(locked["profile_path"]),
        "profile_sha256": locked["profile_sha256"],
        "calibration": str(locked["calibration_path"]),
        "calibration_sha256": locked["calibration_sha256"],
        "offset_json": str(locked["offset_path"]),
        "offset_json_sha256": locked["offset_sha256"],
        "algorithm": algorithm,
        "camera_rig": locked["contract"]["camera_rig"],
        "same_physical_device": locked["contract"]["same_physical_device"],
        "episodes": [episode.name for episode in episodes],
    }
    (batch / "batch_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"[DONE] batch={batch}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
