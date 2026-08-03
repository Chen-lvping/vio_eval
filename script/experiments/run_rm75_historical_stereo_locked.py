#!/usr/bin/env python3
"""Run one fail-closed RM75 historical pure-stereo baseline reproduction."""

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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "data/evaluation/config/rm75_historical_stereo_baseline_locked_20260731.json"
DEFAULT_OUTPUT = ROOT / "data/evaluation/workbench/rm75_historical_stereo_locked_20260731"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", default="main/0004", help="One main/NNNN episode; batch execution is intentionally disabled.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--run", action="store_true", help="Execute after all locks and input gates pass; otherwise only validate and print commands.")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"locked {label} is missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"locked {label} changed: {path}\nexpected={expected}\nactual={actual}")


def verify_runtime(config: dict[str, Any]) -> dict[str, Path]:
    runtime = config["runtime"]
    root = resolve(runtime["root"])
    paths: dict[str, Path] = {"root": root}
    for key in ("executable", "library", "runner", "smoother", "vocabulary"):
        path = root / runtime[key]
        verify_hash(path, runtime[f"{key}_sha256"], key)
        paths[key] = path
    patch = resolve(runtime["anchor_patch"])
    verify_hash(patch, runtime["anchor_patch_sha256"], "anchor patch")
    paths["anchor_patch"] = patch
    return paths


def run_input_audit(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    gate = config["input_gate"]
    command = ["ionice", "-c3", "nice", "-n", "19", sys.executable, str(resolve(gate["audit_script"]))]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in (0, 2):
        raise RuntimeError(f"input audit failed with exit code {completed.returncode}")
    audit_csv = resolve(gate["audit_csv"])
    with audit_csv.open(newline="", encoding="utf-8") as handle:
        return {row["episode"]: row for row in csv.DictReader(handle)}


def run_logged(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN]", " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    episodes = config["episodes"]
    if args.episode not in episodes:
        blocked = config["input_gate"]["blocked_episodes"]
        reason = blocked.get(args.episode, "not present in the locked historical allowlist")
        raise SystemExit(f"refusing {args.episode}: {reason}")
    runtime = verify_runtime(config)
    calibration = resolve(config["calibration"]["path"])
    verify_hash(calibration, config["calibration"]["sha256"], "unified calibration")

    audit = run_input_audit(config)
    row = audit.get(args.episode)
    required_ratio = float(config["input_gate"]["required_timestamp_match_ratio"])
    if row is None or row["status"] != "ready":
        raise SystemExit(f"input gate rejected {args.episode}: {row['reason'] if row else 'missing audit row'}")
    if float(row["historical_camera_timestamp_match_ratio"]) < required_ratio:
        raise SystemExit(f"input gate rejected {args.episode}: historical camera timestamp match below {required_ratio}")

    episode = episodes[args.episode]
    episode_dir = resolve(episode["episode_dir"])
    ground_truth = resolve(episode["ground_truth"])
    for path, label in ((episode_dir / "stereo_right.mkv", "video"), (episode_dir / "fays_data_right.mcap", "MCAP"), (ground_truth, "ground truth")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root.expanduser().resolve() / f"{args.episode.replace('/', '_')}_{stamp}"
    raw = output / "pose_raw.csv"
    strict = output / "pose_smooth_strictsync.csv"
    evaluation = output / "eval"
    trajectory_name = f"historical_locked_{args.episode.replace('/', '_')}_ffba10"
    low_priority = ["ionice", "-c3", "nice", "-n", "19", "taskset", "-c", str(args.cpu)]
    algorithm = config["algorithm"]
    orb_command = low_priority + [
        sys.executable, str(runtime["runner"]), "--offline-accurate", "--tracking-mode", "stereo",
        "--episode-dir", str(episode_dir), "--calibration-json", str(calibration), "--camera-key", "stereo_right",
        "--orbslam-binary", str(runtime["executable"]), "--vocabulary", str(runtime["vocabulary"]),
        "--work-dir", str(output), "--output-csv", str(raw), "--eval-dir", str(output / "eval_raw"),
        "--trajectory-name", trajectory_name, "--gba-iterations", str(algorithm["gba_iterations"]),
        "--full-frame-ba-iterations", str(algorithm["full_frame_ba_iterations"]),
        "--orb-features", str(algorithm["orb_features"]), "--orb-scale-factor", str(algorithm["orb_scale_factor"]),
        "--orb-levels", str(algorithm["orb_levels"]), "--orb-init-fast", str(algorithm["orb_init_fast"]),
        "--orb-min-fast", str(algorithm["orb_min_fast"]), "--overwrite"
    ]
    smooth_command = low_priority + [
        sys.executable, str(runtime["smoother"]), "--input-csv", str(raw), "--output-csv", str(strict),
        "--position-window", str(algorithm["position_window"]), "--position-poly", str(algorithm["position_poly"]),
        "--rotation-window", str(algorithm["rotation_window"]), "--timestamp-offset-sec", str(episode["strict_sync_offset_sec"])
    ]
    eval_command = low_priority + [
        sys.executable, str(runtime["runner"]), "--evaluate-only", "--tracking-mode", "stereo", "--eval-pose-frame", "cam0",
        "--episode-dir", str(episode_dir), "--calibration-json", str(calibration), "--camera-key", "stereo_right",
        "--work-dir", str(output), "--output-csv", str(strict), "--eval-dir", str(evaluation),
        "--robot-json", str(ground_truth), "--trajectory-name", trajectory_name,
        "--t-max-diff", str(algorithm["t_max_diff_sec"])
    ]

    print(f"[LOCKED] runtime={runtime['root']}")
    print(f"[LOCKED] calibration={calibration}")
    print(f"[INPUT] episode={episode_dir}")
    print(f"[INPUT] ground_truth={ground_truth}")
    print(f"[INPUT] historical_camera_timestamp_match_ratio={row['historical_camera_timestamp_match_ratio']}")
    print(f"[REFERENCE] APE={episode['historical_stereo_ape_mm']} mm offset={episode['strict_sync_offset_sec']} s")
    if not args.run:
        for command in (orb_command, smooth_command, eval_command):
            print("[DRY-RUN]", " ".join(command))
        return 0

    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "config": str(config_path), "episode": args.episode, "algorithm": algorithm,
        "runtime_hashes": {key: sha256(runtime[key]) for key in ("executable", "library", "runner", "smoother", "vocabulary")},
        "calibration": {"path": str(calibration), "sha256": sha256(calibration)},
        "inputs": {
            "video": {"path": str(episode_dir / "stereo_right.mkv"), "sha256": sha256(episode_dir / "stereo_right.mkv")},
            "mcap": {"path": str(episode_dir / "fays_data_right.mcap"), "sha256": sha256(episode_dir / "fays_data_right.mcap")},
            "ground_truth": {"path": str(ground_truth), "sha256": sha256(ground_truth)},
            "audit_row": row
        },
        "commands": [orb_command, smooth_command, eval_command]
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    run_logged(orb_command, output / "logs/orbslam3.log")
    run_logged(smooth_command, output / "logs/smooth.log")
    run_logged(eval_command, output / "logs/evaluate.log")
    summary = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
    se3 = summary["results"]["se3"]["all"]
    ape_mm = float(se3["trans_part"]["rmse"]) * 1000.0
    acceptance = float(episode.get("gate_acceptance_ape_mm", float(episode["historical_stereo_ape_mm"]) + 0.75))
    result = {
        "episode": args.episode, "ape_translation_se3_rmse_mm": ape_mm,
        "ape_rotation_se3_rmse_deg": se3["angle_deg"]["rmse"], "pairs": se3["pairs"],
        "historical_stereo_ape_mm": episode["historical_stereo_ape_mm"],
        "acceptance_ape_mm": acceptance, "accepted": ape_mm <= acceptance
    }
    (output / "locked_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("[RESULT]", json.dumps(result, ensure_ascii=True), flush=True)
    return 0 if result["accepted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
