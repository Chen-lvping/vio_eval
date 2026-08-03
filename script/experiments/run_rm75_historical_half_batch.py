#!/usr/bin/env python3
"""Run the first half of the curated RM75 set with the locked stereo baseline."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CURATED_CSV = (
    ROOT
    / "data/evaluation/curated/rm75_main_clean_ape_le_50mm_20260729/episodes.csv"
)
LOCKED_RUNNER = ROOT / "script/run_orbslam3_rm75_best_batch.py"
LOCKED_PROFILE = (
    ROOT / "data/evaluation/config/rm75_historical_stereo_baseline_20260716.json"
)
OUTPUT_ROOT = (
    ROOT / "data/evaluation/workbench/rm75_historical_stereo_half_a_20260730"
)
SELECTED_COUNT = 9


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "episode",
        "dataset",
        "status",
        "historical_stereo_mm",
        "ape_translation_se3_rmse_mm",
        "ape_rotation_se3_rmse_deg",
        "pairs",
        "result_dir",
        "log",
        "returncode",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_curated_rows() -> list[dict[str, str]]:
    with CURATED_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:SELECTED_COUNT]
    if len(rows) != SELECTED_COUNT:
        raise RuntimeError(f"expected {SELECTED_COUNT} curated rows, found {len(rows)}")
    if any(row["episode"] == "main/0005" for row in rows):
        raise RuntimeError("explicitly excluded main/0005 must not enter the queue")
    return rows


def find_result_dir(episode_output: Path) -> Path | None:
    matches = sorted(
        episode_output.glob("orbslam3_rm75_historical_stereo_20260716_*"),
        key=lambda path: path.stat().st_mtime,
    )
    return matches[-1] if matches else None


def load_child_summary(result_dir: Path) -> dict[str, str]:
    path = result_dir / "batch_summary.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle), {})


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logs_dir = OUTPUT_ROOT / "logs"
    results_dir = OUTPUT_ROOT / "results"
    logs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    selected = read_curated_rows()
    started_at = datetime.now().astimezone().isoformat()
    provenance = {
        "started_at": started_at,
        "pid": os.getpid(),
        "selection": "first 9 rows of curated clean-18 CSV",
        "curated_csv": str(CURATED_CSV),
        "locked_runner": str(LOCKED_RUNNER),
        "locked_profile": str(LOCKED_PROFILE),
        "resource_policy": "serial, inherited taskset CPU 0, nice 19, ionice idle",
        "episodes": [row["episode"] for row in selected],
    }
    write_json(OUTPUT_ROOT / "queue_provenance.json", provenance)
    print(f"[QUEUE] pid={os.getpid()} episodes={len(selected)} output={OUTPUT_ROOT}", flush=True)

    completed: list[dict[str, object]] = []
    status_path = OUTPUT_ROOT / "status.json"
    summary_path = OUTPUT_ROOT / "batch_summary.csv"

    for index, source in enumerate(selected, start=1):
        label = source["episode"].replace("/", "_")
        episode = Path(source["raw_dataset"]).resolve()
        ground_truth = Path(source["ground_truth"]).resolve()
        episode_output = results_dir / label
        episode_output.mkdir(exist_ok=True)
        log_path = logs_dir / f"{label}.log"

        status = {
            "state": "running",
            "pid": os.getpid(),
            "started_at": started_at,
            "updated_at": datetime.now().astimezone().isoformat(),
            "current_index": index,
            "total": len(selected),
            "current_episode": source["episode"],
            "completed": completed,
        }
        write_json(status_path, status)
        print(f"[START] {index}/{len(selected)} {source['episode']} dataset={episode}", flush=True)

        cmd = [
            sys.executable,
            str(LOCKED_RUNNER),
            "--episode-root",
            str(episode.parent),
            "--gt-root",
            str(ground_truth.parent),
            "--episode-pattern",
            episode.name,
            "--output-root",
            str(episode_output),
            "--config",
            str(LOCKED_PROFILE),
            "--skip-viewer",
        ]
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            log.write("$ " + " ".join(cmd) + "\n\n")
            log.flush()
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        result_dir = find_result_dir(episode_output)
        child = load_child_summary(result_dir) if result_dir else {}
        row: dict[str, object] = {
            "episode": source["episode"],
            "dataset": str(episode),
            "status": "ok" if result.returncode == 0 and child.get("status") == "ok" else "failed",
            "historical_stereo_mm": source["stereo_mm"],
            "ape_translation_se3_rmse_mm": child.get("ape_translation_se3_rmse_mm", ""),
            "ape_rotation_se3_rmse_deg": child.get("ape_rotation_se3_rmse_deg", ""),
            "pairs": child.get("pairs", ""),
            "result_dir": str(result_dir) if result_dir else "",
            "log": str(log_path),
            "returncode": result.returncode,
        }
        completed.append(row)
        print(
            f"[DONE] {index}/{len(selected)} {source['episode']} "
            f"status={row['status']} ape_mm={row['ape_translation_se3_rmse_mm']}",
            flush=True,
        )
        write_summary(summary_path, completed)
        write_json(
            status_path,
            {
                **status,
                "state": "running" if index < len(selected) else "complete",
                "updated_at": datetime.now().astimezone().isoformat(),
                "current_episode": None if index == len(selected) else selected[index]["episode"],
                "completed": completed,
            },
        )
        time.sleep(2)

    failed = sum(row["status"] != "ok" for row in completed)
    print(f"[QUEUE_DONE] total={len(completed)} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
