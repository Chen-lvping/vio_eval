#!/usr/bin/env python3
"""Gate a candidate RM75 batch against accuracy and regression targets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


APE_COLUMNS = ("ape_mm", "ape_translation_se3_rmse_mm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--exclude-json", type=Path, default=None)
    parser.add_argument("--min-good-count", type=int, default=15)
    parser.add_argument("--target-mm", type=float, default=20.0)
    parser.add_argument("--good-mm", type=float, default=10.0)
    parser.add_argument("--good-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--max-regression-mm", type=float, default=1.0)
    return parser.parse_args()


def load(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, float] = {}
    for row in rows:
        episode = row.get("episode", "").strip()
        value = next((row.get(column, "").strip() for column in APE_COLUMNS if row.get(column, "").strip()), "")
        if episode and value and row.get("status", "ok") == "ok":
            result[episode] = float(value)
    return result


def main() -> int:
    args = parse_args()
    baseline = load(args.baseline.expanduser().resolve())
    candidate = load(args.candidate.expanduser().resolve())
    exclusions = {}
    if args.exclude_json is not None:
        exclusions = json.loads(args.exclude_json.expanduser().resolve().read_text(encoding="utf-8"))
    excluded = set(exclusions)
    rows = []
    for episode, baseline_mm in sorted(baseline.items()):
        if episode in excluded:
            continue
        candidate_mm = candidate.get(episode)
        reasons = []
        if candidate_mm is None:
            reasons.append("missing candidate result")
        else:
            if candidate_mm > args.target_mm:
                reasons.append(f"candidate exceeds {args.target_mm:g} mm")
            if baseline_mm <= args.good_mm and candidate_mm > baseline_mm + args.good_tolerance_mm:
                reasons.append("good baseline regressed beyond tolerance")
            if candidate_mm > baseline_mm + args.max_regression_mm:
                reasons.append("candidate regressed beyond global tolerance")
        rows.append(
            {
                "episode": episode,
                "baseline_mm": baseline_mm,
                "candidate_mm": candidate_mm,
                "delta_mm": None if candidate_mm is None else candidate_mm - baseline_mm,
                "passed": not reasons,
                "reasons": reasons,
            }
        )
    extra = sorted(set(candidate).difference(baseline))
    good_count = sum(
        row["candidate_mm"] is not None and row["candidate_mm"] <= args.good_mm for row in rows
    )
    all_rows_passed = bool(rows) and all(row["passed"] for row in rows)
    count_passed = good_count >= args.min_good_count
    payload = {
        "passed": all_rows_passed and count_passed,
        "thresholds": {
            "target_mm": args.target_mm,
            "good_mm": args.good_mm,
            "good_tolerance_mm": args.good_tolerance_mm,
            "max_regression_mm": args.max_regression_mm,
            "min_good_count": args.min_good_count,
        },
        "counts": {
            "evaluated": len(rows),
            "excluded": len(excluded),
            "within_good_mm": good_count,
            "within_target_mm": sum(
                row["candidate_mm"] is not None and row["candidate_mm"] <= args.target_mm for row in rows
            ),
        },
        "excluded_episodes": exclusions,
        "episodes": rows,
        "extra_candidate_episodes": extra,
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    failed = [row for row in rows if not row["passed"]]
    print(
        f"checked={len(rows)} row_passed={len(rows) - len(failed)} row_failed={len(failed)} "
        f"within_{args.good_mm:g}mm={good_count}/{args.min_good_count}"
    )
    for row in failed:
        print(f"FAIL {row['episode']}: {', '.join(row['reasons'])}")
    if not count_passed:
        print(f"FAIL only {good_count} episodes are within {args.good_mm:g} mm; need {args.min_good_count}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
