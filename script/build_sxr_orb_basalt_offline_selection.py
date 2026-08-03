#!/usr/bin/env python3
"""Create an offline ORB/Basalt evaluation selection manifest.

The manifest is for reporting and post-processing only. It never reads an
episode's head_pose and must not be used as an online algorithm switch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orb-summary", type=Path, required=True)
    parser.add_argument("--basalt-summary", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [row for row in payload if isinstance(row, dict) and isinstance(row.get("episode"), str)]


def valid(row: dict[str, Any]) -> bool:
    gate = row.get("scale_gate")
    return row.get("status") in {"ok", "reused"} and isinstance(gate, dict) and gate.get("passed") is True and isinstance(row.get("ape_rmse_m"), (int, float))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    orb = {row["episode"]: row for row in load_rows(args.orb_summary)}
    basalt: dict[str, dict[str, Any]] = {}
    for path in args.basalt_summary:
        basalt.update({row["episode"]: row for row in load_rows(path)})
    episodes = sorted(set(orb) | set(basalt))
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        orb_row, basalt_row = orb.get(episode), basalt.get(episode)
        orb_valid = orb_row is not None and valid(orb_row)
        basalt_valid = basalt_row is not None and valid(basalt_row)
        if orb_valid and basalt_valid:
            selected = "orb" if orb_row["ape_rmse_m"] <= basalt_row["ape_rmse_m"] else "basalt"
            reason = "lower_offline_ape_after_raw_scale_gate"
        elif orb_valid:
            selected, reason = "orb", "only_orb_passed_raw_scale_gate"
        elif basalt_valid:
            selected, reason = "basalt", "orb_unavailable_or_scale_rejected"
        else:
            selected, reason = "none", "neither_estimate_passed_raw_scale_gate"
        rows.append({
            "episode": episode,
            "selected_for_offline_reporting": selected,
            "reason": reason,
            "orb_ape_rmse_m": None if orb_row is None else orb_row.get("ape_rmse_m"),
            "orb_status": None if orb_row is None else orb_row.get("status"),
            "orb_scale_gate": None if orb_row is None else orb_row.get("scale_gate"),
            "basalt_ape_rmse_m": None if basalt_row is None else basalt_row.get("ape_rmse_m"),
            "basalt_status": None if basalt_row is None else basalt_row.get("status"),
            "basalt_scale_gate": None if basalt_row is None else basalt_row.get("scale_gate"),
        })
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "purpose": "offline evaluation/reporting selection only",
        "prohibited_online_use": "Do not select an online algorithm from APE or head_pose; both are post-run metrics.",
        "selection_rule": "Choose the lower post-run APE only among raw-scale-valid estimates.",
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
