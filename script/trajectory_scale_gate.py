#!/usr/bin/env python3
"""Reject trajectory comparisons whose raw metric scale is clearly invalid."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ScaleGateReport:
    reference_poses: int
    estimate_poses: int
    reference_path_length_m: float
    estimate_path_length_m: float
    path_length_ratio: float
    min_ratio: float
    max_ratio: float
    passed: bool
    reason: str = "ok"

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def tum_positions(path: Path) -> np.ndarray:
    positions: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) != 8:
                raise ValueError(f"{path}:{line_number}: expected 8 TUM fields")
            values = tuple(float(value) for value in fields)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}:{line_number}: non-finite TUM value")
            positions.append(values[1:4])
    if len(positions) < 3:
        raise ValueError(f"{path}: expected at least three poses")
    return np.asarray(positions, dtype=np.float64)


def path_length(positions: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def assess_scale(
    reference: Path,
    estimate: Path,
    *,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
) -> ScaleGateReport:
    if not 0 < min_ratio <= max_ratio:
        raise ValueError("scale ratio bounds must satisfy 0 < min_ratio <= max_ratio")
    reference_positions = tum_positions(reference)
    estimate_positions = tum_positions(estimate)
    reference_length = path_length(reference_positions)
    estimate_length = path_length(estimate_positions)
    if reference_length <= 1e-9:
        return ScaleGateReport(
            reference_poses=len(reference_positions),
            estimate_poses=len(estimate_positions),
            reference_path_length_m=reference_length,
            estimate_path_length_m=estimate_length,
            path_length_ratio=0.0,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            passed=False,
            reason="reference_path_too_short",
        )
    ratio = estimate_length / reference_length
    return ScaleGateReport(
        reference_poses=len(reference_positions),
        estimate_poses=len(estimate_positions),
        reference_path_length_m=reference_length,
        estimate_path_length_m=estimate_length,
        path_length_ratio=ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        passed=min_ratio <= ratio <= max_ratio,
        reason="ok" if min_ratio <= ratio <= max_ratio else "path_length_ratio_out_of_range",
    )


def require_reasonable_scale(
    reference: Path,
    estimate: Path,
    report_path: Path,
    *,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
) -> ScaleGateReport:
    report = assess_scale(
        reference,
        estimate,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.write(report_path)
    if not report.passed:
        raise RuntimeError(
            f"scale gate rejected alignment ({report.reason}): "
            f"estimate/reference path ratio={report.path_length_ratio:.4g}, "
            f"allowed=[{min_ratio:.4g}, {max_ratio:.4g}]"
        )
    return report
