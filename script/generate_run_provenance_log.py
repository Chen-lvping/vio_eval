#!/usr/bin/env python3
"""Generate reproducible provenance logs for ORB-SLAM3 RM75 evaluation batches."""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def normalize_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    return value


def namespace_to_dict(namespace: argparse.Namespace | Mapping[str, object]) -> dict[str, object]:
    if isinstance(namespace, argparse.Namespace):
        payload = vars(namespace)
    else:
        payload = dict(namespace)
    return {str(key): normalize_value(value) for key, value in payload.items()}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_or_none(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metric(value: object) -> str:
    number = numeric_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.6f}"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_file(path: Path, snapshot_dir: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {
            "source": str(resolved),
            "exists": False,
        }
    digest = file_sha256(resolved)
    target = snapshot_dir / f"{resolved.stem}_{digest[:12]}{resolved.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(resolved, target)
    return {
        "source": str(resolved),
        "snapshot": str(target),
        "exists": True,
        "sha256": digest,
        "size_bytes": resolved.stat().st_size,
    }


def git_info_for_path(path_like: str | Path | None) -> dict[str, object]:
    if not path_like:
        return {"available": False, "reason": "empty_path"}
    path = Path(path_like).expanduser().resolve()
    target = path if path.is_dir() else path.parent
    try:
        root = (
            subprocess.run(
                ["git", "-C", str(target), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
        commit = (
            subprocess.run(
                ["git", "-C", root, "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
        branch = (
            subprocess.run(
                ["git", "-C", root, "branch", "--show-current"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
        status = (
            subprocess.run(
                ["git", "-C", root, "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
        return {
            "available": True,
            "repo_root": root,
            "branch": branch,
            "commit": commit,
            "dirty": bool(status),
            "status_short": status.splitlines()[:20],
        }
    except Exception as exc:
        return {
            "available": False,
            "path": str(target),
            "reason": str(exc),
        }


def collect_manifest_records(batch_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for manifest_path in sorted(batch_root.rglob("orbslam3_tcp_eval_manifest.json")):
        try:
            data = load_json(manifest_path)
        except Exception as exc:
            records.append(
                {
                    "path": str(manifest_path),
                    "load_error": str(exc),
                }
            )
            continue
        records.append(
            {
                "path": str(manifest_path),
                "episode": Path(str(data.get("episode_dir", ""))).name,
                "eval_dir": str(data.get("eval_dir", "")),
                "output_dir": str(data.get("output_dir", "")),
                "requested_mode": data.get("requested_mode", ""),
                "mode": data.get("mode", ""),
                "feature_preset": data.get("feature_preset", ""),
                "imu_fast_init": data.get("imu_fast_init", ""),
                "strict_sync_offset_sec": data.get("strict_sync_offset_sec", ""),
                "fallback_to_stereo": data.get("fallback_to_stereo", False),
                "fallback_reason": data.get("fallback_reason", ""),
                "settings_yaml": data.get("settings_yaml", ""),
                "trajectory": data.get("trajectory", ""),
                "estimate_csv_for_eval": data.get("estimate_csv_for_eval", ""),
                "orb_native_log": data.get("orb_native_log", ""),
                "summary_rmse": data.get("summary_rmse", {}),
                "commands": data.get("commands", {}),
                "manifest": data,
            }
        )
    return records


def manifest_lookup(records: Sequence[dict[str, object]]) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    by_eval_dir: dict[str, dict[str, object]] = {}
    by_episode: dict[str, dict[str, object]] = {}
    for record in records:
        eval_dir = str(record.get("eval_dir", "")).strip()
        episode = str(record.get("episode", "")).strip()
        if eval_dir:
            by_eval_dir[eval_dir] = record
        if episode and episode not in by_episode:
            by_episode[episode] = record
    return by_eval_dir, by_episode


def attach_manifest_context(
    rows: Sequence[Mapping[str, object]],
    manifest_by_eval_dir: Mapping[str, dict[str, object]],
    manifest_by_episode: Mapping[str, dict[str, object]],
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        item = {str(key): normalize_value(value) for key, value in row.items()}
        manifest = None
        manifest_path = str(item.get("single_run_manifest", "")).strip()
        eval_dir = str(item.get("eval_dir", "")).strip()
        episode = str(item.get("episode", "")).strip()
        if manifest_path and Path(manifest_path).is_file():
            try:
                data = load_json(Path(manifest_path))
                manifest = {
                    "path": manifest_path,
                    "episode": episode or Path(str(data.get("episode_dir", ""))).name,
                    "eval_dir": str(data.get("eval_dir", "")),
                    "output_dir": str(data.get("output_dir", "")),
                    "requested_mode": data.get("requested_mode", ""),
                    "mode": data.get("mode", ""),
                    "feature_preset": data.get("feature_preset", ""),
                    "imu_fast_init": data.get("imu_fast_init", ""),
                    "strict_sync_offset_sec": data.get("strict_sync_offset_sec", ""),
                    "fallback_to_stereo": data.get("fallback_to_stereo", False),
                    "fallback_reason": data.get("fallback_reason", ""),
                    "settings_yaml": data.get("settings_yaml", ""),
                    "trajectory": data.get("trajectory", ""),
                    "estimate_csv_for_eval": data.get("estimate_csv_for_eval", ""),
                    "orb_native_log": data.get("orb_native_log", ""),
                    "summary_rmse": data.get("summary_rmse", {}),
                    "commands": data.get("commands", {}),
                }
            except Exception as exc:
                manifest = {"path": manifest_path, "load_error": str(exc)}
        elif eval_dir and eval_dir in manifest_by_eval_dir:
            manifest = dict(manifest_by_eval_dir[eval_dir])
        elif episode and episode in manifest_by_episode:
            manifest = dict(manifest_by_episode[episode])
        item["manifest_context"] = manifest or {}
        enriched.append(item)
    return enriched


def build_metrics_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ok_rows = [row for row in rows if str(row.get("status", "")).lower() == "ok"]
    failed_rows = [row for row in rows if str(row.get("status", "")).lower() != "ok"]
    metric_keys = (
        "ape_translation_se3_rmse_mm",
        "rpe_translation_5cm_rmse_mm",
        "ape_rotation_se3_rmse_deg",
        "rpe_rotation_5cm_rmse_deg",
    )
    metrics: dict[str, object] = {}
    for key in metric_keys:
        values = [numeric_or_none(row.get(key)) for row in ok_rows]
        values = [value for value in values if value is not None]
        if values:
            metrics[key] = {
                "count": len(values),
                "min": min(values),
                "median": statistics.median(values),
                "mean": statistics.fmean(values),
                "max": max(values),
            }
    best_episode = None
    if ok_rows:
        ranked = [
            (numeric_or_none(row.get("ape_translation_se3_rmse_mm")), str(row.get("episode", "")))
            for row in ok_rows
        ]
        ranked = [(metric, episode) for metric, episode in ranked if metric is not None]
        if ranked:
            metric, episode = min(ranked, key=lambda item: item[0])
            best_episode = {"episode": episode, "ape_translation_se3_rmse_mm": metric}
    return {
        "total": len(rows),
        "ok": len(ok_rows),
        "failed": len(failed_rows),
        "failed_episodes": [str(row.get("episode", "")) for row in failed_rows],
        "best_episode_by_ape": best_episode,
        "metrics": metrics,
    }


def write_markdown_log(
    path: Path,
    *,
    label: str,
    batch_root: Path,
    batch_args: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    summary_csv: Path | None,
    report_md: Path | None,
    provenance_json: Path,
    snapshots: Sequence[Mapping[str, object]],
    repo_state: Mapping[str, object],
    generated_at: str,
) -> None:
    lines = [
        f"# {label} Run Log",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Batch root: `{batch_root}`",
        f"- Summary CSV: `{summary_csv}`" if summary_csv else "- Summary CSV: ``",
        f"- Report MD: `{report_md}`" if report_md else "- Report MD: ``",
        f"- Provenance JSON: `{provenance_json}`",
        "",
        "## Invocation",
        "",
        f"- Host: `{repo_state['runtime']['host']}`",
        f"- User: `{repo_state['runtime']['user']}`",
        f"- CWD: `{repo_state['runtime']['cwd']}`",
        f"- Python: `{repo_state['runtime']['python']}`",
        f"- Source script: `{repo_state['runtime']['source_script']}`",
        f"- CLI args: `{json.dumps(repo_state['runtime']['argv'], ensure_ascii=False)}`",
        "",
        "## Git State",
        "",
    ]
    vio_eval_git = repo_state["git"].get("vio_eval", {})
    if vio_eval_git.get("available"):
        dirty = "dirty" if vio_eval_git.get("dirty") else "clean"
        lines.append(
            f"- vio_eval: `{vio_eval_git.get('repo_root')}` @ `{vio_eval_git.get('commit')}` "
            f"({vio_eval_git.get('branch') or 'detached'}, {dirty})"
        )
    else:
        lines.append(f"- vio_eval: unavailable (`{vio_eval_git.get('reason', 'unknown')}`)")
    for index, payload in enumerate(repo_state["git"].get("orb_roots", []), start=1):
        if payload.get("available"):
            dirty = "dirty" if payload.get("dirty") else "clean"
            lines.append(
                f"- orb_root[{index}]: `{payload.get('repo_root')}` @ `{payload.get('commit')}` "
                f"({payload.get('branch') or 'detached'}, {dirty})"
            )
        else:
            lines.append(f"- orb_root[{index}]: unavailable (`{payload.get('reason', 'unknown')}`)")
    lines.extend(
        [
            "",
            "## Batch Args",
            "",
        ]
    )
    for key in sorted(batch_args):
        lines.append(f"- {key}: `{batch_args[key]}`")
    lines.extend(
        [
            "",
            "## Snapshots",
            "",
        ]
    )
    if snapshots:
        for item in snapshots:
            if item.get("exists"):
                lines.append(
                    f"- `{item['source']}` -> `{item['snapshot']}` "
                    f"(sha256 `{item['sha256'][:12]}`, {item['size_bytes']} bytes)"
                )
            else:
                lines.append(f"- `{item['source']}` (missing)")
    else:
        lines.append("- No external input files were snapshotted.")
    summary = build_metrics_summary(rows)
    lines.extend(
        [
            "",
            "## Result Summary",
            "",
            f"- Total episodes: `{summary['total']}`",
            f"- OK: `{summary['ok']}`",
            f"- Failed: `{summary['failed']}`",
        ]
    )
    best = summary.get("best_episode_by_ape")
    if best:
        lines.append(
            f"- Best APE episode: `{best['episode']}` "
            f"(`{best['ape_translation_se3_rmse_mm']:.6f}` mm)"
        )
    lines.extend(
        [
            "",
            "## Episodes",
            "",
            "| episode | status | APE mm | RPE mm | mode | fallback | strict sync | manifest | viewer |",
            "| --- | --- | ---: | ---: | --- | --- | ---: | --- | --- |",
        ]
    )
    for row in rows:
        manifest = row.get("manifest_context", {})
        lines.append(
            f"| {row.get('episode', '')} | {row.get('status', '')} | "
            f"{format_metric(row.get('ape_translation_se3_rmse_mm'))} | "
            f"{format_metric(row.get('rpe_translation_5cm_rmse_mm'))} | "
            f"{manifest.get('mode', row.get('mode_used', ''))} | "
            f"{manifest.get('fallback_reason', row.get('fallback_reason', ''))} | "
            f"{format_metric(row.get('strict_sync_offset_sec'))} | "
            f"`{row.get('single_run_manifest', manifest.get('path', ''))}` | "
            f"`{row.get('viewer_html', '')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_batch_run_provenance(
    batch_root: Path,
    *,
    label: str = "ORB-SLAM3 RM75 Batch Evaluation",
    batch_args: Mapping[str, object] | argparse.Namespace | None = None,
    rows: Sequence[Mapping[str, object]] | None = None,
    summary_csv: Path | None = None,
    report_md: Path | None = None,
    source_script: Path | None = None,
    argv: Sequence[str] | None = None,
    copy_files: Iterable[str | Path] | None = None,
) -> dict[str, str]:
    batch_root = batch_root.expanduser().resolve()
    summary_csv = summary_csv.expanduser().resolve() if summary_csv else batch_root / "batch_summary.csv"
    report_md = report_md.expanduser().resolve() if report_md else batch_root / "REPORT.md"
    if rows is None:
        rows = load_csv_rows(summary_csv) if summary_csv.is_file() else []
    normalized_args = namespace_to_dict(batch_args or {})
    manifest_records = collect_manifest_records(batch_root)
    manifest_by_eval_dir, manifest_by_episode = manifest_lookup(manifest_records)
    enriched_rows = attach_manifest_context(rows, manifest_by_eval_dir, manifest_by_episode)

    snapshot_dir = batch_root / "run_inputs_snapshot"
    snapshots: list[dict[str, object]] = []
    seen_sources: set[str] = set()
    for item in copy_files or []:
        path = Path(item).expanduser().resolve()
        source = str(path)
        if source in seen_sources:
            continue
        seen_sources.add(source)
        snapshots.append(snapshot_file(path, snapshot_dir))

    unique_orb_roots = sorted(
        {
            str(record.get("manifest", {}).get("orb_root", "")).strip()
            for record in manifest_records
            if record.get("manifest")
        }
        - {""}
    )
    generated_at = datetime.now().isoformat(timespec="seconds")
    repo_state = {
        "runtime": {
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "cwd": os.getcwd(),
            "python": sys.executable,
            "source_script": str(source_script.expanduser().resolve()) if source_script else "",
            "argv": list(argv or []),
        },
        "git": {
            "vio_eval": git_info_for_path(batch_root),
            "orb_roots": [git_info_for_path(path) for path in unique_orb_roots],
        },
    }
    payload = {
        "label": label,
        "generated_at": generated_at,
        "batch_root": str(batch_root),
        "summary_csv": str(summary_csv),
        "report_md": str(report_md),
        "batch_args": normalized_args,
        "snapshots": snapshots,
        "repo_state": repo_state,
        "summary": build_metrics_summary(enriched_rows),
        "episodes": enriched_rows,
        "manifest_index": [
            {
                key: value
                for key, value in record.items()
                if key != "manifest"
            }
            for record in manifest_records
        ],
    }
    provenance_json = batch_root / "batch_provenance.json"
    provenance_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_log(
        batch_root / "RUN_LOG.md",
        label=label,
        batch_root=batch_root,
        batch_args=normalized_args,
        rows=enriched_rows,
        summary_csv=summary_csv if summary_csv.is_file() else None,
        report_md=report_md if report_md.is_file() else None,
        provenance_json=provenance_json,
        snapshots=snapshots,
        repo_state=repo_state,
        generated_at=generated_at,
    )
    return {
        "provenance_json": str(provenance_json),
        "run_log_md": str(batch_root / "RUN_LOG.md"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--label", default="ORB-SLAM3 RM75 Batch Evaluation")
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--report-md", type=Path, default=None)
    parser.add_argument("--args-json", type=Path, default=None, help="Optional JSON file with batch invocation args.")
    parser.add_argument("--copy-file", type=Path, action="append", default=[], help="Extra files to snapshot into the batch root.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_args: dict[str, object] = {}
    if args.args_json is not None:
        batch_args = load_json(args.args_json.expanduser().resolve())
    outputs = write_batch_run_provenance(
        args.batch_root,
        label=args.label,
        batch_args=batch_args,
        summary_csv=args.summary_csv,
        report_md=args.report_md,
        source_script=Path(__file__),
        argv=sys.argv[1:],
        copy_files=args.copy_file,
    )
    print(f"[OK] wrote {outputs['provenance_json']}")
    print(f"[OK] wrote {outputs['run_log_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
