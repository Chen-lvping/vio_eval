#!/usr/bin/env python3
"""Batch-check mounted EGO episodes and persist summaries on the local PC."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKER = REPO_ROOT / "script" / "check_episode_data.py"
DEFAULT_RESULTS_DIR = REPO_ROOT / "ego_fleet_results"
REGISTRY_NAME = "registry.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="发现当前挂载的 EGO，批量检查 episode，并把日志保存在本地电脑。"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("scan", "summary", "devices"),
        default="scan",
        help="scan=检查当前设备；summary=查看全部历史结果；devices=查看已登记设备。",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--checker", type=Path, default=DEFAULT_CHECKER)
    parser.add_argument("--label", help="给当前唯一挂载设备指定名称，例如 EGO-03。")
    parser.add_argument("--force", action="store_true", help="即使文件未变化也重新检查。")
    parser.add_argument("--dry-run", action="store_true", help="只列出设备和 episode，不读取数据。")
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        metavar="GLOB",
        help="只检查匹配的 episode，可重复，例如 --episode '*_0002'。",
    )
    parser.add_argument(
        "--expected",
        action="append",
        default=[],
        metavar="GLOB=METERS",
        help="可选距离验收映射，例如 --expected '*_0001=0.60'。默认只报告距离。",
    )
    return parser.parse_args(argv)


def empty_registry() -> dict[str, Any]:
    return {"version": 1, "updated_at": None, "devices": {}}


def load_registry(results_dir: Path) -> dict[str, Any]:
    path = results_dir / REGISTRY_NAME
    if not path.is_file():
        return empty_registry()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取本地登记表 {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("devices"), dict):
        raise RuntimeError(f"本地登记表格式错误: {path}")
    return data


def save_registry(results_dir: Path, registry: dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    registry["updated_at"] = now_iso()
    path = results_dir / REGISTRY_NAME
    temporary = results_dir / f".{REGISTRY_NAME}.tmp"
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_local_results_dir(path: Path) -> Path:
    """Reject MTP/GVFS destinations so scans never write back to the device."""

    resolved = path.expanduser().resolve()
    text = str(resolved)
    if (text.startswith("/run/user/") and "/gvfs/" in text) or any(
        part.startswith("mtp:host=") for part in resolved.parts
    ):
        raise ValueError("结果目录不能位于 MTP/GVFS 设备；请保存到电脑本地。")
    return resolved


def discover_devices() -> list[dict[str, Any]]:
    gvfs_root = Path(f"/run/user/{os.getuid()}/gvfs")
    discovered: list[dict[str, Any]] = []
    if not gvfs_root.is_dir():
        return discovered
    for mount in sorted(gvfs_root.glob("mtp:host=*")):
        host = mount.name.removeprefix("mtp:host=")
        try:
            storage_roots = sorted(path for path in mount.iterdir() if path.is_dir())
        except OSError:
            continue
        for storage in storage_roots:
            files_root = storage / "Android" / "data" / "com.ssnwt.helloxr" / "files"
            if not files_root.is_dir():
                continue
            try:
                app_roots = sorted(path for path in files_root.iterdir() if path.is_dir())
            except OSError:
                continue
            for app_root in app_roots:
                dataset_root = app_root / "data" / "dataset"
                if dataset_root.is_dir():
                    discovered.append(
                        {
                            "device_key": app_root.name,
                            "device_id": app_root.name,
                            "mtp_host": host,
                            "mount_path": str(mount),
                            "dataset_root": dataset_root,
                        }
                    )
    unique: dict[str, dict[str, Any]] = {}
    for item in discovered:
        key = str(item["device_key"])
        if key in unique and unique[key]["mtp_host"] != item["mtp_host"]:
            key = f"{key}@{item['mtp_host']}"
            item["device_key"] = key
        unique[key] = item
    return list(unique.values())


def resolve_episode_data_dir(episode_dir: Path) -> Path | None:
    candidates = (episode_dir, episode_dir / "ego")
    for candidate in candidates:
        if (candidate / "rgb.mp4").is_file() and (candidate / "sensor.mcap").is_file():
            return candidate
    return None


def discover_episodes(dataset_root: Path, patterns: Sequence[str]) -> list[tuple[Path, Path]]:
    episodes: list[tuple[Path, Path]] = []
    try:
        children = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    except OSError:
        return episodes
    for episode in children:
        if not episode.name.startswith("episode_"):
            continue
        if patterns and not any(fnmatch.fnmatch(episode.name, pattern) for pattern in patterns):
            continue
        data_dir = resolve_episode_data_dir(episode)
        if data_dir is not None:
            episodes.append((episode, data_dir))
    return episodes


def file_fingerprint(data_dir: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for name in ("rgb.mp4", "sensor.mcap"):
        stat = (data_dir / name).stat()
        values[f"{name}_size"] = int(stat.st_size)
        values[f"{name}_mtime_ns"] = int(stat.st_mtime_ns)
    return values


def next_alias(registry: dict[str, Any]) -> str:
    used: set[int] = set()
    for device in registry["devices"].values():
        match = re.fullmatch(r"EGO-(\d+)", str(device.get("alias", "")), re.IGNORECASE)
        if match:
            used.add(int(match.group(1)))
    number = 1
    while number in used:
        number += 1
    return f"EGO-{number:02d}"


def parse_expected_rules(values: Sequence[str]) -> list[tuple[str, float]]:
    rules: list[tuple[str, float]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--expected 格式应为 GLOB=METERS，收到: {value}")
        pattern, raw_distance = value.rsplit("=", 1)
        distance = float(raw_distance)
        if not pattern or distance <= 0:
            raise ValueError(f"无效的 --expected: {value}")
        rules.append((pattern, distance))
    return rules


def expected_for_episode(name: str, rules: Sequence[tuple[str, float]]) -> float | None:
    for pattern, distance in rules:
        if fnmatch.fnmatch(name, pattern):
            return distance
    return None


def first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else None


def first_text(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).strip() if match else None


def frequency_from_line(text: str, marker: str) -> float | None:
    for line in text.splitlines():
        if marker not in line:
            continue
        for field in ("effective_freq", "nominal_freq", "stream_freq"):
            match = re.search(rf"{field}=([0-9.eE+-]+)\s*Hz", line)
            if match:
                return float(match.group(1))
    return None


def parse_check_log(text: str, return_code: int) -> dict[str, Any]:
    distance = first_float(r"euclidean_distance=([0-9.eE+-]+)", text)
    summary = first_text(r"^\[SUMMARY\]\s+(PASS|FAIL)", text)
    acceptance = first_text(r"distance_acceptance=(PASS|FAIL|REPORT_ONLY)", text)
    start_position = first_text(r"^\[INFO\]\s+start:.*?position=(\[[^\]]+\])", text)
    end_position = first_text(r"^\[INFO\]\s+end:\s+.*?position=(\[[^\]]+\])", text)
    relative_euler = first_text(
        r"relative_euler_xyz_deg\(roll,pitch,yaw\)=(\[[^\]]+\])", text
    )
    return {
        "status": summary or ("PASS" if return_code == 0 else "FAIL"),
        "return_code": return_code,
        "distance_m": distance,
        "distance_cm": round(distance * 100.0, 3) if distance is not None else None,
        "distance_acceptance": acceptance,
        "start_position": start_position,
        "end_position": end_position,
        "relative_euler_xyz_deg": relative_euler,
        "rgb_hz": frequency_from_line(text, "rgb.mp4 可读"),
        "head_pose_hz": frequency_from_line(text, "head_pose: count="),
        "hand_tracking_hz": frequency_from_line(text, "hand_tracking: count="),
        "hand_tracking_present": "存在 hand_tracking topic" in text,
    }


def run_episode_check(
    checker: Path,
    episode_dir: Path,
    log_file: Path,
    expected_distance: float | None,
) -> dict[str, Any]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(checker),
        "--episode-dir",
        str(episode_dir),
        "--log-file",
        str(log_file),
        "--require-hand-tracking",
    ]
    if expected_distance is not None:
        command.extend(("--expected-distance", str(expected_distance)))
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if log_file.is_file():
        text = log_file.read_text(encoding="utf-8", errors="replace")
    else:
        text = completed.stdout
        log_file.write_text(text, encoding="utf-8")
    parsed = parse_check_log(text, completed.returncode)
    parsed["command"] = command
    return parsed


def all_results(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    devices = sorted(registry["devices"].values(), key=lambda item: str(item.get("alias", "")))
    for device in devices:
        for episode_name, result in sorted(device.get("episodes", {}).items()):
            rows.append(
                {
                    "alias": device.get("alias"),
                    "device_id": device.get("device_id"),
                    "episode": episode_name,
                    **result,
                }
            )
    return rows


def write_summaries(results_dir: Path, registry: dict[str, Any]) -> None:
    rows = all_results(registry)
    columns = [
        "alias",
        "device_id",
        "episode",
        "status",
        "distance_m",
        "distance_cm",
        "expected_distance_m",
        "distance_acceptance",
        "rgb_hz",
        "head_pose_hz",
        "hand_tracking_hz",
        "hand_tracking_present",
        "relative_euler_xyz_deg",
        "checked_at",
        "log_file",
        "source_episode",
        "source_deleted_at",
    ]
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "latest_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# EGO 数据检查汇总",
        "",
        f"更新时间：{registry.get('updated_at') or now_iso()}",
        "",
        "| 设备 | Episode | 首尾距离(m) | 首尾距离(cm) | 结果 | hand_tracking | 源数据 | 日志 |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        distance_m = "-" if row.get("distance_m") is None else f"{float(row['distance_m']):.6f}"
        distance_cm = "-" if row.get("distance_cm") is None else f"{float(row['distance_cm']):.2f}"
        hand = "有" if row.get("hand_tracking_present") else "无"
        source_state = "已删除" if row.get("source_deleted_at") else "设备中"
        lines.append(
            f"| {row.get('alias')} | {row.get('episode')} | {distance_m} | {distance_cm} | "
            f"{row.get('status')} | {hand} | {source_state} | `{row.get('log_file')}` |"
        )
    (results_dir / "latest_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(registry: dict[str, Any], results_dir: Path) -> None:
    rows = all_results(registry)
    if not rows:
        print("尚无本地 EGO 检查记录。")
        return
    print("设备     episode                    首尾距离(m)  首尾距离(cm)  结果   hand")
    print("-" * 82)
    for row in rows:
        distance_m = "-" if row.get("distance_m") is None else f"{float(row['distance_m']):.6f}"
        distance_cm = "-" if row.get("distance_cm") is None else f"{float(row['distance_cm']):.2f}"
        hand = "有" if row.get("hand_tracking_present") else "无"
        print(
            f"{str(row.get('alias')):<8} {str(row.get('episode')):<26} "
            f"{distance_m:>12} {distance_cm:>13}  {str(row.get('status')):<6} {hand}"
        )
    print(f"\n本地汇总: {results_dir / 'latest_summary.md'}")
    print(f"CSV 总表: {results_dir / 'latest_summary.csv'}")


def print_devices(registry: dict[str, Any]) -> None:
    if not registry["devices"]:
        print("尚未登记 EGO 设备。")
        return
    for device in sorted(registry["devices"].values(), key=lambda item: str(item.get("alias", ""))):
        print(
            f"{device.get('alias')}: device_id={device.get('device_id')}, "
            f"episodes={len(device.get('episodes', {}))}, last_seen={device.get('last_seen')}"
        )


def scan(args: argparse.Namespace, registry: dict[str, Any], results_dir: Path) -> int:
    if not args.checker.is_file():
        print(f"[FAIL] 找不到底层检查脚本: {args.checker}", file=sys.stderr)
        return 2
    try:
        expected_rules = parse_expected_rules(args.expected)
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    devices = discover_devices()
    if not devices:
        print("[FAIL] 未发现已挂载的 EGO。请先插入设备，并在文件管理器中打开设备。")
        return 2
    if args.label and len(devices) != 1:
        print("[FAIL] --label 只能在当前仅挂载一台 EGO 时使用。", file=sys.stderr)
        return 2

    failures = 0
    for discovered in devices:
        key = str(discovered["device_key"])
        known = registry["devices"].get(key)
        if known is None:
            known = {
                "alias": args.label or next_alias(registry),
                "device_id": discovered["device_id"],
                "first_seen": now_iso(),
                "episodes": {},
            }
            registry["devices"][key] = known
        elif args.label:
            known["alias"] = args.label
        known["mtp_host"] = discovered["mtp_host"]
        known["dataset_root"] = str(discovered["dataset_root"])
        known["last_seen"] = now_iso()

        episodes = discover_episodes(discovered["dataset_root"], args.episode)
        print(f"[INFO] {known['alias']} ({known['device_id']}): 发现 {len(episodes)} 个 episode")
        for episode_dir, data_dir in episodes:
            fingerprint = file_fingerprint(data_dir)
            expected = expected_for_episode(episode_dir.name, expected_rules)
            previous = known["episodes"].get(episode_dir.name)
            previous_log = Path(previous["log_file"]) if previous and previous.get("log_file") else None
            unchanged = bool(
                previous
                and previous.get("fingerprint") == fingerprint
                and previous.get("expected_distance_m") == expected
                and previous_log
                and previous_log.is_file()
            )
            if unchanged and not args.force:
                print(f"[SKIP] {episode_dir.name}: 文件未变化，使用本地已有结果")
                continue
            if args.dry_run:
                print(f"[DRY ] {episode_dir.name}: {episode_dir}")
                continue

            log_file = results_dir / "logs" / str(known["alias"]) / f"{episode_dir.name}.log"
            print(f"[RUN ] {episode_dir.name}: 日志写入本地 {log_file}", flush=True)
            checked = run_episode_check(args.checker, episode_dir, log_file, expected)
            checked.update(
                {
                    "expected_distance_m": expected,
                    "checked_at": now_iso(),
                    "fingerprint": fingerprint,
                    "log_file": str(log_file.resolve()),
                    "source_episode": str(episode_dir),
                }
            )
            known["episodes"][episode_dir.name] = checked
            distance = checked.get("distance_m")
            distance_text = "无法计算" if distance is None else f"{float(distance):.6f} m / {float(distance) * 100:.2f} cm"
            print(f"[DONE] {episode_dir.name}: {distance_text}, {checked['status']}", flush=True)
            if checked["status"] != "PASS":
                failures += 1
            save_registry(results_dir, registry)
            write_summaries(results_dir, registry)

    if not args.dry_run:
        save_registry(results_dir, registry)
        write_summaries(results_dir, registry)
        print_summary(registry, results_dir)
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results_dir = require_local_results_dir(args.results_dir)
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    args.checker = args.checker.expanduser().resolve()
    registry = load_registry(results_dir)
    if args.command == "summary":
        write_summaries(results_dir, registry)
        print_summary(registry, results_dir)
        return 0
    if args.command == "devices":
        print_devices(registry)
        return 0
    return scan(args, registry, results_dir)


if __name__ == "__main__":
    raise SystemExit(main())
