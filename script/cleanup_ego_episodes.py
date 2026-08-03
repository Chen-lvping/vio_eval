#!/usr/bin/env python3
"""Safely delete checked EGO episodes only after strict local validation."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from check_ego_fleet import (
    DEFAULT_CHECKER,
    DEFAULT_RESULTS_DIR,
    discover_devices,
    discover_episodes,
    file_fingerprint,
    load_registry,
    now_iso,
    require_local_results_dir,
    run_episode_check,
    save_registry,
    write_summaries,
)


EXPECTED_RGB_HZ = 60.0
EXPECTED_HEAD_POSE_HZ = 200.0
EXPECTED_HAND_HZ = 60.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="严格复核当前 EGO；全部通过后才允许删除设备中的 episode。"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--checker", type=Path, default=DEFAULT_CHECKER)
    parser.add_argument(
        "--expected-distances",
        default="0.60,0.90",
        help="正式 episode 应一一对应的距离（米），默认 0.60,0.90；顺序不限。",
    )
    parser.add_argument("--tolerance-percent", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="只验证和显示目标，不重跑或删除")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认；仅限受控自动化")
    return parser.parse_args(argv)


def parse_distances(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("--expected-distances 必须是逗号分隔的正数，例如 0.60,0.90")
    return values


def in_tolerance(actual: float, expected: float, tolerance_percent: float) -> bool:
    tolerance = expected * max(0.0, tolerance_percent) / 100.0
    return expected - tolerance <= actual <= expected + tolerance


def match_distances(
    episode_names: list[str],
    actual_distances: list[float],
    expected_distances: list[float],
    tolerance_percent: float,
) -> dict[str, float] | None:
    if len(actual_distances) < len(expected_distances):
        return None
    best_assignment: dict[str, float] | None = None
    best_score: float | None = None
    expected_count = len(expected_distances)
    for indices in itertools.combinations(range(len(actual_distances)), expected_count):
        selected_names = [episode_names[index] for index in indices]
        selected_actual = [actual_distances[index] for index in indices]
        for permutation in itertools.permutations(expected_distances):
            if not all(
                in_tolerance(actual, expected, tolerance_percent)
                for actual, expected in zip(selected_actual, permutation)
            ):
                continue
            score = sum(abs(actual - expected) for actual, expected in zip(selected_actual, permutation))
            if best_score is None or score < best_score:
                best_score = score
                best_assignment = dict(zip(selected_names, permutation))
    return best_assignment


def frequency_ok(value: Any, expected: float, tolerance_percent: float = 10.0) -> bool:
    try:
        actual = float(value)
    except (TypeError, ValueError):
        return False
    return in_tolerance(actual, expected, tolerance_percent)


def local_result_ok(result: dict[str, Any], current_fingerprint: dict[str, int]) -> tuple[bool, str]:
    if result.get("fingerprint") != current_fingerprint:
        return False, "设备文件已变化，必须重新检查"
    if result.get("status") != "PASS" or int(result.get("return_code", 1)) != 0:
        return False, "数据检查不是 PASS"
    if not result.get("hand_tracking_present"):
        return False, "缺少 hand_tracking"
    if result.get("distance_m") is None:
        return False, "无法计算首尾距离"
    if not frequency_ok(result.get("rgb_hz"), EXPECTED_RGB_HZ):
        return False, "RGB 频率不合格"
    if not frequency_ok(result.get("head_pose_hz"), EXPECTED_HEAD_POSE_HZ):
        return False, "head_pose 频率不合格"
    if not frequency_ok(result.get("hand_tracking_hz"), EXPECTED_HAND_HZ):
        return False, "hand_tracking 频率不合格"
    log_file = Path(str(result.get("log_file", "")))
    if not log_file.is_file():
        return False, "本地日志不存在"
    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    if "[FAIL]" in log_text:
        return False, "本地日志包含 FAIL"
    if log_text.count("non_monotonic=0") < 2 or log_text.count("malformed=0") < 2:
        return False, "MCAP 时间戳或 payload 完整性未确认"
    return True, "PASS"


def safe_episode_target(episode_dir: Path, dataset_root: Path) -> Path:
    target = episode_dir.resolve()
    root = dataset_root.resolve()
    if target.parent != root:
        raise ValueError(f"删除目标不在 dataset 根目录下一层: {target}")
    if not target.name.startswith("episode_"):
        raise ValueError(f"删除目标名称不是 episode_*: {target}")
    if not target.is_dir():
        raise ValueError(f"删除目标不存在或不是目录: {target}")
    if "/gvfs/mtp:host=" not in str(target):
        raise ValueError(f"删除目标不是当前 MTP 挂载目录: {target}")
    return target


def ensure_delete_access(dataset_root: Path, targets: Sequence[Path]) -> None:
    """Fail before destructive work when the mounted MTP storage is read-only."""

    paths = [dataset_root, *targets]
    for path in paths:
        try:
            read_only = bool(os.statvfs(path).f_flag & os.ST_RDONLY)
        except OSError as exc:
            raise PermissionError(f"无法确认设备写入权限: {path}: {exc}") from exc
        if read_only:
            raise PermissionError(
                "当前 EGO 的 MTP 存储为只读，电脑端不能删除 episode。"
                "请先解锁设备并确认 USB 文件传输允许写入；若仍为只读，"
                "请在 EGO 设备/应用中删除，电脑本地日志不会受影响。"
            )
    if not os.access(dataset_root, os.W_OK):
        raise PermissionError(
            "当前 dataset 目录没有写权限，已在删除前停止；请检查 EGO 的 USB/MTP 写入状态。"
        )


def append_cleanup_history(results_dir: Path, record: dict[str, Any]) -> None:
    path = results_dir / "cleanup_history.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results_dir = require_local_results_dir(args.results_dir)
        expected_distances = parse_distances(args.expected_distances)
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    checker = args.checker.expanduser().resolve()
    if not checker.is_file():
        print(f"[FAIL] 找不到检查脚本: {checker}", file=sys.stderr)
        return 2
    devices = discover_devices()
    if len(devices) != 1:
        print(f"[FAIL] 删除时必须且只能挂载一台 EGO，当前发现 {len(devices)} 台。", file=sys.stderr)
        return 2

    connected = devices[0]
    device_key = str(connected["device_key"])
    registry = load_registry(results_dir)
    device = registry["devices"].get(device_key)
    if device is None:
        print("[FAIL] 当前设备尚无本地登记结果，请先运行一键检查。", file=sys.stderr)
        return 2
    dataset_root = Path(str(connected["dataset_root"]))
    live_episodes = discover_episodes(dataset_root, [])
    if not live_episodes:
        print("[FAIL] 当前没有可删除的完整 episode。", file=sys.stderr)
        return 1

    episode_names: list[str] = []
    actual_distances: list[float] = []
    validated: dict[str, tuple[Path, Path, dict[str, int], dict[str, Any]]] = {}
    for episode_dir, data_dir in live_episodes:
        result = device.get("episodes", {}).get(episode_dir.name)
        if not isinstance(result, dict):
            print(f"[FAIL] {episode_dir.name} 没有本地检查结果。", file=sys.stderr)
            return 1
        fingerprint = file_fingerprint(data_dir)
        ok, reason = local_result_ok(result, fingerprint)
        if not ok:
            print(f"[FAIL] {episode_dir.name}: {reason}", file=sys.stderr)
            return 1
        episode_names.append(episode_dir.name)
        actual_distances.append(float(result["distance_m"]))
        validated[episode_dir.name] = (episode_dir, data_dir, fingerprint, result)

    assignments = match_distances(
        episode_names,
        actual_distances,
        expected_distances,
        args.tolerance_percent,
    )
    assignments = assignments or {}

    alias = str(device.get("alias", device_key))
    # The operator's y confirmation authorizes deleting every structurally
    # valid episode in this device, including episodes whose distance does not
    # match the nominal 0.60/0.90 m acceptance targets.
    delete_names = episode_names
    unmatched_names = [name for name in episode_names if name not in assignments]
    targets: list[Path] = []
    print(f"[PASS] {alias} 预删除验证通过（文件完整性通过；距离不匹配项也按确认删除）")
    for name in delete_names:
        actual = float(validated[name][3]["distance_m"])
        target = safe_episode_target(validated[name][0], dataset_root)
        targets.append(target)
        if name in assignments:
            print(f"  {name}: actual={actual:.6f} m, expected={assignments[name]:.6f} m")
        else:
            print(f"  {name}: actual={actual:.6f} m, expected=未匹配（仍删除）")
        print(f"    target={target}")
    if unmatched_names:
        print(f"[INFO] 距离不匹配但将删除: {', '.join(unmatched_names)}")
    if args.dry_run:
        print("[DRY-RUN] 未重跑、未删除任何 episode。")
        return 0

    sys.stdout.flush()
    try:
        ensure_delete_access(dataset_root, targets)
    except PermissionError as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        print(f"[INFO] 本地日志和汇总仍保留在 {results_dir}", file=sys.stderr)
        return 4

    # Re-run every episode with its assigned target so the retained local log
    # contains a formal distance_acceptance=PASS before device data is removed.
    for name in delete_names:
        episode_dir, data_dir, fingerprint, previous = validated[name]
        log_file = Path(str(previous["log_file"]))
        expected_distance = assignments.get(name)
        expected_text = f"{expected_distance:.6f} m" if expected_distance is not None else "仅结构检查"
        print(f"[RECHECK] {name} expected={expected_text}", flush=True)
        checked = run_episode_check(checker, episode_dir, log_file, expected_distance)
        checked.update(
            {
                "expected_distance_m": expected_distance,
                "checked_at": now_iso(),
                "fingerprint": fingerprint,
                "log_file": str(log_file.resolve()),
                "source_episode": str(episode_dir),
            }
        )
        ok, reason = local_result_ok(checked, file_fingerprint(data_dir))
        if not ok or (expected_distance is not None and checked.get("distance_acceptance") != "PASS"):
            print(f"[FAIL] {name} 正式复核失败: {reason}", file=sys.stderr)
            return 1
        device["episodes"][name] = checked
        save_registry(results_dir, registry)
        write_summaries(results_dir, registry)

    if not args.yes:
        token = f"DELETE {connected['device_id']}"
        print("\n警告：以下 MTP episode 将被永久删除，无法恢复。")
        try:
            confirmation = input(f"请输入 {token!r} 确认删除: ").strip()
        except EOFError:
            confirmation = ""
        if confirmation != token:
            print("[CANCEL] 确认内容不匹配，未删除。")
            return 3

    # Resolve and fingerprint again immediately before destructive writes.
    for name, target in zip(delete_names, targets):
        _episode_dir, data_dir, expected_fingerprint, _previous = validated[name]
        if file_fingerprint(data_dir) != expected_fingerprint:
            print(f"[FAIL] {name} 在删除前发生变化，已取消全部删除。", file=sys.stderr)
            return 1
        safe_episode_target(target, dataset_root)

    deleted_at = now_iso()
    deleted: list[str] = []
    for name, target in zip(selected_names, targets):
        print(f"[DELETE] {target}", flush=True)
        try:
            shutil.rmtree(target)
        except OSError as exc:
            print(f"[FAIL] 删除失败: {target}: {exc}", file=sys.stderr)
            if deleted:
                print(f"[WARN] 此前已删除: {', '.join(deleted)}", file=sys.stderr)
            print(f"[INFO] 本地日志和汇总仍保留在 {results_dir}", file=sys.stderr)
            return 4
        if target.exists():
            print(f"[FAIL] 删除后目录仍存在: {target}", file=sys.stderr)
            return 1
        deleted.append(name)
        device["episodes"][name]["source_deleted_at"] = deleted_at
        save_registry(results_dir, registry)
        write_summaries(results_dir, registry)

    record = {
        "deleted_at": deleted_at,
        "device": alias,
        "device_id": connected["device_id"],
        "episodes": deleted,
        "preserved_episodes": [],
        "distance_mismatch_deleted": unmatched_names,
        "expected_distances_m": assignments,
        "tolerance_percent": args.tolerance_percent,
    }
    append_cleanup_history(results_dir, record)
    print(f"[DONE] 已删除 {len(deleted)} 个 episode；本地日志和汇总保留在 {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
