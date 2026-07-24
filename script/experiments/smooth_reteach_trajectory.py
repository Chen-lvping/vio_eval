#!/usr/bin/env python3
"""Smooth, resample, and replay RealMan teach pendant trajectory."""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ARM_TRAJECTORY_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_REL = Path("data/robot_record.txt")
DEFAULT_OUTPUT_JSON_REL = Path("data/smoothed_robot_record.json")
DEFAULT_OUTPUT_TXT_REL = Path("data/smoothed_robot_record.txt")


def resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path
    alt = ARM_TRAJECTORY_DIR / path
    if alt.exists():
        return alt
    return ARM_TRAJECTORY_DIR / DEFAULT_INPUT_REL if path == DEFAULT_INPUT_REL else path


def resolve_output_path(raw_path: str, default_rel: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return ARM_TRAJECTORY_DIR / default_rel if path == default_rel else path


def load_reteach_points(path: Path, joint_scale: float) -> List[List[float]]:
    points: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno}: invalid json: {exc}") from exc

            if not isinstance(obj, dict) or "point" not in obj:
                raise ValueError(f"line {lineno}: missing 'point' field")
            arr = obj["point"]
            if not isinstance(arr, list) or len(arr) != 7:
                raise ValueError(f"line {lineno}: 'point' must contain 7 values")

            points.append([float(v) * joint_scale for v in arr])

    if len(points) < 2:
        raise ValueError("trajectory must contain at least 2 points")
    return points


def drop_consecutive_duplicates(points: List[List[float]], eps: float = 1e-9) -> List[List[float]]:
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        prev = out[-1]
        if any(abs(p[j] - prev[j]) > eps for j in range(7)):
            out.append(p)
    return out


def moving_average(points: List[List[float]], window: int, passes: int) -> List[List[float]]:
    if window <= 1 or passes <= 0:
        return [p[:] for p in points]
    if window % 2 == 0:
        raise ValueError("--window must be odd")

    n = len(points)
    half = window // 2
    smoothed = [p[:] for p in points]

    for _ in range(passes):
        next_points = [[0.0] * 7 for _ in range(n)]
        for i in range(n):
            left = max(0, i - half)
            right = min(n - 1, i + half)
            count = right - left + 1
            for j in range(7):
                s = 0.0
                for k in range(left, right + 1):
                    s += smoothed[k][j]
                next_points[i][j] = s / count
        smoothed = next_points
    return smoothed


def suppress_spikes(
    points: List[List[float]],
    passes: int,
    jump_deg: float,
    ratio: float,
) -> tuple[List[List[float]], int]:
    """Replace local spike points by neighbor interpolation."""
    if passes <= 0 or len(points) < 3:
        return [p[:] for p in points], 0

    out = [p[:] for p in points]
    total_fixed = 0
    for _ in range(passes):
        fixed_this_pass = 0
        updated = [p[:] for p in out]
        for i in range(1, len(out) - 1):
            prev_p = out[i - 1]
            cur_p = out[i]
            next_p = out[i + 1]
            d_prev = math.sqrt(sum((cur_p[j] - prev_p[j]) * (cur_p[j] - prev_p[j]) for j in range(7)))
            d_next = math.sqrt(sum((next_p[j] - cur_p[j]) * (next_p[j] - cur_p[j]) for j in range(7)))
            d_span = math.sqrt(sum((next_p[j] - prev_p[j]) * (next_p[j] - prev_p[j]) for j in range(7)))
            major = max(d_prev, d_next)
            if major < jump_deg:
                continue
            if major <= ratio * max(d_span, 1e-9):
                continue
            for j in range(7):
                updated[i][j] = 0.5 * (prev_p[j] + next_p[j])
            fixed_this_pass += 1
        out = updated
        total_fixed += fixed_this_pass
        if fixed_this_pass == 0:
            break
    return out, total_fixed


def cumulative_joint_arclength(points: Sequence[Sequence[float]]) -> List[float]:
    cumulative = [0.0]
    total = 0.0
    for i in range(1, len(points)):
        d2 = 0.0
        for j in range(7):
            dv = float(points[i][j]) - float(points[i - 1][j])
            d2 += dv * dv
        total += math.sqrt(d2)
        cumulative.append(total)
    return cumulative


def resample_by_arclength(
    points: List[List[float]], target_count: int | None, step: float | None
) -> List[List[float]]:
    if len(points) < 2:
        return [p[:] for p in points]

    cumulative = cumulative_joint_arclength(points)
    total = cumulative[-1]
    if total <= 1e-12:
        return [points[0][:], points[-1][:]]

    if step is not None and step > 0:
        target_count = max(2, int(math.floor(total / step)) + 1)
    elif target_count is None:
        target_count = len(points)
    else:
        target_count = max(2, target_count)

    targets = [i * total / (target_count - 1) for i in range(target_count)]
    out: List[List[float]] = []
    seg = 0

    for s in targets:
        while seg < len(cumulative) - 2 and cumulative[seg + 1] < s:
            seg += 1
        s0 = cumulative[seg]
        s1 = cumulative[seg + 1]
        if s1 <= s0 + 1e-12:
            alpha = 0.0
        else:
            alpha = (s - s0) / (s1 - s0)

        p = [0.0] * 7
        for j in range(7):
            v0 = points[seg][j]
            v1 = points[seg + 1][j]
            p[j] = (1.0 - alpha) * v0 + alpha * v1
        out.append(p)

    out[0] = points[0][:]
    out[-1] = points[-1][:]
    return out


def enforce_max_joint_step(points: List[List[float]], max_joint_step_deg: float) -> List[List[float]]:
    """Subdivide segments so each joint delta per command is bounded."""
    if len(points) < 2 or max_joint_step_deg <= 0:
        return [p[:] for p in points]

    out: List[List[float]] = [points[0][:]]
    for i in range(1, len(points)):
        prev = out[-1]
        cur = points[i]
        max_delta = max(abs(cur[j] - prev[j]) for j in range(7))
        if max_delta <= max_joint_step_deg:
            out.append(cur[:])
            continue
        splits = int(math.ceil(max_delta / max_joint_step_deg))
        for s in range(1, splits + 1):
            alpha = s / float(splits)
            inter = [(1.0 - alpha) * prev[j] + alpha * cur[j] for j in range(7)]
            out.append(inter)
    return out


def step_stats(points: Sequence[Sequence[float]]) -> Dict[str, float]:
    if len(points) < 2:
        return {"count": float(len(points)), "path": 0.0, "mean_step": 0.0, "std_step": 0.0, "cv": 0.0}
    steps: List[float] = []
    for i in range(1, len(points)):
        d2 = 0.0
        for j in range(7):
            dv = float(points[i][j]) - float(points[i - 1][j])
            d2 += dv * dv
        steps.append(math.sqrt(d2))
    path = sum(steps)
    mean = path / len(steps)
    var = sum((x - mean) * (x - mean) for x in steps) / len(steps)
    std = math.sqrt(max(0.0, var))
    cv = 0.0 if mean < 1e-12 else std / mean
    return {"count": float(len(points)), "path": path, "mean_step": mean, "std_step": std, "cv": cv}


def save_points_json(path: Path, source_file: str, dt: float, points: Sequence[Sequence[float]]) -> None:
    data: Dict[str, Any] = {
        "source_file": source_file,
        "joint_unit": "deg",
        "dt_sec": dt,
        "dof": 7,
        "points": [[float(v) for v in p] for p in points],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_points_reteach_txt(path: Path, points: Sequence[Sequence[float]], inv_scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in points:
            raw = [int(round(float(v) * inv_scale)) for v in p]
            f.write(json.dumps({"point": raw}, ensure_ascii=False))
            f.write("\n")


def replay_joint_trajectory(
    points: Sequence[Sequence[float]],
    dt: float,
    ip: str,
    port: int,
    speed: int,
    blend_radius: int,
    connect: int,
    block: int,
    settle_sec: float,
    control_api: str,
    canfd_follow: bool,
    canfd_trajectory_mode: int,
    canfd_radio: int,
    send_mode: str,
    send_interval_sec: float,
    retry_count: int,
    retry_wait_sec: float,
    spin_wait_ms: float,
    final_lock: str,
    final_lock_speed: int,
    finish_threshold_deg: float,
    finish_timeout_sec: float,
    finish_stall_sec: float,
    finish_stall_delta_deg: float,
    continue_on_error: bool,
) -> Dict[str, float]:
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(ip, port)
    print(f"[INFO] connected robot arm id={getattr(handle, 'id', 'unknown')}")
    motion_start = time.perf_counter()
    stream_start = motion_start
    send_done = motion_start
    motion_end = motion_start
    reached = False
    final_err = float("nan")
    used_final_lock = False

    try:
        first = [float(v) for v in points[0]]
        status = arm.rm_movej(first, speed, blend_radius, 0, 1)
        if status != 0:
            raise RuntimeError(f"failed to move to first point, status={status}")
        if settle_sec > 0:
            time.sleep(settle_sec)

        def send_joint(pose: List[float]) -> int:
            if control_api == "movej_follow":
                return arm.rm_movej_follow(pose)
            if control_api == "movej_canfd":
                return arm.rm_movej_canfd(
                    pose,
                    bool(canfd_follow),
                    0.0,
                    int(canfd_trajectory_mode),
                    int(canfd_radio),
                )
            return arm.rm_movej(pose, speed, blend_radius, connect, block)

        start = time.perf_counter()
        stream_start = start
        for idx in range(1, len(points)):
            if send_mode == "timed":
                target_time = start + idx * dt
                wait = target_time - time.perf_counter()
                if wait > 0:
                    coarse = wait - max(0.0, spin_wait_ms / 10.0)
                    if coarse > 0:
                        time.sleep(coarse)
                    while time.perf_counter() < target_time:
                        pass
            elif send_mode == "burst" and send_interval_sec > 0:
                time.sleep(send_interval_sec)

            pose = [float(v) for v in points[idx]]
            status = send_joint(pose)
            if status != 0 and retry_count > 0:
                for _ in range(retry_count):
                    time.sleep(retry_wait_sec)
                    status = send_joint(pose)
                    if status == 0:
                        break
            if status != 0:
                msg = f"move failed at idx={idx}, status={status}"
                if continue_on_error:
                    print(f"[WARN] {msg}")
                else:
                    raise RuntimeError(msg)

            if idx % 200 == 0 or idx == len(points) - 1:
                print(f"[PROGRESS] {idx + 1}/{len(points)} sent")
        send_done = time.perf_counter()

        if block == 0:
            # For non-blocking replay, wait until the robot is close to final joints
            # before disconnecting, otherwise trajectory may be cut short.
            goal = [float(v) for v in points[-1]]
            should_final_lock = final_lock == "on" or (
                final_lock == "auto" and control_api in ("movej_follow", "movej_canfd")
            )
            if should_final_lock:
                used_final_lock = True
                lock_speed = final_lock_speed if final_lock_speed > 0 else speed
                lock_status = arm.rm_movej(goal, lock_speed, 0, 0, 1)
                if lock_status != 0:
                    msg = f"final lock movej failed, status={lock_status}"
                    if continue_on_error:
                        print(f"[WARN] {msg}")
                    else:
                        raise RuntimeError(msg)

            deadline = time.time() + finish_timeout_sec
            last_print = 0.0
            best_err = float("inf")
            best_time = time.time()
            timed_out = True
            while time.time() < deadline:
                status, current = arm.rm_get_joint_degree()
                if status != 0:
                    time.sleep(0.2)
                    continue
                if not isinstance(current, Sequence) or len(current) < len(goal):
                    time.sleep(0.2)
                    continue
                max_err = max(abs(float(current[j]) - goal[j]) for j in range(len(goal)))
                final_err = max_err
                now = time.time()
                if now - last_print >= 1.0:
                    print(f"[WAIT] finish max_joint_err={max_err:.3f} deg")
                    last_print = now
                if max_err <= finish_threshold_deg:
                    reached = True
                    print(f"[INFO] final joint reached within {finish_threshold_deg:.3f} deg")
                    break
                if max_err + finish_stall_delta_deg < best_err:
                    best_err = max_err
                    best_time = now
                if now - best_time >= finish_stall_sec:
                    print(
                        "[WARN] finish error stalled, "
                        f"max_joint_err={max_err:.3f}deg, stall={finish_stall_sec:.1f}s; stop waiting."
                    )
                    timed_out = False
                    break
                time.sleep(0.1)

            if not reached and timed_out:
                print(
                    "[WARN] timeout waiting final joint, "
                    f"waited={finish_timeout_sec:.1f}s, threshold={finish_threshold_deg:.3f}deg"
                )
        motion_end = time.perf_counter()
    finally:
        arm.rm_delete_robot_arm()
        print("[INFO] disconnected robot arm")

    send_phase_sec = max(0.0, send_done - stream_start)
    total_sec = max(0.0, motion_end - motion_start)
    wait_phase_sec = max(0.0, motion_end - send_done)
    cmd_count = max(0, len(points) - 1)
    point_rate_hz = 0.0 if send_phase_sec <= 1e-12 else cmd_count / send_phase_sec
    return {
        "points": float(len(points)),
        "commands": float(cmd_count),
        "send_phase_sec": send_phase_sec,
        "wait_phase_sec": wait_phase_sec,
        "motion_total_sec": total_sec,
        "point_rate_hz": point_rate_hz,
        "final_max_err_deg": final_err,
        "final_reached": 1.0 if reached else 0.0,
        "final_lock_used": 1.0 if used_final_lock else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smooth and uniformly resample teach-pendant trajectory, then optionally replay it."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_REL), help="input re-teach txt path")
    parser.add_argument("--joint-scale", type=float, default=0.001, help="raw value -> degree scale")
    parser.add_argument("--window", type=int, default=21, help="moving-average window (odd)")
    parser.add_argument("--passes", type=int, default=5, help="moving-average passes")
    parser.add_argument("--resample-count", type=int, default=None, help="target point count after resample")
    parser.add_argument("--resample-step", type=float, default=0.10, help="joint-space step (deg) for resample")
    parser.add_argument("--dt", type=float, default=0.04, help="replay interval seconds")
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=120.0,
        help="target trajectory execution time in seconds (excluding move-to-start), overrides --dt",
    )
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON_REL), help="output json path")
    parser.add_argument(
        "--output-txt",
        default=str(DEFAULT_OUTPUT_TXT_REL),
        help="output txt path in {'point':[...]} format",
    )

    parser.add_argument("--ip", default="192.168.1.18", help="robot controller ip")
    parser.add_argument("--port", type=int, default=8080, help="robot controller port")
    parser.add_argument("--speed", type=int, default=10, help="rm_movej speed percentage")
    parser.add_argument("--blend-radius", type=int, default=0, help="rm_movej blend radius")
    parser.add_argument("--connect", type=int, default=1, help="rm_movej connect flag")
    parser.add_argument("--block", type=int, default=0, help="rm_movej block flag")
    parser.add_argument("--settle-sec", type=float, default=1.0, help="wait after reaching first point")
    parser.add_argument(
        "--control-api",
        choices=["movej_follow", "movej", "movej_canfd"],
        default="movej_canfd",
        help="joint command API. movej_follow is usually smoother for dense trajectory",
    )
    parser.add_argument(
        "--canfd-follow",
        action="store_true",
        help="only for movej_canfd: enable high-follow mode (requires stable <=10ms cycle)",
    )
    parser.add_argument(
        "--canfd-trajectory-mode",
        type=int,
        default=2,
        help="only for movej_canfd: 0 passthrough, 1 curve-fit, 2 filter",
    )
    parser.add_argument(
        "--canfd-radio",
        type=int,
        default=800,
        help="only for movej_canfd smoothing coefficient",
    )
    parser.add_argument(
        "--send-mode",
        choices=["burst", "timed"],
        default="timed",
        help="burst: continuous stream for smoother motion; timed: send by --dt cadence",
    )
    parser.add_argument(
        "--send-interval-sec",
        type=float,
        default=0.002,
        help="small gap between commands in burst mode",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=5,
        help="retry times when one command send fails",
    )
    parser.add_argument(
        "--retry-wait-sec",
        type=float,
        default=0.01,
        help="wait between retries",
    )
    parser.add_argument(
        "--spin-wait-ms",
        type=float,
        default=2.0,
        help="busy-wait time slice in timed mode to reduce scheduling jitter (ms)",
    )
    parser.add_argument(
        "--final-lock",
        choices=["auto", "on", "off"],
        default="on",
        help="after streaming, optionally send one blocking movej to final point for precise settle",
    )
    parser.add_argument(
        "--final-lock-speed",
        type=int,
        default=6,
        help="speed percentage for final lock movej",
    )
    parser.add_argument(
        "--finish-threshold-deg",
        type=float,
        default=1.8,
        help="max joint error threshold to consider finished (non-block mode)",
    )
    parser.add_argument(
        "--finish-timeout-sec",
        type=float,
        default=180.0,
        help="max wait time for final joint in non-block mode",
    )
    parser.add_argument(
        "--finish-stall-sec",
        type=float,
        default=4.0,
        help="stop waiting if final error does not improve for this many seconds",
    )
    parser.add_argument(
        "--finish-stall-delta-deg",
        type=float,
        default=0.02,
        help="minimum improvement considered as progress in final-error waiting",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="continue replay when one point fails")
    parser.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        default=True,
        help="actually move robot arm (default: enabled)",
    )
    parser.add_argument(
        "--dry-run",
        dest="execute",
        action="store_false",
        help="preprocess only, do not execute robot motion",
    )
    parser.add_argument("--post-window", type=int, default=15, help="post-smoothing window (odd, after resample)")
    parser.add_argument("--post-passes", type=int, default=4, help="post-smoothing passes")
    parser.add_argument(
        "--spike-guard-passes",
        type=int,
        default=3,
        help="passes of local spike suppression (0 to disable)",
    )
    parser.add_argument(
        "--spike-jump-deg",
        type=float,
        default=0.8,
        help="minimum local jump size considered as potential spike (joint-space norm, deg)",
    )
    parser.add_argument(
        "--spike-ratio",
        type=float,
        default=1.8,
        help="spike ratio threshold against neighbor span (larger -> less aggressive)",
    )
    parser.add_argument(
        "--max-joint-step-deg",
        type=float,
        default=0.35,
        help="max allowed single-command delta of any joint; extra points will be inserted",
    )
    parser.add_argument(
        "--continuous-mode",
        action="store_true",
        help="aggressive preset for highly continuous motion (follow control + denser path + faster cycle)",
    )
    args = parser.parse_args()

    if args.joint_scale <= 0:
        raise ValueError("--joint-scale must be > 0")
    if args.window < 1:
        raise ValueError("--window must be >= 1")
    if args.passes < 0:
        raise ValueError("--passes must be >= 0")
    if args.dt <= 0:
        raise ValueError("--dt must be > 0")
    if args.duration_sec is not None and args.duration_sec <= 0:
        raise ValueError("--duration-sec must be > 0")
    if args.send_interval_sec < 0:
        raise ValueError("--send-interval-sec must be >= 0")
    if args.retry_count < 0:
        raise ValueError("--retry-count must be >= 0")
    if args.retry_wait_sec < 0:
        raise ValueError("--retry-wait-sec must be >= 0")
    if args.spin_wait_ms < 0:
        raise ValueError("--spin-wait-ms must be >= 0")
    if not (1 <= args.final_lock_speed <= 100):
        raise ValueError("--final-lock-speed must be in [1, 100]")
    if args.finish_threshold_deg <= 0:
        raise ValueError("--finish-threshold-deg must be > 0")
    if args.finish_timeout_sec < 0:
        raise ValueError("--finish-timeout-sec must be >= 0")
    if args.finish_stall_sec < 0:
        raise ValueError("--finish-stall-sec must be >= 0")
    if args.finish_stall_delta_deg < 0:
        raise ValueError("--finish-stall-delta-deg must be >= 0")
    if args.post_window < 1:
        raise ValueError("--post-window must be >= 1")
    if args.post_window % 2 == 0:
        raise ValueError("--post-window must be odd")
    if args.post_passes < 0:
        raise ValueError("--post-passes must be >= 0")
    if args.spike_guard_passes < 0:
        raise ValueError("--spike-guard-passes must be >= 0")
    if args.spike_jump_deg <= 0:
        raise ValueError("--spike-jump-deg must be > 0")
    if args.spike_ratio <= 0:
        raise ValueError("--spike-ratio must be > 0")
    if args.max_joint_step_deg <= 0:
        raise ValueError("--max-joint-step-deg must be > 0")
    if args.canfd_trajectory_mode not in (0, 1, 2):
        raise ValueError("--canfd-trajectory-mode must be 0/1/2")
    if args.canfd_radio < 0:
        raise ValueError("--canfd-radio must be >= 0")
    if args.resample_count is not None and args.resample_count < 2:
        raise ValueError("--resample-count must be >= 2")
    if args.resample_step is not None and args.resample_step <= 0:
        raise ValueError("--resample-step must be > 0")

    if args.continuous_mode:
        # Preset tuned for one smooth, coherent and uninterrupted motion.
        args.control_api = "movej_canfd"
        args.canfd_follow = True
        args.canfd_trajectory_mode = 2
        args.canfd_radio = max(args.canfd_radio, 400)
        args.send_mode = "timed"
        args.dt = min(args.dt, 0.006)
        args.window = max(args.window, 15)
        args.passes = max(args.passes, 3)
        args.post_window = max(args.post_window, 11)
        args.post_passes = max(args.post_passes, 2)
        if args.resample_step is None and args.resample_count is None:
            args.resample_step = 0.18
        args.retry_count = max(args.retry_count, 10)
        args.retry_wait_sec = min(args.retry_wait_sec, 0.005)
        args.spin_wait_ms = max(args.spin_wait_ms, 1.0)
        args.final_lock = "auto"
        args.finish_threshold_deg = max(args.finish_threshold_deg, 1.8)
        args.finish_stall_sec = min(args.finish_stall_sec, 4.0)
        args.spike_guard_passes = max(args.spike_guard_passes, 2)
        args.max_joint_step_deg = min(args.max_joint_step_deg, 0.45)

    src = resolve_input_path(args.input)
    if not src.is_file():
        raise FileNotFoundError(
            f"input file not found: {src} "
            f"(current directory: {Path.cwd()}, default file: {ARM_TRAJECTORY_DIR / DEFAULT_INPUT_REL})"
        )
    raw = load_reteach_points(src, args.joint_scale)
    dedup = drop_consecutive_duplicates(raw)
    de_spiked, spike_fixed = suppress_spikes(
        dedup,
        passes=args.spike_guard_passes,
        jump_deg=args.spike_jump_deg,
        ratio=args.spike_ratio,
    )
    smooth = moving_average(de_spiked, window=args.window, passes=args.passes)
    final = resample_by_arclength(smooth, target_count=args.resample_count, step=args.resample_step)
    if args.post_passes > 0 and args.post_window > 1:
        final = moving_average(final, window=args.post_window, passes=args.post_passes)
    final = enforce_max_joint_step(final, max_joint_step_deg=args.max_joint_step_deg)
    # Keep start/end exactly the same as source trajectory for safer replay.
    final[0] = dedup[0][:]
    final[-1] = dedup[-1][:]

    if args.duration_sec is not None:
        if len(final) < 2:
            raise ValueError("trajectory must contain at least 2 points for --duration-sec")
        if args.send_mode != "timed":
            print("[INFO] --duration-sec is set, force send-mode to timed")
            args.send_mode = "timed"
        args.dt = args.duration_sec / float(len(final) - 1)
        print(
            "[INFO] duration control enabled: "
            f"target={args.duration_sec:.3f}s, points={len(final)}, auto_dt={args.dt:.6f}s"
        )

    before = step_stats(raw)
    after = step_stats(final)
    print(
        "[SUMMARY] "
        f"raw={int(before['count'])} pts, smooth={len(smooth)} pts, final={int(after['count'])} pts, "
        f"path(raw)={before['path']:.3f}deg, path(final)={after['path']:.3f}deg"
    )
    print(
        "[UNIFORMITY] "
        f"raw_step_cv={before['cv']:.4f}, final_step_cv={after['cv']:.4f}, "
        f"raw_mean={before['mean_step']:.4f}deg, final_mean={after['mean_step']:.4f}deg"
    )
    print(
        "[FILTER] "
        f"spike_fixed={spike_fixed}, max_joint_step={args.max_joint_step_deg:.3f}deg, "
        f"spike_guard_passes={args.spike_guard_passes}"
    )
    if after["mean_step"] > 0.35:
        print(
            "[WARN] final mean step is relatively large; for smoother motion "
            "try smaller --resample-step (e.g. 0.12~0.20 deg)."
        )
    print(f"[PREVIEW] first={final[0]}")
    print(f"[PREVIEW] last ={final[-1]}")
    if args.control_api == "movej" and args.connect == 1 and args.blend_radius == 0:
        print("[WARN] blend-radius=0 may still cause stop-go. try --blend-radius 1~5 for smoother transitions.")
    if args.control_api != "movej":
        print(
            f"[INFO] control_api={args.control_api}, "
            "movej params (speed/blend/connect/block) are only used for initial move-to-start."
        )
    print(
        "[INFO] finish_policy: "
        f"final_lock={args.final_lock}, threshold={args.finish_threshold_deg:.3f}deg, "
        f"stall={args.finish_stall_sec:.1f}s, timeout={args.finish_timeout_sec:.1f}s"
    )
    if args.control_api == "movej_follow" and args.dt > 0.02 and args.send_mode == "timed":
        print("[WARN] movej_follow建议更高刷新率，建议 --dt 0.004~0.012")
    if args.control_api == "movej_canfd" and args.dt > 0.01 and args.send_mode == "timed":
        print("[WARN] movej_canfd高跟随建议周期<=10ms，建议 --dt 0.002~0.010")
    if args.send_mode == "timed":
        planned_traj_sec = (len(final) - 1) * args.dt
        print(f"[TIME] planned_trajectory={planned_traj_sec:.3f}s by dt={args.dt:.6f}s")
    else:
        planned_traj_sec = (len(final) - 1) * args.send_interval_sec
        print(f"[TIME] planned_trajectory~{planned_traj_sec:.3f}s by send_interval={args.send_interval_sec:.6f}s")

    output_json = resolve_output_path(args.output_json, DEFAULT_OUTPUT_JSON_REL)
    output_txt = resolve_output_path(args.output_txt, DEFAULT_OUTPUT_TXT_REL)
    save_points_json(output_json, str(src), args.dt, final)
    save_points_reteach_txt(output_txt, final, inv_scale=1.0 / args.joint_scale)
    print(f"[INFO] saved json: {output_json}")
    print(f"[INFO] saved txt : {output_txt}")

    if not args.execute:
        print("[DRY-RUN] preprocessing finished. remove --dry-run to execute robot motion.")
        return

    exec_stats = replay_joint_trajectory(
        points=final,
        dt=args.dt,
        ip=args.ip,
        port=args.port,
        speed=args.speed,
        blend_radius=args.blend_radius,
        connect=args.connect,
        block=args.block,
        settle_sec=args.settle_sec,
        control_api=args.control_api,
        canfd_follow=args.canfd_follow,
        canfd_trajectory_mode=args.canfd_trajectory_mode,
        canfd_radio=args.canfd_radio,
        send_mode=args.send_mode,
        send_interval_sec=args.send_interval_sec,
        retry_count=args.retry_count,
        retry_wait_sec=args.retry_wait_sec,
        spin_wait_ms=args.spin_wait_ms,
        final_lock=args.final_lock,
        final_lock_speed=args.final_lock_speed,
        finish_threshold_deg=args.finish_threshold_deg,
        finish_timeout_sec=args.finish_timeout_sec,
        finish_stall_sec=args.finish_stall_sec,
        finish_stall_delta_deg=args.finish_stall_delta_deg,
        continue_on_error=args.continue_on_error,
    )
    print(
        "[TIME] "
        f"send_phase={exec_stats['send_phase_sec']:.3f}s, "
        f"wait_phase={exec_stats['wait_phase_sec']:.3f}s, "
        f"motion_total={exec_stats['motion_total_sec']:.3f}s, "
        f"rate={exec_stats['point_rate_hz']:.1f}Hz"
    )
    print(
        "[TIME] "
        f"planned={planned_traj_sec:.3f}s, "
        f"actual_send={exec_stats['send_phase_sec']:.3f}s, "
        f"error={exec_stats['send_phase_sec'] - planned_traj_sec:+.3f}s"
    )
    print(
        "[FINISH] "
        f"reached={bool(exec_stats['final_reached'])}, "
        f"final_err={exec_stats['final_max_err_deg']:.3f}deg, "
        f"final_lock={bool(exec_stats['final_lock_used'])}"
    )


if __name__ == "__main__":
    main()
