#!/usr/bin/env python3
"""Replay an RM75 TCP pose trajectory via pose -> IK -> joint servo."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


DEFAULT_IP = "192.168.1.18"
DEFAULT_PORT = 8080


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="RM75 TCP trajectory JSON")
    parser.add_argument("--output-plan", type=Path, default=None, help="Optional output plan JSON path")
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"robot controller ip (default: {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"robot controller port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--servo-rate-hz",
        type=float,
        default=200.0,
        help="Resample the pose timeline to this servo rate before IK/streaming; use 0 to keep source timestamps",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Execution time scale. 1.0 = original timing, 2.0 = 2x slower, 0.5 = 2x faster",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=60.0,
        help="Override total execution duration in seconds. If set, this takes precedence over --time-scale",
    )
    parser.add_argument(
        "--follow",
        choices=["auto", "on", "off"],
        default="on",
        help="CANFD follow mode. auto enables high-follow only when cycle <= 10ms",
    )
    parser.add_argument(
        "--trajectory-mode",
        type=int,
        default=2,
        help="CANFD high-follow mode: 0 passthrough, 1 curve-fit, 2 filter",
    )
    parser.add_argument(
        "--radio",
        type=int,
        default=950,
        help="CANFD smoothing coefficient for curve-fit/filter modes",
    )
    parser.add_argument(
        "--move-to-start-speed",
        type=int,
        default=8,
        help="Blocking movej speed percentage for moving to the first IK solution",
    )
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=1.5,
        help="Wait time after moving to the first point before streaming",
    )
    parser.add_argument(
        "--finish-threshold-deg",
        type=float,
        default=1.5,
        help="Final max-joint-error threshold to consider the replay finished",
    )
    parser.add_argument(
        "--finish-timeout-sec",
        type=float,
        default=120.0,
        help="Timeout while waiting for the final joint position",
    )
    parser.add_argument(
        "--finish-poll-sec",
        type=float,
        default=0.1,
        help="Polling interval while waiting for the final joint position",
    )
    parser.add_argument(
        "--ik-fallback-arm-angle-deg",
        type=float,
        default=None,
        help="Optional RM75 arm-angle IK fallback when the default IK fails",
    )
    parser.add_argument(
        "--verify-final-pose",
        action="store_true",
        help="Run FK on the final joint state and print position/orientation error against the target pose",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solve IK / generate plan only, do not move the robot",
    )
    return parser.parse_args()


def normalize_quaternion_wxyz(q: Sequence[float]) -> np.ndarray:
    out = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(out))
    if norm <= 1e-12:
        raise ValueError("zero-norm quaternion")
    return out / norm


def slerp_wxyz(q0: Sequence[float], q1: Sequence[float], alpha: float) -> np.ndarray:
    qa = normalize_quaternion_wxyz(q0)
    qb = normalize_quaternion_wxyz(q1)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion_wxyz((1.0 - alpha) * qa + alpha * qb)
    theta_0 = float(math.acos(dot))
    sin_theta_0 = float(math.sin(theta_0))
    theta = theta_0 * alpha
    sin_theta = float(math.sin(theta))
    s0 = float(math.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(sin_theta / sin_theta_0)
    return normalize_quaternion_wxyz(s0 * qa + s1 * qb)


def quat_wxyz_to_rot(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quaternion_angle_deg(q0: Sequence[float], q1: Sequence[float]) -> float:
    qa = normalize_quaternion_wxyz(q0)
    qb = normalize_quaternion_wxyz(q1)
    dot = abs(float(np.dot(qa, qb)))
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(2.0 * math.acos(dot)))


def sample_to_pose(sample: dict) -> dict:
    pos = sample["position_m"]
    quat = sample["quaternion_wxyz"]
    return {
        "timestamp_s": float(sample["timestamp_s"]),
        "position_m": np.array([float(pos["x"]), float(pos["y"]), float(pos["z"])], dtype=float),
        "quaternion_wxyz": normalize_quaternion_wxyz(
            [float(quat["w"]), float(quat["x"]), float(quat["y"]), float(quat["z"])]
        ),
    }


def load_pose_samples(path: Path) -> tuple[dict, List[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples_raw = payload.get("samples")
    if not isinstance(samples_raw, list) or len(samples_raw) < 2:
        raise ValueError(f"{path}: expected at least 2 samples")
    samples = [sample_to_pose(sample) for sample in samples_raw]
    times = [sample["timestamp_s"] for sample in samples]
    if any(t1 <= t0 for t0, t1 in zip(times[:-1], times[1:])):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    frame = payload.get("frame")
    if frame is not None:
        frame_text = str(frame).strip().lower()
        accepted = {
            "base_to_tcp",
            "current work frame / current tool tcp",
        }
        if frame_text not in accepted and "tcp" not in frame_text:
            raise ValueError(f"{path}: unsupported frame={frame!r}; expected a TCP pose frame")
    return payload, samples


def build_target_times(samples: Sequence[dict], servo_rate_hz: float) -> np.ndarray:
    start = float(samples[0]["timestamp_s"])
    end = float(samples[-1]["timestamp_s"])
    if servo_rate_hz <= 0:
        return np.asarray([sample["timestamp_s"] for sample in samples], dtype=float)
    dt = 1.0 / servo_rate_hz
    count = max(2, int(math.floor((end - start) / dt)) + 1)
    times = np.linspace(start, end, count, dtype=float)
    if abs(times[-1] - end) > 1e-9:
        times = np.r_[times, end]
    return np.unique(times)


def interpolate_pose_sequence(samples: Sequence[dict], target_times: np.ndarray) -> List[dict]:
    source_times = np.asarray([sample["timestamp_s"] for sample in samples], dtype=float)
    out: List[dict] = []
    left = 0
    for t in target_times:
        while left + 1 < len(samples) - 1 and source_times[left + 1] < t:
            left += 1
        if t <= source_times[0]:
            out.append(
                {
                    "timestamp_s": float(t),
                    "position_m": samples[0]["position_m"].copy(),
                    "quaternion_wxyz": samples[0]["quaternion_wxyz"].copy(),
                }
            )
            continue
        if t >= source_times[-1]:
            out.append(
                {
                    "timestamp_s": float(t),
                    "position_m": samples[-1]["position_m"].copy(),
                    "quaternion_wxyz": samples[-1]["quaternion_wxyz"].copy(),
                }
            )
            continue
        s0 = samples[left]
        s1 = samples[left + 1]
        t0 = float(s0["timestamp_s"])
        t1 = float(s1["timestamp_s"])
        alpha = 0.0 if abs(t1 - t0) <= 1e-12 else float((t - t0) / (t1 - t0))
        pos = (1.0 - alpha) * s0["position_m"] + alpha * s1["position_m"]
        quat = slerp_wxyz(s0["quaternion_wxyz"], s1["quaternion_wxyz"], alpha)
        out.append({"timestamp_s": float(t), "position_m": pos, "quaternion_wxyz": quat})
    return out


def pose7_wxyz(sample: dict) -> List[float]:
    p = sample["position_m"]
    q = sample["quaternion_wxyz"]
    return [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def choose_follow_mode(mode: str, dt: float) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return dt <= 0.010 + 1e-9


def format_pose7(pose7: Sequence[float]) -> str:
    return (
        f"pos=[{pose7[0]:.5f}, {pose7[1]:.5f}, {pose7[2]:.5f}] "
        f"quat_wxyz=[{pose7[3]:.5f}, {pose7[4]:.5f}, {pose7[5]:.5f}, {pose7[6]:.5f}]"
    )


def solve_joint_plan(
    arm: Any,
    poses: Sequence[dict],
    seed_joints_deg: Sequence[float],
    arm_angle_deg: float | None,
) -> List[List[float]]:
    from Robotic_Arm.rm_robot_interface import rm_inverse_kinematics_params_t

    current = [float(v) for v in seed_joints_deg]
    plan: List[List[float]] = []
    for idx, sample in enumerate(poses):
        params = rm_inverse_kinematics_params_t(q_in=current, q_pose=pose7_wxyz(sample), flag=0)
        status, joints = arm.rm_algo_inverse_kinematics(params)
        if status != 0 and arm_angle_deg is not None:
            status, joints = arm.rm_algo_inverse_kinematics_rm75_for_arm_angle(params, float(arm_angle_deg))
        if status != 0:
            raise RuntimeError(
                f"IK failed at sample {idx + 1}/{len(poses)} status={status} target={format_pose7(pose7_wxyz(sample))}"
            )
        current = [float(v) for v in joints]
        plan.append(current[:])
    return plan


def joint_step_stats(plan: Sequence[Sequence[float]]) -> dict:
    if len(plan) < 2:
        return {"max_joint_step_deg": 0.0, "mean_joint_step_deg": 0.0}
    max_step = 0.0
    norms: List[float] = []
    for prev, cur in zip(plan[:-1], plan[1:]):
        per_joint = [abs(float(c) - float(p)) for p, c in zip(prev, cur)]
        max_step = max(max_step, max(per_joint))
        norms.append(float(np.linalg.norm(np.asarray(cur, dtype=float) - np.asarray(prev, dtype=float))))
    return {
        "max_joint_step_deg": max_step,
        "mean_joint_step_deg": float(np.mean(norms)) if norms else 0.0,
    }


def write_plan_json(
    path: Path,
    source_path: Path,
    source_payload: dict,
    poses: Sequence[dict],
    joints: Sequence[Sequence[float]],
    dt_exec: float,
    time_scale: float,
    follow: bool,
    trajectory_mode: int,
    radio: int,
) -> None:
    rows = []
    t0 = float(poses[0]["timestamp_s"])
    for idx, (sample, joint) in enumerate(zip(poses, joints)):
        rows.append(
            {
                "index": idx,
                "time_from_start_s": (float(sample["timestamp_s"]) - t0) * float(time_scale),
                "pose_wxyz": pose7_wxyz(sample),
                "joint_deg": [float(v) for v in joint],
            }
        )
    payload = {
        "source_file": str(source_path),
        "source_frame": source_payload.get("frame", "base_to_tcp"),
        "servo_dt_s": float(dt_exec),
        "time_scale": float(time_scale),
        "follow": bool(follow),
        "trajectory_mode": int(trajectory_mode),
        "radio": int(radio),
        "sample_count": len(rows),
        "samples": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def move_to_start(arm: Any, joints_deg: Sequence[float], speed: int, settle_sec: float) -> None:
    status = arm.rm_movej([float(v) for v in joints_deg], int(speed), 0, 0, 1)
    if status != 0:
        raise RuntimeError(f"failed to move to start joint position, status={status}")
    if settle_sec > 0:
        time.sleep(settle_sec)


def stream_joint_plan(
    arm: Any,
    joints: Sequence[Sequence[float]],
    dt_exec: float,
    follow: bool,
    trajectory_mode: int,
    radio: int,
) -> dict:
    start = time.perf_counter()
    last_send = start
    for idx in range(1, len(joints)):
        target_time = start + idx * dt_exec
        wait = target_time - time.perf_counter()
        if wait > 0:
            coarse = max(0.0, wait - 0.002)
            if coarse > 0:
                time.sleep(coarse)
            while time.perf_counter() < target_time:
                pass
        status = arm.rm_movej_canfd(
            [float(v) for v in joints[idx]],
            bool(follow),
            0.0,
            int(trajectory_mode),
            int(radio),
        )
        if status != 0:
            raise RuntimeError(f"movej_canfd failed at command {idx + 1}/{len(joints)} status={status}")
        last_send = time.perf_counter()
        if idx % 200 == 0 or idx == len(joints) - 1:
            print(f"[PROGRESS] {idx + 1}/{len(joints)} sent")
    return {
        "send_phase_sec": max(0.0, last_send - start),
        "commands": max(0, len(joints) - 1),
    }


def wait_final_joint(
    arm: Any,
    target_joints_deg: Sequence[float],
    threshold_deg: float,
    timeout_sec: float,
    poll_sec: float,
) -> dict:
    deadline = time.time() + timeout_sec
    final_err = float("nan")
    while time.time() < deadline:
        status, current = arm.rm_get_joint_degree()
        if status == 0 and isinstance(current, Sequence) and len(current) >= len(target_joints_deg):
            final_err = max(abs(float(c) - float(t)) for c, t in zip(current, target_joints_deg))
            if final_err <= threshold_deg:
                return {"reached": True, "final_max_joint_err_deg": final_err}
        time.sleep(poll_sec)
    return {"reached": False, "final_max_joint_err_deg": final_err}


def final_pose_error(
    arm: Any,
    target_pose: dict,
) -> dict:
    status, current_joints = arm.rm_get_joint_degree()
    if status != 0:
        return {"fk_ok": False, "status": int(status)}
    fk = arm.rm_algo_forward_kinematics([float(v) for v in current_joints], 0)
    fk_pos = np.asarray(fk[:3], dtype=float)
    fk_quat = normalize_quaternion_wxyz(fk[3:7])
    target_pos = np.asarray(target_pose["position_m"], dtype=float)
    target_quat = normalize_quaternion_wxyz(target_pose["quaternion_wxyz"])
    return {
        "fk_ok": True,
        "position_error_mm": float(np.linalg.norm(fk_pos - target_pos) * 1000.0),
        "rotation_error_deg": float(quaternion_angle_deg(fk_quat, target_quat)),
        "fk_pose_wxyz": [float(v) for v in np.r_[fk_pos, fk_quat]],
    }


def main() -> int:
    args = parse_args()
    if args.time_scale <= 0:
        raise ValueError("--time-scale must be > 0")
    if args.duration_sec is not None and args.duration_sec <= 0:
        raise ValueError("--duration-sec must be > 0")
    if args.servo_rate_hz < 0:
        raise ValueError("--servo-rate-hz must be >= 0")
    if args.trajectory_mode not in (0, 1, 2):
        raise ValueError("--trajectory-mode must be 0/1/2")
    if args.radio < 0:
        raise ValueError("--radio must be >= 0")

    src = args.input.expanduser().resolve()
    payload, source_samples = load_pose_samples(src)
    target_times = build_target_times(source_samples, float(args.servo_rate_hz))
    poses = interpolate_pose_sequence(source_samples, target_times)
    if len(poses) < 2:
        raise RuntimeError("not enough interpolated poses to replay")
    src_duration = float(source_samples[-1]["timestamp_s"] - source_samples[0]["timestamp_s"])
    if args.duration_sec is not None:
        exec_duration = float(args.duration_sec)
        resolved_time_scale = exec_duration / src_duration if src_duration > 1e-12 else 1.0
    else:
        exec_duration = src_duration * float(args.time_scale)
        resolved_time_scale = float(args.time_scale)
    dt_exec = exec_duration / float(len(poses) - 1)
    follow = choose_follow_mode(args.follow, dt_exec)

    print(f"[INFO] source samples: {len(source_samples)}, duration={src_duration:.3f}s")
    print(
        f"[INFO] replay samples: {len(poses)}, execution_duration={exec_duration:.3f}s, "
        f"dt={dt_exec:.6f}s, resolved_time_scale={resolved_time_scale:.4f}"
    )
    print(
        "[INFO] stream mode: "
        f"follow={'high' if follow else 'low'}, trajectory_mode={args.trajectory_mode}, radio={args.radio}"
    )
    if follow and dt_exec > 0.010 + 1e-9:
        print("[WARN] high-follow usually expects <= 10ms cycle")

    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(args.ip, args.port)
    print(f"[INFO] connected robot arm id={getattr(handle, 'id', 'unknown')}")
    try:
        status, current_joints = arm.rm_get_joint_degree()
        if status != 0:
            raise RuntimeError(f"failed to read current joints, status={status}")
        current_joints = [float(v) for v in current_joints]
        print(f"[INFO] current joints: {[round(v, 3) for v in current_joints]}")

        joint_plan = solve_joint_plan(
            arm=arm,
            poses=poses,
            seed_joints_deg=current_joints,
            arm_angle_deg=args.ik_fallback_arm_angle_deg,
        )
        stats = joint_step_stats(joint_plan)
        print(
            "[PLAN] "
            f"max_joint_step={stats['max_joint_step_deg']:.4f}deg, "
            f"mean_joint_step={stats['mean_joint_step_deg']:.4f}deg"
        )
        print(f"[PLAN] first pose: {format_pose7(pose7_wxyz(poses[0]))}")
        print(f"[PLAN] last  pose: {format_pose7(pose7_wxyz(poses[-1]))}")

        if args.output_plan:
            out_plan = args.output_plan.expanduser().resolve()
            write_plan_json(
                out_plan,
                src,
                payload,
                poses,
                joint_plan,
                dt_exec,
                resolved_time_scale,
                follow,
                int(args.trajectory_mode),
                int(args.radio),
            )
            print(f"[OK] wrote plan: {out_plan}")

        if args.dry_run:
            print("[DRY-RUN] IK solved and plan generated. No robot motion executed.")
            return 0

        move_to_start(arm, joint_plan[0], int(args.move_to_start_speed), float(args.settle_sec))
        print("[INFO] reached start pose, begin streaming")
        stream_stats = stream_joint_plan(
            arm=arm,
            joints=joint_plan,
            dt_exec=dt_exec,
            follow=follow,
            trajectory_mode=int(args.trajectory_mode),
            radio=int(args.radio),
        )
        wait_stats = wait_final_joint(
            arm=arm,
            target_joints_deg=joint_plan[-1],
            threshold_deg=float(args.finish_threshold_deg),
            timeout_sec=float(args.finish_timeout_sec),
            poll_sec=float(args.finish_poll_sec),
        )
        print(
            "[TIME] "
            f"planned={exec_duration:.3f}s, actual_send={stream_stats['send_phase_sec']:.3f}s, "
            f"commands={stream_stats['commands']}, rate={stream_stats['commands'] / max(stream_stats['send_phase_sec'], 1e-9):.1f}Hz"
        )
        print(
            "[FINISH] "
            f"reached={wait_stats['reached']}, "
            f"final_max_joint_err={wait_stats['final_max_joint_err_deg']:.4f}deg"
        )
        if args.verify_final_pose:
            pose_err = final_pose_error(arm, poses[-1])
            if pose_err.get("fk_ok"):
                print(
                    "[POSE] "
                    f"final_position_error={pose_err['position_error_mm']:.3f}mm, "
                    f"final_rotation_error={pose_err['rotation_error_deg']:.4f}deg"
                )
            else:
                print(f"[POSE] final FK unavailable, status={pose_err.get('status')}")
        return 0
    finally:
        arm.rm_delete_robot_arm()
        print("[INFO] disconnected robot arm")


if __name__ == "__main__":
    raise SystemExit(main())
