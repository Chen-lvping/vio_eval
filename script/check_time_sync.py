#!/usr/bin/env python3
"""
Check time synchronization between the local PC and the robot arm controller.

This script connects to the robot arm, queries its system time via the C library's
rm_get_cur_time() function, and compares it with the local PC time.

Usage:
    python check_time_sync.py
    python check_time_sync.py --ip 192.168.1.18 --port 8080
"""

import ctypes
import os
import sys
import time
from datetime import datetime


def load_rm_library():
    """Load the RM C library and set up function signatures."""
    lib_path = None

    # Search common locations for the library
    search_paths = [
        os.path.expanduser("~/.local/lib/python3.10/site-packages/Robotic_Arm/libs/linux_x86/libapi_c.so"),
        os.path.expanduser("~/.local/lib/python3.10/site-packages/Robotic_Arm/libs/linux_x86/libapi_c.so"),
    ]

    for p in search_paths:
        if os.path.exists(p):
            lib_path = p
            break

    if lib_path is None:
        # Try to find it via the Robotic_Arm package
        try:
            import Robotic_Arm.rm_ctypes_wrap as wrap
            lib_path = os.path.join(os.path.dirname(wrap.__file__), "libs", "linux_x86", "libapi_c.so")
            if not os.path.exists(lib_path):
                raise FileNotFoundError(lib_path)
        except (ImportError, FileNotFoundError):
            raise RuntimeError("Could not locate libapi_c.so")

    lib = ctypes.cdll.LoadLibrary(lib_path)

    # Set up rm_get_cur_time: void → char* (returns pointer to static formatted string)
    lib.rm_get_cur_time.argtypes = []
    lib.rm_get_cur_time.restype = ctypes.c_char_p

    # Set up rm_get_cur_times: void → char* (returns pointer to static formatted string)
    lib.rm_get_cur_times.argtypes = []
    lib.rm_get_cur_times.restype = ctypes.c_char_p

    return lib


def parse_arm_time(time_str):
    """
    Parse the formatted time string from rm_get_cur_time.
    Format: "YYYY-MM-DD HH_MM_SS"
    Example: "2026-06-13 10_30_45"
    """
    try:
        # Replace underscores with colons for parsing
        normalized = time_str.replace("_", ":")
        dt = datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S")
        return dt
    except ValueError as e:
        print(f"[ERROR] Failed to parse arm time string: {time_str!r}")
        print(f"  {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Check time sync between PC and robot arm"
    )
    parser.add_argument("--ip", default="192.168.1.18", help="robot controller IP")
    parser.add_argument("--port", type=int, default=8080, help="robot controller port")
    parser.add_argument("--count", type=int, default=5, help="number of samples")
    parser.add_argument("--interval", type=float, default=1.0, help="interval between samples (s)")
    args = parser.parse_args()

    print("=" * 65)
    print("  PC ↔ Robot Arm Time Synchronization Check")
    print("=" * 65)

    # Step 1: Load the C library
    print("\n[1] Loading RM C library ...")
    lib = load_rm_library()
    print(f"    ✓ Loaded: {lib._name}")

    # Step 2: Connect to robot arm (to ensure communication is working)
    print(f"\n[2] Connecting to robot arm at {args.ip}:{args.port} ...")
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(args.ip, args.port)
    arm_id = getattr(handle, 'id', 'unknown')
    print(f"    ✓ Connected, arm id={arm_id}")

    # Step 3: Try both time functions
    print(f"\n[3] Querying arm controller time ({args.count} samples) ...")
    print()

    samples = []
    for i in range(args.count):
        # Get arm time
        arm_time_str = lib.rm_get_cur_time().decode('utf-8')
        arm_dt = parse_arm_time(arm_time_str)

        # Get local time immediately after
        local_now = time.time()
        local_dt = datetime.fromtimestamp(local_now)

        if arm_dt is None:
            continue

        arm_ts = arm_dt.timestamp()
        diff = local_now - arm_ts  # positive = local ahead

        samples.append({
            "arm_str": arm_time_str,
            "arm_dt": arm_dt,
            "local_dt": local_dt,
            "diff_sec": diff,
        })

        # Determine sync status
        if abs(diff) < 1.0:
            status = "✓ SYNCED"
        elif abs(diff) < 5.0:
            status = "⚠ Near-sync"
        elif abs(diff) < 60.0:
            status = "✗ Out of sync"
        else:
            status = "✗✗ WAY OFF"

        direction = "ahead" if diff > 0 else "behind"
        print(f"  Sample {i+1}:")
        print(f"    Arm time  : {arm_time_str}")
        print(f"    PC time   : {local_dt.strftime('%Y-%m-%d %H:%M:%S')} ({local_now:.3f})")
        print(f"    Difference: {abs(diff):.3f} s  (PC is {direction})")
        print(f"    Status    : {status}")
        print()

        if i < args.count - 1:
            time.sleep(args.interval)

    # Summary
    if len(samples) >= 2:
        diffs = [s["diff_sec"] for s in samples]
        avg_diff = sum(diffs) / len(diffs)
        min_diff = min(diffs, key=abs)
        max_abs = max(diffs, key=abs)

        print("=" * 65)
        print("  SUMMARY")
        print("=" * 65)
        print(f"  Samples        : {len(samples)}")
        print(f"  Avg difference : {avg_diff:+.3f} s")
        print(f"  Min |diff|     : {abs(min_diff):.3f} s")
        print(f"  Max |diff|     : {abs(max_abs):.3f} s")

        if all(abs(d) < 1.0 for d in diffs):
            print("\n  ✓ Your PC and the robot arm are TIME SYNCED!")
            print("    (difference < 1 second in all samples)")
        elif all(abs(d) < 5.0 for d in diffs):
            print("\n  ⚠ Times are nearly synchronized (diff < 5 s)")
            print("    Likely within NTP tolerance but not identical.")
        else:
            print("\n  ✗ Times are NOT synchronized!")
            if abs(avg_diff) > 60:
                print(f"    The arm is {abs(avg_diff)/60:.1f} minutes off from your PC.")
            print("    Consider setting up NTP sync or manually setting the arm time.")
    elif len(samples) == 1:
        print(f"\n  Single sample difference: {samples[0]['diff_sec']:.3f} s")
    else:
        print("\n  No valid samples collected.")

    # Cleanup
    arm.rm_delete_robot_arm()
    print("\n[Done] Disconnected from robot arm.")


if __name__ == "__main__":
    main()
