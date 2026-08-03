#!/usr/bin/env python3
"""Publish one SXR side-by-side RGB recording as timestamped ROS1 stereo images."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--image-dir", type=Path,
                        help="Directory containing <source-index>_{left,right}.jpg pairs.")
    parser.add_argument("--timestamps", type=Path, required=True,
                        help="Lines: decoded-frame index and camera-clock timestamp in nanoseconds.")
    parser.add_argument("--rate", type=float, default=60.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.video is None) == (args.image_dir is None):
        raise ValueError("provide exactly one of --video or --image-dir")
    timestamps = [tuple(map(int, line.split())) for line in args.timestamps.read_text().splitlines() if line.strip()]
    if not timestamps or any(len(item) != 2 for item in timestamps):
        raise ValueError("timestamp file is empty")
    rospy.init_node("sxr_offline_stereo_publisher", anonymous=True)
    left_pub = rospy.Publisher("/cam0/image_raw", Image, queue_size=3)
    right_pub = rospy.Publisher("/cam1/image_raw", Image, queue_size=3)
    deadline = time.monotonic() + args.startup_timeout
    while (left_pub.get_num_connections() < 1 or right_pub.get_num_connections() < 1) and not rospy.is_shutdown():
        if time.monotonic() > deadline:
            raise RuntimeError("OV2SLAM did not subscribe to both stereo topics")
        time.sleep(0.1)

    capture = cv2.VideoCapture(str(args.video)) if args.video else None
    if capture is not None and not capture.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    bridge = CvBridge()
    period = 1.0 / args.rate if args.rate > 0 else 0.0
    sent = 0
    try:
        decoded_index = -1
        for expected_index, stamp_ns in timestamps:
            if args.image_dir:
                left_gray = cv2.imread(str(args.image_dir / f"{expected_index}_left.jpg"), cv2.IMREAD_GRAYSCALE)
                right_gray = cv2.imread(str(args.image_dir / f"{expected_index}_right.jpg"), cv2.IMREAD_GRAYSCALE)
                if left_gray is None or right_gray is None:
                    raise RuntimeError(f"missing exported stereo pair for source frame {expected_index}")
            else:
                image = None
                while decoded_index < expected_index:
                    ok, image = capture.read()
                    decoded_index += 1
                    if not ok:
                        raise RuntimeError(f"video ended at source frame {decoded_index}, expected {expected_index}")
                if image.shape[:2] != (1748, 4656):
                    raise ValueError(f"unexpected video size {image.shape[1]}x{image.shape[0]}")
                left_gray = cv2.cvtColor(image[:, :2328], cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(image[:, 2328:], cv2.COLOR_BGR2GRAY)
            stamp = rospy.Time(secs=stamp_ns // 1_000_000_000, nsecs=stamp_ns % 1_000_000_000)
            left = bridge.cv2_to_imgmsg(left_gray, encoding="mono8")
            right = bridge.cv2_to_imgmsg(right_gray, encoding="mono8")
            left.header.stamp = stamp; left.header.frame_id = "cam0"; left.header.seq = sent
            right.header.stamp = stamp; right.header.frame_id = "cam1"; right.header.seq = sent
            start = time.monotonic()
            left_pub.publish(left)
            right_pub.publish(right)
            sent += 1
            remaining = period - (time.monotonic() - start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if capture is not None:
            capture.release()
    rospy.loginfo("published %d stereo pairs", sent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
