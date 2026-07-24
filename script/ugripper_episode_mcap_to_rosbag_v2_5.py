#!/usr/bin/env python3
"""Convert a UGripper latest-format H265 stereo MCAP side into a ROS1 bag for VINS-Fusion."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import rosbag
import rospy
from mcap.reader import make_reader
from mcap.stream_reader import StreamReader
from rosbags.typesys import Stores, get_typestore
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Header


DEFAULT_CAM0_TOPIC = "/fays/atrak/cam0"
DEFAULT_CAM1_TOPIC = "/fays/atrak/cam1"
DEFAULT_IMU_TOPIC = "/fays/atrak/imu"
DEFAULT_CAM0_FRAME_ID = "cam0"
DEFAULT_CAM1_FRAME_ID = "cam1"
DEFAULT_IMU_FRAME_ID = "imu_link"
INPUT_IMU_TOPIC_TEMPLATE = "/imu_{side}"
COMPRESSED_IMAGE_TYPENAME = "sensor_msgs/msg/CompressedImage"
IMU_TYPENAME = "sensor_msgs/msg/Imu"
STEREO_IMAGE_CODEC = "h265"
STEREO_STREAM_SUFFIX = ".h265"


@dataclass(frozen=True)
class RawImageRecord:
    fallback_stamp_ns: int
    log_time_ns: int
    publish_time_ns: int | None


@dataclass(frozen=True)
class RawImuRecord:
    parsed: dict[str, Any]
    fallback_stamp_ns: int
    log_time_ns: int
    publish_time_ns: int | None


@dataclass(frozen=True)
class ConverterConfig:
    episode_dir: Path | None
    mcap_file: Path
    side: str
    output_bag: Path
    tail_imu_sec: float
    keep_temp_stream: bool
    temp_dir: Path | None
    max_frames: int
    cam0_topic: str = DEFAULT_CAM0_TOPIC
    cam1_topic: str = DEFAULT_CAM1_TOPIC
    imu_topic: str = DEFAULT_IMU_TOPIC
    cam0_frame_id: str = DEFAULT_CAM0_FRAME_ID
    cam1_frame_id: str = DEFAULT_CAM1_FRAME_ID
    imu_frame_id: str = DEFAULT_IMU_FRAME_ID

    @property
    def image_topic(self) -> str:
        return f"/stereo_{self.side}/image_compressed"

    @property
    def input_imu_topic(self) -> str:
        return INPUT_IMU_TOPIC_TEMPLATE.format(side=self.side)


def _ns_to_ros_time(timestamp_ns: int) -> rospy.Time:
    secs = int(timestamp_ns // 1_000_000_000)
    nsecs = int(timestamp_ns % 1_000_000_000)
    return rospy.Time(secs, nsecs)


@lru_cache(maxsize=1)
def _ros2_typestore():
    return get_typestore(Stores.ROS2_HUMBLE)


def _normalize_typename(schema_name: str | None, fallback: str) -> str:
    if not schema_name:
        return fallback
    return str(schema_name)


def _require_cdr_encoding(message_encoding: str | None, topic_type: str) -> None:
    if str(message_encoding or "cdr").lower() != "cdr":
        raise RuntimeError(
            f"{topic_type} message encoding must be ROS2 CDR for new-format MCAP, "
            f"got {message_encoding!r}"
        )


def _stamp_to_ns(stamp: Any) -> int:
    sec = int(getattr(stamp, "sec", getattr(stamp, "secs", 0)))
    nanosec = int(getattr(stamp, "nanosec", getattr(stamp, "nsec", 0)))
    return sec * 1_000_000_000 + nanosec


def _align(offset: int, size: int) -> int:
    return offset + ((-(offset - 4)) % size)


def _read_u32(buf: bytes, offset: int) -> tuple[int, int]:
    import struct

    offset = _align(offset, 4)
    return struct.unpack_from("<I", buf, offset)[0], offset + 4


def _read_i32(buf: bytes, offset: int) -> tuple[int, int]:
    import struct

    offset = _align(offset, 4)
    return struct.unpack_from("<i", buf, offset)[0], offset + 4


def _read_f64(buf: bytes, offset: int) -> tuple[float, int]:
    import struct

    offset = _align(offset, 8)
    return struct.unpack_from("<d", buf, offset)[0], offset + 8


def _read_string(buf: bytes, offset: int) -> tuple[str, int]:
    size, offset = _read_u32(buf, offset)
    raw = buf[offset : offset + size]
    offset += size
    if raw.endswith(b"\x00"):
        raw = raw[:-1]
    return raw.decode("utf-8", errors="replace"), offset


def _read_bytes(buf: bytes, offset: int) -> tuple[bytes, int]:
    size, offset = _read_u32(buf, offset)
    return buf[offset : offset + size], offset + size


def _read_f64_array(buf: bytes, offset: int, count: int) -> tuple[list[float], int]:
    values = []
    for _index in range(count):
        value, offset = _read_f64(buf, offset)
        values.append(value)
    return values, offset


def parse_compressed_image(
    data: bytes,
    schema_name: str | None = COMPRESSED_IMAGE_TYPENAME,
    message_encoding: str | None = "cdr",
) -> dict[str, Any]:
    _require_cdr_encoding(message_encoding, "CompressedImage")
    typename = _normalize_typename(schema_name, COMPRESSED_IMAGE_TYPENAME)
    msg = _ros2_typestore().deserialize_cdr(data, typename)
    return {
        "stamp_ns": _stamp_to_ns(msg.header.stamp),
        "frame_id": str(msg.header.frame_id),
        "format": str(msg.format),
        "payload": bytes(msg.data),
    }


def parse_imu(
    data: bytes,
    schema_name: str | None = IMU_TYPENAME,
    message_encoding: str | None = "cdr",
) -> dict[str, Any]:
    _require_cdr_encoding(message_encoding, "Imu")
    typename = _normalize_typename(schema_name, IMU_TYPENAME)
    msg = _ros2_typestore().deserialize_cdr(data, typename)
    return {
        "stamp_ns": _stamp_to_ns(msg.header.stamp),
        "frame_id": str(msg.header.frame_id),
        "orientation": [
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        ],
        "orientation_covariance": [float(value) for value in msg.orientation_covariance],
        "angular_velocity": [
            float(msg.angular_velocity.x),
            float(msg.angular_velocity.y),
            float(msg.angular_velocity.z),
        ],
        "angular_velocity_covariance": [float(value) for value in msg.angular_velocity_covariance],
        "linear_acceleration": [
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
            float(msg.linear_acceleration.z),
        ],
        "linear_acceleration_covariance": [float(value) for value in msg.linear_acceleration_covariance],
    }


def iter_mcap_messages(path: Path, topics: set[str]) -> Iterator[tuple[Any, Any, Any]]:
    with path.open("rb") as stream:
        reader = make_reader(stream)
        yielded = False
        try:
            for schema, channel, message in reader.iter_messages(topics=sorted(topics)):
                yielded = True
                yield schema, channel, message
        except Exception:
            if yielded:
                raise

        if yielded:
            return

        stream.seek(0)
        schemas = {}
        channels = {}
        for record in StreamReader(stream).records:
            name = type(record).__name__
            if name == "Schema":
                schemas[record.id] = record
            elif name == "Channel":
                channels[record.id] = record
            elif name == "Message":
                channel = channels.get(record.channel_id)
                if channel is not None and channel.topic in topics:
                    yield schemas.get(channel.schema_id), channel, record


def resolve_new_format_mcap_file(episode_dir: Path) -> Path:
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.is_file() or metadata_path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty metadata.json: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require_files = metadata.get("require_files") if isinstance(metadata, dict) else None
    if not isinstance(require_files, list):
        raise RuntimeError(f"{metadata_path} metadata.require_files must be a list")

    candidates: list[Path] = []
    for item in require_files:
        if not isinstance(item, str) or not item.endswith(".mcap"):
            continue
        path = Path(item)
        candidates.append(path if path.is_absolute() else episode_dir / path)

    if len(candidates) != 1:
        raise RuntimeError(f"{metadata_path} require_files must contain exactly one .mcap file")

    candidate = candidates[0].resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _message_time_ns(message: Any) -> int:
    return int(getattr(message, "log_time", getattr(message, "logTime", 0)))


def _message_publish_time_ns(message: Any) -> int | None:
    publish_time = getattr(message, "publish_time", getattr(message, "publishTime", None))
    if publish_time is None:
        return None
    return int(publish_time)


def _require_message_publish_time_ns(message: Any, topic: str) -> int:
    publish_time_ns = _message_publish_time_ns(message)
    if publish_time_ns is None:
        raise RuntimeError(
            f"message on {topic} has no publish_time "
            f"(log_time={_message_time_ns(message)}); "
            "cannot align image/IMU timestamps safely"
        )
    return publish_time_ns


def _fallback_stamp_ns(parsed_stamp_ns: int, message: Any) -> int:
    return int(parsed_stamp_ns) or _message_time_ns(message)


def _compute_time_offset_ns(records: list[RawImuRecord], topic: str) -> int:
    first = records[0]
    if first.publish_time_ns is None:
        raise RuntimeError(
            f"first IMU message on {topic} has no publish_time; "
            "cannot align image/IMU timestamps safely"
        )
    offset_ns = first.log_time_ns - first.publish_time_ns
    print(
        f"[Time] offset from {topic}: first_log={first.log_time_ns} "
        f"first_publish={first.publish_time_ns} offset={offset_ns} ns "
        f"({offset_ns / 1e9:.9f}s)"
    )
    return offset_ns


def _aligned_stamp_ns(
    record: RawImageRecord | RawImuRecord,
    offset_ns: int,
    topic: str,
) -> int:
    if record.publish_time_ns is None:
        raise RuntimeError(
            f"message on {topic} has no publish_time "
            f"(log_time={record.log_time_ns}, header_or_log_stamp={record.fallback_stamp_ns}); "
            "refusing to fall back to header/log timestamp"
        )
    return int(record.publish_time_ns + offset_ns)


def create_imu_msg(parsed: dict[str, Any], config: ConverterConfig) -> Imu:
    msg = Imu()
    msg.header = Header()
    msg.header.stamp = _ns_to_ros_time(int(parsed["stamp_ns"]))
    msg.header.frame_id = parsed.get("frame_id") or config.imu_frame_id

    orientation = list(parsed["orientation"])
    if all(abs(value) < 1e-15 for value in orientation):
        msg.orientation.w = 1.0
        msg.orientation_covariance[0] = -1.0
    else:
        msg.orientation.x = orientation[0]
        msg.orientation.y = orientation[1]
        msg.orientation.z = orientation[2]
        msg.orientation.w = orientation[3]
        msg.orientation_covariance = list(parsed["orientation_covariance"])

    angular_velocity = parsed["angular_velocity"]
    msg.angular_velocity.x = angular_velocity[0]
    msg.angular_velocity.y = angular_velocity[1]
    msg.angular_velocity.z = angular_velocity[2]
    msg.angular_velocity_covariance = list(parsed["angular_velocity_covariance"])

    linear_acceleration = parsed["linear_acceleration"]
    msg.linear_acceleration.x = linear_acceleration[0]
    msg.linear_acceleration.y = linear_acceleration[1]
    msg.linear_acceleration.z = linear_acceleration[2]
    msg.linear_acceleration_covariance = list(parsed["linear_acceleration_covariance"])
    return msg


def load_side_data(config: ConverterConfig, stream_path: Path) -> tuple[list[int], list[tuple[int, Imu]]]:
    image_records: list[RawImageRecord] = []
    image_timestamps: list[int] = []
    raw_imu_records: list[RawImuRecord] = []
    imu_records: list[tuple[int, Imu]] = []
    image_count = 0
    topics = {config.image_topic, config.input_imu_topic}

    print(f"[MCAP] reading {config.mcap_file}")
    print(f"[MCAP] image topic: {config.image_topic}")
    print(f"[MCAP] imu topic:   {config.input_imu_topic}")

    with stream_path.open("wb") as stream:
        for schema, channel, message in iter_mcap_messages(config.mcap_file, topics):
            topic = channel.topic
            if topic == config.image_topic:
                if config.max_frames > 0 and image_count >= config.max_frames:
                    continue
                parsed = parse_compressed_image(
                    bytes(message.data),
                    getattr(schema, "name", COMPRESSED_IMAGE_TYPENAME),
                    getattr(channel, "message_encoding", "cdr"),
                )
                publish_time_ns = _require_message_publish_time_ns(message, topic)
                fmt = str(parsed["format"]).lower()
                if STEREO_IMAGE_CODEC not in fmt:
                    raise RuntimeError(f"{topic} format is not {STEREO_IMAGE_CODEC}: {parsed['format']}")
                payload = bytes(parsed["payload"])
                if not payload:
                    print(f"[MCAP] WARNING: empty {STEREO_IMAGE_CODEC.upper()} payload at frame {image_count}")
                    continue
                stream.write(payload)
                image_records.append(
                    RawImageRecord(
                        fallback_stamp_ns=_fallback_stamp_ns(int(parsed["stamp_ns"]), message),
                        log_time_ns=_message_time_ns(message),
                        publish_time_ns=publish_time_ns,
                    )
                )
                image_count += 1
            elif topic == config.input_imu_topic:
                parsed = parse_imu(
                    bytes(message.data),
                    getattr(schema, "name", IMU_TYPENAME),
                    getattr(channel, "message_encoding", "cdr"),
                )
                publish_time_ns = _require_message_publish_time_ns(message, topic)
                raw_imu_records.append(
                    RawImuRecord(
                        parsed=parsed,
                        fallback_stamp_ns=_fallback_stamp_ns(int(parsed["stamp_ns"]), message),
                        log_time_ns=_message_time_ns(message),
                        publish_time_ns=publish_time_ns,
                    )
                )

    if not image_records:
        raise RuntimeError(f"no image messages found on {config.image_topic}")
    if not raw_imu_records:
        raise RuntimeError(f"no IMU messages found on {config.input_imu_topic}")

    offset_ns = _compute_time_offset_ns(raw_imu_records, config.input_imu_topic)

    image_timestamps = [
        _aligned_stamp_ns(record, offset_ns, config.image_topic)
        for record in image_records
    ]

    for record in raw_imu_records:
        stamp_ns = _aligned_stamp_ns(record, offset_ns, config.input_imu_topic)
        parsed = dict(record.parsed)
        parsed["stamp_ns"] = stamp_ns
        imu_records.append((stamp_ns, create_imu_msg(parsed, config)))

    imu_records.sort(key=lambda item: item[0])
    print(
        f"[MCAP] loaded images={len(image_timestamps)} "
        f"imu={len(imu_records)} {STEREO_IMAGE_CODEC}_bytes={stream_path.stat().st_size}"
    )
    print(
        f"[MCAP] image time: {image_timestamps[0]} -> {image_timestamps[-1]} "
        f"({(image_timestamps[-1] - image_timestamps[0]) / 1e9:.3f}s)"
    )
    print(
        f"[MCAP] imu time:   {imu_records[0][0]} -> {imu_records[-1][0]} "
        f"({(imu_records[-1][0] - imu_records[0][0]) / 1e9:.3f}s)"
    )
    return image_timestamps, imu_records


def _is_gray_bgr(frame: np.ndarray) -> bool:
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        return False
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    return bool(np.array_equal(b, g) and np.array_equal(g, r))


def make_image_msg(frame: np.ndarray, stamp_ns: int, frame_id: str) -> Image:
    msg = Image()
    msg.header = Header()
    msg.header.stamp = _ns_to_ros_time(stamp_ns)
    msg.header.frame_id = frame_id
    msg.height = int(frame.shape[0])
    msg.width = int(frame.shape[1])
    msg.is_bigendian = 0

    if len(frame.shape) == 2:
        msg.encoding = "mono8"
        msg.step = int(msg.width)
        msg.data = frame.tobytes()
        return msg

    if frame.shape[2] == 1:
        mono = frame[:, :, 0]
        msg.encoding = "mono8"
        msg.step = int(msg.width)
        msg.data = mono.tobytes()
        return msg

    if _is_gray_bgr(frame):
        mono = frame[:, :, 0]
        msg.encoding = "mono8"
        msg.step = int(msg.width)
        msg.data = mono.tobytes()
        return msg

    msg.encoding = "bgr8"
    msg.step = int(msg.width) * 3
    msg.data = frame.tobytes()
    return msg


def iter_stereo_image_messages(
    stream_path: Path,
    image_timestamps: list[int],
    config: ConverterConfig,
) -> Iterator[tuple[int, str, Image]]:
    cap = cv2.VideoCapture(str(stream_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open temporary {STEREO_IMAGE_CODEC.upper()} stream: {stream_path}")

    decoded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if decoded >= len(image_timestamps):
                print("[Stereo] WARNING: decoded more frames than MCAP timestamps; dropping extras")
                break

            if len(frame.shape) < 2:
                raise RuntimeError(f"decoded frame {decoded} has invalid shape: {frame.shape}")
            height = int(frame.shape[0])
            if height % 2 != 0:
                raise RuntimeError(f"decoded frame {decoded} height is odd: {height}")

            half_height = height // 2
            stamp_ns = image_timestamps[decoded]
            cam0 = frame[:half_height, :]
            cam1 = frame[half_height:, :]

            if decoded == 0:
                print(f"[Stereo] decoded frame shape: {frame.shape}")
                print(f"[Stereo] cam0/cam1 shape: {cam0.shape} / {cam1.shape}")

            yield stamp_ns, config.cam0_topic, make_image_msg(cam0, stamp_ns, config.cam0_frame_id)
            yield stamp_ns, config.cam1_topic, make_image_msg(cam1, stamp_ns, config.cam1_frame_id)
            decoded += 1
    finally:
        cap.release()

    if decoded == 0:
        raise RuntimeError(f"no frames decoded from temporary {STEREO_IMAGE_CODEC.upper()} stream: {stream_path}")
    if decoded < len(image_timestamps):
        print(
            f"[Stereo] WARNING: decoded {decoded} frames but MCAP has "
            f"{len(image_timestamps)} image timestamps"
        )
    print(f"[Stereo] decoded stereo frames={decoded}, ros image messages={decoded * 2}")


def filter_imu_records(
    imu_records: list[tuple[int, Imu]],
    image_timestamps: list[int],
    tail_imu_sec: float,
) -> list[tuple[int, Imu]]:
    if tail_imu_sec < 0:
        return imu_records
    end_ns = image_timestamps[-1] + int(tail_imu_sec * 1e9)
    filtered = [(stamp_ns, msg) for stamp_ns, msg in imu_records if stamp_ns <= end_ns]
    print(
        f"[IMU] keeping {len(filtered)}/{len(imu_records)} samples "
        f"through image_end+{tail_imu_sec:.3f}s"
    )
    return filtered


def merge_and_write(
    bag: rosbag.Bag,
    image_iter: Iterable[tuple[int, str, Image]],
    imu_records: list[tuple[int, Imu]],
    config: ConverterConfig,
) -> None:
    image_iterator = iter(image_iter)
    image_msg = next(image_iterator, None)
    imu_index = 0
    total = 0
    image_count = 0
    imu_count = 0

    while image_msg is not None or imu_index < len(imu_records):
        if image_msg is not None and imu_index < len(imu_records):
            next_imu = imu_records[imu_index]
            if image_msg[0] < next_imu[0]:
                stamp_ns, topic, msg = image_msg
                bag.write(topic, msg, _ns_to_ros_time(stamp_ns))
                image_msg = next(image_iterator, None)
                image_count += 1
            else:
                stamp_ns, msg = next_imu
                bag.write(config.imu_topic, msg, _ns_to_ros_time(stamp_ns))
                imu_index += 1
                imu_count += 1
        elif image_msg is not None:
            stamp_ns, topic, msg = image_msg
            bag.write(topic, msg, _ns_to_ros_time(stamp_ns))
            image_msg = next(image_iterator, None)
            image_count += 1
        else:
            stamp_ns, msg = imu_records[imu_index]
            bag.write(config.imu_topic, msg, _ns_to_ros_time(stamp_ns))
            imu_index += 1
            imu_count += 1

        total += 1
        if total % 5000 == 0:
            print(f"[Bag] written {total} messages...")

    print(f"[Bag] done: total={total}, images={image_count}, imu={imu_count}")


def print_bag_info(path: Path) -> None:
    bag = rosbag.Bag(str(path), "r")
    try:
        info = bag.get_type_and_topic_info()
        print("=== Bag Info ===")
        for topic_name, topic_info in sorted(info.topics.items()):
            print(f"  {topic_name}: {topic_info.message_count} msgs, type={topic_info.msg_type}")
    finally:
        bag.close()


def run(config: ConverterConfig) -> int:
    if not config.mcap_file.is_file():
        raise FileNotFoundError(config.mcap_file)
    config.output_bag.parent.mkdir(parents=True, exist_ok=True)

    temp_root = config.temp_dir or config.output_bag.parent
    temp_root.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{config.mcap_file.parent.name}_{config.side}_",
        suffix=STEREO_STREAM_SUFFIX,
        dir=str(temp_root),
    )
    os.close(fd)
    stream_path = Path(temp_name)

    try:
        image_timestamps, imu_records = load_side_data(config, stream_path)
        imu_records = filter_imu_records(imu_records, image_timestamps, config.tail_imu_sec)
        image_iter = iter_stereo_image_messages(stream_path, image_timestamps, config)
        with rosbag.Bag(str(config.output_bag), "w") as bag:
            merge_and_write(bag, image_iter, imu_records, config)
        print(f"[Done] ROS1 bag saved: {config.output_bag}")
        print_bag_info(config.output_bag)
        return 0
    finally:
        if config.keep_temp_stream:
            print(f"[Temp] kept {STEREO_IMAGE_CODEC.upper()} stream: {stream_path}")
        else:
            stream_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, help="Episode directory containing a new-format .mcap.")
    parser.add_argument("--mcap-file", type=Path, help="Explicit new-format .mcap path.")
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--output-bag", required=True, type=Path)
    parser.add_argument(
        "--tail-imu-sec",
        type=float,
        default=0.2,
        help="Keep IMU samples up to last image timestamp plus this many seconds. Use -1 to keep all.",
    )
    parser.add_argument("--keep-temp-stream", action="store_true", help="Keep the temporary H265 bitstream.")
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Debug only: stop after this many compressed image frames. 0 means all.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ConverterConfig:
    episode_dir = args.episode_dir.resolve() if args.episode_dir else None
    if args.mcap_file:
        mcap_file = args.mcap_file.resolve()
    elif episode_dir is not None:
        mcap_file = resolve_new_format_mcap_file(episode_dir)
    else:
        raise SystemExit("--episode-dir or --mcap-file is required")

    return ConverterConfig(
        episode_dir=episode_dir,
        mcap_file=mcap_file,
        side=args.side,
        output_bag=args.output_bag.resolve(),
        tail_imu_sec=args.tail_imu_sec,
        keep_temp_stream=args.keep_temp_stream,
        temp_dir=args.temp_dir.resolve() if args.temp_dir else None,
        max_frames=max(0, args.max_frames),
    )


def main() -> int:
    return run(build_config(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
