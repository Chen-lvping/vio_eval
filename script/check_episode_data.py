#!/usr/bin/env python3
"""Check an SXR ego episode and summarize RGB/MCAP data quality.

The SXR ego recorder stores the following binary payloads in ``sensor.mcap``
(``message_encoding=binary``):

* ``head_pose``: ``int64 timestamp_ns, float32 pos[3], float32 quat[4]``
* ``hand_tracking``: timestamp is the first ``int64`` field (the remainder is
  the recorder's hand/joint payload).

The script deliberately reports the position in the payload's original unit.
The MCAP schema does not declare a unit, so silently converting it to metres
would be misleading.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import unquote, urlparse

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised on installations without OpenCV
    cv2 = None

try:
    import av  # type: ignore
except Exception:  # pragma: no cover - optional fallback for unusual MP4 metadata
    av = None

from mcap.reader import make_reader
from mcap.stream_reader import StreamReader


HEAD_POSE_TOPIC = "head_pose"
HAND_TRACKING_TOPIC = "hand_tracking"
HEAD_POSE_BINARY_SIZE = struct.calcsize("<q7f")
MIN_HAND_TRACKING_BINARY_SIZE = struct.calcsize("<q")
EXPECTED_RGB_HZ = 60.0
EXPECTED_HEAD_POSE_HZ = 200.0
EXPECTED_HAND_TRACKING_HZ = 60.0
DEFAULT_FREQUENCY_TOLERANCE_PERCENT = 10.0


@dataclass
class Series:
    name: str
    topic_names: set[str] = field(default_factory=set)
    timestamps_ns: list[int] = field(default_factory=list)
    poses: list[tuple[int, tuple[float, float, float], tuple[float, float, float, float]]] = field(
        default_factory=list
    )
    malformed_count: int = 0
    timestamp_fallback_count: int = 0
    payload_sizes: set[int] = field(default_factory=set)


@dataclass
class CheckResult:
    lines: list[str] = field(default_factory=list)
    errors: int = 0
    warnings: int = 0

    def info(self, message: str) -> None:
        self.lines.append(f"[INFO] {message}")

    def ok(self, message: str) -> None:
        self.lines.append(f"[ OK ] {message}")

    def core(self, message: str) -> None:
        self.lines.append(f"[CORE] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        self.lines.append(f"[WARN] {message}")

    def error(self, message: str) -> None:
        self.errors += 1
        self.lines.append(f"[FAIL] {message}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 episode 的 rgb.mp4 和 sensor.mcap（head_pose/hand_tracking）数据。"
    )
    parser.add_argument(
        "--episode-dir",
        required=True,
        type=str,
        help="episode 根目录，或直接指向包含 rgb.mp4/sensor.mcap 的 ego 目录。",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="输出日志路径；默认写到 ~/ego_episode_check_logs/<episode>_episode_data_check.log。",
    )
    parser.add_argument(
        "--require-hand-tracking",
        action="store_true",
        help="将缺少 hand_tracking 从警告升级为失败。默认只报告是否存在。",
    )
    parser.add_argument(
        "--quaternion-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="head_pose 四元数顺序，默认 xyzw（SXR ego 当前格式）。",
    )
    parser.add_argument(
        "--frequency-tolerance-percent",
        type=float,
        default=DEFAULT_FREQUENCY_TOLERANCE_PERCENT,
        help="频率验收容差百分比，默认 10%%（RGB/hand 60 Hz，head_pose 200 Hz）。",
    )
    parser.add_argument(
        "--expected-distance",
        type=float,
        default=None,
        help="核心验收：期望的起点到终点欧氏距离，单位必须与 head_pose payload 一致。",
    )
    parser.add_argument(
        "--distance-tolerance-percent",
        type=float,
        default=10.0,
        help="核心距离验收容差百分比，默认 10%%。仅在设置 --expected-distance 时生效。",
    )
    return parser.parse_args(argv)


def episode_arg_to_path(episode_arg: str) -> Path:
    """Resolve a local path or a mounted GVFS MTP URI to a filesystem path."""

    if "://" not in episode_arg:
        return Path(episode_arg).expanduser()
    parsed = urlparse(episode_arg)
    if parsed.scheme != "mtp":
        raise ValueError(
            "--episode-dir 仅支持本地路径或 mtp:// URI；"
            f"不支持的 URI scheme: {parsed.scheme!r}"
        )
    if not parsed.netloc:
        raise ValueError(f"无效的 MTP URI（缺少设备名）: {episode_arg}")
    mount_root = Path(f"/run/user/{os.getuid()}/gvfs") / f"mtp:host={parsed.netloc}"
    return mount_root.joinpath(*[unquote(part) for part in parsed.path.split("/") if part])


def resolve_data_dir(episode_arg: str) -> tuple[Path, Path]:
    """Return (original episode argument, directory containing the data files)."""

    original = episode_arg_to_path(episode_arg)
    if not original.exists():
        suffix = "（请确认 MTP 设备已在文件管理器中挂载）" if episode_arg.startswith("mtp://") else ""
        raise FileNotFoundError(f"episode directory does not exist: {original}{suffix}")
    if not original.is_dir():
        raise NotADirectoryError(original)

    direct_score = int((original / "rgb.mp4").is_file()) + int((original / "sensor.mcap").is_file())
    ego = original / "ego"
    ego_score = int((ego / "rgb.mp4").is_file()) + int((ego / "sensor.mcap").is_file()) if ego.is_dir() else 0
    if direct_score == 2:
        return original, original
    if ego_score == 2:
        return original, ego
    if direct_score or ego_score:
        data_dir = original if direct_score >= ego_score else ego
        return original, data_dir
    raise FileNotFoundError(
        f"在 {original} 或 {ego} 下没有找到 rgb.mp4 和 sensor.mcap；"
        "请传 episode 根目录或 ego 目录。"
    )


def normalize_topic(topic: str) -> str:
    return topic.strip().strip("/").split("/")[-1]


def iter_mcap_messages(path: Path, topics: set[str] | None = None) -> Iterator[tuple[Any, Any, Any]]:
    """Iterate MCAP messages, including files without a usable summary index."""

    with path.open("rb") as stream:
        yielded = False
        matched = False
        try:
            for schema, channel, message in make_reader(stream).iter_messages(
                topics=sorted(topics) if topics else None
            ):
                if topics and str(getattr(channel, "topic", "")) not in topics:
                    continue
                yielded = True
                matched = True
                yield schema, channel, message
        except Exception:
            if yielded and (topics is None or matched):
                raise

        # Some recorder-generated MCAPs have a summary but no usable chunk
        # index.  StreamReader is the supported sequential fallback.
        if yielded and (topics is None or matched):
            return
        stream.seek(0)
        schemas: dict[int, Any] = {}
        channels: dict[int, Any] = {}
        for record in StreamReader(stream).records:
            record_name = type(record).__name__
            if record_name == "Schema":
                schemas[record.id] = record
            elif record_name == "Channel":
                channels[record.id] = record
            elif record_name == "Message":
                channel = channels.get(record.channel_id)
                if channel is not None and (topics is None or str(channel.topic) in topics):
                    yield schemas.get(channel.schema_id), channel, record


def discover_summary(path: Path) -> tuple[list[Any], list[Any]]:
    """Return channels and schemas when available, without requiring an index."""

    try:
        with path.open("rb") as stream:
            summary = make_reader(stream).get_summary()
        if summary is not None:
            return list(summary.channels.values()), list(summary.schemas.values())
    except Exception:
        pass
    return [], []


def finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def json_find(obj: Any, names: set[str]) -> Any:
    """Find a likely field in a nested JSON payload."""

    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in names:
                return value
        for value in obj.values():
            found = json_find(value, names)
            if found is not None:
                return found
    return None


def decode_json_head_pose(data: bytes) -> tuple[int, tuple[float, float, float], tuple[float, float, float, float]] | None:
    try:
        value = json.loads(data.decode("utf-8"))
    except Exception:
        return None
    timestamp = json_find(value, {"timestamp_ns", "timestamp", "time_ns", "ts_ns"})
    position = json_find(value, {"pos", "position", "translation"})
    quaternion = json_find(value, {"quat", "quaternion", "orientation"})
    if isinstance(position, dict):
        position = [position.get(k) for k in ("x", "y", "z")]
    if isinstance(quaternion, dict):
        quaternion = [quaternion.get(k) for k in ("x", "y", "z", "w")]
    try:
        if timestamp is None or not isinstance(position, (list, tuple)) or not isinstance(quaternion, (list, tuple)):
            return None
        if len(position) < 3 or len(quaternion) < 4:
            return None
        return int(timestamp), tuple(float(x) for x in position[:3]), tuple(float(x) for x in quaternion[:4])
    except (TypeError, ValueError):
        return None


def decode_head_pose(data: bytes, schema_encoding: str | None, log_time_ns: int) -> tuple[int, tuple[float, float, float], tuple[float, float, float, float]] | None:
    if schema_encoding and "json" in str(schema_encoding).lower():
        decoded = decode_json_head_pose(data)
        if decoded is not None:
            return decoded
    if len(data) < HEAD_POSE_BINARY_SIZE:
        return None
    values = struct.unpack_from("<q7f", data)
    timestamp = int(values[0]) or int(log_time_ns)
    position = tuple(float(value) for value in values[1:4])
    quaternion = tuple(float(value) for value in values[4:8])
    if not finite(position) or not finite(quaternion):
        return None
    return timestamp, position, quaternion


def decode_timestamp(data: bytes, schema_encoding: str | None, log_time_ns: int) -> int | None:
    if schema_encoding and "json" in str(schema_encoding).lower():
        try:
            value = json.loads(data.decode("utf-8"))
            timestamp = json_find(value, {"timestamp_ns", "timestamp", "time_ns", "ts_ns"})
            if timestamp is not None:
                return int(timestamp) or int(log_time_ns)
        except Exception:
            pass
    if len(data) < MIN_HAND_TRACKING_BINARY_SIZE:
        return None
    timestamp = int(struct.unpack_from("<q", data)[0])
    return timestamp or int(log_time_ns)


def mcap_schema_encoding(schema: Any) -> str | None:
    return str(getattr(schema, "encoding", "")) if schema is not None else None


def inspect_mcap(path: Path, result: CheckResult) -> tuple[Series, Series, list[str], list[str]]:
    channels, schemas = discover_summary(path)
    channel_topics = [str(getattr(channel, "topic", "")) for channel in channels]
    schema_names = [str(getattr(schema, "name", "")) for schema in schemas]
    if channels:
        result.info(f"MCAP channels ({len(channels)}): {', '.join(channel_topics)}")
    if schemas:
        result.info(f"MCAP schemas ({len(schemas)}): {', '.join(schema_names)}")

    head = Series(HEAD_POSE_TOPIC)
    hand = Series(HAND_TRACKING_TOPIC)
    target_topics = {
        str(getattr(channel, "topic", ""))
        for channel in channels
        if normalize_topic(str(getattr(channel, "topic", ""))) in {HEAD_POSE_TOPIC, HAND_TRACKING_TOPIC}
    }
    try:
        for schema, channel, message in iter_mcap_messages(path, target_topics or None):
            raw_topic = str(getattr(channel, "topic", ""))
            topic = normalize_topic(raw_topic)
            target = head if topic == HEAD_POSE_TOPIC else hand if topic == HAND_TRACKING_TOPIC else None
            if target is None:
                continue
            target.topic_names.add(raw_topic)
            data = bytes(getattr(message, "data", b""))
            target.payload_sizes.add(len(data))
            log_time_ns = int(getattr(message, "log_time", getattr(message, "logTime", 0)) or 0)
            encoding = mcap_schema_encoding(schema)
            if target is head:
                decoded = decode_head_pose(data, encoding, log_time_ns)
                if decoded is None:
                    target.malformed_count += 1
                    continue
                timestamp_ns, position, quaternion = decoded
                if timestamp_ns <= 0 or not finite(position) or not finite(quaternion):
                    target.malformed_count += 1
                    continue
                norm = math.sqrt(sum(value * value for value in quaternion))
                if not 0.5 <= norm <= 1.5:
                    target.malformed_count += 1
                    continue
                if int(struct.unpack_from("<q", data)[0]) == 0 if len(data) >= 8 else False:
                    target.timestamp_fallback_count += 1
                target.timestamps_ns.append(timestamp_ns)
                target.poses.append((timestamp_ns, position, quaternion))
            else:
                timestamp_ns = decode_timestamp(data, encoding, log_time_ns)
                if timestamp_ns is None or timestamp_ns <= 0:
                    target.malformed_count += 1
                    continue
                if int(struct.unpack_from("<q", data)[0]) == 0 if len(data) >= 8 else False:
                    target.timestamp_fallback_count += 1
                target.timestamps_ns.append(timestamp_ns)
    except Exception as exc:
        result.error(f"读取 sensor.mcap 失败: {type(exc).__name__}: {exc}")
        return head, hand, channel_topics, schema_names

    return head, hand, channel_topics, schema_names


def series_stats(series: Series) -> dict[str, float | int | str]:
    timestamps = sorted(series.timestamps_ns)
    stats: dict[str, float | int | str] = {
        "count": len(timestamps),
        "malformed": series.malformed_count,
        "timestamp_fallback": series.timestamp_fallback_count,
        "non_monotonic": sum(1 for left, right in zip(series.timestamps_ns, series.timestamps_ns[1:]) if right <= left),
    }
    if not timestamps:
        return stats
    stats["start_ns"] = timestamps[0]
    stats["end_ns"] = timestamps[-1]
    stats["duration_s"] = (timestamps[-1] - timestamps[0]) / 1e9
    positive_dt = [right - left for left, right in zip(timestamps, timestamps[1:]) if right > left]
    if positive_dt:
        median_period_s = statistics.median(positive_dt) / 1e9
        mean_period_s = statistics.fmean(positive_dt) / 1e9
        stats["effective_hz"] = (len(timestamps) - 1) / stats["duration_s"] if stats["duration_s"] > 0 else 0.0
        stats["median_hz"] = 1.0 / median_period_s if median_period_s > 0 else 0.0
        stats["mean_hz"] = 1.0 / mean_period_s if mean_period_s > 0 else 0.0
        stats["median_period_ms"] = median_period_s * 1000.0
        stats["min_period_ms"] = min(positive_dt) / 1e6
        stats["max_period_ms"] = max(positive_dt) / 1e6
    return stats


def format_series_report(series: Series, result: CheckResult) -> None:
    if not series.topic_names:
        return
    stats = series_stats(series)
    result.info(f"{series.name} topic: {', '.join(sorted(series.topic_names))}")
    result.info(
        f"{series.name}: count={stats['count']}, duration={float(stats.get('duration_s', 0.0)):.6f} s, "
        f"effective_freq={float(stats.get('effective_hz', 0.0)):.3f} Hz, "
        f"median_freq={float(stats.get('median_hz', 0.0)):.3f} Hz, "
        f"period_median={float(stats.get('median_period_ms', 0.0)):.3f} ms"
    )
    result.info(
        f"{series.name}: timestamp={stats.get('start_ns', 'N/A')} -> {stats.get('end_ns', 'N/A')} ns, "
        f"period_range={float(stats.get('min_period_ms', 0.0)):.3f}..{float(stats.get('max_period_ms', 0.0)):.3f} ms, "
        f"non_monotonic={stats.get('non_monotonic', 0)}, malformed={series.malformed_count}"
    )
    if series.payload_sizes:
        result.info(f"{series.name}: payload_size(s)={sorted(series.payload_sizes)} bytes")
    if series.malformed_count:
        result.error(f"{series.name} 有 {series.malformed_count} 条消息无法按当前格式解码。")


def quaternion_xyzw(quaternion: Sequence[float], order: str) -> tuple[float, float, float, float]:
    if order == "xyzw":
        x, y, z, w = quaternion
    else:
        w, x, y, z = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("zero-norm quaternion")
    return x / norm, y / norm, z / norm, w / norm


def quat_conjugate(q: Sequence[float]) -> tuple[float, float, float, float]:
    return -q[0], -q[1], -q[2], q[3]


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quaternion_to_euler_xyz_deg(q: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def format_vector(values: Sequence[float], digits: int = 6) -> str:
    return "[" + ", ".join(f"{float(value):.{digits}f}" for value in values) + "]"


def wrapped_angle_delta_deg(start: Sequence[float], end: Sequence[float]) -> tuple[float, float, float]:
    """Component-wise end-start Euler delta, wrapped to [-180, 180)."""

    values = [(float(b) - float(a) + 180.0) % 360.0 - 180.0 for a, b in zip(start, end)]
    return values[0], values[1], values[2]


def report_head_pose_delta(series: Series, result: CheckResult, quaternion_order: str) -> float | None:
    if len(series.poses) < 2:
        if series.topic_names:
            result.error(f"head_pose 有效消息不足 2 条，无法计算起点/终点位姿。")
        return None
    first, last = sorted(series.poses, key=lambda item: item[0])[0], sorted(series.poses, key=lambda item: item[0])[-1]
    t0, p0, raw_q0 = first
    t1, p1, raw_q1 = last
    q0 = quaternion_xyzw(raw_q0, quaternion_order)
    q1 = quaternion_xyzw(raw_q1, quaternion_order)
    relative_q = quat_multiply(quat_conjugate(q0), q1)
    position_delta = tuple(end - start for start, end in zip(p0, p1))
    position_distance = math.sqrt(sum(value * value for value in position_delta))
    euler0 = quaternion_to_euler_xyz_deg(q0)
    euler1 = quaternion_to_euler_xyz_deg(q1)
    relative_euler = quaternion_to_euler_xyz_deg(relative_q)
    relative_angle = math.degrees(2.0 * math.atan2(math.sqrt(sum(value * value for value in relative_q[:3])), abs(relative_q[3])))

    result.info(f"head_pose 起点/终点（按 payload 原始位置单位；四元数按 {quaternion_order} 解码）:")
    result.info(f"  start: timestamp_ns={t0}, position={format_vector(p0)}, quaternion={format_vector(raw_q0)}")
    result.info(f"  end:   timestamp_ns={t1}, position={format_vector(p1)}, quaternion={format_vector(raw_q1)}")
    result.core("========== 核心验收项：head_pose 起点到终点欧式距离 ==========")
    result.core(
        f"euclidean_distance={position_distance:.6f}（按 payload 原始位置单位）; "
        f"position_delta={format_vector(position_delta)}"
    )
    result.info(f"  start_euler_xyz_deg(roll,pitch,yaw)={format_vector(euler0, 4)}")
    result.info(f"  end_euler_xyz_deg(roll,pitch,yaw)={format_vector(euler1, 4)}")
    result.info(f"  euler_component_delta_xyz_deg(end-start,wrapped)={format_vector(wrapped_angle_delta_deg(euler0, euler1), 4)}")
    result.info(f"  relative_euler_xyz_deg(roll,pitch,yaw)={format_vector(relative_euler, 4)}")
    result.info(f"  relative_rotation_angle_deg(axis-angle magnitude)={relative_angle:.4f}")
    return position_distance


def check_expected_distance(
    result: CheckResult,
    actual_distance: float | None,
    expected_distance: float | None,
    tolerance_percent: float,
) -> None:
    if actual_distance is None:
        return
    if expected_distance is None:
        result.core("distance_acceptance=REPORT_ONLY（未设置 --expected-distance，未进行目标距离判定）")
        return
    if expected_distance <= 0:
        result.error("--expected-distance 必须大于 0。")
        return
    tolerance = expected_distance * max(0.0, tolerance_percent) / 100.0
    lower, upper = expected_distance - tolerance, expected_distance + tolerance
    if lower <= actual_distance <= upper:
        result.core(
            f"distance_acceptance=PASS actual={actual_distance:.6f}, expected={expected_distance:.6f}, "
            f"tolerance=±{tolerance_percent:.1f}%"
        )
    else:
        result.core(
            f"distance_acceptance=FAIL actual={actual_distance:.6f}, expected={expected_distance:.6f}, "
            f"tolerance=±{tolerance_percent:.1f}%"
        )
        result.error("起点到终点欧式距离未通过核心验收。")


def inspect_rgb(video_path: Path, result: CheckResult) -> float | None:
    if cv2 is None:
        result.error("未安装 OpenCV，无法检查 rgb.mp4；请安装 opencv-python。")
        return None
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        result.error(f"rgb.mp4 无法打开或无法被当前 OpenCV 解码器读取: {video_path}")
        return None
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
    width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)))
    height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)))
    ok, _frame = capture.read()
    capture.release()
    if not ok:
        result.error("rgb.mp4 已打开，但第一帧解码失败。")
        return None
    # Some H.265 MP4 files produced by the SXR recorder advertise nb_frames=1
    # even though the stream contains hundreds of frames.  Re-count only in
    # this suspicious case so normal episodes do not pay for a full decode.
    if frame_count <= 1 and av is not None:
        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            decoded_count = 0
            first_pts = None
            last_pts = None
            for frame in container.decode(stream):
                decoded_count += 1
                if frame.pts is not None:
                    first_pts = frame.pts if first_pts is None else first_pts
                    last_pts = frame.pts
            avg_rate = float(stream.average_rate) if stream.average_rate else fps
            stream_duration = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None and stream.time_base is not None
                else (frame_count / avg_rate if avg_rate > 0 else 0.0)
            )
            frame_count = decoded_count
            fps = avg_rate
            duration = stream_duration
            if first_pts is not None and last_pts is not None and last_pts > first_pts:
                effective_fps = (decoded_count - 1) / float((last_pts - first_pts) * stream.time_base)
            else:
                effective_fps = (decoded_count / duration) if duration > 0 else fps
            container.close()
            result.ok(
                f"rgb.mp4 可读（PyAV 实际解码）: {width}x{height}, frames={frame_count}, "
                f"stream_freq={fps:.3f} Hz, effective_freq={effective_fps:.3f} Hz, duration={duration:.3f} s"
            )
            return effective_fps
        except Exception as exc:
            result.warn(f"rgb.mp4 容器帧数为 {frame_count}，PyAV 实际计数失败: {type(exc).__name__}: {exc}")
    if fps <= 0 or frame_count <= 0:
        result.warn(f"rgb.mp4 可解码但无法获取可靠的 fps/frame_count (fps={fps}, frames={frame_count})")
        return None
    else:
        result.ok(f"rgb.mp4 可读: {width}x{height}, frames={frame_count}, nominal_freq={fps:.3f} Hz, duration={frame_count / fps:.3f} s")
        return fps


def check_frequency(
    result: CheckResult,
    name: str,
    actual_hz: float | None,
    expected_hz: float,
    tolerance_percent: float,
) -> None:
    if actual_hz is None or not math.isfinite(actual_hz) or actual_hz <= 0:
        return
    tolerance = abs(expected_hz) * max(0.0, tolerance_percent) / 100.0
    lower, upper = expected_hz - tolerance, expected_hz + tolerance
    if lower <= actual_hz <= upper:
        result.ok(f"{name} 频率验收通过: {actual_hz:.3f} Hz（期望 {expected_hz:.1f} ± {tolerance_percent:.1f}%）")
    else:
        result.error(f"{name} 频率不符合要求: {actual_hz:.3f} Hz（期望 {expected_hz:.1f} ± {tolerance_percent:.1f}%）")


def default_log_path(data_dir: Path) -> Path:
    """Use a stable, local per-episode log path, even when input is MTP."""

    episode_name = data_dir.parent.name if data_dir.name == "ego" else data_dir.name
    if not episode_name:
        episode_name = "episode"
    return Path.home() / "ego_episode_check_logs" / f"{episode_name}_episode_data_check.log"


def inspect_episode(args: argparse.Namespace) -> tuple[CheckResult, Path]:
    result = CheckResult()
    _, data_dir = resolve_data_dir(args.episode_dir)
    result.info(f"episode 输入: {args.episode_dir}")
    result.info(f"数据目录: {data_dir}")

    rgb_path = data_dir / "rgb.mp4"
    mcap_path = data_dir / "sensor.mcap"
    if rgb_path.is_file() and rgb_path.stat().st_size > 0:
        result.ok(f"找到 rgb.mp4 ({rgb_path.stat().st_size} bytes)")
        rgb_frequency = inspect_rgb(rgb_path, result)
        check_frequency(result, "rgb.mp4", rgb_frequency, EXPECTED_RGB_HZ, args.frequency_tolerance_percent)
    elif not rgb_path.exists():
        result.error(f"缺少 rgb.mp4: {rgb_path}")
    else:
        result.error(f"rgb.mp4 为空: {rgb_path}")

    if not mcap_path.is_file() or mcap_path.stat().st_size <= 0:
        result.error(f"缺少或为空的 sensor.mcap: {mcap_path}")
        return result, data_dir
    result.ok(f"找到 sensor.mcap ({mcap_path.stat().st_size} bytes)")
    head, hand, _channels, _schemas = inspect_mcap(mcap_path, result)

    if head.topic_names:
        result.ok(f"存在 head_pose topic: {', '.join(sorted(head.topic_names))}")
    else:
        result.error("sensor.mcap 中未找到 head_pose topic。")
    if hand.topic_names:
        result.ok(f"存在 hand_tracking topic: {', '.join(sorted(hand.topic_names))}")
    elif args.require_hand_tracking:
        result.error("sensor.mcap 中未找到 hand_tracking topic（--require-hand-tracking 已启用）。")
    else:
        result.warn("sensor.mcap 中未找到 hand_tracking topic。")

    format_series_report(head, result)
    format_series_report(hand, result)
    head_stats = series_stats(head)
    hand_stats = series_stats(hand)
    if head.timestamps_ns:
        check_frequency(
            result,
            "head_pose",
            float(head_stats.get("effective_hz", 0.0)),
            EXPECTED_HEAD_POSE_HZ,
            args.frequency_tolerance_percent,
        )
    if hand.timestamps_ns:
        check_frequency(
            result,
            "hand_tracking",
            float(hand_stats.get("effective_hz", 0.0)),
            EXPECTED_HAND_TRACKING_HZ,
            args.frequency_tolerance_percent,
        )
    if head.topic_names and not head.timestamps_ns:
        result.error("head_pose topic 存在，但没有可用的有效时间戳/位姿。")
    if hand.topic_names and not hand.timestamps_ns:
        result.error("hand_tracking topic 存在，但没有可用的有效时间戳。")
    for series in (head, hand):
        if series.timestamps_ns and series_stats(series).get("non_monotonic", 0):
            result.warn(f"{series.name} payload timestamp 存在非递增消息，请检查采集时间戳。")
    actual_distance = report_head_pose_delta(head, result, args.quaternion_order)
    check_expected_distance(result, actual_distance, args.expected_distance, args.distance_tolerance_percent)
    return result, data_dir


def write_log(log_path: Path, result: CheckResult) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(result.lines) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] 无法写入日志 {log_path}: {type(exc).__name__}: {exc}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result, data_dir = inspect_episode(args)
        log_path = args.log_file.expanduser() if args.log_file else default_log_path(data_dir)
    except Exception as exc:
        result = CheckResult()
        result.error(f"检查失败: {type(exc).__name__}: {exc}")
        try:
            _original = episode_arg_to_path(args.episode_dir)
            data_dir = _original / "ego" if (_original / "ego").is_dir() else _original
        except Exception:
            data_dir = Path.cwd()
        log_path = args.log_file.expanduser() if args.log_file else default_log_path(data_dir)

    result.lines.append("")
    if result.errors:
        result.lines.append(f"[SUMMARY] FAIL: errors={result.errors}, warnings={result.warnings}")
    else:
        result.lines.append(f"[SUMMARY] PASS: errors=0, warnings={result.warnings}")
    result.lines.append(f"[SUMMARY] log_file={log_path}")
    write_log(log_path, result)
    print("\n".join(result.lines))
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
