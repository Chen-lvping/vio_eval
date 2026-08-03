#!/usr/bin/env python3
"""Run the local OpenVINS container on an SXR RGB stereo-IMU episode.

It creates a short-lived ROS1 bag from rgb.mp4 + sensor.mcap, derives KB4
fisheye/IMU calibration from calibration.json, and saves OpenVINS output.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap.stream_reader import StreamReader

ROOT = Path(__file__).resolve().parents[1]
META = "<qqqIIII"
VEC3 = "<q3f"
HEAD = "<q7f"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode-dir", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--image-scale", type=float, default=0.5)
    p.add_argument("--stream", choices=("rgb", "tracking", "ctrl"), default="rgb",
                   help="Stereo video/calibration stream to feed into VIO.")
    p.add_argument("--profile", choices=("robust", "default", "static-bootstrap"), default="robust")
    p.add_argument("--keep-bag", action="store_true")
    p.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def resolve(value: str) -> Path:
    if "://" not in value:
        path = Path(value).expanduser()
    else:
        uri = urlparse(value)
        if uri.scheme != "mtp" or not uri.netloc:
            raise ValueError("only local directories or mtp:// URIs are supported")
        path = Path(f"/run/user/{os.getuid()}/gvfs") / f"mtp:host={uri.netloc}"
        path = path.joinpath(*[unquote(x) for x in uri.path.split("/") if x])
    if (path / "ego").is_dir() and not (path / "sensor.mcap").exists():
        path /= "ego"
    missing = [x for x in ("rgb.mp4", "sensor.mcap", "metadata.json", "calibration.json") if not (path / x).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {path}")
    return path.resolve()


def messages(path: Path):
    with path.open("rb") as f:
        yielded = False
        try:
            for item in make_reader(f).iter_messages():
                yielded = True
                yield item
        except Exception:
            if yielded:
                raise
    if yielded:
        return
    with path.open("rb") as f:
        schemas, channels = {}, {}
        for item in StreamReader(f).records:
            if type(item).__name__ == "Schema": schemas[item.id] = item
            elif type(item).__name__ == "Channel": channels[item.id] = item
            elif type(item).__name__ == "Message" and item.channel_id in channels:
                ch = channels[item.channel_id]
                yield schemas.get(ch.schema_id), ch, item


def records(path: Path, frame_topic: str = "rgb_metainfo"):
    frames, acc, gyro, head = [], [], [], []
    for _, channel, message in messages(path):
        topic, data = str(channel.topic).strip("/"), bytes(message.data)
        if topic == frame_topic and len(data) >= struct.calcsize(META):
            _, mid, _, index, _, _, _ = struct.unpack_from(META, data)
            frames.append((int(index), int(mid)))
        elif topic in ("imu/accel", "imu/gyro") and len(data) >= struct.calcsize(VEC3):
            t, x, y, z = struct.unpack_from(VEC3, data)
            (acc if topic.endswith("accel") else gyro).append((int(t), np.array((x, y, z))))
        elif topic == "head_pose" and len(data) >= struct.calcsize(HEAD):
            item = struct.unpack_from(HEAD, data)
            head.append((int(item[0]), item[1:]))
    return sorted(frames), sorted(acc), sorted(gyro), sorted(head)


def stream_spec(calibration: dict, stream: str) -> tuple[dict, int, int, tuple[str, str]]:
    image = calibration["observation"]["images"][stream]
    width = int(image["shape"][1] // 2)
    height = int(image["shape"][0])
    offsets = {
        "rgb": ("rgb-left", "rgb-right"),
        "tracking": ("trackingA", "trackingB"),
        "ctrl": ("ctrl-trackingA", "ctrl-trackingB"),
    }
    return image, width, height, offsets[stream]


def join_imu(acc, gyro):
    out, idx = [], 0
    for time_ns, omega in gyro:
        while idx + 1 < len(acc) and abs(acc[idx + 1][0] - time_ns) <= abs(acc[idx][0] - time_ns): idx += 1
        if abs(acc[idx][0] - time_ns) <= 2_000_000: out.append((time_ns, acc[idx][1], omega))
    if len(out) < 100: raise ValueError("not enough matched IMU samples")
    return out


def matrix(f, value):
    for row in value:
        f.write("    - [" + ", ".join(f"{float(x):.15g}" for x in row) + "]\n")


def config(out: Path, calib: dict, scale: float, profile: str, stream: str):
    d = out / "openvins_config"; d.mkdir(parents=True, exist_ok=True)
    robust = profile in ("robust", "static-bootstrap")
    static_bootstrap = profile == "static-bootstrap"
    (d / "estimator_config.yaml").write_text(f'''%YAML:1.0
verbosity: "INFO"
use_fej: true
integration: "rk4"
use_stereo: true
max_cameras: 2
calib_cam_extrinsics: false
calib_cam_intrinsics: false
calib_cam_timeoffset: false
calib_imu_intrinsics: false
calib_imu_g_sensitivity: false
max_clones: 11
max_slam: 50
max_slam_in_update: 25
max_msckf_in_update: 40
dt_slam_delay: 1
gravity_mag: 9.81
feat_rep_msckf: "GLOBAL_3D"
feat_rep_slam: "ANCHORED_MSCKF_INVERSE_DEPTH"
feat_rep_aruco: "ANCHORED_MSCKF_INVERSE_DEPTH"
try_zupt: true
zupt_chi2_multipler: 0.5
zupt_max_velocity: 0.1
zupt_noise_multiplier: 10.0
zupt_max_disparity: 0.5
zupt_only_at_beginning: true
init_window_time: 2.0
init_imu_thresh: {0.8 if robust else 1.5}
init_max_disparity: {15 if robust else 10}
init_max_features: {80 if robust else 50}
init_dyn_use: {"false" if static_bootstrap else "true"}
init_dyn_mle_opt_calib: false
init_dyn_mle_max_iter: 50
init_dyn_mle_max_time: 0.05
init_dyn_mle_max_threads: 6
init_dyn_num_pose: 6
init_dyn_min_deg: {5 if robust else 10}
init_dyn_inflation_ori: 10
init_dyn_inflation_vel: 100
init_dyn_inflation_bg: 10
init_dyn_inflation_ba: 100
init_dyn_min_rec_cond: 1e-12
init_dyn_bias_g: [0.0, 0.0, 0.0]
init_dyn_bias_a: [0.0, 0.0, 0.0]
save_total_state: true
record_timing_information: false
record_timing_filepath: "/tmp/traj_timing.txt"
filepath_est: "/output/openvins_estimate.txt"
filepath_std: "/output/openvins_estimate_std.txt"
filepath_gt: "/output/openvins_groundtruth.txt"
use_klt: true
num_pts: {420 if robust else 220}
fast_threshold: {10 if robust else 15}
grid_x: {8 if robust else 5}
grid_y: 5
min_px_dist: {8 if robust else 12}
knn_ratio: 0.70
track_frequency: 30.0
downsample_cameras: false
num_opencv_threads: 4
histogram_method: "NONE"
use_aruco: false
num_aruco: 1024
downsize_aruco: true
up_msckf_sigma_px: {1.5 if robust else 1.0}
up_msckf_chi2_multipler: 1
up_slam_sigma_px: {1.5 if robust else 1.0}
up_slam_chi2_multipler: 1
up_aruco_sigma_px: 1
up_aruco_chi2_multipler: 1
use_mask: false
relative_config_imu: "kalibr_imu_chain.yaml"
relative_config_imucam: "kalibr_imucam_chain.yaml"
''', encoding="utf-8")
    stereo, width, height, offset_names = stream_spec(calib, stream)
    imu = calib["observation"]["imu"]
    extr, offsets = stereo["extrinsics"], imu.get("time_alignment_s", {}).get("cameras", {})
    with (d / "kalibr_imucam_chain.yaml").open("w", encoding="utf-8") as f:
        f.write("%YAML:1.0\n\n")
        resolution_name = f"{width}x{height}"
        for index, (name, offset) in enumerate(zip(("cam0", "cam1"), offset_names)):
            cam, intr = stereo[name], stereo[name]["intrinsics"][resolution_name]
            # SXR's T_ic_imu0_cam* maps IMU coordinates into its camera
            # coordinates.  OpenVINS T_imu_cam is explicitly camera -> IMU.
            f.write(f"cam{index}:\n  T_imu_cam:\n"); matrix(f, np.linalg.inv(np.asarray(extr[f"T_ic_imu0_{name}"], dtype=float)))
            coeff = ", ".join(f"{float(x):.15g}" for x in cam["distortion_coeffs"][:4])
            vals = ", ".join(f"{float(intr[k])*scale:.15g}" for k in ("fx", "fy", "ppx", "ppy"))
            f.write(f"  cam_overlaps: [{1-index}]\n  camera_model: pinhole\n  distortion_coeffs: [{coeff}]\n  distortion_model: equidistant\n  intrinsics: [{vals}]\n  resolution: [{round(width*scale)}, {round(height*scale)}]\n  rostopic: /sxr/cam{index}\n  timeshift_cam_imu: {float(offsets.get(offset, 0)):.15g}\n")
    n, rate = imu["noise"], 1013.8
    with (d / "kalibr_imu_chain.yaml").open("w", encoding="utf-8") as f:
        f.write("%YAML:1.0\n\nimu0:\n  T_i_b:\n"); matrix(f, np.eye(4))
        f.write(f"  accelerometer_noise_density: {float(n['accel_noise_std_mps2'][0])/math.sqrt(rate):.15g}\n  accelerometer_random_walk: {float(n['accel_bias_std_mps2'][0]):.15g}\n  gyroscope_noise_density: {float(n['gyro_noise_std_rads'][0])/math.sqrt(rate):.15g}\n  gyroscope_random_walk: {float(n['gyro_bias_std_rads'][0]):.15g}\n  rostopic: /sxr/imu\n  time_offset: 0.0\n  update_rate: {rate}\n  model: \"kalibr\"\n")
        for key in ("Tw", "R_IMUtoGYRO", "Ta", "R_IMUtoACC"):
            f.write(f"  {key}:\n"); matrix(f, np.eye(3))
        f.write("  Tg:\n    - [0.0, 0.0, 0.0]\n    - [0.0, 0.0, 0.0]\n    - [0.0, 0.0, 0.0]\n")


def export_bag(episode: Path, out: Path, scale: float, calibration: dict, stream: str):
    import rosbag
    import rospy
    from sensor_msgs.msg import Image, Imu
    meta = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    frames, acc, gyro, head = records(episode / "sensor.mcap", f"{stream}_metainfo")
    if not frames or not head: raise ValueError("rgb_metainfo or head_pose missing")
    start_us = next(x["start_offset_us"] for x in meta["head_pose_details"] if x["name"] == "head_pose")
    aligned = [(i, head[0][0] + mid - int(start_us)*1000) for i, mid in frames]
    imu = join_imu(acc, gyro)
    # The MCAP payload is in physical units but carries the device's residual
    # factory calibration. Apply the documented bias, scale, and small
    # non-orthogonality correction before OpenVINS preintegration.
    imu_calib = calibration["observation"]["imu"]
    def correction(kind: str, bias_key: str):
        bias = np.asarray(imu_calib["bias"][bias_key], dtype=np.float64)
        scale_factor = np.asarray(imu_calib["scale_factor"][kind], dtype=np.float64)
        n0, n1, n2 = imu_calib["nonorthogonality"][kind]
        nonorthogonal = np.array(((1.0, n0, n1), (0.0, 1.0, n2), (0.0, 0.0, 1.0)), dtype=np.float64)
        return bias, nonorthogonal @ np.diag(1.0 + scale_factor)
    accel_bias, accel_matrix = correction("accelerometer", "accelerometer_mps2")
    gyro_bias, gyro_matrix = correction("gyroscope", "gyroscope_rads")
    imu = [(t, accel_matrix @ (a - accel_bias), gyro_matrix @ (w - gyro_bias)) for t, a, w in imu]
    aligned = [(i, t) for i, t in aligned if imu[0][0] <= t <= imu[-1][0]]
    bag = out / "sxr_openvins_input.bag"; wanted = dict(aligned); written = 0
    # The container's OpenCV build may open this H.264 MP4 but yield no frames.
    # ffmpeg has a software H.264 decoder in the same image, so stream raw BGR
    # frames without ever materialising a large image sequence on disk.
    _stereo, half_width, height, _offsets = stream_spec(calibration, stream)
    width, channels = half_width * 2, 3
    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(episode / f"{stream}.mp4"), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert decoder.stdout is not None
    with rosbag.Bag(str(bag), "w", compression="lz4") as b:
        imu_index = 0
        def write_imu(t, a, w):
            stamp = rospy.Time(t//1_000_000_000, t%1_000_000_000); msg = Imu(); msg.header.stamp = stamp; msg.header.frame_id = "imu0"
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = a; msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = w
            b.write("/sxr/imu", msg, t=stamp)
        for index in range(max(wanted) + 1):
            raw = decoder.stdout.read(height * width * channels)
            if len(raw) != height * width * channels: break
            image = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, channels))
            t = wanted.get(index); index += 1
            if t is None: continue
            # Write messages in strict sensor-time order. Bag insertion order
            # matters for the ROS subscriber callbacks used by OpenVINS.
            while imu_index < len(imu) and imu[imu_index][0] <= t:
                write_imu(*imu[imu_index]); imu_index += 1
            if image.shape[:2] != (height, width): raise ValueError(f"unexpected {stream} shape {image.shape}")
            stamp = rospy.Time(t//1_000_000_000, t%1_000_000_000)
            for side, half in enumerate((image[:, :half_width], image[:, half_width:])):
                gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
                if scale != 1: gray = cv2.resize(gray, (round(gray.shape[1]*scale),round(gray.shape[0]*scale)), interpolation=cv2.INTER_AREA)
                msg = Image(); msg.header.stamp = stamp; msg.header.frame_id = f"cam{side}"; msg.height,msg.width=gray.shape; msg.encoding="mono8"; msg.step=gray.shape[1]; msg.data=gray.tobytes()
                b.write(f"/sxr/cam{side}", msg, t=stamp)
            written += 1
        while imu_index < len(imu):
            write_imu(*imu[imu_index]); imu_index += 1
    decoder.stdout.close(); decoder.wait(timeout=60)
    if written != len(aligned):
        raise RuntimeError(f"decoded {written}/{len(aligned)} aligned RGB frames")
    with (out / "head_pose_reference.tum").open("w", encoding="utf-8") as f:
        for t, x in head: f.write(f"{t*1e-9:.9f} {' '.join(f'{float(v):.9f}' for v in x)}\n")
    return {"frames":written,"imu":len(imu),"head_pose":len(head)}


def inside(args, episode, out):
    if not .2 <= args.image_scale <= 1: raise ValueError("image-scale must be in [0.2, 1]")
    out.mkdir(parents=True, exist_ok=True); calib=json.loads((episode/"calibration.json").read_text()); config(out,calib,args.image_scale,args.profile,args.stream); stats=export_bag(episode,out,args.image_scale,calib,args.stream)
    name="sxr_"+out.name.replace("-","_"); dest=Path("/root/openvins_ws/src/open_vins/config")/name
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(out/"openvins_config",dest)
    shell=f'''set -e; source /opt/ros/noetic/setup.bash; source /root/openvins_ws/devel/setup.bash; roscore >/tmp/sxr_roscore.log 2>&1 & R=$!; trap "rosnode kill /ov_msckf >/dev/null 2>&1 || true; kill $R >/dev/null 2>&1 || true" EXIT; sleep 2; rosparam set /use_sim_time true; roslaunch ov_msckf subscribe.launch config:={name} >/tmp/sxr_openvins_node.log 2>&1 & O=$!; sleep 3; rostopic echo -p /ov_msckf/poseimu > /output/openvins_poseimu.csv 2>/tmp/sxr_poseimu_echo.log & P=$!; rosbag play --clock /output/sxr_openvins_input.bag; sleep 3; kill $P >/dev/null 2>&1 || true; rosnode kill /ov_msckf >/dev/null 2>&1 || true; wait $O || true; cat /tmp/sxr_openvins_node.log'''
    result=subprocess.run(["bash","-lc",shell],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
    (out/"openvins.log").write_text(result.stdout,encoding="utf-8")
    (out/"run_provenance.json").write_text(json.dumps({"algorithm":"OpenVINS MSCKF stereo-inertial","stream":args.stream,"image_scale":args.image_scale,"profile":args.profile,"records":stats,"returncode":result.returncode,"head_pose_note":"Device-provided reference, not independently verified ground truth."},indent=2)+"\n")
    if not args.keep_bag: (out/"sxr_openvins_input.bag").unlink(missing_ok=True)
    return result.returncode


def main():
    args=parse_args(); episode=resolve(args.episode_dir); out=args.output_dir.expanduser().resolve()
    if args.inside: return inside(args,episode,out)
    out.mkdir(parents=True,exist_ok=True)
    # Docker's daemon cannot bind-mount a FUSE/GVFS MTP path even though the
    # current user can read it.  Stage only the four inputs required here to a
    # local temporary directory; it is removed once the run ends.
    stage = None
    if str(episode).startswith("/run/user/") and "/gvfs/" in str(episode):
        stage = tempfile.TemporaryDirectory(prefix=f"sxr_{episode.name}_", dir="/tmp")
        staged_episode = Path(stage.name) / "episode"
        staged_episode.mkdir()
        for name in ("rgb.mp4", "sensor.mcap", "metadata.json", "calibration.json"):
            shutil.copy2(episode / name, staged_episode / name)
        episode = staged_episode
    # OpenVINS uses Boost interprocess mutexes during startup.  The locally
    # validated OpenVINS container uses the host IPC namespace; without this
    # option a fresh container can abort before subscribing to any topic.
    cmd=["docker","run","--rm","--net=host","--ipc=host",
         "--mount",f"type=bind,source={ROOT},target=/workspace,readonly",
         "--mount",f"type=bind,source={episode},target=/input,readonly",
         "--mount",f"type=bind,source={out},target=/output",
         "openvins-noetic-ready:latest","bash","-lc",f"source /opt/ros/noetic/setup.bash && source /root/openvins_ws/devel/setup.bash && python3 /workspace/script/run_sxr_openvins.py --inside --episode-dir /input --output-dir /output --stream {args.stream} --image-scale {args.image_scale} --profile {args.profile}" + (" --keep-bag" if args.keep_bag else "")]
    (out/"docker_command.txt").write_text(" ".join(cmd)+"\n")
    try:
        return subprocess.run(cmd,check=False).returncode
    finally:
        if stage is not None: stage.cleanup()


if __name__ == "__main__": raise SystemExit(main())
