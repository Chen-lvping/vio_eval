#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
REPO_ROOT="$(realpath "$(dirname "$SCRIPT_PATH")/..")"
PY_DEPS_DIR="$REPO_ROOT/.deps/rosbags"

EPISODE_DIR="$REPO_ROOT/data/0704/episode_20260704_0009"
OUTPUT_DIR="$REPO_ROOT/data/evaluation/workbench/orb_0704_0009_docker"
DOCKER_IMAGE="my_orb_slam3:mcap"
HOST_HOME="/home/chenlvping"
ORB_ROOT="$HOST_HOME/已完成项目/orb3_build/ORB_SLAM3"
VINS_CONFIG=""
GEN_CONFIG="$HOST_HOME/5_skill/lwm/vinsfusion_ws/scripts/gen_orbslam3_config_from_calib.py"
CONVERTER="$REPO_ROOT/script/ugripper_episode_mcap_to_rosbag_v2_5.py"
CAMERA_RIG="stereo_right"
SIDE="right"
MODE="stereo-inertial"
FEATURE_PRESET="low-texture"
PLAY_RATE="1.0"
SHUTDOWN_WAIT="3"
TAIL_IMU_SEC="0.2"
MAX_FRAMES="0"
TIMEOUT_SEC="0"
IMU_FAST_INIT="0"
RUN_TS="$(date '+%Y%m%d_%H%M%S')"

usage() {
    cat <<EOF
Usage: bash script/run_orbslam3_ugripper_mcap_docker.sh [options]

Options:
  --episode-dir <dir>      Episode directory containing metadata.json + single .mcap
  --output-dir <dir>       Output directory
  --docker-image <image>   Docker image (default: $DOCKER_IMAGE)
  --orb-root <dir>         Host ORB_SLAM3 ROS workspace root
  --vins-config <file>     Optional VINS config to reuse when episode lacks one
  --camera-rig <name>      stereo_left or stereo_right
  --side <name>            left or right, must match camera-rig
  --mode <name>            stereo or stereo-inertial
  --feature-preset <name>  baseline, low-texture, aggressive
  --play-rate <x>          rosbag play rate
  --shutdown-wait <sec>    Wait after rosbag play before stopping ORB
  --tail-imu-sec <sec>     Keep IMU until image_end + this value
  --max-frames <n>         Debug-only frame limit, 0 means all
  --timeout-sec <n>        Kill docker run after this many seconds, 0 disables
  --imu-fast-init <0|1>    ORB IMU.fastInit passed to config generator
  -h, --help               Show help
EOF
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --episode-dir) EPISODE_DIR="$2"; shift 2 ;;
            --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
            --docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
            --orb-root) ORB_ROOT="$2"; shift 2 ;;
            --vins-config) VINS_CONFIG="$2"; shift 2 ;;
            --camera-rig) CAMERA_RIG="$2"; shift 2 ;;
            --side) SIDE="$2"; shift 2 ;;
            --mode) MODE="$2"; shift 2 ;;
            --feature-preset) FEATURE_PRESET="$2"; shift 2 ;;
            --play-rate) PLAY_RATE="$2"; shift 2 ;;
            --shutdown-wait) SHUTDOWN_WAIT="$2"; shift 2 ;;
            --tail-imu-sec) TAIL_IMU_SEC="$2"; shift 2 ;;
            --max-frames) MAX_FRAMES="$2"; shift 2 ;;
            --timeout-sec) TIMEOUT_SEC="$2"; shift 2 ;;
            --imu-fast-init) IMU_FAST_INIT="$2"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) die "Unknown argument: $1" ;;
        esac
    done
}

require_file() {
    [[ -e "$1" ]] || die "Missing path: $1"
}

resolve_paths() {
    EPISODE_DIR="$(realpath "$EPISODE_DIR")"
    OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
    ORB_ROOT="$(realpath "$ORB_ROOT")"
    GEN_CONFIG="$(realpath "$GEN_CONFIG")"
    CONVERTER="$(realpath "$CONVERTER")"
    mkdir -p "$OUTPUT_DIR"
    require_file "$EPISODE_DIR"
    require_file "$ORB_ROOT"
    require_file "$GEN_CONFIG"
    require_file "$CONVERTER"
    require_file "$EPISODE_DIR/metadata.json"
    require_file "$ORB_ROOT/Vocabulary/ORBvoc.txt"
    require_file "$ORB_ROOT/Examples_old/ROS/ORB_SLAM3/Stereo_Inertial"
}

resolve_vins_config() {
    if [[ -n "$VINS_CONFIG" ]]; then
        VINS_CONFIG="$(realpath "$VINS_CONFIG")"
        require_file "$VINS_CONFIG"
        echo "$VINS_CONFIG"
        return
    fi
    local candidate
    candidate="$EPISODE_DIR/pose_data_vins/vio_log/$SIDE/generated_config/StereoIMU-vinsfusion.yaml"
    require_file "$candidate"
    echo "$candidate"
}

find_mcap() {
    local count
    count="$(find "$EPISODE_DIR" -maxdepth 1 -type f -name '*.mcap' | wc -l)"
    [[ "$count" == "1" ]] || die "Expected exactly one .mcap under $EPISODE_DIR, found $count"
    find "$EPISODE_DIR" -maxdepth 1 -type f -name '*.mcap' | head -n 1
}

extract_calibration_json() {
    local mcap_path="$1"
    local out_json="$2"
    python3 - <<PY
from pathlib import Path
from mcap.reader import make_reader
import sys
mcap_path = Path(${mcap_path@Q})
out_json = Path(${out_json@Q})
with mcap_path.open("rb") as handle:
    reader = make_reader(handle)
    for _schema, channel, message in reader.iter_messages():
        if channel.topic == "/calibration":
            out_json.write_text(bytes(message.data).decode("utf-8"), encoding="utf-8")
            print(out_json)
            sys.exit(0)
raise SystemExit("missing /calibration in " + str(mcap_path))
PY
}

write_inner_script() {
    local path="$1"
    cat > "$path" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/noetic/setup.bash
export ROS_PACKAGE_PATH="/opt/ros/noetic/share:${ORB_ROOT}/Examples_old/ROS:${ROS_PACKAGE_PATH:-}"
export LD_LIBRARY_PATH="${ORB_ROOT}/lib:${ORB_ROOT}/Thirdparty/DBoW2/lib:${ORB_ROOT}/Thirdparty/g2o/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PY_DEPS_DIR}:${PYTHONPATH:-}"
export ROS_MASTER_URI="http://127.0.0.1:${ROS_PORT}"
export ROS_IP="127.0.0.1"
export ORB_SLAM3_TRAJECTORY_DIR="${OUTPUT_DIR}"

if ! python3 - <<'PY' >/dev/null 2>&1
import importlib
mods = ["rosbags", "mcap", "sensor_msgs", "rosbag", "rospy", "cv2", "numpy"]
missing = []
for name in mods:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit(1)
PY
then
    mkdir -p "${PY_DEPS_DIR}"
    python3 -m pip install --quiet --target "${PY_DEPS_DIR}" rosbags
fi

python3 "${GEN_CONFIG}" \
    "${SHIM_EPISODE_DIR}" \
    "${OUTPUT_DIR}/orbslam3_${CAMERA_RIG}_${MODE}.yaml" \
    --camera-rig "${CAMERA_RIG}" \
    --mode "${MODE}" \
    --feature-preset "${FEATURE_PRESET}" \
    --vins-config "${VINS_CONFIG}" \
    --vins-noise-mode orb_from_vins \
    --imu-fast-init "${IMU_FAST_INIT}" \
    > "${OUTPUT_DIR}/config_gen.log" 2>&1

python3 "${CONVERTER}" \
    --episode-dir "${EPISODE_DIR}" \
    --side "${SIDE}" \
    --output-bag "${OUTPUT_DIR}/aligned.bag" \
    --tail-imu-sec "${TAIL_IMU_SEC}" \
    --max-frames "${MAX_FRAMES}" \
    > "${OUTPUT_DIR}/convert_rosbag.log" 2>&1

roscore -p "${ROS_PORT}" > "${OUTPUT_DIR}/roscore.log" 2>&1 &
roscore_pid=$!
slam_pid=""

cleanup() {
    set +e
    if [[ -n "${slam_pid:-}" ]] && kill -0 "${slam_pid}" 2>/dev/null; then
        rosnode kill "/${ROS_NODE_NAME}" >/dev/null 2>&1 || kill -INT "${slam_pid}" 2>/dev/null || true
        sleep 1
        kill -TERM "${slam_pid}" 2>/dev/null || true
    fi
    if kill -0 "${roscore_pid}" 2>/dev/null; then
        kill -TERM "${roscore_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 80); do
    if rostopic list >/dev/null 2>&1; then
        break
    fi
    sleep 0.25
done
rostopic list >/dev/null 2>&1

cd "${OUTPUT_DIR}"
if [[ "${MODE}" == "stereo-inertial" ]]; then
    rosrun ORB_SLAM3 Stereo_Inertial \
        "${ORB_ROOT}/Vocabulary/ORBvoc.txt" \
        "${OUTPUT_DIR}/orbslam3_${CAMERA_RIG}_${MODE}.yaml" \
        false \
        false \
        /camera/left/image_raw:=/fays/atrak/cam0 \
        /camera/right/image_raw:=/fays/atrak/cam1 \
        /imu:=/fays/atrak/imu \
        _use_viewer:=false > "${OUTPUT_DIR}/orbslam3.log" 2>&1 &
    ROS_NODE_NAME="Stereo_Inertial"
else
    rosrun ORB_SLAM3 Stereo \
        "${ORB_ROOT}/Vocabulary/ORBvoc.txt" \
        "${OUTPUT_DIR}/orbslam3_${CAMERA_RIG}_${MODE}.yaml" \
        false \
        /camera/left/image_raw:=/fays/atrak/cam0 \
        /camera/right/image_raw:=/fays/atrak/cam1 \
        _use_viewer:=false > "${OUTPUT_DIR}/orbslam3.log" 2>&1 &
    ROS_NODE_NAME="Stereo"
fi
slam_pid=$!

sleep 2
rosbag play --quiet --clock -r "${PLAY_RATE}" "${OUTPUT_DIR}/aligned.bag" > "${OUTPUT_DIR}/rosbag_play.log" 2>&1
sleep "${SHUTDOWN_WAIT}"

rosnode kill "/${ROS_NODE_NAME}" > "${OUTPUT_DIR}/rosnode_kill.log" 2>&1 || kill -INT "${slam_pid}" 2>/dev/null || true

    for _ in $(seq 1 360); do
        if ! kill -0 "${slam_pid}" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done

if kill -0 "${slam_pid}" 2>/dev/null; then
    kill -TERM "${slam_pid}" 2>/dev/null || true
    sleep 1
fi
if kill -0 "${slam_pid}" 2>/dev/null; then
    kill -KILL "${slam_pid}" 2>/dev/null || true
fi
wait "${slam_pid}" 2>/dev/null || true
INNER
    chmod +x "$path"
}

main() {
    parse_args "$@"
    resolve_paths
    local vins_config mcap_path job_dir shim_episode_dir ros_port
    vins_config="$(resolve_vins_config)"
    mcap_path="$(find_mcap)"
    job_dir="$OUTPUT_DIR/docker_job_${RUN_TS}"
    shim_episode_dir="$OUTPUT_DIR/episode_shim"
    mkdir -p "$job_dir" "$shim_episode_dir"
    extract_calibration_json "$mcap_path" "$shim_episode_dir/calibration.json" >/dev/null
    ros_port="$((12311 + (RANDOM % 2000)))"
    write_inner_script "$job_dir/run_inside_container.sh"

    local docker_cmd=(
        docker run --rm
        --name "orbslam3_ugripper_${RUN_TS}"
        --net=host
        --ipc=host
        -e "EPISODE_DIR=$EPISODE_DIR"
        -e "OUTPUT_DIR=$OUTPUT_DIR"
        -e "SHIM_EPISODE_DIR=$shim_episode_dir"
        -e "PY_DEPS_DIR=$PY_DEPS_DIR"
        -e "ORB_ROOT=$ORB_ROOT"
        -e "GEN_CONFIG=$GEN_CONFIG"
        -e "CONVERTER=$CONVERTER"
        -e "VINS_CONFIG=$vins_config"
        -e "CAMERA_RIG=$CAMERA_RIG"
        -e "SIDE=$SIDE"
        -e "MODE=$MODE"
        -e "FEATURE_PRESET=$FEATURE_PRESET"
        -e "PLAY_RATE=$PLAY_RATE"
        -e "SHUTDOWN_WAIT=$SHUTDOWN_WAIT"
        -e "TAIL_IMU_SEC=$TAIL_IMU_SEC"
        -e "MAX_FRAMES=$MAX_FRAMES"
        -e "IMU_FAST_INIT=$IMU_FAST_INIT"
        -e "ROS_PORT=$ros_port"
        -v "$HOST_HOME:$HOST_HOME"
        -v "$REPO_ROOT:$REPO_ROOT"
        -v "$PY_DEPS_DIR:$PY_DEPS_DIR"
        -v "$OUTPUT_DIR:$OUTPUT_DIR"
        -v "$job_dir:$job_dir"
        "$DOCKER_IMAGE"
        bash "$job_dir/run_inside_container.sh"
    )

    echo "[RUN] ${docker_cmd[*]}"
    if [[ "$TIMEOUT_SEC" != "0" ]]; then
        timeout "$TIMEOUT_SEC" "${docker_cmd[@]}"
    else
        "${docker_cmd[@]}"
    fi
    echo "[OK] output dir: $OUTPUT_DIR"
}

main "$@"
