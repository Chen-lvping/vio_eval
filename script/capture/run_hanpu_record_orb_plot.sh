#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/v4l/by-id/usb-YCTC_YCTC_SC233HGS_0152312181647-video-index0}"
IMU_PORT="${IMU_PORT:-/dev/ttyACM0}"
DURATION_SEC="${DURATION_SEC:-30}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/hanpu/runs}"
ORB_ROOT="${ORB_ROOT:-/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_clean}"
CALIBRATION_JSON="${CALIBRATION_JSON:-${REPO_ROOT}/data/hanpu/calibration/stereo_calibration.json}"

CAPTURE_OUTPUT="$({
  cd "${REPO_ROOT}"
  python3 script/capture/run_hanpu_orbslam3_smoketest.py \
    --video-device "${VIDEO_DEVICE}" \
    --imu-port "${IMU_PORT}" \
    --duration-sec "${DURATION_SEC}" \
    --output-root "${OUTPUT_ROOT}" \
    --keep-raw \
    --no-orb \
    --no-top-crop
} | tee /dev/stderr)"

RUN_ROOT="$(printf '%s\n' "${CAPTURE_OUTPUT}" | sed -n 's/^run_root=//p' | tail -n1)"
if [[ -z "${RUN_ROOT}" || ! -d "${RUN_ROOT}" ]]; then
  echo "Could not determine capture run directory" >&2
  exit 4
fi

REAL_YAML="${RUN_ROOT}/hanpu_stereo_real_1920x1080.yaml"
python3 "${REPO_ROOT}/script/capture/hanpu_calibration_to_orbslam3_yaml.py" \
  --calibration-json "${CALIBRATION_JSON}" \
  --output "${REAL_YAML}" \
  --width 1920 \
  --height 1080 \
  --fps 30

export LD_LIBRARY_PATH="${ORB_ROOT}/lib:${ORB_ROOT}/Thirdparty/DBoW2/lib:${ORB_ROOT}/Thirdparty/g2o/lib:/home/chenlvping/1_DM_work/orbslam3_fresh/Pangolin_v06/install/lib:${LD_LIBRARY_PATH:-}"
ORB_DIR="${RUN_ROOT}/orbslam3_real_run"
mkdir -p "${ORB_DIR}"
cd "${ORB_DIR}"

ORB_SLAM3_ENABLE_VIEWER="${ORB_SLAM3_ENABLE_VIEWER:-0}" \
  "${ORB_ROOT}/Examples/Stereo/stereo_euroc" \
  "${ORB_ROOT}/Vocabulary/ORBvoc.txt" \
  "${REAL_YAML}" \
  "${RUN_ROOT}/euroc_export" \
  "${RUN_ROOT}/euroc_export/times.txt" \
  hanpu_real_stereo

echo "run_root=${RUN_ROOT}"
echo "trajectory=${ORB_DIR}/f_hanpu_real_stereo.txt"
evo_traj euroc "${ORB_DIR}/f_hanpu_real_stereo.txt" --plot --plot_mode xyz
