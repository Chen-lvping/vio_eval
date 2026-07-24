#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORB_ROOT="${ORB_ROOT:-/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_clean}"
PANGOLIN_ROOT="${PANGOLIN_ROOT:-/home/chenlvping/1_DM_work/orbslam3_fresh/Pangolin_v06/install}"
BUILD_DIR="${REPO_ROOT}/data/hanpu/live_build"
BIN_PATH="${BUILD_DIR}/hanpu_live_orbslam3_stereo"
SETTINGS_PATH="${SETTINGS_PATH:-${REPO_ROOT}/data/hanpu/hanpu_live_orbslam3_stereo_placeholder.yaml}"
SRC_PATH="${REPO_ROOT}/script/capture/hanpu_live_orbslam3_stereo.cc"
VIDEO_DEVICE="${VIDEO_DEVICE:-}"
GENERATE_PLACEHOLDER_SETTINGS="${GENERATE_PLACEHOLDER_SETTINGS:-1}"

mkdir -p "${BUILD_DIR}"

if [[ -z "${VIDEO_DEVICE}" ]]; then
  for cand in /dev/v4l/by-id/*YCTC*video-index0 /dev/video0 /dev/video1 /dev/video2; do
    if [[ -e "${cand}" ]]; then
      VIDEO_DEVICE="$(readlink -f "${cand}" 2>/dev/null || printf '%s' "${cand}")"
      break
    fi
  done
fi

if [[ -z "${VIDEO_DEVICE}" ]]; then
  echo "No YCTC video device found. Try: ls -l /dev/v4l/by-id /dev/video*" >&2
  exit 2
fi

if [[ "${GENERATE_PLACEHOLDER_SETTINGS}" == "1" ]]; then
  python3 - <<'PY' "${SETTINGS_PATH}"
from pathlib import Path
import sys
from script.capture.run_hanpu_orbslam3_smoketest import write_placeholder_orb_settings

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
write_placeholder_orb_settings(path, 1920, 540, 30.0, 0.06)
print(path)
PY
else
  if [[ ! -f "${SETTINGS_PATH}" ]]; then
    echo "Settings file not found: ${SETTINGS_PATH}" >&2
    exit 3
  fi
  echo "Using settings: ${SETTINGS_PATH}"
fi

g++ -std=c++17 -O2 -w -include GL/glew.h \
  -I"${ORB_ROOT}" \
  -I"${ORB_ROOT}/include" \
  -I"${ORB_ROOT}/include/CameraModels" \
  -I"${ORB_ROOT}/Thirdparty/Sophus" \
  -I"${PANGOLIN_ROOT}/include" \
  -I/usr/include/eigen3 \
  $(pkg-config --cflags opencv4) \
  "${SRC_PATH}" \
  -L"${ORB_ROOT}/lib" \
  -L"${PANGOLIN_ROOT}/lib" \
  -Wl,-rpath,"${ORB_ROOT}/lib" \
  -Wl,-rpath,"${PANGOLIN_ROOT}/lib" \
  -lORB_SLAM3 \
  -lpangolin \
  $(pkg-config --libs opencv4) \
  -lGLEW -lGL \
  -lpthread \
  -o "${BIN_PATH}"

export LD_LIBRARY_PATH="${ORB_ROOT}/lib:${ORB_ROOT}/Thirdparty/DBoW2/lib:${ORB_ROOT}/Thirdparty/g2o/lib:/home/chenlvping/1_DM_work/orbslam3_fresh/Pangolin_v06/install/lib:${LD_LIBRARY_PATH:-}"
exec "${BIN_PATH}" "${ORB_ROOT}/Vocabulary/ORBvoc.txt" "${SETTINGS_PATH}" --video-device "${VIDEO_DEVICE}" "$@"
