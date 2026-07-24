#!/usr/bin/env bash
# Offline ablation: separate camera-IMU phase correction from final visual BA.
set -euo pipefail

ROOT="/home/chenlvping/1_DM_work/vio_eval"
OUT="$ROOT/data/evaluation/workbench/rm75_0708_imu_upper_bound_ablation_20260723"
VINS="$ROOT/data/gripper/gripper_data2/episode_20260618_0004/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml"
RUNNER="$ROOT/script/run_orbslam3_tcp_eval.py"

mkdir -p "$OUT"

run_variant() {
  local episode="$1"
  local gt="$2"
  local label="$3"
  local ba_iters="$4"
  local shift_sec="$5"
  local work="$OUT/${episode}_${label}/run"
  local eval="$OUT/${episode}_${label}/evaluation"

  echo "[$(date --iso-8601=seconds)] episode=$episode label=$label ba=$ba_iters camera_imu_shift_sec=$shift_sec" >&2
  python3 "$RUNNER" \
    --episode-dir "$ROOT/data/0708/$episode" --ground-truth "$ROOT/data/0708/$gt" \
    --output-dir "$work" --eval-dir "$eval" --camera-rig stereo_right --mode stereo-inertial \
    --feature-preset low-texture --nfeatures 3000 --ini-fast 12 --min-fast 3 --imu-fast-init 0 \
    --vins-config "$VINS" --vins-noise-mode orb_from_vins --vins-noise-only \
    --camera-time-shift-sec "$shift_sec" --final-ba-iters "$ba_iters" \
    --min-inertial-coverage 0.25 --strict-sync-offset-scan-span-ms 180 \
    --strict-sync-offset-scan-step-ms 10 --strict-sync-offset-scan-score composite \
    --timeout-sec 1200 --skip-viewer
}

# The per-episode calibration reports cam0->IMU shifts of about -1.677 ms.
for pair in "episode_20260708_0001 rm75_pose_traj01.json" "episode_20260708_0004 rm75_pose_traj04.json"; do
  read -r episode gt <<< "$pair"
  run_variant "$episode" "$gt" ba0_calibrated_td 0 -0.00167747994
  run_variant "$episode" "$gt" ba10_calibrated_td 10 -0.00167747994
done
