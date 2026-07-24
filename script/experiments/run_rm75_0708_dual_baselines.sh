#!/usr/bin/env bash
# Run comparable offline baselines sequentially after the targeted IMU retest.
set -euo pipefail

ROOT="/home/chenlvping/1_DM_work/vio_eval"
OUT_ROOT="$ROOT/data/evaluation/workbench/rm75_0708_dual_baseline_20260723"
VINS_CONFIG="$ROOT/data/gripper/gripper_data2/episode_20260618_0004/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml"
EMPTY_OFFSETS="$ROOT/data/evaluation/config/rm75_0708_empty_strict_sync_offsets.json"
RUNNER="$ROOT/script/run_orbslam3_rm75_batch_eval.py"

mkdir -p "$OUT_ROOT"

run_batch() {
  local dataset="$1"
  local label="$2"
  local mode="$3"
  local final_ba_iters="$4"
  shift 4

  echo "[$(date --iso-8601=seconds)] START dataset=$dataset label=$label mode=$mode ba=$final_ba_iters" >&2
  python3 "$RUNNER" \
    --episode-root "$ROOT/data/$dataset" \
    --gt-root "$ROOT/data/$dataset" \
    --output-root "$OUT_ROOT/$dataset/$label" \
    --episode-pattern 'episode_20260708_*' \
    --camera-rig stereo_right \
    --mode "$mode" \
    --feature-preset low-texture \
    --nfeatures 3000 --ini-fast 12 --min-fast 3 \
    --imu-fast-init 0 \
    --final-ba-iters "$final_ba_iters" \
    --min-inertial-coverage 0.25 \
    --strict-sync-offset-json "$EMPTY_OFFSETS" \
    --strict-sync-offset-scan-span-ms 180 \
    --strict-sync-offset-scan-step-ms 10 \
    --strict-sync-offset-scan-score composite \
    --timeout-sec 1200 \
    --skip-viewer \
    "$@"
  echo "[$(date --iso-8601=seconds)] DONE dataset=$dataset label=$label" >&2
}

# The stereo baseline uses the proven final BA10 post-processing.  The IMU
# baseline uses the historical VINS noise model but retains each episode's
# calibration-derived camera/IMU extrinsics.
run_batch 0708 stereo_ba10 stereo 10
run_batch 0708 stereo_imu_ba0_vins_noise stereo-inertial 0 \
  --vins-noise-mode orb_from_vins \
  --vins-noise-only \
  --fallback-vins-config "$VINS_CONFIG"
run_batch 0708_2 stereo_ba10 stereo 10
run_batch 0708_2 stereo_imu_ba0_vins_noise stereo-inertial 0 \
  --vins-noise-mode orb_from_vins \
  --vins-noise-only \
  --fallback-vins-config "$VINS_CONFIG"
