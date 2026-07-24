#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/chenlvping/1_DM_work/vio_eval"
OUT="$ROOT/data/evaluation/workbench/rm75_0708_imu_ba10_vins_noise_20260723"
VINS="$ROOT/data/gripper/gripper_data2/episode_20260618_0004/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml"
OFFSETS="$ROOT/data/evaluation/config/rm75_0708_empty_strict_sync_offsets.json"

run_standard_dataset() {
  local dataset="$1"
  python3 "$ROOT/script/run_orbslam3_rm75_batch_eval.py" \
    --episode-root "$ROOT/data/$dataset" --gt-root "$ROOT/data/$dataset" \
    --output-root "$OUT/$dataset" --episode-pattern 'episode_20260708_*' \
    --camera-rig stereo_right --mode stereo-inertial --feature-preset low-texture \
    --nfeatures 3000 --ini-fast 12 --min-fast 3 --imu-fast-init 0 --final-ba-iters 10 \
    --vins-noise-mode orb_from_vins --vins-noise-only --fallback-vins-config "$VINS" \
    --min-inertial-coverage 0.25 --strict-sync-offset-json "$OFFSETS" \
    --strict-sync-offset-scan-span-ms 180 --strict-sync-offset-scan-step-ms 10 \
    --strict-sync-offset-scan-score composite --timeout-sec 1200 --skip-viewer
}

run_standard_dataset 0708
run_standard_dataset 0708_2

source /opt/ros/humble/setup.bash
exec /usr/bin/python3 "$ROOT/script/experiments/run_rm75_0708_xht_imu_ba10.py" \
  --episode-root "$ROOT/data/0708_xht_mcap" --gt-root "$ROOT/data/0708_xht_mcap" \
  --output-root "$OUT/0708_xht_mcap" --vins-config "$VINS" \
  --timeout-sec 1200 --offset-span-ms 180 --offset-step-ms 10
