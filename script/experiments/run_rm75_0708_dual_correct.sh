#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/chenlvping/1_DM_work/vio_eval"
OUT="$ROOT/data/evaluation/workbench/rm75_0708_dual_correct_20260723"
VINS="$ROOT/data/gripper/gripper_data2/episode_20260618_0004/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml"
OFFSETS="$ROOT/data/evaluation/config/rm75_0708_empty_strict_sync_offsets.json"

run_pure_eval() {
  local dataset="$1"
  local source="$2"
  python3 "$ROOT/script/experiments/reevaluate_rm75_0708_stereo_ba10.py" \
    --episode-root "$ROOT/data/$dataset" --gt-root "$ROOT/data/$dataset" \
    --source-batch-root "$source" --output-root "$OUT/$dataset/stereo_ba10" \
    --offset-span-ms 180 --offset-step-ms 10
}

run_imu() {
  local dataset="$1"
  python3 "$ROOT/script/run_orbslam3_rm75_batch_eval.py" \
    --episode-root "$ROOT/data/$dataset" --gt-root "$ROOT/data/$dataset" \
    --output-root "$OUT/$dataset/stereo_imu_ba0_vins_noise" \
    --episode-pattern 'episode_20260708_*' --camera-rig stereo_right \
    --mode stereo-inertial --feature-preset low-texture \
    --nfeatures 3000 --ini-fast 12 --min-fast 3 --imu-fast-init 0 --final-ba-iters 0 \
    --vins-noise-mode orb_from_vins --vins-noise-only --fallback-vins-config "$VINS" \
    --min-inertial-coverage 0.25 --strict-sync-offset-json "$OFFSETS" \
    --strict-sync-offset-scan-span-ms 180 --strict-sync-offset-scan-step-ms 10 \
    --strict-sync-offset-scan-score composite --timeout-sec 1200 --skip-viewer
}

run_pure_eval 0708 "$ROOT/data/evaluation/workbench/rm75_0708_shadow_gate_20260722_r2/0708/orbslam3_batch_runs_20260722_170046"
run_imu 0708
run_pure_eval 0708_2 "$ROOT/data/evaluation/workbench/rm75_0708_shadow_gate_20260722_r2/0708_2/orbslam3_batch_runs_20260722_182408"
run_imu 0708_2
