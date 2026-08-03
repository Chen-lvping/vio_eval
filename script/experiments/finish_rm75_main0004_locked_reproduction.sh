#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/chenlvping/1_DM_work/vio_eval
RUNTIME=/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_exact_20260716
OUTPUT="$ROOT/data/evaluation/workbench/rm75_historical_exact_reproduction_20260731/main_0004"
ACTIVE_PARENT_PID=2237512

if kill -0 "$ACTIVE_PARENT_PID" 2>/dev/null; then
  tail --pid="$ACTIVE_PARENT_PID" -f /dev/null
fi

test -s "$OUTPUT/pose_raw.csv"

ionice -c3 nice -n 19 taskset -c 0 \
  python3 "$RUNTIME/scripts/smooth_pose_csv.py" \
  --input-csv "$OUTPUT/pose_raw.csv" \
  --output-csv "$OUTPUT/pose_smooth_strictsync_m115.csv" \
  --position-window 21 \
  --position-poly 2 \
  --rotation-window 9 \
  --timestamp-offset-sec -0.115

ionice -c3 nice -n 19 taskset -c 0 \
  python3 "$RUNTIME/scripts/run_fays_orbslam3_stereo_right.py" \
  --evaluate-only \
  --tracking-mode stereo \
  --eval-pose-frame cam0 \
  --episode-dir "$ROOT/data/gripper/gripper_data2/episode_20260618_0004" \
  --calibration-json "$ROOT/data/evaluation/config/rm75_stereo_right_unified_historical_calibration_locked_20260716.json" \
  --camera-key stereo_right \
  --orbslam-binary "$RUNTIME/Examples/Stereo/stereo_euroc_offline" \
  --vocabulary "$RUNTIME/Vocabulary/ORBvoc.txt" \
  --work-dir "$OUTPUT" \
  --output-csv "$OUTPUT/pose_smooth_strictsync_m115.csv" \
  --eval-dir "$OUTPUT/eval_m115" \
  --robot-json "$ROOT/data/ground_truth/rm75/rm75_pose_traj_4.json" \
  --trajectory-name historical_exact_main_0004_ffba10_m115 \
  --t-max-diff 0.01
