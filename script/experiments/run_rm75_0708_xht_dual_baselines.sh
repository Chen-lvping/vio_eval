#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
ROOT="/home/chenlvping/1_DM_work/vio_eval"
exec /usr/bin/python3 "$ROOT/script/experiments/run_rm75_0708_xht_dual_baselines.py" \
  --episode-root "$ROOT/data/0708_xht_mcap" \
  --gt-root "$ROOT/data/0708_xht_mcap" \
  --output-root "$ROOT/data/evaluation/workbench/rm75_0708_xht_dual_baseline_20260723" \
  --vins-config "$ROOT/data/gripper/gripper_data2/episode_20260618_0004/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml" \
  --timeout-sec 1200 --offset-span-ms 180 --offset-step-ms 10
