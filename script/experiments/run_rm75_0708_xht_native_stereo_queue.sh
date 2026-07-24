#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/chenlvping/1_DM_work/vio_eval"
OUT="$ROOT/data/evaluation/workbench/rm75_0708_xht_native_stereo_20260724"

source /opt/ros/humble/setup.bash
exec /usr/bin/python3 "$ROOT/script/experiments/run_rm75_0708_xht_native_stereo.py" \
  --episode-root "$ROOT/data/0708_xht_mcap" \
  --gt-root "$ROOT/data/0708_xht_mcap" \
  --output-root "$OUT" \
  --camera-time-shift-sec -0.00150879554 \
  --timeout-sec 1200 \
  --offset-span-ms 180 \
  --offset-step-ms 10
