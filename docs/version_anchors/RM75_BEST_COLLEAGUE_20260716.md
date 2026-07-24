# RM75 Best Colleague ORB-SLAM3 Anchor

## Pipeline

- ORB repo: `/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_colleague_sync`
- Sensor mode: **stereo only** (the generated YAML contains IMU calibration fields,
  but this anchor executes `Examples/Stereo/stereo_euroc_offline`; IMU is not fused)
- Backend: offline stereo, `GBA=100`, `Full-frame BA=10`
- Features: `nFeatures=1200`, `iniThFAST=20`, `minThFAST=7`
- Smoothing: position `window=21, poly=2`; rotation `window=9`
- Timing: per-episode offsets in `data/evaluation/config/rm75_colleague_best_strict_sync_offsets.json`
- Main entry: `script/run_orbslam3_rm75_best_batch.py`

## Formal baseline

- Result root: `data/evaluation/workbench/rm75_best_all_gt_20260716/`
- Episodes: `22`
- SE3 translation APE RMSE <= 10 mm: `12/22`
- Mean SE3 translation APE RMSE: `15.040 mm`
- Metric: TCP trajectory, rigid SE3 alignment (`scale=1`)

This is the accuracy floor to restore before accepting stereo+IMU changes. Do not
label these results as stereo-inertial merely because the generated settings file
contains `IMU.*` entries.

## Verification

The minimal strict-sync test is stored under:

`data/evaluation/workbench/colleague_all_gt_20260716/strictsync_minimal_test/`

For the ten tested episodes, this changed the mean APE from `17.682 mm` to
`10.080 mm` and the count under `10 mm` from `5/10` to `8/10`.

## Rollback

The previous clean RM75 anchor remains available through:

`script/dev/restore_orbslam3_best_version.sh`

This new anchor uses the separately maintained colleague repository. Its exact
repository state should be preserved outside this worktree before changing that
repository further.
