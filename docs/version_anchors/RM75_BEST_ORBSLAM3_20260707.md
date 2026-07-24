# RM75 Best ORB-SLAM3 Anchor

## Summary

- Date: `2026-07-07`
- Scope: `ORB_SLAM3_clean` stereo-inertial RM75 baseline used by `vio_eval`
- ORB repo: `/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_clean`
- ORB base commit: `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`
- ORB tag: `rm75-best-base-20260707`

## Preserved Delta

The anchored working-tree delta is stored at:

- [orbslam3_clean_rm75_best_20260707.patch](/home/chenlvping/1_DM_work/vio_eval/docs/version_anchors/patches/orbslam3_clean_rm75_best_20260707.patch)

This anchor intentionally preserves the small RM75-specific fixes that were kept
after rolling back the later visual-first initialization experiments:

- `Tracking.cc`: derive `mImuPer` from configured IMU frequency
- `stereo_inertial_euroc.cc`: guard the IMU index upper bound during export playback

## Reference Evaluation

Best-known reference batch:

- [batch_summary.csv](/home/chenlvping/1_DM_work/vio_eval/data/evaluation/workbench/orbslam3_rm75_batch_eval_20260630_103032/batch_summary.csv)

Mean metrics on `gripper_data_6_24`:

- APE RMSE: `7.601054 mm`
- RPE RMSE: `1.601635 mm`

## Restore

Run:

```bash
/home/chenlvping/1_DM_work/vio_eval/script/dev/restore_orbslam3_best_version.sh
```
