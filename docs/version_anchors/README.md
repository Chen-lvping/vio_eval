# Version Anchors

This directory preserves rollback points for important working versions that may
exist as a base git commit plus local uncommitted deltas.

## Files

- `RM75_BEST_ORBSLAM3_20260707.md`
  Human-readable note for the current best-known RM75 ORB-SLAM3 baseline.
- `patches/*.patch`
  Exact code delta snapshots used to recreate the anchored working tree.

## Restore

Use:

- [restore_orbslam3_best_version.sh](/home/chenlvping/1_DM_work/vio_eval/script/dev/restore_orbslam3_best_version.sh)

Review the script before running it. It is intentionally explicit because the
restore flow resets the target repo to a known base commit before applying the
anchored patch.
