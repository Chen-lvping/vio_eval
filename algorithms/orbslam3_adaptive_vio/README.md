# Adaptive ORB-SLAM3 VIO

This package makes the RM75 adaptive stereo-inertial algorithm reproducible
without vendoring an entire ORB-SLAM3 checkout. It pins the public upstream
source revision and carries the small, auditable RM75 delta as a patch.

## What Changes

- Uses the configured `IMU.Frequency` to set the IMU sample period.
- Optionally delays inertial initialization until the stereo trajectory has
  sufficient cumulative translation or rotation.
- Lets offline playback wait for Local Mapping after each frame, removing
  thread scheduling as a source of run-to-run variation.

All changes are disabled by default except the IMU-period correction. Enable
the adaptive guard through the environment variables recorded in
`docs/ALGORITHM_REPRODUCTION.md`.

## Prepare

```bash
script/setup_orbslam3_adaptive_vio.sh --orb-root /opt/ORB_SLAM3_adaptive_vio --build
```

The preparation script clones the exact upstream commit, checks the patch hash,
and refuses to modify a non-empty target directory. Review the upstream
ORB-SLAM3 license and install its build dependencies before running `--build`.
