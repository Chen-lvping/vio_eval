# Adaptive Stereo-Inertial Algorithm

## Reproducible Source

The optional RM75 adaptive VIO implementation is a patch over public
`ORB_SLAM3` commit `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`, rather than a
machine-local source snapshot. Its exact repository URL, commit, patch path,
and checksum are recorded in `algorithms/orbslam3_adaptive_vio/UPSTREAM.lock`.

Prepare a new checkout outside this repository:

```bash
script/setup_orbslam3_adaptive_vio.sh --orb-root /opt/ORB_SLAM3_adaptive_vio --build
```

The script accepts only an empty target. It checks the patch before applying
it, then runs `git diff --check`. ORB-SLAM3 build dependencies and its license
remain those of the upstream project.

## Algorithm Contract

The patched code makes three bounded changes:

1. `mImuPer` is `1 / IMU.Frequency` in both ORB-SLAM3 settings loaders.
2. Offline stereo-inertial playback can wait for Local Mapping to become idle
   after every frame.
3. An opt-in gate postpones IMU initialization until the metric stereo path
   has enough accumulated translation **or** rotation.

No robot ground truth, TCP alignment, strict-sync offset, or smoothing result
is used by the online initialization decision.

## One-Episode Run

```bash
export ORB_SLAM3_OFFLINE_WAIT_LOCAL_MAPPING=1
export ORB_SLAM3_OFFLINE_WAIT_TIMEOUT_SEC=10
export ORB_SLAM3_ADAPTIVE_IMU_INIT=1
export ORB_SLAM3_ADAPTIVE_IMU_INIT_MIN_TIME_SEC=1.5
export ORB_SLAM3_ADAPTIVE_IMU_INIT_MIN_TRANSLATION_M=0.08
export ORB_SLAM3_ADAPTIVE_IMU_INIT_MIN_ROTATION_DEG=5.0

python3 script/run_orbslam3_tcp_eval.py \
  --orb-root /opt/ORB_SLAM3_adaptive_vio \
  --episode-dir /data/episode_20260708_0001 \
  --ground-truth /data/rm75_pose_traj01.json \
  --camera-rig stereo_right --mode stereo-inertial \
  --no-smooth-trajectory --strict-sync-offset-sec 0
```

Keep the `[ADAPTIVE_IMU_INIT] defer` and `gate pass` logs with each run. Use a
fixed zero offset and no smoothing for algorithm ablations; run TCP evaluation
only after trajectory generation has completed.

## Ablation

The provided runner compares baseline, translation-only, rotation-only, and
combined gates under the same deterministic offline policy:

```bash
python3 script/run_adaptive_imu_init_ablation.py \
  --orb-root /opt/ORB_SLAM3_adaptive_vio \
  --episode-root /data/episodes --gt-root /data/ground_truth \
  --output-root local/adaptive_vio_ablation
```

It writes a CSV manifest with environment values, commands, output paths, and
post-run metrics. Threshold fitting should use leave-one-date-out validation;
the published visualization portfolio is not a held-out benchmark.
