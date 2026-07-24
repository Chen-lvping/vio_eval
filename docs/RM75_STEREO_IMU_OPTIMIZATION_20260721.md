# RM75 Unified Stereo-IMU Optimization Status

## Delivered mainline

- One ORB-SLAM3 stereo-inertial algorithm configuration: `3000 / 12 / 3`,
  `IMU.fastInit=0`, validated calibration profiles and 25% inertial coverage gate.
- Every episode keeps its own camera intrinsics, camera-IMU extrinsics and
  strict-sync offset. These are calibration/evaluation inputs, not algorithm
  tuning parameters.
- Stereo-only remains a degradation path for failed/incomplete inertial
  initialization.
- A GT-free stereo shadow gate now compares the stereo-inertial trajectory
  against the validated offline stereo ORB-SLAM3 branch after SE3 alignment.
  At more than `15 mm` disagreement it selects offline stereo (`GBA=100`,
  full-frame BA=10, smoothing 21/2/9) and records the decision in provenance.
- Batch execution is serial. ORB shutdown now waits until LocalMapping,
  LoopClosing and any running GBA have stopped before trajectory export.
- Final full inertial BA is implemented and exposed by `--final-ba-iters`, but
  stays disabled by default because the high-accuracy regression probe became
  worse (`10.97 mm` without final BA, `11.63 mm` with 10 iterations).

Main entry point:

```bash
python3 script/run_rm75_unified_stereo_imu.py --skip-viewer
```

## Verified results and current gap

### Camera-IMU convention A/B

An attempted switch from the historically validated VINS Tbc/Tlr convention to
the raw `calibration.json::T_ic_cam0_to_imu0` direction was rejected. Although
it helped two isolated post-process probes, fresh 20260618 runs diverged to
`764 mm` and `6519 mm` with severe inertial scale failure. The mainline therefore
retains the 20260707 VINS-generated convention and the established TCP evaluator
chain. Values above 50 mm are treated as configuration failures, not accepted as
algorithm results.

The shadow gate was verified on three representative episodes:

- `episode_gripper_0002`: disagreement `46.405 mm`, fallback selected, APE
  improved from `46.752 mm` to `13.387 mm`.
- `episode_gripper_0012`: disagreement `3.887 mm`, stereo-inertial retained,
  APE `4.133 mm`.
- `episode_gripper_0008`: disagreement `404.847 mm`, fallback selected, but
  offline stereo still gives `59.430 mm`; this remains a special data-quality
  outlier rather than an accepted algorithm result.

The stale complete gate reported `13/20 <= 20 mm` and `4/20 <= 10 mm` because
it mixed incorrect 6/24 offsets and forced Tbc values. After restoring the
validated calibration/time profiles, the auditable recovered baseline is
`14/20 <= 20 mm` and `6/20 <= 10 mm`. It still does not meet the requested
`15 <= 10 mm` target and must not be presented as achieved.

The 6/24 strict-sync offsets had been overwritten by unrelated approximately
`-85 ms` values. Restoring the previously verified offsets reproduces the
historical direction where five of seven 6/24 episodes were below 10 mm:
`5.50, 11.58, 5.94, 5.27, 5.79, 7.16, 12.29 mm`. The fresh recovery batch
reproduced the baseline direction at
`5.62, 10.95, 6.22, 5.24, 5.77, 7.11, 12.85 mm` (five of seven below 10 mm).
A shared smoothing sweep then replaced `w7/p2` with `w5/p1`; all seven improved
to `4.46, 10.18, 4.66, 3.87, 4.64, 4.86, 12.23 mm` without changing the
five-of-seven sub-10-mm count.
The recovered cross-dataset baseline and gate are stored under
`data/evaluation/workbench/rm75_recovered_baseline_20260721/`.

Episode 0008 is a genuine algorithmic outlier: stereo-inertial APE is hundreds
of millimetres while local RPE stays near 4 mm. Stereo-only reduces it to about
56 mm but does not reach 20 mm. This points to low-frequency inertial
scale/bias drift, not feature count or timestamp association.

## External engineering references

- UMI uses a dedicated ORB-SLAM3 fork and explicitly calls ORB-SLAM3 the most
  fragile part of its pipeline. Its practical lesson is to combine careful
  calibration/data quality with explicit success-rate gating:
  <https://github.com/real-stanford/universal_manipulation_interface>
- Basalt provides stereo visual-inertial odometry/mapping and square-root
  marginalization. It is the strongest next candidate for a deterministic
  sliding-window comparison on these EuRoC exports:
  <https://github.com/VladyslavUsenko/basalt>
- Kimera-VIO is a stereo+IMU pipeline and documents sequential execution
  (`parallel_run=false`) for deterministic evaluation:
  <https://github.com/MIT-SPARK/Kimera-VIO>
- ORB-SLAM3 paper/repository remains the selected mainline because this
  workspace already has calibrated exporters, evaluation and the best verified
  RM75 results on it: <https://github.com/UZ-SLAMLab/ORB_SLAM3>

## Next algorithm work

1. Run the unified entry point serially over all 22 episodes and regenerate the
   regression gate with corrected calibration/time inputs.
2. Export IMU bias, gravity, inertial initialization state and map resets from
   ORB. Use those internal signals—not GT—to detect 0008-like low-frequency
   divergence and trigger stereo degradation.
3. Run Basalt on the same EuRoC exports for a fair fixed-config comparison.
   Replace ORB only if held-out episodes improve without regressing the current
   sub-10-mm set.
