# Full stereo vs stereo-IMU comparison

All values are translation APE RMSE in mm after each candidate's independent
strict-sync offset scan. `IMU BA0` and `IMU BA10` are raw stereo-inertial
outputs: no stereo shadow-gate or automatic stereo fallback was allowed for
the newly completed cells.

## Unified calibration

All rows use the `stereo_right` rig. The canonical calibration is
`data/evaluation/config/rm75_stereo_right_unified_historical_calibration.json`,
which resolves to `episode_20260623_0004/calibration.json` (SHA-256
`f02bf510fa4c74fe74950546257b0d721e5ba20c1cac9d9c5518e412dc861647`). The
main and 0708 calibration files differ only in inactive `stereo_left` and
left-side camera fields; the active `stereo_right` calibration block is the
same. Therefore the listed right-rig results are already on one calibration
definition, without an unnecessary 38-episode ORB rerun.

| Dataset / EP | Stereo | IMU BA0 | IMU BA10 | Status |
|---|---:|---:|---:|---|
| main/0001 | 10.546 | 15.934 | 14.955 | complete |
| main/0002 | 26.000 | 52.268 | 37.977 | BA10 completed 2026-07-24 |
| main/0003 | 13.178 | 8.904 | 10.471 | complete |
| main/0004 | 6.216 | 8.238 | 6.485 | BA10 completed 2026-07-24 |
| main/0005 | 36.825 | 25.182 | 36.045 | complete |
| main/0006 | 48.466 | 283.554 | 46.519 | BA10 from preserved Euroc input, historical calibration, recovered GT |
| main/0007 | 7.332 | 6.820 | 8.087 | complete |
| main/0008 | 11.554 | 575.008 | 681.155 | BA10 completed 2026-07-24 |
| main/0009 | 7.377 | 4.391 | 4.348 | complete |
| main/0010 | 13.007 | 12.318 | 9.000 | complete |
| main/0011 | 15.107 | 4.448 | 4.584 | complete |
| main/0012 | 3.760 | 3.468 | 4.028 | BA10 completed 2026-07-24 |
| main/0013 | 5.180 | 5.083 | 4.702 | BA10 completed 2026-07-24 |
| main/0014 | 5.227 | 5.155 | 5.073 | BA10 completed 2026-07-24 |
| main/0015 | 13.237 | 13.250 | 10.895 | complete |
| main/0016 | 4.510 | 3.913 | 3.938 | BA0 and BA10 completed 2026-07-24 |
| main/0017 | 4.382 | 4.113 | 4.518 | BA0 and BA10 completed 2026-07-24 |
| main/0018 | 8.617 | 7.932 | 6.622 | complete |
| main/0019 | 12.049 | 4.532 | 5.235 | complete |
| main/0020 | 5.329 | 7.659 | 6.493 | complete |
| main/0021 | 6.984 | 5.795 | 7.491 | complete |
| main/0022 | 8.186 | 7.866 | 9.376 | complete |
| 0708/0001 | 3.646 | 3.454 | 3.490 | complete |
| 0708/0004 | 4.243 | 4.739 | 4.249 | complete |
| 0708/0005 | 6.733 | 6.141 | 6.492 | complete |
| 0708/0006 | 7.402 | 6.440 | 7.488 | complete |
| 0708/0007 | 6.830 | 6.715 | 6.920 | complete |
| 0708/0008 | 6.358 | 6.726 | 6.866 | complete |
| 0708_2/0001 | 8.555 | 5.651 | 5.991 | complete |
| 0708_2/0002 | 8.677 | 6.759 | 6.314 | complete |
| 0708_2/0003 | 6.532 | 6.232 | 6.251 | complete |
| 0708_2/0004 | 10.394 | 8.235 | 7.471 | complete |
| 0708_2/0005 | 8.193 | 7.551 | 6.979 | complete |
| 0708_xht/0006 | 9.751 | 8.982 | 7.659 | complete |
| 0708_xht/0007 | 14.394 | 13.607 | 13.156 | complete |
| 0708_xht/0008 | 14.618 | 8.990 | 7.655 | complete |
| 0708_xht/0009 | 14.712 | 233.759 | 64.121 | complete |
| 0708_xht/0010 | 13.005 | 10.947 | 10.516 | historical calibration BA10 result |

## Newly completed run provenance

- BA10 raw IMU manifests: `missing_direct_imu_ba10/selected_runs.json`
- BA10 scan results: `missing_direct_imu_ba10_screen/fast_screen_20260724_130516/candidate_summary.csv`
- BA0 raw IMU manifests: `missing_direct_imu_ba0/selected_runs.json`
- BA0 scan results: `missing_direct_imu_ba0_screen/fast_screen_20260724_130848/candidate_summary.csv`
- main/0006 preserved-input BA10: `main_0006_historical_calib_ba10_screen/fast_screen_20260724_134155/candidate_summary.csv`
- 0708_xht/0010 historical-calibration BA10: `../rm75_0708_historical_calib_20260723/0708_xht_mcap/ba10/episode_20260708_0010/evaluation/summary.csv`
