# Script Layout

`script/` 现在分成两层：

- 根目录保留最常用的主入口
- 子目录收纳诊断、实验、扫描、可视化和后处理工具

## Delivery Entry Point

For final delivery, start with `vio_eval.py`, not the historical runners:

```bash
python3 script/vio_eval.py demo --output-dir local/minimal_reproduction
python3 script/vio_eval.py evaluate --help
```

It evaluates one exported trajectory against RM75 TCP ground truth and keeps the evaluation contract independent from machine-local ORB-SLAM3 builds. See `docs/DELIVERY_SOP.md` for acceptance and handoff steps. The remaining scripts are retained as integration, acquisition, diagnostics, or research tools.

## Root Entrypoints

- `check_time_sync.py`
- `evaluate_vins_accuracy.py`
- `evaluate_vio_tcp_camera_evo.py`
- `get_rm75_end_pose.py`
- `handeye_calibrate_aprilgrid.py`
- `record_trajectory.py`
- `replay_rm75_tcp_pose_trajectory.py`
- `run_orbslam3_rm75_batch_eval.py`
- `run_orbslam3_tcp_eval.py`
- `sync_remote_ptp_to_local.py`
- `visualize_single_tcp_trajectory_3d.py`

## Subdirectories

- `capture/`
  采集辅助和 SDK 检查脚本
- `calibration/`
  标定和姿态重建辅助脚本
- `diagnose/`
  一次性诊断和跨 episode 分析
- `eval/`
  备用评估入口和旧流程 runner
- `experiments/`
  RM75 / ORB 时间偏移、残差、迁移实验
- `postprocess/`
  轨迹平滑、重采样、CSV 转换、残差校正
- `scan/`
  参数和约定扫描脚本
- `visualize/`
  专项 viewer 和对比可视化
- `vendor/`
  固化的外部 ORB-SLAM3 运行辅助脚本；不包含第三方 SLAM 源码或构建产物

## Quick Map

- 采集机械臂数据：先看 `check_time_sync.py`、`sync_remote_ptp_to_local.py`、`get_rm75_end_pose.py`、`record_trajectory.py`
- 探测 YCTC 双目 IMU 设备：先看 `script/capture/probe_yctc_hanpu.py`
- 录 Hanpu/YCTC 双目 IMU 并做 ORB-SLAM3 smoke test：先看 `script/capture/run_hanpu_orbslam3_smoketest.py`
- 手动按 `q` 结束 Hanpu 录制，再自动导出、跑 ORB 和画轨迹：先看 `script/capture/run_hanpu_manual_record_orb_plot.py`
- 实时看 Hanpu/YCTC 的 ORB-SLAM3 stereo 效果：先看 `script/capture/run_hanpu_live_orbslam3_stereo.sh`
- 按 TCP 位姿严格回放：先看 `replay_rm75_tcp_pose_trajectory.py`
- 时间同步 SOP：见 [SOP_rm75_time_sync.md](/home/chenlvping/1_DM_work/vio_eval/script/SOP_rm75_time_sync.md:1)
- 重教轨迹平滑回放 SOP：见 [SOP_reteach_trajectory_replay.md](/home/chenlvping/1_DM_work/vio_eval/script/SOP_reteach_trajectory_replay.md:1)
- 主评估链路：先看 `run_orbslam3_tcp_eval.py`、`run_orbslam3_rm75_batch_eval.py`、`evaluate_vio_tcp_camera_evo.py`
- SXR CSV tracking：先看 `run_sxr_tracking_basalt_batch.py`；单 episode runner 位于 `run_sxr_csv_*.py` 与 `run_sxr_*.py`
- RM75 最终可视化：先看 `visualize/build_rm75_camera_trajectory_portfolio.py`
- 排错：优先去 `diagnose/`、`scan/`、`experiments/`
- 额外可视化：去 `visualize/`
# 主推荐入口

RM75 当前最佳方案使用：

```bash
python3 script/run_orbslam3_rm75_best_batch.py \
  --episode-root data/gripper/gripper_data_6_24 \
  --gt-root data/ground_truth/rm75_6_24 \
  --episode-pattern 'episode_20260624_*'
```

默认配置为同事版 ORB-SLAM3 offline **纯双目**、`GBA=100`、`Full-frame BA=10`、平移平滑 `window=21/poly=2`、旋转平滑 `window=9`，并读取 `data/evaluation/config/rm75_colleague_best_strict_sync_offsets.json`。这条历史基线为 22 条中 12 条 APE <= 10 mm、平均 APE 15.040 mm；生成的 YAML 虽含 `IMU.*` 字段，但实际运行的是 `Examples/Stereo/stereo_euroc_offline`，不融合 IMU。双目+IMU 优化使用 `run_rm75_unified_stereo_imu.py`，评估时必须与该基线使用相同 TCP/SE3 口径。

旧版入口仍保留：`script/run_orbslam3_rm75_batch_eval.py`。
