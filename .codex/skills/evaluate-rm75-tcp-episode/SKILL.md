---
name: evaluate-rm75-tcp-episode
description: 为 `vio_eval` 项目执行单条 RM75 episode 的 TCP 精度评估。用于用户提出“评估这条 VIO / ORB / VINS 结果 / 跑 TCP evo / 对齐 ground truth / strict-sync offset / 看这一条 episode 指标”等请求时：先阅读 `README.md`、`script/README.md` 与相关评估脚本，确认输入轨迹、`camera_rig`、坐标链和 GT 文件，再优先复用仓库现有 runner，而不是临时拼评估链。
---

# evaluate-rm75-tcp-episode

## 固定执行顺序

1. 先阅读 `README.md` 和 `script/README.md`，确认当前主评估链。
2. 找到用户要评估的输入类型：
   - 直接已有 `pose_data.csv` / `*.tum`
   - 某条 `episode_*` 需要先跑 ORB-SLAM3
   - 已有某个实验输出目录，只需重评估
3. 确认 4 个关键前提：
   - `ground_truth` 用的是哪条 RM75 JSON
   - `camera_rig` 是 `stereo_left` 还是 `stereo_right`
   - 轨迹 pose 是 `world->imu`、`world->camera` 还是已经到 TCP
   - 是否需要 `strict-sync offset`
4. 优先复用现有脚本：
   - 通用底座：`python3 script/evaluate_vins_accuracy.py ...`
   - 当前主入口：`python3 script/evaluate_vio_tcp_camera_evo.py ...`
   - 若要从 episode 直接跑 ORB：`python3 script/run_orbslam3_tcp_eval.py ...`
5. 若用户只要求看某个已有结果，不重复跑长流程，优先读取已有 `summary.csv`、`manifest`、viewer 输出。

## 当前默认口径

- RM75 主线优先使用 2026-06-17 raw-pose GT 和 2026-06-18 ORB/VINS 对比链。
- 评估时必须显式说明坐标系和变换方向，尤其是 `T_imu_cam`、`T_cam_imu`、`T_world_imu`。
- 不要默认把 `pose_data.csv` 当作 camera pose；若定义不清，先查脚本或 manifest。
- `strict-sync offset` 是后处理补偿项，不是物理真值本身。

## 常用入口

- `script/evaluate_vins_accuracy.py`
- `script/evaluate_vio_tcp_camera_evo.py`
- `script/run_orbslam3_tcp_eval.py`
- `script/postprocess/resample_pose_csv_to_gt_timestamps.py`
- `script/postprocess/apply_tcp_residual_correction.py`

## 输出要求

1. 本次评估使用的输入轨迹、GT、`camera_rig` 和 offset 口径。
2. 关键指标，至少包含 `APE / RPE`，必要时补姿态误差。
3. 若结果异常，先说明更像是时间相位、pose 定义、外参链还是 GT 语义问题。
4. 若复用了已有结果，指出结果目录和核心文件。
