# AI Onboarding Prompt For `vio_eval`

把下面这段直接发给新的 AI 对话窗口：

```text
你现在位于项目 `/home/chenlvping/1_DM_work/vio_eval`。

这是一个独立的 VIO / 手眼标定 / 轨迹精度评估项目，请把它当成一个新的工作区，不要再默认引用旧项目 `/home/chenlvping/1_DM_work/Auto_calibration/ARM_trajectory` 下的脚本和数据，除非我明确说明要回看旧项目。

这个项目的目标主要有四类：
1. 采集机械臂静态位姿与轨迹真值
2. 基于 AprilGrid 和机械臂位姿做手眼标定
3. 评估 VINS / DynaVINS / 其他 VIO 轨迹精度
4. 做轨迹、姿态误差、对齐结果的可视化与诊断

项目结构：
- `script/`：主脚本目录，内部已经按 `capture/`、`diagnose/`、`eval/`、`experiments/`、`postprocess/`、`scan/`、`visualize/` 分组
- `data/`：标定输入、轨迹样本、评估结果、可视化输出
- `README.md`：项目概览

请你先做这些事，再开始回答我后续问题：
1. 先阅读 `README.md`
2. 再阅读这些核心脚本，理解职责和输入输出：
   - `script/capture/capture_static_pose.py`
   - `script/handeye_calibrate_aprilgrid.py`
   - `script/record_trajectory.py`
   - `script/evaluate_vins_accuracy.py`
   - `script/evaluate_vio_tcp_camera_evo.py`
   - `script/diagnose/diagnose_tcp_orientation.py`
   - `script/visualize/visualize_tcp_trajectory_comparison.py`
3. 再浏览这些关键数据目录：
   - `data/handeye_0615`
   - `data/static_pose_samples_0615`
   - `data/image_0615`
   - `data/trajectory_samples0617`
   - `data/evo_vio_tcp_0616`
   - `data/evo_dynavins_tcp_0616`
   - `data/evo_tcp_visualization_0616`

工作要求：
- 优先使用这个新项目里的相对路径和本地数据
- 如果发现脚本默认路径仍然指向旧绝对路径，先指出，再根据当前任务决定是否修正
- 做评估时要明确坐标系定义、变换方向、姿态表示方式
- 如果需要排查误差，优先区分：
  - VIO pose 定义问题
  - 相机 / IMU 外参链问题
  - 手眼标定方向问题
  - 机械臂姿态定义问题
- 输出时尽量给出明确结论，不要只给模糊猜测

当前这轮排查已经得到一些比较明确的结论，请直接继承，不要重复从零假设：

1. 关于 VINS 中 Rs / 外参链的定义
- `Rs` 在 VINS 里是 `R_world_imu`，也就是 IMU/body 到 world 的旋转。
- `Ps` 对应 `p_world_imu`。
- `qic/ric, tic` 对应 `T_imu_camera`。
- VINS 配置里的 `body_T_cam0` 就是 `T_imu_cam0`。
- 投影链可参考 `projection_factor.cpp` 中：
  - `pts_imu_i = qic * pts_camera_i + tic;`
  - `pts_w = Qi * pts_imu_i + Pi;`
  - `pts_imu_j = Qj.inverse() * (pts_w - Pj);`
  由此可判断 `Qi/Rs = R_world_imu`。

2. 关于 camera-IMU 外参方向
- 已核对 `calibration.json` 与 VINS config：
  - `calibration.json` 中 `stereo_right/cam0` 的内参与手眼标定使用的 `camera_cam0_640x400_intrinsics.yaml` 一致。
  - `body_T_cam0 == inverse(T_ic_cam0_to_imu0)`，数值误差约 `1.66e-7`，说明方向没有抄反。
- 评估脚本中的 `T_LEFT_CAMERA_IMU` 使用的是 `T_cam_imu`，再取 `inverse` 得到 `T_imu_cam`，与 VINS 里的 `body_T_cam0` 一致。

3. 已明确排除的解释
- `60 deg` 量级的姿态误差，不是简单的 camera/IMU 外参方向错误。
- 也不是把 `pose_data.csv` 误当成 camera pose 就能解释。
- 也不是只需要把 quaternion 取反就能解释。
- 使用 `frame_level_optimized_pose.csv` 评估后姿态更差，约 `69.37 deg`，所以不是 IMU 高频 propagated pose 本身导致。

4. 已跑过的对照结果
- 把 `pose_data.csv` 当作 `T_world_camera`：姿态误差约 `140 deg`，明显不对。
- 只把 quaternion 取反、不反 position：姿态误差约 `136 deg`，明显不对。
- 分层评估结果：
  - IMU 层姿态 RMSE：`66.51 deg`
  - Camera 层姿态 RMSE：`67.21 deg`
  - TCP 层姿态 RMSE：`61.22 deg`
- 这说明错误在进入 TCP 链之前就已经存在，因此 handeye 和 camera-IMU 固定变换不是主因。

5. 当前主流程
- 现在的主评估流程以 0617 的 raw-pose episode 为准，尤其是
  `episode_20260617_0005` 和 `episode_20260617_0006`。
- 这些 episode 的 ground truth 文件包含：
  - `timestamp`
  - `position_m`
  - `quaternion_xyzw`
  - `orientation_raw`
  - `orientation_mode`
  - `raw_pose`
- 这意味着可以像手眼静态样本那样，继续验证 `raw_rpy / rotvec / quaternion`
  三种机器人姿态解释。

6. 0614 现状
- `episode_20260614_0239` 仍可用于历史诊断，但不再作为主基线。
- 0614 的 `trajectory_001.json` 没有 `raw_pose`，所以它更适合做辅助分析，
  不适合作为主流程的精度基准。

如果你已经完成初步阅读，请先给我一段简短总结：
- 这个项目是做什么的
- 关键脚本分别负责什么
- 你认为接下来最容易出错的 2 到 4 个点是什么
```

推荐你在新窗口第一句再补一句：

```text
请先读项目，不要直接沿用旧对话里的路径假设。
```
