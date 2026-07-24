# vio_eval

独立的 VIO / SLAM / 手眼标定 / 机械臂 TCP 真值评估工作区。  
当前最活跃的方向已经不是早期 0614 的泛化排查，而是：

- RM75 机械臂 TCP ground truth 采集与评估
- 0617 raw-pose GT 链路校验
- 2026-06-18 `gripper_data2` / `stereo_right` 上的 ORB-SLAM3 stereo-inertial 优化
- 时间戳对齐、strict-sync 后处理、viewer 可视化

README 的目标是帮你继续开发，而不是从零理解整个仓库。

## 当前重点

现在项目里有两条最重要的主线：

1. `20260617` raw-pose GT 主线  
用于验证机械臂姿态定义、手眼链路、TCP 评估链是否自洽。  
核心 episode 是：
- `episode_20260617_0005`
- `episode_20260617_0006`

2. `20260618` RM75 ORB / VINS 对比主线  
用于把 ORB-SLAM3 / ORB-VINS3 stereo-inertial 的 TCP 精度压近 VINS baseline。  
当前最常用 rig / mode 是：
- `camera_rig = stereo_right`
- `mode = stereo-inertial`

## 目录

- `script/`
  主脚本目录，根目录只保留最常用入口；详细分组见 [script/README.md](/home/chenlvping/1_DM_work/vio_eval/script/README.md:1)
- `docs/HANPU_YCTC_SLAM_DEVELOPMENT.md`
  Hanpu/YCTC SC233HGS 设备标定、手动录制、相机质量报告、ORB 运行和后续 VINS 开发说明
- `data/calibration/`
  静态位姿、标定图像、相机内参、手眼结果
- `data/ground_truth/`
  机械臂轨迹真值
- `data/evaluation/workbench/`
  新实验默认输出目录
- `data/evaluation/core/`
  历史核心结果
- `data/archive/`
  已归档实验

## 当前主入口

当前 RM75 最佳主链路：

- `script/run_orbslam3_rm75_best_batch.py`
  同事版 ORB-SLAM3 offline stereo + `GBA=100` + `Full-frame BA=10` + `21/2/9` 平滑 + per-episode strict-sync。
- `data/evaluation/config/rm75_colleague_best_strict_sync_offsets.json`
  当前已验证的 episode 时间偏移配置。

示例：

```bash
python3 script/run_orbslam3_rm75_best_batch.py \
  --episode-root data/gripper/gripper_data_6_24 \
  --gt-root data/ground_truth/rm75_6_24 \
  --episode-pattern 'episode_20260624_*'
```

旧版 RM75 主链路仍由 `script/run_orbslam3_rm75_batch_eval.py` 保留，旧 ORB-SLAM3 代码可通过 `script/dev/restore_orbslam3_best_version.sh` 回退。

- `script/get_rm75_end_pose.py`
  当前 RM75 末端位姿采集脚本
- `script/check_time_sync.py`
  机械臂控制器与主机时间同步检查
- `script/sync_remote_ptp_to_local.py`
  通过 SSH 把远端设备系统时钟和 PHC 对齐到本机时钟
- `script/SOP_rm75_time_sync.md`
  RM75 时间同步标准操作流程
- `script/record_trajectory.py`
  较完整的轨迹录制脚本，保留 `raw_pose` / before-after timestamp 等字段
- `script/replay_rm75_tcp_pose_trajectory.py`
  按 TCP 位姿 JSON 进行 pose -> IK -> joint servo 回放
- `script/handeye_calibrate_aprilgrid.py`
  AprilGrid 手眼标定
- `script/evaluate_vins_accuracy.py`
  通用轨迹评估底座
- `script/evaluate_vio_tcp_camera_evo.py`
  当前 TCP evo 评估主入口
- `script/generate_run_provenance_log.py`
  为 batch 结果生成参数追溯日志和配置快照
- `script/run_orbslam3_tcp_eval.py`
  单个 episode 的 ORB-SLAM3 运行 + 导出 + TCP 评估
- `script/run_orbslam3_rm75_batch_eval.py`
  RM75 多 episode 批量 ORB 评估
- `script/visualize_single_tcp_trajectory_3d.py`
  单次评估 viewer

## 已验证结论

下面这些结论已经验证过，后续开发默认继承，不要每次从零重查。

### 1. evaluator 不是当前主因

- evaluator 早先对 `left_camera` 的写死歧义已经修过
- 现在显式读取 `calibration.json` + `camera_rig`
- 修完之后指标基本不变，所以 evaluator 不是当前 RM75 ORB 精度瓶颈

### 2. `Tlr` / rig 外参不是当前主因

- fresh 配置里的 `Tlr` 与 raw / VINS 推导出的 `cam0 -> cam1` 一致
- old-style `Tlr` 反而明显退化
- 因此不要优先怀疑 stereo rig 左右外参

### 3. ORB stereo-inertial 的大收益来自 inertial init 策略

对 `episode_20260618_0004`：

- VINS baseline: `APE 34.431 mm`, `RPE 4.789 mm`
- ORB baseline, `IMU.fastInit=1`: `APE 83.823 mm`, `RPE 42.647 mm`
- ORB 改成 `IMU.fastInit=0`: `APE 39.585 mm`, `RPE 23.912 mm`

结论：
- 这批数据前约 3 秒基本静止
- fast inertial init 不适合这批低激励数据
- 后续 ORB stereo-inertial 实验默认从 `imu_fast_init=0` 开始

### 4. `mImuPer` 修复有效，但不是决定性飞跃

已修改：
- `ORB_SLAM3_clean/src/Tracking.cc`
- 两处 `mImuPer = 0.001`
- 改为 `mImuPer = 1.0 / (double)mImuFreq`

在约 `1024 Hz` IMU 上，这修掉了约 `2.4%` 的周期误差。

结果：
- `fastInit=0` 基线: `APE 39.585 mm`, `RPE 23.912 mm`
- 加 `mImuPer` 修复后: `APE 37.991 mm`, `RPE 23.804 mm`

结论：
- 有真实收益
- 但主要改善的是起图后的 IMU 时间/预积分精度
- 不是更早初始化

### 5. 平滑只能去一点高频抖动

对 ORB 平移做 `w7 p2` 平滑后：

- `APE 37.625 mm`
- `RPE 22.402 mm`

结论：
- 确实存在高频抖动
- 但主体误差不是纯平移噪声
- 仍然有系统性偏差 / 时间相位 / IMU 融合问题

### 6. 机械臂 GT 时间戳带主机侧语义

当前 RM75 采集脚本与历史脚本分析表明：

- GT 时间戳并不是“控制器原生采样时刻”
- 更像“主机侧 SDK 读取时刻”或其近似
- 因此会天然带固定延迟和调度抖动

这也是为什么：
- 小幅 fixed offset / strict-sync 会明显影响 APE/RPE
- viewer 看起来已经很好，但换几毫秒到几十毫秒仍可能继续改善

### 7. 时间偏差更像链路语义不统一，不只是 ORB 算法误差

当前判断更接近：

- 机械臂 GT 用主机轮询时间
- camera / IMU 导出使用各自 topic 的时间语义重建
- 最终在评估层表现成固定时间相位差

所以 strict-sync / fixed offset 是当前很重要的后处理手段。

## 当前 ORB 主链路

当前最常用命令：

```bash
python3 script/run_orbslam3_tcp_eval.py \
  --mode stereo-inertial \
  --camera-rig stereo_right \
  --feature-preset low-texture \
  --skip-viewer \
  --vins-config data/gripper_data2/episode_20260618_0004/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml \
  --vins-noise-mode orb_from_vins \
  --imu-fast-init 0
```

这个链路默认包含：

- ORB EuRoC 导出
- ORB 原生轨迹生成
- TUM -> pose CSV 转换
- 可选平滑
- strict-sync 到 GT 时间轴
- TCP evo 评估
- viewer 生成

## strict-sync / 时间对齐说明

这里要特别注意两件事：

1. `strict-sync offset` 是“后处理补偿项”，不是物理真值本身  
2. 不同 episode、不同导出链、不同 postprocess 组合，最优 offset 可能不同

当前标准化 batch offset 配置在：

- [data/evaluation/config/rm75_episode_strict_sync_offsets.json](/home/chenlvping/1_DM_work/vio_eval/data/evaluation/config/rm75_episode_strict_sync_offsets.json:1)

当前内容：

- `episode_20260618_0001 -> 0 ms`
- `episode_20260618_0002 -> +15 ms`
- `episode_20260618_0003 -> -5 ms`
- `episode_20260618_0004 -> -115 ms`

但要注意：

- `0004` 还有一个特定 `ts_aligned + smooth + strictsync_p435ms` viewer 实验
- 那条实验达到了约 `APE 7.535 mm`, `RPE 5.511 mm`
- 它是某条特定后处理链上的“最佳 viewer 结果”
- 不应直接当作所有 `0004` 运行的通用默认 offset

换句话说：

- `offset json` 里的值是“当前批量链路默认值”
- `p435ms` 是“特定实验链路下的最好 viewer 结果”
- 两者不要混为一谈

## 各 episode 已知最好 ORB viewer

按 APE 主指标，你目前手工确认过的最好 ORB viewer 是：

- `0001`: `APE 12.138 mm`, `RPE 11.948 mm`
- `0002`: `APE 30.662 mm`, `RPE 15.052 mm`
- `0003`: `APE 12.949 mm`, `RPE 9.995 mm`
- `0004`: `APE 7.535 mm`, `RPE 5.511 mm`

这些结果的重要含义是：

- ORB viewer 的主观效果已经可以很好
- 现在一个很大的收益点在“时间对齐后处理”
- 后续开发优先级应放在时间对齐语义统一、offset 自动化和稳定化，而不是盲调视觉参数

## 推荐开发顺序

如果你接下来继续做这个项目，建议优先顺序如下：

1. 固化 README / 脚本目录 / 主入口  
避免再出现“脚本太乱，不知道该从哪跑”的问题

2. 固化 ORB 默认链路  
以 `stereo_right + stereo-inertial + low-texture + imu_fast_init=0 + mImuPer fix` 作为默认起点

3. 固化时间对齐后处理  
把 strict-sync offset 管理、sweep 输出、最佳结果落盘做成标准流程

4. 再做更深的 IMU noise / bias / init 策略验证  
只有在时间对齐收益见顶之后，再继续往 IMU 语义深挖

## 不建议重复浪费时间的方向

除非有新的证据，否则不要优先回到这些方向：

- 反复怀疑 evaluator
- 反复怀疑 `left_camera` 写死
- 反复怀疑 `Tlr`
- 反复怀疑 raw timestamp 导出链本身是否“完全错了”
- 只靠平滑想把 ORB 直接抹到 VINS 水平

## 常见工作流

### 1. 跑单个 ORB episode

```bash
python3 script/run_orbslam3_tcp_eval.py \
  --episode-dir data/gripper_data2/episode_20260618_0004 \
  --ground-truth data/ground_truth/rm75/rm75_pose_traj_4.json \
  --camera-rig stereo_right \
  --mode stereo-inertial \
  --feature-preset low-texture \
  --vins-config data/gripper_data2/episode_20260618_0004/right/vio_log/generated_config/StereoIMU-vinsfusion.yaml \
  --vins-noise-mode orb_from_vins \
  --imu-fast-init 0
```

### 2. 跑 RM75 batch

```bash
python3 script/run_orbslam3_rm75_batch_eval.py \
  --episode-root data/gripper_data2 \
  --gt-root data/ground_truth/rm75 \
  --camera-rig stereo_right \
  --mode stereo-inertial \
  --feature-preset low-texture \
  --imu-fast-init 0 \
  --strict-sync-offset-json data/evaluation/config/rm75_episode_strict_sync_offsets.json
```

### 3. 看单次评估 viewer

```bash
python3 script/visualize_single_tcp_trajectory_3d.py \
  --eval-dir data/evaluation/workbench/<your_eval_dir> \
  --output-dir data/evaluation/workbench/<your_eval_dir>
```

## 开发注意事项

- 优先复用 `data/evaluation/workbench/`，不要污染旧核心结果
- 新实验必须尽量单变量改动
- 最好保存 command / config / summary / manifest / log / viewer 路径
- `run_orbslam3_rm75_batch_eval.py` 会自动生成 `RUN_LOG.md` 和 `batch_provenance.json`
- 如果某方向没收益，要明确降级，不要无限盲调

## 参考文件

- [script/README.md](/home/chenlvping/1_DM_work/vio_eval/script/README.md:1)
- [script/run_orbslam3_tcp_eval.py](/home/chenlvping/1_DM_work/vio_eval/script/run_orbslam3_tcp_eval.py:1)
- [script/run_orbslam3_rm75_batch_eval.py](/home/chenlvping/1_DM_work/vio_eval/script/run_orbslam3_rm75_batch_eval.py:1)
- [script/evaluate_vio_tcp_camera_evo.py](/home/chenlvping/1_DM_work/vio_eval/script/evaluate_vio_tcp_camera_evo.py:1)
- [script/generate_run_provenance_log.py](/home/chenlvping/1_DM_work/vio_eval/script/generate_run_provenance_log.py:1)
- [script/get_rm75_end_pose.py](/home/chenlvping/1_DM_work/vio_eval/script/get_rm75_end_pose.py:1)
- [data/evaluation/config/rm75_episode_strict_sync_offsets.json](/home/chenlvping/1_DM_work/vio_eval/data/evaluation/config/rm75_episode_strict_sync_offsets.json:1)
