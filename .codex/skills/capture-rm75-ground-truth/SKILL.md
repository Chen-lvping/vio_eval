---
name: capture-rm75-ground-truth
description: 为 `vio_eval` 项目执行 RM75 TCP ground truth 采集与时间同步准备。用于用户提出“采集 RM75 真值 / 录机械臂轨迹 / 检查时间同步 / 回放 TCP 轨迹 / 生成 raw_pose GT”等请求时：先阅读 `README.md`、`script/README.md` 和 `script/SOP_rm75_time_sync.md` 建立上下文，再优先执行只读检查，随后使用仓库现有采集脚本推进，不要回退到旧项目路径或旧工作区假设。
---

# capture-rm75-ground-truth

## 固定执行顺序

1. 先阅读 `README.md` 与 `script/README.md`，确认当前仓库主入口和默认数据目录。
2. 若请求涉及采集前准备，继续阅读 `script/SOP_rm75_time_sync.md`。
3. 默认先执行只读检查：
   - 时间同步检查：`python3 script/check_time_sync.py`
   - 如需要跨机校时，再看 `python3 script/sync_remote_ptp_to_local.py --help`
4. 真正采集前，先确认目标是：
   - 单点静态位姿
   - 轨迹真值
   - 带 `raw_pose` 的 RM75 episode ground truth
5. 采集动作优先复用仓库脚本：
   - 末端位姿采集：`python3 script/get_rm75_end_pose.py`
   - 轨迹录制：`python3 script/record_trajectory.py ...`
   - TCP 轨迹回放：`python3 script/replay_rm75_tcp_pose_trajectory.py ...`
6. 若用户只要求排查或确认流程，不自动执行真实采集或回放。

## 默认原则

- 优先使用当前仓库相对路径，不默认引用旧项目 `Auto_calibration/ARM_trajectory`。
- 先明确时间戳语义，再讨论误差；RM75 GT 默认带主机侧读取时间语义，不要把它直接当成控制器原生采样时刻。
- 若用户只说“帮我看看能不能录”，默认停在检查和命令准备阶段，不主动驱动机械臂。
- 若需要写出采集命令，明确输出目录、文件命名和是否保留 `raw_pose` 字段。

## 常用入口

- 时间同步：`script/check_time_sync.py`
- 远端 PTP/系统时钟对齐：`script/sync_remote_ptp_to_local.py`
- 时间同步 SOP：`script/SOP_rm75_time_sync.md`
- 末端位姿采集：`script/get_rm75_end_pose.py`
- 轨迹录制：`script/record_trajectory.py`
- TCP 轨迹回放：`script/replay_rm75_tcp_pose_trajectory.py`

## 输出要求

1. 当前目标属于哪类采集任务。
2. 已执行的检查或准备动作。
3. 若尚未真正采集，给出下一条推荐命令。
4. 若发现风险，明确是时间同步、姿态定义、路径设置还是回放链路问题。
