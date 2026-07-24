---
name: diagnose-vio-eval
description: 为 `vio_eval` 项目执行 RM75 / VIO / ORB 评估误差诊断。用于用户提出“为什么误差这么大 / 帮我排查姿态问题 / 看看是不是外参错了 / 分析 0617 raw-pose / 诊断 strict-sync 或 handeye 问题”等请求时：先阅读仓库 README 与现有诊断脚本，再优先执行只读分析，按“pose 定义 -> 外参链 -> 手眼方向 -> 机器人姿态解释 -> 时间相位”顺序收敛，不从旧项目继承无关假设。
---

# diagnose-vio-eval

## 固定执行顺序

1. 先阅读 `README.md`，继承当前已验证结论，不重复从零假设。
2. 若问题涉及上下文切换，再补读 `AI_ONBOARDING_PROMPT.md`。
3. 将问题归到最接近的类别：
   - VIO pose 定义问题
   - camera/IMU 外参链问题
   - handeye 方向或 TCP 链问题
   - 机器人姿态解释问题
   - 时间同步 / strict-sync / phase offset 问题
4. 默认先做只读排查，不修改脚本：
   - 阅读 `summary.csv`、`manifest`、GT JSON
   - 运行 `script/diagnose/*.py`、`script/scan/*.py`、`script/experiments/*.py` 中最贴近的问题脚本
5. 若已有明确 baseline，优先做对照，不要重新发明口径。

## 当前默认排查顺序

1. 先确认轨迹 pose 定义是不是用对了。
2. 再确认 `T_imu_cam` / `T_cam_imu`、`body_T_cam0` 等固定外参方向。
3. 再看 handeye 与 TCP 固定链。
4. 最后看 RM75 GT 时间戳语义和固定 offset。

## 推荐脚本

- `script/diagnose/diagnose_tcp_orientation.py`
- `script/diagnose/diagnose_0614_orientation_hypotheses.py`
- `script/diagnose/diagnose_rm75_cross_episode.py`
- `script/diagnose/diagnose_rm75_state_generalization.py`
- `script/scan/scan_vio_tcp_conventions.py`
- `script/scan/scan_tcp_handeye_rotation.py`
- `script/scan/scan_robot_gt_adjustments.py`

## 输出要求

1. 当前最可能的问题类别。
2. 已核实的证据与已排除的解释。
3. 下一步最小验证动作。
4. 明确区分“证据支持的结论”和“待验证的猜测”。
