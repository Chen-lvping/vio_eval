---
name: smooth-rm75-ground-truth
description: 为 `vio_eval` 项目执行 RM75 ground-truth 轨迹平滑、偏差约束和可选 TUM 导出。用于用户提出“平滑 RM75 GT / 限制姿态偏差 / 导出 TUM / 处理 raw_pose 轨迹抖动”等请求时：优先使用 `script/smooth_rm75_ground_truth.py`，保持原有时间戳和 JSON schema，不手写临时平滑脚本。
---

# smooth-rm75-ground-truth

## 固定执行顺序

1. 先确认输入文件是 RM75 GT JSON，且包含 `samples` 列表。
2. 阅读 `script/smooth_rm75_ground_truth.py`，确认当前支持的参数。
3. 明确本轮平滑目标：
   - 仅轻度去抖
   - 限制最大位置偏差
   - 限制最大姿态偏差
   - 同时导出 TUM
4. 默认命令模板：
   - `python3 script/smooth_rm75_ground_truth.py --input <src.json> --output <dst.json> --window 11`
5. 若用户关心和原轨迹偏离太大，优先加：
   - `--max-pos-dev-mm <value>`
   - `--max-rot-dev-deg <value>`
6. 若后续要接入 evo 或 viewer，可同时导出：
   - `--export-tum <dst.tum>`

## 默认原则

- 保留原始时间戳和整体 schema，不把平滑结果写成另一个不兼容格式。
- `window` 必须为奇数；若用户给偶数，应先改成最近的合理奇数或说明约束。
- 平滑只解决高频抖动，不应把它当成时间偏差、姿态解释错误或外参链错误的替代修复。

## 输出要求

1. 输入文件、输出文件和主要参数。
2. 是否启用了偏差约束。
3. 是否额外导出了 TUM。
4. 若用户是为了评估提分，提醒区分“平滑收益”和“时间对齐收益”。
