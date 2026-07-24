---
name: run-rm75-batch-eval
description: 为 `vio_eval` 项目执行 RM75 多 episode 批量评估与结果汇总。用于用户提出“批量跑 ORB 评估 / 批量比较多个 episode / 生成 workbench 报告 / 做 strict-sync offset 扫描”等请求时：优先使用 `script/run_orbslam3_rm75_batch_eval.py` 与仓库既有输出结构，保留 provenance、日志与每条 episode 的独立结果目录。
---

# run-rm75-batch-eval

## 固定执行顺序

1. 先阅读 `README.md`，确认当前批量主线和默认数据根目录。
2. 再阅读 `script/run_orbslam3_rm75_batch_eval.py`，确认支持的批量参数。
3. 在启动批量任务前，先明确：
   - `episode-root`
   - `gt-root`
   - `camera-rig`
   - `mode`
   - `feature-preset`
   - `imu-fast-init`
   - 是否启用 `strict-sync offset json` 或 offset scan
4. 默认输出到 `data/evaluation/workbench/`，不要把新结果写回历史核心目录。
5. 批量完成后，优先汇总：
   - 每条 episode 的 `APE / RPE`
   - 最佳 offset 候选（若启用了 scan）
   - 失败条目及对应日志

## 默认原则

- 先复用批量 runner，不要手工循环拼很多单条命令，除非用户明确只想跑 1 到 2 条做试验。
- 结果应保留 `logs/`、每条 `eval_episode_*` 子目录和 provenance 记录。
- 若批量范围很大，先帮助用户缩小到一个明确的 episode pattern。
- 若已有 workbench 目录且用户是在做 review，优先读现成结果，不重复跑。

## 常用入口

- `script/run_orbslam3_rm75_batch_eval.py`
- `script/run_orbslam3_tcp_eval.py`
- `script/generate_run_provenance_log.py`
- `data/evaluation/config/rm75_episode_strict_sync_offsets.json`

## 输出要求

1. 本轮批量配置摘要。
2. 成功/失败 episode 概览。
3. 最值得继续追的 episode 或参数差异。
4. 结果根目录与关键日志位置。
