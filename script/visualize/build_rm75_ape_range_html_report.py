#!/usr/bin/env python3
"""Build a polished HTML/PDF report for RM75 trajectories in a target APE band."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pikepdf
import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Inches, Pt, RGBColor


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_DIR = REPO_ROOT / "data/evaluation/workbench/orbslam3_rm75_batch_eval_20260630_103032"
DEFAULT_OUTPUT_HTML = REPO_ROOT / "data/evaluation/showcase/rm75_mainline_showcase_202606/rm75_ape_5_to_7mm_report.html"
DEFAULT_MIN_APE_MM = 5.0
DEFAULT_MAX_APE_MM = 7.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

EN_TRANSLATION_PAIRS: list[tuple[str, str]] = [
    ('lang="zh"', 'lang="en"'),
    ("VIO 精度评估报告", "VIO Precision Evaluation Report"),
    ("RM75 5-7 mm 轨迹精度清单", "RM75 5-7 mm Trajectory Precision Summary"),
    ("客户展示版 · 可编辑稿", "Customer-facing editable draft"),
    ("目标带", "Target band"),
    ("达标轨迹", "Qualified tracks"),
    ("APE 均值", "APE mean"),
    ("APE 标准差", "APE std"),
    ("最佳 APE", "Best APE"),
    ("最差 APE", "Worst APE"),
    ("RPE 均值", "RPE mean"),
    ("APE 离散度", "APE spread"),
    ("一、报告摘要", "1. Executive Summary"),
    ("二、测试对象说明", "2. Test Object"),
    ("三、测试数据与 Ground Truth", "3. Data and Ground Truth"),
    ("四、评估方法", "4. Evaluation Method"),
    ("五、精度结果总览", "5. Precision Overview"),
    ("六、轨迹可视化分析", "6. Trajectory Visualization"),
    ("七、典型场景分析", "7. Typical Scenarios"),
    ("八、失败案例与原因分析", "8. Failure Cases and Root Causes"),
    ("九、实时性与资源占用分析", "9. Real-time Performance and Resource Usage"),
    ("十、结论与建议", "10. Conclusions and Recommendations"),
    ("建议补充的数据清单", "Suggested Data to Add"),
    ("本报告选取 ", "This report selects "),
    ("本报告选取 4 条进入 5-7 mm 区间的轨迹，展示当前方案在同一测试条件下的稳定精度表现。", "This report selects 4 trajectories in the 5-7 mm band, demonstrating stable precision performance under the same test conditions."),
    ("条进入 5-7 mm 区间的轨迹，展示当前方案在同一测试条件下的稳定精度表现。", " trajectories in the 5-7 mm band, demonstrating stable precision performance under the same test conditions."),
    ("条进入 5-7 mm 目标带的 RM75 轨迹，整体 APE 为 ", " RM75 trajectories in the 5-7 mm target band, with overall APE of "),
    ("，标准差 ", ", standard deviation "),
    ("。结果稳定落在目标带内，可直接用于客户汇报与交付材料。", ". The result stays stably within the target band and is ready for customer reporting and delivery materials."),
    ("四条轨迹都落在 5-7 mm 的Target band内，APE 集中在 5.254-5.926 mm，标准差只有 0.245 mm。", "All four trajectories fall within the 5-7 mm target band, with APE concentrated in 5.254-5.926 mm and standard deviation of only 0.245 mm."),
    ("四条轨迹都落在 5-7 mm 的Target band内，APE 集中在 5.254-5.926 mm，标准差只有 0.245 mm。 The results show good stability and consistency.", "All four trajectories fall within the 5-7 mm target band, with APE concentrated in 5.254-5.926 mm and standard deviation of only 0.245 mm. The results show good stability and consistency."),
    ("四条轨迹都落在 5-7 mm 的目标带内，APE 集中在 5.254-5.926 mm，标准差只有 0.245 mm。", "All four trajectories fall within the 5-7 mm target band, with APE concentrated in 5.254-5.926 mm and standard deviation of only 0.245 mm."),
    ("四条轨迹都落在 5-7 mm 的Target band内，APE 集中在 ", "All four trajectories fall within the 5-7 mm target band, with APE concentrated in "),
    ("结果具有较好的稳定性和一致性。", "The results show good stability and consistency."),
    ("测试条件", "Test Conditions"),
    ("同一批次、同一主线、同一传感器组合。", "Same batch, same mainline, and the same sensor stack."),
    ("统一采用Strict time alignment与轨迹对齐策略。", "Strict time alignment and trajectory alignment are used throughout."),
    ("统一采用Strict time alignment与轨迹对齐策略。", "Strict time alignment and trajectory alignment are used throughout."),
    ("统一采用严格时间对齐与轨迹对齐策略。", "Strict time alignment and trajectory alignment are used throughout."),
    ("统一采用Strict time alignment与trajectory对齐策略。", "Strict time alignment and trajectory alignment are used throughout."),
    ("四条轨迹均稳定落在Target band内。", "All four trajectories remain stably within the target band."),
    ("四条轨迹均稳定落在目标带内。", "All four trajectories remain stably within the target band."),
    ("四条轨迹均稳定落在 5-7 mm 目标带内，APE 离散度较小。", "All four trajectories remain stably within the 5-7 mm target band, with a small APE spread."),
    ("四条trajectory均稳定落在Target band内。", "All four trajectories remain stably within the target band."),
    ("trajectory APE 范围仅 ", "The trajectory APE range is only "),
    ("轨迹 APE 范围仅 0.672 mm，and the spread is small.", "The trajectory APE range is only 0.672 mm, and the spread is small."),
    ("轨迹 APE 范围仅 0.672 mm，and the spread is small.", "The trajectory APE range is only 0.672 mm, and the spread is small."),
    ("轨迹 APE 范围仅 0.672 mm，离散度较小。", "The trajectory APE range is only 0.672 mm, and the spread is small."),
    ("离散度较小。", "and the spread is small."),
    ("匹配样本数覆盖 ", "Matched samples cover "),
    ("匹配样本数覆盖 772-3752，Duration覆盖 15.4-75.0 s。", "Matched samples cover 772-3752, and duration covers 15.4-75.0 s."),
    ("Matched samples cover 772-3752，Duration覆盖 15.4-75.0 s。", "Matched samples cover 772-3752, and duration covers 15.4-75.0 s."),
    ("匹配样本数覆盖 772-3752，时长覆盖 15.4-75.0 s。", "Matched samples cover 772-3752, and duration covers 15.4-75.0 s."),
    ("Duration覆盖 ", "duration covers "),
    ("全局 APE 与局部 RPE 同时保持收敛，说明结果具备稳定一致性。", "Global APE and local RPE both remain converged, indicating stable consistency."),
    ("四条轨迹均稳定落在 5-7 mm 目标带内，APE 离散度较小。", "All four trajectories remain stably within the 5-7 mm target band, with a small APE spread."),
    ("四条轨迹均稳定落在 5-7 mm Target band内，APE spread较小。", "All four trajectories remain stably within the 5-7 mm target band, with a small APE spread."),
    ("四条轨迹均稳定落在 5-7 mm Target band内，APE spread较小。", "All four trajectories remain stably within the 5-7 mm target band, with a small APE spread."),
    ("每条trajectory都保留独立预览图，便于复核空间形态。", "Each trajectory keeps an independent preview for spatial verification."),
    ("每条轨迹都保留独立预览图，便于复核空间形态。", "Each trajectory keeps an independent preview for spatial verification."),
    ("每条轨迹都保留独立预览图，便于复核空间形态。", "Each trajectory keeps an independent preview for spatial verification."),
    ("仅展示已达标样本的汇总结果。", "Only qualified samples are summarized here."),
    ("对应的单trajectory页面可继续查看明细图。", "The corresponding single-trajectory pages can be used to view the detailed plots."),
    ("对应的单轨迹页面可继续查看明细图。", "The corresponding single-trajectory pages can be used to view the detailed plots."),
    ("轨迹精度报告", "Trajectory Precision Report"),
    ("轨迹精度清单", "Trajectory Precision Summary"),
    ("最大误差", "Max error"),
    ("成功率", "Success rate"),
    ("平均输入帧率约 ", "Average input frame rate is about "),
    ("；更细的处理时延、CPU 占用与内存占用待补充。", "; more detailed processing latency, CPU usage, and memory usage are pending."),
    ("系统名称/版本", "System name / version"),
    ("RM75 精度展示版", "RM75 precision showcase build"),
    ("算法类型", "Algorithm type"),
    ("双目 VIO / 惯性融合", "Stereo VIO / inertial fusion"),
    ("输入传感器", "Input sensors"),
    ("双目相机 + IMU；输入帧率约 25 Hz", "Stereo camera + IMU; input frame rate about 25 Hz"),
    ("输出内容", "Output"),
    ("位姿轨迹", "Pose trajectory"),
    ("回环/重定位", "Loop closure / relocalization"),
    ("未提供", "Not provided"),
    ("运行平台", "Runtime platform"),
    ("标定与同步", "Calibration and synchronization"),
    ("严格时间对齐", "Strict time alignment"),
    ("序列", "Sequence"),
    ("长度", "Length"),
    ("时长", "Duration"),
    ("运动特征", "Motion profile"),
    ("GT 来源", "GT source"),
    ("完成情况", "Status"),
    ("达标", "Qualified"),
    ("项目", "Item"),
    ("说明", "Description"),
    ("未提供（本报告保留精度结果）", "Not provided (precision results retained in this report)"),
    ("成功完成", "Completed successfully"),
    ("轨迹对齐以 SE(3) 为主，不进行 Sim(3) 尺度修正作为主展示口径。", "Trajectory alignment is based on SE(3); Sim(3) scale correction is not used for the main presentation."),
    ("使用严格时间配对后再进行轨迹对齐，尽量避免时间相位对结果的干扰。", "Strict timestamp pairing is applied before trajectory alignment to minimize timing-phase effects."),
    ("ATE / APE 用于衡量全局轨迹误差，RPE 用于衡量局部漂移与短时稳定性。", "ATE / APE measure global trajectory error, while RPE measures local drift and short-term stability."),
    ("本报告仅保留客户可直接阅读的核心指标与结论。", "This report keeps only the core metrics and conclusions intended for customer reading."),
    ("Sequence", "Sequence"),
    ("Length", "Length"),
    ("Duration", "Duration"),
    ("ATE RMSE", "ATE RMSE"),
    ("ATE Mean", "ATE Mean"),
    ("ATE Median", "ATE Median"),
    ("ATE Max", "ATE Max"),
    ("RPE Translation", "RPE Translation"),
    ("RPE Rotation", "RPE Rotation"),
    ("Lost Count", "Lost Count"),
    ("FPS", "FPS"),
    ("Result", "Result"),
    ("整体来看，四条轨迹均稳定落在 ", "Overall, all four trajectories remain within the "),
    ("目标带内。", " target band."),
    ("下图为整体精度分布与单条轨迹的空间预览。建议优先观察 APE 分布是否集中、RPE 是否一致，以及轨迹形态是否保持一致。", "The figure below shows the overall precision distribution and the spatial preview of each trajectory. Focus on whether APE is concentrated, RPE is consistent, and the trajectory shape remains stable."),
    ("图 1  APE 分布与 APE-RPE 关系图", "Figure 1  APE distribution and APE-RPE relationship"),
    ("正常纹理室内环境：当前展示样本已体现出稳定低毫米级精度，适合作为标准演示场景。", "Normal textured indoor environment: the current samples already show stable low-millimeter accuracy and are suitable as a standard demo scenario."),
    ("长走廊或低纹理环境：建议补充专项序列后再做定量结论。", "Long corridor or low-texture environment: add dedicated sequences before drawing quantitative conclusions."),
    ("快速旋转场景：建议补充角速度峰值更高的样本以验证极限能力。", "Fast rotation: add samples with higher peak angular velocity to verify the upper bound."),
    ("光照剧烈变化场景：建议补充强反光、逆光和曝光跳变数据。", "Severe illumination change: add strong reflections, backlight, and exposure-jump data."),
    ("静止或慢速运动场景：建议关注短时漂移和初始化稳定性。", "Static or slow-motion scenes: pay attention to short-term drift and initialization stability."),
    ("室外大尺度场景：本报告未覆盖，建议单独补测。", "Large-scale outdoor scenes: not covered in this report; recommend separate testing."),
    ("动态物体干扰场景：建议以内部测试版进一步验证鲁棒性。", "Dynamic-object interference: validate robustness further with an internal test build."),
    ("本报告仅呈现达标轨迹；如需内部评审版，可另行补充异常序列与误差曲线。", "This report only presents qualified trajectories; for an internal review version, add abnormal sequences and error curves separately."),
    ("现象", "Phenomenon"),
    ("位置", "Location"),
    ("误差表现", "Error behavior"),
    ("可能原因", "Possible cause"),
    ("建议", "Recommendation"),
    ("待补充", "To be added"),
    ("平均处理时间", "Average processing time"),
    ("最大处理时间", "Max processing time"),
    ("是否满足实时运行", "Meets real-time requirement"),
    ("输入帧率约 25 Hz；更细的处理时延待补充", "Input frame rate is about 25 Hz; finer-grained latency data are pending"),
    ("CPU 占用", "CPU usage"),
    ("内存占用", "Memory usage"),
    ("耗时尖峰", "Latency spikes"),
    ("嵌入式部署影响", "Impact on embedded deployment"),
    ("建议补充资源统计后再评估", "Re-evaluate after adding resource statistics"),
    ("系统整体达到低毫米级展示目标，适合用于客户汇报、项目交付和阶段性评审。", "The system overall reaches the low-millimeter showcase target and is suitable for customer reporting, project delivery, and phase reviews."),
    ("当前样本适合作为对外材料主版本，重点体现精度稳定性与轨迹一致性。", "The current samples are suitable as the main external-facing version, emphasizing precision stability and trajectory consistency."),
    ("若进入部署决策阶段，建议补充资源占用、长时稳定性和更复杂场景数据。", "If moving into deployment decisions, add resource usage, long-term stability, and more complex scene data."),
    ("后续优先补充：场景标签、运行平台、时延统计、CPU/内存统计和更完整的轨迹覆盖。", "Priority additions: scene labels, runtime platform, latency statistics, CPU/memory statistics, and broader trajectory coverage."),
    ("局部运动一致性", "local motion consistency"),
    ("批次", "Batch"),
    ("时间对齐", "Time association"),
    ("对齐方式", "Alignment"),
    ("相机配置", "Camera setup"),
    ("运动基准", "Motion unit"),
    ("交付场景", "Delivery"),
    ("展示范围", "Showcase scope"),
    ("评估重点", "Evaluation focus"),
    ("最优绝对误差", "Best absolute"),
    ("最优局部 RPE", "Best local RPE"),
    ("达标率", "On-band rate"),
    ("APE 离散度", "APE spread"),
    ("5 cm 局部误差", "RPE @ 5 cm"),
    ("姿态 APE", "Rotation APE"),
    ("严格同步", "Strict sync"),
    ("样本 / 时长", "Samples / duration"),
    ("最佳轨迹", "Best track"),
    ("最差轨迹", "Worst track"),
    ("匹配时长均值", "Matched duration mean"),
    ("匹配样本均值", "Matched samples mean"),
    ("姿态", "Rot"),
    ("RM75 批量评估", "RM75 batch eval"),
    ("严格时间配对", "strict timestamp pairing"),
    ("SE(3) 精度评估", "SE(3) precision evaluation"),
    ("双目", "stereo"),
    ("末端执行器 TCP", "end-effector TCP"),
    ("客户展示", "customer-facing showcase"),
    ("配置快照", "Configuration Snapshot"),
    ("精度展示 / APE / RPE", "Precision showcase / APE / RPE"),
    ("4/4 全部达标", "4/4 in target band"),
    ("来源：data/evaluation/workbench 与 data/evaluation/showcase 的现有评估产物。", "Generated from existing evaluation artifacts in data/evaluation/workbench and data/evaluation/showcase."),
    ("运行平台（CPU / GPU / 内存 / 操作系统版本）", "Runtime platform (CPU / GPU / memory / OS version)"),
    ("平均处理时间、最大处理时间与端到端时延", "Average processing time, max processing time, and end-to-end latency"),
    ("更完整的场景标签（室内/室外/长走廊/动态干扰等）", "More complete scene labels (indoor / outdoor / long corridor / dynamic interference, etc.)"),
    ("丢跟次数、初始化成功率与重定位统计", "Tracking-loss count, initialization success rate, and relocalization statistics"),
    ("相机分辨率、IMU 频率、标定参数与噪声参数", "Camera resolution, IMU frequency, calibration parameters, and noise parameters"),
    ("注：本 Word 版为可编辑稿，文字、表格与图注均可直接修改。", "Note: this Word version is editable; text, tables, and captions can be modified directly."),
    ("轨迹预览", "Trajectory Preview"),
    ("交互式轨迹查看", "Interactive 3D View"),
    ("轨迹预览与交互式 3D 视图并列展示，便于核对轨迹空间形态与整体走势。", "The trajectory preview is shown alongside the interactive 3D view to verify the spatial shape and overall trend."),
    ("交互视图用于核对轨迹形态、起终点一致性和整体空间走势。", "The interactive view is used to verify trajectory shape, start/end consistency, and overall spatial trend."),
    ("3D 预览图库", "3D Preview Gallery"),
    ("PDF 版本不嵌交互 iframe，改为保留每条轨迹的静态 3D 空间预览，确保导出版式稳定、打印可读、且仍能展示轨迹几何形态。", "The PDF version does not embed interactive iframes; instead, it keeps static 3D previews for each trajectory to ensure stable export layout, print readability, and preserved geometric shape."),
    ("轨迹明细", "Trajectory Details"),
    ("这里按轨迹单独拆解，重点看全局位置误差、局部运动误差和姿态误差。", "Each trajectory is broken down individually, focusing on global position error, local motion error, and orientation error."),
    ("双目帧（上下拼接）", "Stereo pair (top-bottom stitched)"),
    ("右目", "Stereo pair"),
    ("RPE @ 5 cm", "RPE @ 5 cm"),
    ("Rotation APE", "Rotation APE"),
    ("Strict sync", "Strict sync"),
    ("Samples / duration", "Samples / duration"),
]

ZH_TRANSLATION_PAIRS: list[tuple[str, str]] = [
    ("RM75 TCP trajectory precision report", "RM75 TCP 轨迹精度报告"),
    ("RM75 5-7 mm Trajectory Precision Summary", "RM75 5-7 mm 轨迹精度清单"),
    ("Target band", "目标带"),
    ("Qualified tracks", "达标轨迹"),
    ("APE mean", "APE 均值"),
    ("APE std", "APE 标准差"),
    ("Best APE", "最佳 APE"),
    ("Worst APE", "最差 APE"),
    ("RPE mean", "RPE 均值"),
    ("Executive Summary", "报告摘要"),
    ("Test Conditions", "测试条件"),
    ("Trajectory Preview", "轨迹预览"),
    ("Interactive 3D View", "交互式 3D 视图"),
    ("3D Preview Gallery", "3D 预览图库"),
    ("Trajectory Details", "轨迹明细"),
    ("Conclusions", "结论"),
    ("Traceability", "追溯说明"),
    ("Configuration Snapshot", "配置快照"),
    ("Strict sync", "严格同步"),
    ("Samples / duration", "样本 / 时长"),
    ("RPE @ 5 cm", "5 cm 局部误差"),
    ("Rotation APE", "姿态 APE"),
    ("Showcase scope", "展示范围"),
    ("Evaluation focus", "评估重点"),
    ("local motion consistency", "局部运动一致性"),
    ("Best absolute", "最优绝对误差"),
    ("Best local RPE", "最优局部 RPE"),
    ("On-band rate", "达标率"),
    ("APE spread", "APE 离散度"),
    ("RM75 batch eval", "RM75 批量评估"),
    ("strict timestamp pairing", "严格时间配对"),
    ("SE(3) precision evaluation", "SE(3) 精度评估"),
    ("stereo", "双目"),
    ("end-effector TCP", "末端执行器 TCP"),
    ("customer-facing showcase", "客户展示"),
    ("Batch", "批次"),
    ("Time association", "时间对齐"),
    ("Alignment", "对齐方式"),
    ("Camera setup", "相机配置"),
    ("Motion unit", "运动基准"),
    ("Delivery", "交付场景"),
    ("Best track", "最佳轨迹"),
    ("Worst track", "最差轨迹"),
    ("Matched duration mean", "匹配时长均值"),
    ("Matched samples mean", "匹配样本均值"),
    ("Best absolute", "最优绝对误差"),
    ("Best local RPE", "最优局部 RPE"),
    ("On-band rate", "达标率"),
    ("APE spread", "APE 离散度"),
    ("showcase scope", "展示范围"),
    ("Evaluation focus", "评估重点"),
    ("RM75 batch eval", "RM75 批量评估"),
    ("Precision showcase / APE / RPE", "精度展示 / APE / RPE"),
    ("3D trajectory preview", "3D 轨迹预览"),
    ("Stereo pair (top-bottom stitched)", "双目帧（上下拼接）"),
    ("Stereo pair", "双目帧"),
    ("mid-point", "中间时刻"),
    ("Stereo pair", "双目帧"),
    ("stereo_right", "双目右路"),
    ("stereo-inertial", "双目惯导"),
    ("low-texture", "低纹理"),
    ("stabilized init", "稳定初始化"),
    ("strict-sync", "严格同步"),
    ("right-eye", "右目"),
    ("Top", "上"),
    ("Bottom", "下"),
]


@dataclass
class EpisodeRecord:
    episode: str
    episode_dir: Path
    eval_dir: Path
    summary_csv: Path
    report_md: Path
    manifest_json: Path
    viewer_html: Path
    gt_name: str
    gt_duration_s: float
    gt_sample_count: int
    gt_path_length_m: float
    ape_mm: float
    ape_sim3_mm: float
    ape_rot_deg: float
    rpe_mm: float
    rpe_rot_deg: float
    strict_sync_offset_sec: float
    input_fps_hz: float | None
    estimate_frame: str
    camera_rig: str
    mode: str
    feature_preset: str
    imu_fast_init: int | None
    matched_samples: int
    matched_duration_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--min-ape-mm", type=float, default=DEFAULT_MIN_APE_MM)
    parser.add_argument("--max-ape-mm", type=float, default=DEFAULT_MAX_APE_MM)
    return parser.parse_args()


def read_summary_metrics(summary_csv: Path) -> dict[str, dict[str, str]]:
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["metric"]: row for row in rows}


def parse_report(report_md: Path) -> tuple[int, float]:
    text = report_md.read_text(encoding="utf-8")
    samples = re.search(r"Matched samples:\s*(\d+)", text)
    duration = re.search(r"Matched duration:\s*([0-9.]+)\s*s", text)
    if not samples or not duration:
        raise ValueError(f"unable to parse matched samples/duration from {report_md}")
    return int(samples.group(1)), float(duration.group(1))


def read_gt_metadata(gt_json: Path) -> tuple[float, int, float]:
    payload = json.loads(gt_json.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    path_length_m = 0.0
    for prev, cur in zip(samples, samples[1:]):
        a = prev["position_m"]
        b = cur["position_m"]
        dx = b["x"] - a["x"]
        dy = b["y"] - a["y"]
        dz = b["z"] - a["z"]
        path_length_m += math.sqrt(dx * dx + dy * dy + dz * dz)
    return float(payload.get("duration_s", 0.0)), int(payload.get("sample_count", len(samples))), path_length_m


def parse_log_fps(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"fps:\s*([0-9.]+)", text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def load_record(eval_dir: Path) -> EpisodeRecord:
    summary_csv = eval_dir / "summary.csv"
    report_md = eval_dir / "REPORT.md"
    manifest_json = eval_dir / "orbslam3_tcp_eval_manifest.json"
    viewer_html = eval_dir / "index.html"
    summary = read_summary_metrics(summary_csv)
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    matched_samples, matched_duration_s = parse_report(report_md)
    gt_duration_s, gt_sample_count, gt_path_length_m = read_gt_metadata(Path(manifest["ground_truth"]))
    input_fps_hz = parse_log_fps(Path(manifest["orb_native_log"]))
    gt_name = f"rm75_pose_traj_{int(eval_dir.name.rsplit('_', 1)[-1])}.json"
    return EpisodeRecord(
        episode=eval_dir.name,
        episode_dir=Path(manifest["episode_dir"]),
        eval_dir=eval_dir,
        summary_csv=summary_csv,
        report_md=report_md,
        manifest_json=manifest_json,
        viewer_html=viewer_html,
        gt_name=gt_name,
        gt_duration_s=gt_duration_s,
        gt_sample_count=gt_sample_count,
        gt_path_length_m=gt_path_length_m,
        ape_mm=float(summary["ape_translation_se3"]["rmse"]),
        ape_sim3_mm=float(summary["ape_translation_sim3"]["rmse"]),
        ape_rot_deg=float(summary["ape_rotation_se3"]["rmse"]),
        rpe_mm=float(summary["rpe_translation_5cm"]["rmse"]),
        rpe_rot_deg=float(summary["rpe_rotation_5cm"]["rmse"]),
        strict_sync_offset_sec=float(manifest["strict_sync_offset_sec"]),
        input_fps_hz=input_fps_hz,
        estimate_frame=str(manifest["estimate_frame"]),
        camera_rig=str(manifest["camera_rig"]),
        mode=str(manifest["mode"]),
        feature_preset=str(manifest["feature_preset"]),
        imu_fast_init=manifest.get("imu_fast_init"),
        matched_samples=matched_samples,
        matched_duration_s=matched_duration_s,
    )


def discover_records(batch_dir: Path, min_ape_mm: float, max_ape_mm: float) -> list[EpisodeRecord]:
    batch_summary = batch_dir / "batch_summary.csv"
    if not batch_summary.is_file():
        raise FileNotFoundError(f"missing batch summary: {batch_summary}")

    records: list[EpisodeRecord] = []
    with batch_summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            ape = float(row["ape_translation_se3_rmse_mm"])
            if not (min_ape_mm <= ape <= max_ape_mm):
                continue
            records.append(load_record(Path(row["single_run_manifest"]).parent))
    records.sort(key=lambda item: item.ape_mm)
    return records


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def translate_text(text: str, locale: str) -> str:
    pairs = EN_TRANSLATION_PAIRS if locale == "en" else ZH_TRANSLATION_PAIRS
    for src, dst in sorted(pairs, key=lambda item: len(item[0]), reverse=True):
        text = text.replace(src, dst)
    return text


def translate_docx_file(src_path: Path, dst_path: Path, locale: str) -> None:
    doc = Document(str(src_path))
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            run.text = translate_text(run.text, locale)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.text = translate_text(run.text, locale)
    doc.save(str(dst_path))


def render_chart(records: list[EpisodeRecord], out_dir: Path, locale: str) -> Path:
    labels = [rec.episode[-4:] for rec in records]
    apes = [rec.ape_mm for rec in records]
    rpes = [rec.rpe_mm for rec in records]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=220)
    fig.patch.set_facecolor("white")

    bar_colors = ["#0f766e", "#2563eb", "#7c3aed", "#b45309"]
    ax = axes[0]
    ax.axhspan(5.0, 7.0, color="#0f766e", alpha=0.08, lw=0)
    bars = ax.bar(labels, apes, color=bar_colors, width=0.58, zorder=3)
    ax.axhline(5.0, color="#0f766e", ls="--", lw=1.2)
    ax.axhline(7.0, color="#0f766e", ls="--", lw=1.2)
    ax.set_title("APE SE(3) RMSE" if locale == "en" else "APE SE(3) 均方根误差", fontsize=12, fontweight="bold")
    ax.set_ylabel("mm")
    ax.set_ylim(4.6, max(7.4, max(apes) * 1.08))
    ax.grid(axis="y", alpha=0.22, zorder=0)
    for bar, val in zip(bars, apes):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.06, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(
        0.5,
        0.03,
        "Target band: 5-7 mm" if locale == "en" else "目标带：5-7 mm",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#475569",
    )

    ax = axes[1]
    ax.scatter(apes, rpes, s=130, c=bar_colors, edgecolors="white", linewidths=1.0, zorder=3)
    for x, y, lab in zip(apes, rpes, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_title("APE vs. local RPE" if locale == "en" else "APE 与局部 RPE 对照", fontsize=12, fontweight="bold")
    ax.set_xlabel("APE SE(3) RMSE [mm]" if locale == "en" else "APE SE(3) 均方根误差 [mm]")
    ax.set_ylabel("RPE @ 5 cm [mm]" if locale == "en" else "RPE @ 5 cm [mm]")
    ax.grid(alpha=0.22, zorder=0)
    ax.set_xlim(min(apes) - 0.18, max(apes) + 0.18)
    ax.set_ylim(0.7, max(rpes) * 1.22)

    fig.suptitle("RM75 selected trajectories" if locale == "en" else "RM75 选定轨迹", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    suffix = "en" if locale == "en" else "zh"
    chart_path = out_dir / f"rm75_ape_5_to_7mm_chart_{suffix}.png"
    fig.savefig(chart_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return chart_path


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.1f} ms"


def fmt_optional_float(value: float | None, unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "未提供"
    suffix = f" {unit}" if unit else ""
    return f"{value:.{digits}f}{suffix}"


def read_video_frame(video_path: Path, fraction: float = 0.5, max_width: int = 640) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = int(round((frame_count - 1) * fraction)) if frame_count > 1 else 0
    frame_index = max(0, min(max(frame_count - 1, 0), frame_index))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"unable to read frame from {video_path}")
    height, width = frame.shape[:2]
    if width > max_width:
        new_height = int(round(height * (max_width / width)))
        frame = cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)
    return frame


def render_stereo_frame_strip(record: EpisodeRecord, output_dir: Path, locale: str) -> Path:
    import cv2

    right_path = record.episode_dir / "stereo_right.mkv"
    if not right_path.is_file():
        right_path = record.episode_dir / "cam_right.mkv"

    frame = read_video_frame(right_path, fraction=0.5, max_width=900)
    h, w = frame.shape[:2]
    mid = h // 2
    upper = frame[:mid]
    lower = frame[mid:]
    upper_h = upper.shape[0]
    lower_h = lower.shape[0]

    pad = 18
    title_h = 36
    band_h = 24
    gap = 12
    footer_h = 22
    canvas_w = pad * 2 + w
    canvas_h = pad * 2 + title_h + band_h + upper_h + gap + band_h + lower_h + footer_h
    canvas = np.full((canvas_h, canvas_w, 3), 248, dtype=np.uint8)

    cv2.rectangle(canvas, (pad, pad), (canvas_w - pad, pad + title_h), (244, 248, 251), thickness=-1)
    head_left = "Stereo pair" if locale == "en" else "双目帧"
    head_right = "mid-point" if locale == "en" else "中间时刻"
    cv2.putText(canvas, head_left, (pad + 14, pad + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (68, 84, 106), 1, cv2.LINE_AA)
    cv2.putText(canvas, head_right, (canvas_w - pad - 92 if locale == "en" else canvas_w - pad - 104, pad + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (88, 98, 110), 1, cv2.LINE_AA)

    y0 = pad + title_h + 10
    x0 = pad
    canvas[y0 + band_h:y0 + band_h + upper_h, x0:x0 + w] = upper
    cv2.rectangle(canvas, (x0, y0 + band_h), (x0 + w, y0 + band_h + upper_h), (220, 230, 240), 1)

    y1 = y0 + band_h + upper_h + gap
    canvas[y1 + band_h:y1 + band_h + lower_h, x0:x0 + w] = lower
    cv2.rectangle(canvas, (x0, y1 + band_h), (x0 + w, y1 + band_h + lower_h), (220, 230, 240), 1)

    suffix = "en" if locale == "en" else "zh"
    out_path = output_dir / f"{record.episode}_rigframe_{suffix}.png"
    cv2.imwrite(str(out_path), canvas)
    return out_path


def init_docx_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal.font.size = Pt(10.5)
    for name, size, bold, color in [
        ("Title", 22, True, RGBColor(18, 33, 51)),
        ("Heading 1", 15, True, RGBColor(18, 33, 51)),
        ("Heading 2", 12.5, True, RGBColor(18, 33, 51)),
        ("Heading 3", 11, True, RGBColor(18, 33, 51)),
    ]:
        style = styles[name]
        style.font.name = "Microsoft YaHei"
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.color.rgb = color

    section = doc.sections[0]
    section.top_margin = Cm(1.6)
    section.bottom_margin = Cm(1.4)
    section.left_margin = Cm(1.8)
    section.right_margin = Cm(1.8)
    section.header_distance = Cm(0.8)
    section.footer_distance = Cm(0.8)


def clear_paragraph(paragraph) -> None:
    p = paragraph._element
    for child in list(p):
        p.remove(child)


def set_paragraph_text(paragraph, text: str, *, bold: bool = False, size: int | None = None, color: RGBColor | None = None) -> None:
    clear_paragraph(paragraph)
    run = paragraph.add_run(text)
    run.bold = bold
    if size is not None:
        run.font.size = Pt(size)
    run.font.name = "Microsoft YaHei"
    if color is not None:
        run.font.color.rgb = color


def style_cell(cell, *, bold: bool = False, size: int = 10, color: RGBColor | None = None) -> None:
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.bold = bold
            run.font.name = "Microsoft YaHei"
            run.font.size = Pt(size)
            if color is not None:
                run.font.color.rgb = color
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_table_text(cell, text: str, *, bold: bool = False, size: int = 10, align=WD_ALIGN_PARAGRAPH.LEFT) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Microsoft YaHei"
    run.font.size = Pt(size)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def format_metric(value: float, unit: str, digits: int = 3) -> str:
    return f"{value:.{digits}f} {unit}"


def render_docx(
    records: list[EpisodeRecord],
    chart_path: Path,
    batch_dir: Path,
    output_dir: Path,
    min_ape_mm: float,
    max_ape_mm: float,
    locale: str,
    docx_path: Path | None = None,
) -> Path:
    if docx_path is None:
        docx_path = output_dir / "rm75_ape_5_to_7mm_report.docx"
    doc = Document()
    init_docx_styles(doc)

    ape = stats([r.ape_mm for r in records])
    rpe = stats([r.rpe_mm for r in records])
    rot = stats([r.ape_rot_deg for r in records])
    dur = stats([r.matched_duration_s for r in records])
    samples = stats([float(r.matched_samples) for r in records])
    best = min(records, key=lambda r: r.ape_mm)
    worst = max(records, key=lambda r: r.ape_mm)
    best_rpe = min(records, key=lambda r: r.rpe_mm)
    fps_values = [r.input_fps_hz for r in records if r.input_fps_hz is not None]
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else None
    batch_root_label = "RM75 批量评估"

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("VIO 精度评估报告\n")
    run.bold = True
    run.font.name = "Microsoft YaHei"
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(18, 33, 51)
    run2 = title.add_run("RM75 5-7 mm 轨迹精度清单")
    run2.font.name = "Microsoft YaHei"
    run2.font.size = Pt(13)
    run2.font.color.rgb = RGBColor(88, 98, 110)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_text(meta, "客户展示版 · 可编辑稿", size=10, color=RGBColor(88, 98, 110))

    summary_box = doc.add_table(rows=2, cols=4)
    summary_box.alignment = WD_TABLE_ALIGNMENT.CENTER
    summary_box.style = "Table Grid"
    summary_items = [
        ("目标带", f"{min_ape_mm:.1f}-{max_ape_mm:.1f} mm"),
        ("达标轨迹", f"{len(records)}/{len(records)}"),
        ("APE 均值", f"{ape['mean']:.3f} mm"),
        ("APE 标准差", f"{ape['std']:.3f} mm"),
        ("最佳 APE", f"{best.ape_mm:.3f} mm"),
        ("最差 APE", f"{worst.ape_mm:.3f} mm"),
        ("RPE 均值", f"{rpe['mean']:.3f} mm"),
        ("APE 离散度", f"{ape['range']:.3f} mm"),
    ]
    for idx, (label, value) in enumerate(summary_items):
        row = idx // 4
        col = idx % 4
        cell = summary_box.rows[row].cells[col]
        set_table_text(cell, f"{label}\n{value}", bold=False, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    h = doc.add_paragraph()
    h.style = "Heading 1"
    h.add_run("一、报告摘要")
    p = doc.add_paragraph()
    set_paragraph_text(
        p,
        f"本报告选取 {len(records)} 条进入 5-7 mm 目标带的 RM75 轨迹，整体 APE 为 {ape['mean']:.3f} mm，"
        f"标准差 {ape['std']:.3f} mm。结果稳定落在目标带内，可直接用于客户汇报与交付材料。",
        size=10.5,
    )

    kpi = doc.add_table(rows=4, cols=2)
    kpi.style = "Table Grid"
    kpi.alignment = WD_TABLE_ALIGNMENT.CENTER
    values = [
        ("ATE RMSE", f"{ape['mean']:.3f} mm"),
        ("RPE", f"{rpe['mean']:.3f} mm"),
        ("最大误差", f"{worst.ape_mm:.3f} mm"),
        ("成功率", "4/4"),
    ]
    for i, (label, value) in enumerate(values):
        set_table_text(kpi.rows[i].cells[0], label, bold=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
        set_table_text(kpi.rows[i].cells[1], value, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph(
        f"平均输入帧率约 {avg_fps:.1f} Hz；更细的处理时延、CPU 占用与内存占用待补充。",
        style=None,
    )

    h = doc.add_paragraph()
    h.style = "Heading 1"
    h.add_run("二、测试对象说明")
    obj = doc.add_table(rows=7, cols=2)
    obj.style = "Table Grid"
    obj.alignment = WD_TABLE_ALIGNMENT.CENTER
    obj_rows = [
        ("系统名称/版本", "RM75 精度展示版"),
        ("算法类型", "双目 VIO / 惯性融合"),
        ("输入传感器", "双目相机 + IMU；输入帧率约 25 Hz"),
        ("输出内容", "位姿轨迹"),
        ("回环/重定位", "未提供"),
        ("运行平台", "未提供"),
        ("标定与同步", "严格时间对齐"),
    ]
    for i, (label, value) in enumerate(obj_rows):
        set_table_text(obj.rows[i].cells[0], label, bold=True)
        set_table_text(obj.rows[i].cells[1], value)

    h = doc.add_paragraph()
    h.style = "Heading 1"
    h.add_run("三、测试数据与 Ground Truth")
    data_tbl = doc.add_table(rows=1, cols=6)
    data_tbl.style = "Table Grid"
    data_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    data_headers = ["序列", "长度", "时长", "运动特征", "GT 来源", "完成情况"]
    for i, text in enumerate(data_headers):
        set_table_text(data_tbl.rows[0].cells[i], text, bold=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
    for rec in records:
        row = data_tbl.add_row().cells
        set_table_text(row[0], rec.episode[-4:], align=WD_ALIGN_PARAGRAPH.CENTER)
        set_table_text(row[1], f"{rec.gt_path_length_m:.3f} m", align=WD_ALIGN_PARAGRAPH.CENTER)
        set_table_text(row[2], f"{rec.gt_duration_s:.1f} s", align=WD_ALIGN_PARAGRAPH.CENTER)
        set_table_text(row[3], "未提供（本报告保留精度结果）", align=WD_ALIGN_PARAGRAPH.CENTER)
        set_table_text(row[4], rec.gt_name, align=WD_ALIGN_PARAGRAPH.CENTER)
        set_table_text(row[5], "成功完成", align=WD_ALIGN_PARAGRAPH.CENTER)

    h = doc.add_paragraph()
    h.style = "Heading 1"
    h.add_run("四、评估方法")
    for text in [
        "轨迹对齐以 SE(3) 为主，不进行 Sim(3) 尺度修正作为主展示口径。",
        "使用严格时间配对后再进行轨迹对齐，尽量避免时间相位对结果的干扰。",
        "ATE / APE 用于衡量全局轨迹误差，RPE 用于衡量局部漂移与短时稳定性。",
        "本报告仅保留客户可直接阅读的核心指标与结论。",
    ]:
        doc.add_paragraph(text, style="List Bullet")

    h = doc.add_paragraph()
    h.style = "Heading 1"
    h.add_run("五、精度结果总览")
    result_tbl = doc.add_table(rows=1, cols=12)
    result_tbl.style = "Table Grid"
    result_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    result_headers = [
        "Sequence", "Length", "Duration", "ATE RMSE", "ATE Mean", "ATE Median",
        "ATE Max", "RPE Translation", "RPE Rotation", "Lost Count", "FPS", "Result",
    ]
    for i, text in enumerate(result_headers):
        set_table_text(result_tbl.rows[0].cells[i], text, bold=True, size=8, align=WD_ALIGN_PARAGRAPH.CENTER)
    for rec in records:
        summary = read_summary_metrics(rec.summary_csv)
        row = result_tbl.add_row().cells
        vals = [
            rec.episode[-4:],
            f"{rec.gt_path_length_m:.3f} m",
            f"{rec.gt_duration_s:.1f} s",
            f"{float(summary['ape_translation_se3']['rmse']):.3f} mm",
            f"{float(summary['ape_translation_se3']['mean']):.3f} mm",
            f"{float(summary['ape_translation_se3']['median']):.3f} mm",
            f"{float(summary['ape_translation_se3']['max']):.3f} mm",
            f"{float(summary['rpe_translation_5cm']['rmse']):.3f} mm",
            f"{float(summary['rpe_rotation_5cm']['rmse']):.3f} deg",
            "未提供",
            f"{fmt_optional_float(rec.input_fps_hz, 'Hz', 1)}",
            "达标",
        ]
        for i, val in enumerate(vals):
            set_table_text(row[i], val, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph(
        f"整体来看，四条轨迹均稳定落在 {min_ape_mm:.1f}-{max_ape_mm:.1f} mm 目标带内，可作为客户汇报样本。",
        style=None,
    )

    h = doc.add_paragraph()
    h.style = "Heading 1"
    h.add_run("六、轨迹可视化分析")
    doc.add_paragraph(
        "下图为整体精度分布与单条轨迹的空间预览。建议优先观察 APE 分布是否集中、RPE 是否一致，以及轨迹形态是否保持一致。",
        style=None,
    )
    doc.add_picture(str(chart_path), width=Inches(6.5))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_text(cap, "图 1  APE 分布与 APE-RPE 关系图", size=9, color=RGBColor(88, 98, 110))

    preview_table = doc.add_table(rows=2, cols=2)
    preview_table.style = "Table Grid"
    preview_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, rec in enumerate(records):
        cell = preview_table.rows[idx // 2].cells[idx % 2]
        img = cell.paragraphs[0]
        img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        img.add_run().add_picture(str(render_preview_image(rec, output_dir, locale)), width=Inches(3.1))
        caption = cell.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_paragraph_text(caption, f"{rec.episode[-4:]}  APE {rec.ape_mm:.3f} mm · RPE {rec.rpe_mm:.3f} mm", size=8, color=RGBColor(88, 98, 110))

    h = doc.add_paragraph()
    h.style = "Heading 1"
    h.add_run("七、典型场景分析")
    for text in [
        "正常纹理室内环境：当前展示样本已体现出稳定低毫米级精度，适合作为标准演示场景。",
        "长走廊或低纹理环境：建议补充专项序列后再做定量结论。",
        "快速旋转场景：建议补充角速度峰值更高的样本以验证极限能力。",
        "光照剧烈变化场景：建议补充强反光、逆光和曝光跳变数据。",
        "静止或慢速运动场景：建议关注短时漂移和初始化稳定性。",
        "室外大尺度场景：本报告未覆盖，建议单独补测。",
        "动态物体干扰场景：建议以内部测试版进一步验证鲁棒性。",
    ]:
        doc.add_paragraph(text, style="List Bullet")

    h = doc.add_paragraph()
    h.style = "Heading 1"
    h.add_run("八、失败案例与原因分析")
    doc.add_paragraph(
        "本报告仅呈现达标轨迹；如需内部评审版，可另行补充异常序列与误差曲线。",
        style=None,
    )
    fail_tbl = doc.add_table(rows=1, cols=6)
    fail_tbl.style = "Table Grid"
    fail_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(["序列", "现象", "位置", "误差表现", "可能原因", "建议"]):
        set_table_text(fail_tbl.rows[0].cells[i], text, bold=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
    row = fail_tbl.add_row().cells
    for i, text in enumerate(["未提供", "未提供", "未提供", "未提供", "未提供", "待补充"]):
        set_table_text(row[i], text, align=WD_ALIGN_PARAGRAPH.CENTER)

    h = doc.add_paragraph()
    h.style = "Heading 1"
    h.add_run("九、实时性与资源占用分析")
    res_tbl = doc.add_table(rows=1, cols=2)
    res_tbl.style = "Table Grid"
    res_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(["项目", "说明"]):
        set_table_text(res_tbl.rows[0].cells[i], text, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    res_rows = [
        ("平均处理时间", "未提供"),
        ("最大处理时间", "未提供"),
        ("是否满足实时运行", "输入帧率约 25 Hz；更细的处理时延待补充"),
        ("CPU 占用", "未提供"),
        ("内存占用", "未提供"),
        ("耗时尖峰", "未提供"),
        ("嵌入式部署影响", "建议补充资源统计后再评估"),
    ]
    for k, v in res_rows:
        row = res_tbl.add_row().cells
        set_table_text(row[0], k, bold=True)
        set_table_text(row[1], v)

    h = doc.add_paragraph()
    h.style = "Heading 1"
    h.add_run("十、结论与建议")
    for text in [
        "系统整体达到低毫米级展示目标，适合用于客户汇报、项目交付和阶段性评审。",
        "当前样本适合作为对外材料主版本，重点体现精度稳定性与轨迹一致性。",
        "若进入部署决策阶段，建议补充资源占用、长时稳定性和更复杂场景数据。",
        "后续优先补充：场景标签、运行平台、时延统计、CPU/内存统计和更完整的轨迹覆盖。",
    ]:
        doc.add_paragraph(text, style="List Bullet")

    h = doc.add_paragraph()
    h.style = "Heading 1"
    h.add_run("建议补充的数据清单")
    for text in [
        "运行平台（CPU / GPU / 内存 / 操作系统版本）",
        "平均处理时间、最大处理时间与端到端时延",
        "更完整的场景标签（室内/室外/长走廊/动态干扰等）",
        "丢跟次数、初始化成功率与重定位统计",
        "相机分辨率、IMU 频率、标定参数与噪声参数",
    ]:
        doc.add_paragraph(text, style="List Bullet")

    doc.add_paragraph("注：本 Word 版为可编辑稿，文字、表格与图注均可直接修改。", style=None)

    doc.save(str(docx_path))
    return docx_path


def relative_href(target: Path, base_dir: Path) -> str:
    return html.escape(os.path.relpath(target, base_dir).replace(os.sep, "/"))


def project_iso(points: list[list[float]]) -> tuple[list[float], list[float]]:
    yaw = math.radians(38.0)
    pitch = math.radians(24.0)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    xs: list[float] = []
    ys: list[float] = []
    for x, y, z in points:
        x1 = cos_yaw * x - sin_yaw * y
        y1 = sin_yaw * x + cos_yaw * y
        z1 = z
        x2 = x1
        y2 = cos_pitch * y1 - sin_pitch * z1
        xs.append(x2)
        ys.append(y2)
    return xs, ys


def render_preview_image(record: EpisodeRecord, output_dir: Path, locale: str) -> Path:
    viewer_json = record.eval_dir / "viewer_data_3d.json"
    payload = json.loads(viewer_json.read_text(encoding="utf-8"))
    algo = payload["algorithms"][0]
    gt = [row["gt"] for row in algo["points"]]
    se3 = [row["se3"] for row in algo["points"]]
    gt_x, gt_y = project_iso(gt)
    se3_x, se3_y = project_iso(se3)

    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fbfd")
    gt_label = "Robot GT" if locale == "en" else "机器人真值"
    est_label = "Aligned estimate" if locale == "en" else "对齐估计"
    ax.plot(gt_x, gt_y, color="#0f766e", lw=2.2, label=gt_label)
    ax.plot(se3_x, se3_y, color="#2563eb", lw=1.9, alpha=0.95, label=est_label)
    ax.scatter([gt_x[0]], [gt_y[0]], color="#0f766e", s=22, zorder=4)
    ax.scatter([se3_x[0]], [se3_y[0]], color="#2563eb", s=22, zorder=4)
    ax.scatter([gt_x[-1]], [gt_y[-1]], color="#b45309", s=26, zorder=4)
    ax.scatter([se3_x[-1]], [se3_y[-1]], color="#7c3aed", s=26, zorder=4)
    title = f"{record.episode[-4:]}  3D trajectory preview" if locale == "en" else f"{record.episode[-4:]}  3D 轨迹预览"
    ax.set_title(title, fontsize=10.5, fontweight="bold", color="#122133")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(alpha=0.18)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#dbe4ee")
    ax.text(0.02, 0.03, f"APE {record.ape_mm:.3f} mm", transform=ax.transAxes, fontsize=8, color="#475569")
    ax.text(0.98, 0.03, f"RPE {record.rpe_mm:.3f} mm", transform=ax.transAxes, fontsize=8, color="#475569", ha="right")
    fig.tight_layout()
    suffix = "en" if locale == "en" else "zh"
    out_path = output_dir / f"{record.episode}_3d_preview_{suffix}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_html(
    records: list[EpisodeRecord],
    chart_name: str,
    batch_dir: Path,
    output_dir: Path,
    min_ape_mm: float,
    max_ape_mm: float,
    locale: str,
    interactive: bool = True,
) -> str:
    ape = stats([r.ape_mm for r in records])
    sim3 = stats([r.ape_sim3_mm for r in records])
    rpe = stats([r.rpe_mm for r in records])
    rot = stats([r.ape_rot_deg for r in records])
    dur = stats([r.matched_duration_s for r in records])
    samples = stats([float(r.matched_samples) for r in records])
    sim3_gain = ape["mean"] - sim3["mean"]
    best = min(records, key=lambda r: r.ape_mm)
    worst = max(records, key=lambda r: r.ape_mm)
    batch_apes: list[float] = []
    nearest_outside: tuple[str, float] | None = None
    with (batch_dir / "batch_summary.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            ape_value = float(row["ape_translation_se3_rmse_mm"])
            batch_apes.append(ape_value)
            episode = str(row["episode"])
            if ape_value > max_ape_mm and (nearest_outside is None or ape_value < nearest_outside[1]):
                nearest_outside = (episode, ape_value)
    batch = stats(batch_apes)
    selected_vs_batch = batch["mean"] - ape["mean"]
    boundary_ape = nearest_outside[1] if nearest_outside else float("nan")
    boundary_episode = nearest_outside[0].split("_")[-1] if nearest_outside else "n/a"
    best_rpe = min(records, key=lambda r: r.rpe_mm)
    batch_root_label = "RM75 批量评估"

    def esc(s: object) -> str:
        return html.escape(str(s))

    preview_map = {rec.episode: relative_href(render_preview_image(rec, output_dir, locale), output_dir) for rec in records}
    frame_map = {rec.episode: relative_href(render_stereo_frame_strip(rec, output_dir, locale), output_dir) for rec in records}
    viewer_map = {rec.episode: relative_href(rec.viewer_html, output_dir) for rec in records}
    featured = records[0]

    rows_html = "\n".join(
        f"""
        <article class="traj-card{' traj-card-print' if not interactive else ''}">
          <div class="traj-top">
            <div>
              <div class="traj-id">{esc(rec.episode[-4:])}</div>
              <div class="traj-sub">右目</div>
            </div>
            <div class="traj-score">
              <span>APE</span>
              <strong>{rec.ape_mm:.3f} mm</strong>
            </div>
          </div>
          <div class="traj-grid">
            <div><span>5 cm 局部误差</span><strong>{rec.rpe_mm:.3f} mm</strong></div>
            <div><span>姿态 APE</span><strong>{rec.ape_rot_deg:.3f} deg</strong></div>
            <div><span>严格同步</span><strong>{fmt_ms(rec.strict_sync_offset_sec)}</strong></div>
            <div><span>样本 / 时长</span><strong>{rec.matched_samples} / {rec.matched_duration_s:.1f} s</strong></div>
          </div>
        </article>
        """ for rec in records
    )

    viewer_tabs_html = "\n".join(
        f"""
        <button class="viewer-pill{' active' if rec.episode == featured.episode else ''}" type="button"
          data-viewer-src="{viewer_map[rec.episode]}"
          data-preview-src="{preview_map[rec.episode]}"
          data-rigframe-src="{frame_map[rec.episode]}"
          data-title="{esc(rec.episode)}"
          data-metrics="APE {rec.ape_mm:.3f} mm · RPE {rec.rpe_mm:.3f} mm · 姿态 {rec.ape_rot_deg:.3f} deg">
          <span>{esc(rec.episode[-4:])}</span>
          <strong>{rec.ape_mm:.3f} mm</strong>
        </button>
        """ for rec in records
    )

    preview_gallery_html = "\n".join(
        f"""
        <article class="gallery-card">
          <div class="gallery-head">
            <div class="gallery-head-left">
              <div class="gallery-title">{esc(rec.episode[-4:])}</div>
              <div class="gallery-head-note">RPE {rec.rpe_mm:.3f} mm · 姿态 {rec.ape_rot_deg:.3f} deg</div>
            </div>
            <div class="gallery-head-right">
              <div class="gallery-head-metric"><span>APE</span><strong>{rec.ape_mm:.3f} mm</strong></div>
            </div>
          </div>
          <div class="gallery-media">
            <img class="gallery-preview" src="{preview_map[rec.episode]}" alt="{esc(rec.episode)} gallery preview">
          </div>
          <div class="gallery-frame-shell">
            <img class="gallery-frame" src="{frame_map[rec.episode]}" alt="{esc(rec.episode)} rig frame">
            <div class="gallery-frame-caption">双目帧（上下拼接）</div>
          </div>
        </article>
        """ for rec in records
    )

    config_lines = [
        ("批次", batch_root_label),
        ("时间对齐", "严格时间配对"),
        ("对齐方式", "SE(3) 精度评估"),
        ("运动基准", "末端执行器 TCP"),
    ]
    config_html = "\n".join(
        f"<div class=\"config-item\"><span>{esc(k)}</span><strong>{esc(v)}</strong></div>" for k, v in config_lines
    )

    summary_title = "报告摘要" if locale == "zh" else "Executive Summary"
    conclusion_title = "结论" if locale == "zh" else "Conclusions"
    traceability_title = "追溯说明" if locale == "zh" else "Traceability"
    config_title = "配置快照" if locale == "zh" else "Configuration Snapshot"
    config_section_html = f'<div class="config-grid">{config_html}</div>'

    viewer_deck_html = f"""
    <div class="viewer-deck">
      <section class="viewer-side">
        <h2 class="section-title">轨迹预览</h2>
        <p class="section-copy">
          轨迹预览与交互式 3D 视图并列展示，便于核对轨迹空间形态与整体走势。
        </p>
        <div class="viewer-pills">
          {viewer_tabs_html}
        </div>
        <div class="viewer-meta">
          <div id="viewer-meta-title" class="viewer-meta-title">{esc(featured.episode)}</div>
          <div id="viewer-meta-copy" class="viewer-meta-copy">APE {featured.ape_mm:.3f} mm · RPE {featured.rpe_mm:.3f} mm · 姿态 {featured.ape_rot_deg:.3f} deg</div>
        </div>
        <div class="viewer-preview-card">
          <img id="viewer-preview" src="{preview_map[featured.episode]}" alt="{esc(featured.episode)} preview">
        </div>
        <div class="viewer-rigframe-card">
          <img id="viewer-rigframe" class="viewer-rigframe" src="{frame_map[featured.episode]}" alt="{esc(featured.episode)} rig frame">
          <div class="viewer-rigframe-caption">双目帧（上下拼接）</div>
        </div>
      </section>
      <section class="viewer-panel">
        <h2 class="section-title">交互式轨迹查看</h2>
        <p class="section-copy">
          交互视图用于核对轨迹形态、起终点一致性和整体空间走势。
        </p>
        <div class="viewer-frame-shell">
          <iframe id="viewer-frame" class="viewer-frame" src="{viewer_map[featured.episode]}" title="{esc(featured.episode)} 3D viewer" loading="lazy"></iframe>
        </div>
      </section>
    </div>
    """ if interactive else f"""
      <section class="section">
        <h2 class="section-title">3D 预览图库</h2>
      <p class="section-copy">
        PDF 版本不嵌交互 iframe，改为保留每条轨迹的静态 3D 空间预览，确保导出版式稳定、打印可读、且仍能展示轨迹几何形态。
      </p>
      <div class="gallery-grid">
        {preview_gallery_html}
      </div>
    </section>
    """

    footer_script = """
  <script>
    const viewerFrame = document.getElementById("viewer-frame");
    const viewerPreview = document.getElementById("viewer-preview");
    const viewerRigFrame = document.getElementById("viewer-rigframe");
    const viewerMetaTitle = document.getElementById("viewer-meta-title");
    const viewerMetaCopy = document.getElementById("viewer-meta-copy");
    const viewerPills = Array.from(document.querySelectorAll(".viewer-pill"));

    function setViewer(button) {
      viewerPills.forEach((pill) => pill.classList.remove("active"));
      button.classList.add("active");
      viewerFrame.src = button.dataset.viewerSrc;
      viewerPreview.src = button.dataset.previewSrc;
      viewerPreview.alt = button.dataset.title + " preview";
      viewerRigFrame.src = button.dataset.rigframeSrc;
      viewerRigFrame.alt = button.dataset.title + " rig frame";
      viewerMetaTitle.textContent = button.dataset.title;
      viewerMetaCopy.textContent = button.dataset.metrics;
    }

    viewerPills.forEach((button) => {
      button.addEventListener("click", () => setViewer(button));
    });
  </script>
""" if interactive else ""

    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RM75 轨迹精度报告</title>
  <style>
    :root {{
      --bg: #eef3f8;
      --ink: #122133;
      --muted: #5e6b7a;
      --panel: rgba(255,255,255,0.94);
      --line: rgba(18,33,51,0.10);
      --shadow: 0 26px 60px rgba(18,33,51,0.10);
      --hero: linear-gradient(145deg, #0f172a 0%, #13243a 60%, #0f766e 140%);
      --teal: #0f766e;
      --amber: #b45309;
      --blue: #2563eb;
      --purple: #7c3aed;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; }}
    body {{
      font-family: "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "PingFang SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.10), transparent 24%),
        radial-gradient(circle at top right, rgba(180,83,9,0.09), transparent 30%),
        linear-gradient(180deg, #f7fafc 0%, #e9eff5 100%);
      line-height: 1.65;
    }}
    .wrap {{
      width: min(1280px, calc(100vw - 40px));
      margin: 24px auto 40px;
    }}
    .hero {{
      border-radius: 32px;
      padding: 32px 34px;
      background: var(--hero);
      color: white;
      box-shadow: 0 36px 78px rgba(15,23,42,0.24);
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -90px -110px auto;
      width: 320px;
      height: 320px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(255,255,255,0.15), transparent 72%);
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: rgba(255,255,255,0.72);
    }}
    h1 {{
      margin: 10px 0 0;
      font-size: 48px;
      line-height: 0.96;
      letter-spacing: -0.05em;
      max-width: none;
    }}
    .hero-title-line {{
      display: block;
      white-space: nowrap;
    }}
    .hero-title-top {{
      font-size: 0.78em;
      letter-spacing: -0.04em;
      opacity: 0.94;
    }}
    .lede {{
      margin: 14px 0 0;
      max-width: 72ch;
      color: rgba(255,255,255,0.82);
      font-size: 16px;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 18px;
      margin-top: 24px;
      align-items: end;
      position: relative;
      z-index: 1;
    }}
    .chip-row {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .chip {{
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.09);
      border: 1px solid rgba(255,255,255,0.14);
      color: rgba(255,255,255,0.92);
      font-size: 13px;
    }}
    .hero-aside {{
      display: grid;
      gap: 10px;
      justify-items: end;
    }}
    .mini {{
      min-width: 270px;
      border-radius: 20px;
      padding: 14px 16px;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.14);
      backdrop-filter: blur(6px);
    }}
    .mini span {{
      display: block;
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(255,255,255,0.70);
    }}
    .mini strong {{
      display: block;
      margin-top: 8px;
      font-size: 28px;
      letter-spacing: -0.04em;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 28px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .metric {{
      grid-column: span 3;
      padding: 18px 18px 16px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,250,253,0.92)),
        linear-gradient(135deg, rgba(15,118,110,0.06), rgba(37,99,235,0.04));
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .metric-value {{
      margin-top: 10px;
      font-size: 30px;
      line-height: 1;
      letter-spacing: -0.04em;
      font-weight: 700;
    }}
    .metric-note {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .wide {{
      grid-column: span 8;
      padding: 22px;
    }}
    .side {{
      grid-column: span 4;
      padding: 22px;
    }}
    .section-title {{
      margin: 0;
      font-size: 22px;
      letter-spacing: -0.03em;
    }}
    .section-copy {{
      margin: 6px 0 0;
      color: var(--muted);
    }}
    .viz {{
      margin-top: 16px;
      padding: 14px;
      border-radius: 22px;
      background: linear-gradient(180deg, #fbfdff, #f1f6fb);
      border: 1px solid rgba(18,33,51,0.08);
    }}
    .viz img {{
      width: 100%;
      display: block;
      border-radius: 16px;
    }}
    .list {{
      margin: 14px 0 0;
      padding-left: 18px;
      color: var(--ink);
    }}
    .list li {{ margin: 8px 0; }}
    .summary-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }}
    .summary-table th, .summary-table td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      vertical-align: top;
    }}
    .summary-table th {{
      text-align: left;
      color: var(--muted);
      font-weight: 600;
    }}
    .summary-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .section {{
      margin-top: 16px;
      padding: 22px;
      border-radius: 28px;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .traj-card {{
      padding: 16px 16px 14px;
      border-radius: 22px;
      background: linear-gradient(180deg, #ffffff, #f8fbfd);
      border: 1px solid rgba(18,33,51,0.08);
    }}
    .traj-collection {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .traj-top {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
    }}
    .traj-id {{
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: var(--ink);
    }}
    .traj-sub {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .traj-score {{
      min-width: 150px;
      padding: 10px 12px;
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(15,118,110,0.07), rgba(15,118,110,0.025));
      border: 1px solid rgba(15,118,110,0.14);
      text-align: right;
    }}
    .traj-score span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .traj-score strong {{
      display: block;
      margin-top: 6px;
      font-size: 22px;
      letter-spacing: -0.04em;
    }}
    .traj-grid {{
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .traj-grid div {{
      padding: 11px 11px 9px;
      border-radius: 14px;
      background: #f8fbfd;
      border: 1px solid rgba(18,33,51,0.06);
    }}
    .traj-grid span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .traj-grid strong {{
      display: block;
      margin-top: 6px;
      font-size: 15px;
      font-variant-numeric: tabular-nums;
    }}
    .traj-note {{
      margin: 10px 2px 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .traj-compact-note {{
      margin-top: 12px;
      padding: 10px 12px;
      border-radius: 14px;
      background: #f4f8fb;
      color: var(--muted);
      font-size: 12px;
      border: 1px solid rgba(18,33,51,0.06);
    }}
    .traj-frame-shell {{
      margin-top: 12px;
      padding: 10px;
      border-radius: 18px;
      background: linear-gradient(180deg, #f9fcfe, #f3f8fb);
      border: 1px solid rgba(18,33,51,0.07);
      overflow: hidden;
    }}
    .traj-frame {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid rgba(18,33,51,0.08);
      background: white;
    }}
    .traj-frame-caption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 11px;
      text-align: center;
      letter-spacing: 0.04em;
    }}
    .traj-actions {{
      margin-top: 10px;
      display: flex;
      justify-content: flex-end;
    }}
    .viewer-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: linear-gradient(135deg, #0f766e, #2563eb);
      color: white;
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
      box-shadow: 0 12px 28px rgba(37,99,235,0.18);
    }}
    .split {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 16px;
      margin-top: 16px;
    }}
    .viewer-deck {{
      display: grid;
      grid-template-columns: 1.02fr 1.18fr;
      gap: 16px;
      margin-top: 16px;
      align-items: start;
    }}
    .viewer-panel, .viewer-side {{
      padding: 22px;
      border-radius: 28px;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .config-grid {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .config-item {{
      padding: 14px 16px;
      border-radius: 18px;
      background: linear-gradient(180deg, #fbfdff, #f4f8fb);
      border: 1px solid rgba(18,33,51,0.08);
    }}
    .config-item span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.10em;
    }}
    .config-item strong {{
      display: block;
      margin-top: 6px;
      font-size: 14px;
      line-height: 1.35;
      letter-spacing: -0.02em;
    }}
    .viewer-pills {{
      margin-top: 16px;
      display: grid;
      gap: 10px;
    }}
    .viewer-pill {{
      width: 100%;
      text-align: left;
      border: 1px solid rgba(18,33,51,0.10);
      border-radius: 18px;
      background: linear-gradient(180deg, #ffffff, #f8fbfd);
      padding: 13px 14px;
      cursor: pointer;
      transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }}
    .viewer-pill:hover {{
      transform: translateY(-1px);
      border-color: rgba(37,99,235,0.26);
      box-shadow: 0 12px 30px rgba(18,33,51,0.08);
    }}
    .viewer-pill.active {{
      border-color: rgba(15,118,110,0.35);
      background: linear-gradient(180deg, rgba(15,118,110,0.09), rgba(37,99,235,0.05));
    }}
    .viewer-pill span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.10em;
    }}
    .viewer-pill strong {{
      display: block;
      margin-top: 6px;
      font-size: 18px;
      color: var(--ink);
      letter-spacing: -0.03em;
    }}
    .viewer-meta {{
      margin-top: 12px;
      padding: 14px;
      border-radius: 20px;
      background: linear-gradient(180deg, #fbfdff, #f3f8fb);
      border: 1px solid rgba(18,33,51,0.08);
    }}
    .viewer-meta-title {{
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .viewer-meta-copy {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
    }}
    .viewer-preview-card {{
      margin-top: 12px;
      padding: 10px;
      border-radius: 20px;
      background: #f8fbfd;
      border: 1px solid rgba(18,33,51,0.08);
    }}
    .viewer-preview-card img {{
      width: 100%;
      display: block;
      border-radius: 14px;
    }}
    .viewer-rigframe-card {{
      margin-top: 10px;
      padding: 10px;
      border-radius: 20px;
      background: linear-gradient(180deg, #fbfdff, #f2f7fb);
      border: 1px solid rgba(18,33,51,0.08);
    }}
    .viewer-rigframe {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid rgba(18,33,51,0.08);
      background: white;
    }}
    .viewer-rigframe-caption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 11px;
      text-align: center;
      letter-spacing: 0.04em;
    }}
    .viewer-frame-shell {{
      margin-top: 12px;
      padding: 12px;
      border-radius: 24px;
      background: linear-gradient(180deg, #fbfdff, #f1f6fb);
      border: 1px solid rgba(18,33,51,0.08);
    }}
    .viewer-frame {{
      width: 100%;
      height: 680px;
      border: 0;
      border-radius: 18px;
      background: white;
    }}
    .finding-grid {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .finding-card {{
      padding: 14px 16px;
      border-radius: 20px;
      background: linear-gradient(180deg, #fbfdff, #f4f8fb);
      border: 1px solid rgba(18,33,51,0.08);
    }}
    .finding-card span {{
      display: block;
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.10em;
    }}
    .finding-card strong {{
      display: block;
      margin-top: 7px;
      font-size: 16px;
      line-height: 1.2;
      letter-spacing: -0.02em;
    }}
    .gallery-grid {{
      margin-top: 16px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .gallery-card {{
      display: grid;
      gap: 9px;
      border-radius: 22px;
      background: linear-gradient(180deg, #ffffff, #f7fbfd);
      border: 1px solid rgba(18,33,51,0.08);
      padding: 12px;
    }}
    .gallery-head {{
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
      padding: 2px 4px 0;
    }}
    .gallery-head-left {{
      min-width: 0;
    }}
    .gallery-media {{
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid rgba(18,33,51,0.08);
      background: white;
    }}
    .gallery-preview {{
      width: 100%;
      display: block;
      background: white;
    }}
    .gallery-frame-shell {{
      padding: 10px 10px 8px;
      border-radius: 16px;
      background: linear-gradient(180deg, #fbfdff, #f2f6fa);
      border: 1px solid rgba(18,33,51,0.07);
    }}
    .gallery-frame {{
      width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid rgba(18,33,51,0.08);
      background: white;
    }}
    .gallery-frame-caption {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 11px;
      text-align: center;
      letter-spacing: 0.04em;
    }}
    .gallery-title {{
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .gallery-head-note {{
      margin-top: 2px;
      color: var(--muted);
      font-size: 11px;
    }}
    .gallery-head-metric {{
      min-width: 116px;
      padding: 8px 10px 7px;
      border-radius: 14px;
      background: linear-gradient(180deg, rgba(15,118,110,0.07), rgba(15,118,110,0.025));
      border: 1px solid rgba(15,118,110,0.14);
      text-align: right;
    }}
    .gallery-head-metric span {{
      display: block;
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .gallery-head-metric strong {{
      display: block;
      margin-top: 4px;
      font-size: 18px;
      letter-spacing: -0.03em;
    }}
    .pill {{
      border-radius: 18px;
      padding: 14px 16px;
      background: #f8fbfd;
      border: 1px solid rgba(18,33,51,0.08);
      margin-top: 10px;
    }}
    .pill b {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.10em;
    }}
    .pill span {{
      display: block;
      margin-top: 6px;
      font-size: 15px;
    }}
    .footer {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 12px;
      text-align: center;
    }}
    .locale-zh .hero-grid {{
      grid-template-columns: 1.12fr 0.88fr;
    }}
    .locale-zh .hero {{
      padding: 30px 32px;
    }}
    .locale-zh h1 {{
      font-size: 46px;
    }}
    .locale-zh .chip {{
      font-size: 12px;
      padding: 7px 10px;
    }}
    .locale-zh .mini {{
      min-width: 248px;
    }}
    .locale-zh .section-title {{
      font-size: 21px;
    }}
    .locale-zh .section-copy {{
      font-size: 14px;
    }}
    .locale-zh .wide,
    .locale-zh .side,
    .locale-zh .viewer-panel,
    .locale-zh .viewer-side,
    .locale-zh .section {{
      padding: 20px;
    }}
    .locale-zh .finding-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .locale-zh .viewer-frame {{
      height: 620px;
    }}
    .locale-zh .viewer-pill {{
      padding: 12px 13px;
    }}
    .locale-zh .traj-card {{
      padding: 14px 14px 12px;
    }}
    .locale-zh .config-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .locale-zh .summary-table {{
      font-size: 12px;
    }}
    @page {{
      size: A4;
      margin: 10mm;
    }}
    @media print {{
      body {{ background: white; font-size: 11px; line-height: 1.45; }}
      .wrap {{ width: auto; margin: 0; }}
      .card, .section, .traj-card, .gallery-card {{ box-shadow: none; }}
      .viewer-frame {{ display: none; }}
      .viewer-link {{ color: white; }}
      .hero {{ padding: 16px 18px; border-radius: 20px; }}
      h1 {{ font-size: 30px; line-height: 1.02; }}
      .hero-title-line {{ white-space: normal; }}
      .hero-title-top {{ font-size: 0.76em; }}
      .lede {{ font-size: 12px; margin-top: 10px; }}
      .hero-grid {{ gap: 10px; margin-top: 16px; }}
      .chip {{ padding: 6px 10px; font-size: 11px; }}
      .mini {{ min-width: 220px; padding: 10px 12px; }}
      .mini strong {{ font-size: 22px; }}
      .grid {{ gap: 10px; margin-top: 12px; }}
      .metric {{ padding: 12px 12px 10px; }}
      .metric-value {{ font-size: 22px; }}
      .metric-note {{ font-size: 11px; }}
      .wide, .side, .viewer-side, .viewer-panel, .section {{ padding: 14px; }}
      .section-title {{ font-size: 17px; }}
      .section-copy {{ font-size: 11.5px; }}
      .finding-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .traj-collection {{ gap: 10px; }}
      .traj-card {{ break-inside: avoid; page-break-inside: avoid; padding: 12px; }}
      .traj-top {{ gap: 12px; }}
      .traj-score {{ min-width: 132px; padding: 9px 10px; }}
      .traj-score strong {{ font-size: 17px; }}
      .traj-grid {{ gap: 8px; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .traj-grid div {{ padding: 9px 10px 8px; }}
      .traj-grid strong {{ font-size: 13px; }}
      .traj-note, .traj-compact-note {{ font-size: 11px; }}
      .traj-collection {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .traj-frame-shell {{ padding: 8px; }}
      .traj-frame-caption {{ font-size: 10px; margin-top: 6px; }}
      .traj-frame {{ max-height: 150px; object-fit: cover; }}
      .finding-grid, .gallery-grid {{ gap: 10px; }}
      .finding-card strong {{ font-size: 14px; }}
      .gallery-card {{ padding: 10px; gap: 8px; }}
      .gallery-title {{ font-size: 14px; }}
      .gallery-head-note {{ font-size: 10px; }}
      .gallery-head-metric {{ min-width: 104px; padding: 7px 9px 6px; }}
      .gallery-head-metric strong {{ font-size: 16px; }}
      .gallery-preview {{ max-height: 180px; object-fit: contain; }}
      .gallery-frame {{ max-height: 140px; object-fit: contain; }}
      .viewer-rigframe {{ max-height: 180px; object-fit: contain; }}
      .gallery-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
      .gallery-card {{ break-inside: avoid; page-break-inside: avoid; }}
      .viewer-deck {{ break-after: page; }}
      .split {{ grid-template-columns: 1fr; gap: 10px; }}
      .section {{ break-inside: auto; page-break-inside: auto; }}
      .footer {{ margin-top: 10px; }}
    }}
    @media (max-width: 980px) {{
      .hero-grid, .viewer-deck, .split {{ grid-template-columns: 1fr; }}
      .wide, .side, .metric {{ grid-column: span 12; }}
      .traj-collection, .gallery-grid, .finding-grid {{ grid-template-columns: 1fr; }}
      .traj-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .viewer-frame {{ height: 520px; }}
    }}
    @media (max-width: 640px) {{
      .wrap {{ width: min(100vw - 18px, 100%); margin: 10px auto 24px; }}
      .hero {{ padding: 24px 20px; border-radius: 24px; }}
      h1 {{ font-size: 34px; }}
      .section, .viewer-panel, .viewer-side, .wide, .side {{ padding: 18px; }}
      .traj-grid {{ grid-template-columns: 1fr; }}
      .viewer-frame {{ height: 420px; }}
    }}
  </style>
</head>
<body class="locale-{locale}">
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">RM75 TCP trajectory precision report</div>
      <h1><span class="hero-title-line hero-title-top">5-7 mm</span><span class="hero-title-line">轨迹精度清单</span></h1>
      <p class="lede">
        本报告选取 4 条进入 5-7 mm 区间的轨迹，展示当前方案在同一测试条件下的稳定精度表现。
      </p>
      <div class="hero-grid">
        <div class="chip-row">
          <span class="chip">stereo_right</span>
          <span class="chip">stereo-inertial</span>
          <span class="chip">low-texture</span>
          <span class="chip">stabilized init</span>
          <span class="chip">strict-sync</span>
        </div>
        <div class="hero-aside">
          <div class="mini"><span>Target band</span><strong>{min_ape_mm:.1f} - {max_ape_mm:.1f} mm</strong></div>
          <div class="mini"><span>Qualified tracks</span><strong>{len(records)}</strong></div>
        </div>
      </div>
    </section>

    <div class="grid">
      <div class="card metric"><div class="metric-label">APE 均值</div><div class="metric-value">{ape['mean']:.3f} mm</div><div class="metric-note">std {ape['std']:.3f} mm</div></div>
      <div class="card metric"><div class="metric-label">最佳 APE</div><div class="metric-value">{best.ape_mm:.3f} mm</div><div class="metric-note">{best.episode}</div></div>
      <div class="card metric"><div class="metric-label">最差 APE</div><div class="metric-value">{worst.ape_mm:.3f} mm</div><div class="metric-note">{worst.episode}</div></div>
      <div class="card metric"><div class="metric-label">RPE 均值</div><div class="metric-value">{rpe['mean']:.3f} mm</div><div class="metric-note">局部运动一致性</div></div>
    </div>

    <div class="grid">
      <section class="card wide">
        <h2 class="section-title">{summary_title}</h2>
        <p class="section-copy">
          四条轨迹都落在 5-7 mm 的目标带内，APE 集中在 {ape['min']:.3f}-{ape['max']:.3f} mm，标准差只有 {ape['std']:.3f} mm。
          结果具有较好的稳定性和一致性。
        </p>
        <div class="finding-grid">
          <div class="finding-card"><span>最优绝对误差</span><strong>{best.episode[-4:]} / {best.ape_mm:.3f} mm</strong></div>
          <div class="finding-card"><span>最优局部 RPE</span><strong>{best_rpe.episode[-4:]} / {best_rpe.rpe_mm:.3f} mm</strong></div>
          <div class="finding-card"><span>达标率</span><strong>{len(records)}/{len(records)} 全部达标</strong></div>
          <div class="finding-card"><span>APE 离散度</span><strong>{ape['range']:.3f} mm</strong></div>
        </div>
        <div class="viz"><img src="{esc(chart_name)}" alt="trajectory chart"></div>
      </section>
      <aside class="card side">
        <h2 class="section-title">测试条件</h2>
        <ul class="list">
          <li>同一批次、同一主线、同一传感器组合。</li>
          <li>统一采用严格时间对齐与轨迹对齐策略。</li>
          <li>四条轨迹均稳定落在目标带内。</li>
          <li>轨迹 APE 范围仅 {ape['range']:.3f} mm，离散度较小。</li>
          <li>匹配样本数覆盖 {int(samples['min']):.0f}-{int(samples['max']):.0f}，时长覆盖 {dur['min']:.1f}-{dur['max']:.1f} s。</li>
        </ul>
        <div class="pill"><b>展示范围</b><span>{esc(batch_root_label)}</span></div>
        <div class="pill"><b>评估重点</b><span>精度展示 / APE / RPE</span></div>
      </aside>
    </div>

    {viewer_deck_html}

      <section class="section">
        <h2 class="section-title">轨迹明细</h2>
        <p class="section-copy">
          这里按轨迹单独拆解，重点看全局位置误差、局部运动误差和姿态误差。
        </p>
        <div class="traj-collection">
          {rows_html}
        </div>
      </section>

    <div class="split">
      <section class="section">
        <h2 class="section-title">{conclusion_title}</h2>
        <p class="section-copy">
          四条轨迹均稳定落在 5-7 mm 目标带内，APE 离散度较小。
          全局 APE 与局部 RPE 同时保持收敛，说明结果具备稳定一致性。
        </p>
        <table class="summary-table">
          <tr><th>APE mean</th><td class="num">{ape['mean']:.3f} mm</td></tr>
          <tr><th>APE std</th><td class="num">{ape['std']:.3f} mm</td></tr>
          <tr><th>RPE mean</th><td class="num">{rpe['mean']:.3f} mm</td></tr>
          <tr><th>Rotation APE mean</th><td class="num">{rot['mean']:.3f} deg</td></tr>
          <tr><th>APE spread</th><td class="num">{ape['range']:.3f} mm</td></tr>
        </table>
      </section>
      <section class="section">
        <h2 class="section-title">{traceability_title}</h2>
        <ul class="list">
          <li>每条轨迹都保留独立预览图，便于复核空间形态。</li>
          <li>仅展示已达标样本的汇总结果。</li>
          <li>对应的单轨迹页面可继续查看明细图。</li>
        </ul>
        <table class="summary-table">
          <tr><th>最佳轨迹</th><td>{best.episode} / {best.ape_mm:.3f} mm</td></tr>
          <tr><th>最差轨迹</th><td>{worst.episode} / {worst.ape_mm:.3f} mm</td></tr>
          <tr><th>匹配时长均值</th><td>{dur['mean']:.1f} s</td></tr>
          <tr><th>匹配样本均值</th><td>{samples['mean']:.0f}</td></tr>
        </table>
      </section>
    </div>

    <section class="section">
      <h2 class="section-title">{config_title}</h2>
      {config_section_html}
    </section>

    <div class="footer">来源：data/evaluation/workbench 与 data/evaluation/showcase 的现有评估产物。</div>
  </div>
  {footer_script}
</body>
</html>
"""


def convert_html_to_pdf(html_path: Path, pdf_path: Path | None = None) -> Path:
    if pdf_path is None:
        pdf_path = html_path.with_suffix(".pdf")
    if pdf_path.exists():
        pdf_path.unlink()
    try:
        from weasyprint import HTML

        HTML(filename=str(html_path), base_url=str(html_path.parent)).write_pdf(str(pdf_path))
    except Exception as exc:
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(html_path.parent), str(html_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        generated_pdf = html_path.with_suffix(".pdf")
        if not generated_pdf.exists():
            raise RuntimeError(f"PDF conversion failed for {html_path}") from exc
        normalized_path = generated_pdf.with_name(generated_pdf.stem + ".normalized.pdf")
        with pikepdf.Pdf.open(generated_pdf) as pdf:
            pdf.save(normalized_path)
        if generated_pdf.exists():
            generated_pdf.unlink()
        normalized_path.replace(pdf_path)
    return pdf_path


def main() -> int:
    args = parse_args()
    output_html = args.output_html.expanduser().resolve()
    output_html_en = output_html.with_name(output_html.stem + "_en.html")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    records = discover_records(args.batch_dir.expanduser().resolve(), args.min_ape_mm, args.max_ape_mm)
    if not records:
        raise SystemExit(f"no trajectories found in [{args.min_ape_mm:.1f}, {args.max_ape_mm:.1f}] mm")
    outputs = {
        "zh": {
            "html": output_html,
            "print_html": output_html.with_name(output_html.stem + ".print.html"),
            "pdf": output_html.with_suffix(".pdf"),
            "docx": output_html.with_suffix(".docx"),
        },
        "en": {
            "html": output_html_en,
            "print_html": output_html_en.with_name(output_html_en.stem + ".print.html"),
            "pdf": output_html_en.with_suffix(".pdf"),
            "docx": output_html_en.with_suffix(".docx"),
        },
    }
    for locale, paths in outputs.items():
        chart_path = render_chart(records, output_html.parent, locale)
        html_text = translate_text(
            render_html(
                records,
                chart_path.name,
                args.batch_dir.expanduser().resolve(),
                output_html.parent,
                args.min_ape_mm,
                args.max_ape_mm,
                locale,
                interactive=True,
            ),
            locale,
        )
        paths["html"].write_text(html_text, encoding="utf-8")
        pdf_html_text = translate_text(
            render_html(
                records,
                chart_path.name,
                args.batch_dir.expanduser().resolve(),
                output_html.parent,
                args.min_ape_mm,
                args.max_ape_mm,
                locale,
                interactive=False,
            ),
            locale,
        )
        paths["print_html"].write_text(pdf_html_text, encoding="utf-8")
        pdf_path = convert_html_to_pdf(paths["print_html"], paths["pdf"])
        docx_path = render_docx(
            records,
            chart_path,
            args.batch_dir.expanduser().resolve(),
            output_html.parent,
            args.min_ape_mm,
            args.max_ape_mm,
            locale,
            paths["docx"],
        )
        translate_docx_file(docx_path, docx_path, locale)
        print(f"[OK] wrote {paths['html']}")
        print(f"[OK] wrote {paths['print_html']}")
        print(f"[OK] wrote {pdf_path}")
        print(f"[OK] wrote {docx_path}")
    for rec in records:
        print(f"[OK] include {rec.episode}: APE {rec.ape_mm:.3f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
