#!/usr/bin/env python3
"""Build a full-trajectory overlay viewer for robot TCP GT and VINS-derived TCP.

This viewer intentionally does not timestamp-match samples.  The optional SE(3)
display transform is estimated from arc-length-normalized resampling of the two
complete trajectories, so it is useful for shape inspection rather than metric
accuracy reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GT = REPO_ROOT / "data/ground_truth/trajectory_samples0617/trajectory_sync_rawpose_004.json"
DEFAULT_VINS = REPO_ROOT / "data/gripper_data/episode_20260617_0004/right/vio_log/frame_level_optimized_pose_cam0.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/evaluation/workbench/episode_20260617_0004_full_trajectory_overlay"

T_TCP_LEFT_CAMERA = np.array(
    [
        [0.854672738, -0.422689316, 0.301443615, 0.020087823],
        [0.519166541, 0.694970001, -0.497476432, -0.097333617],
        [0.000783703, 0.581678983, 0.813418064, 0.052300458],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>episode_20260617_0004 full trajectory overlay</title>
<style>
:root{color-scheme:dark;--bg:#101214;--panel:#171b1f;--line:#2a3138;--text:#eef2f5;--muted:#9aa7b2;--gt:#35d07f;--vins:#4ea1ff;--se3:#ffb84d;--grid:#263039}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{padding:16px 20px;border-bottom:1px solid var(--line);background:#12161a}h1{margin:0 0 6px;font-size:20px}.sub{color:var(--muted)}.layout{display:grid;grid-template-columns:380px minmax(0,1fr);min-height:calc(100vh - 82px)}aside{background:var(--panel);border-right:1px solid var(--line);padding:16px;overflow:auto}main{padding:16px;overflow:hidden}.row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}button{border:1px solid var(--line);background:#20262c;color:var(--text);border-radius:6px;padding:7px 10px;cursor:pointer}button.active{border-color:#6aa9ff;background:#22344a}.legend{display:grid;gap:8px;margin:14px 0}.legend span{display:inline-flex;gap:8px;align-items:center}.dot{width:10px;height:10px;border-radius:50%;display:inline-block}table{width:100%;border-collapse:collapse;margin:14px 0}td{padding:7px 5px;border-bottom:1px solid var(--line);vertical-align:top}td:last-child{text-align:right;color:var(--text)}pre{white-space:pre-wrap;word-break:break-word;color:var(--muted);background:#111519;border:1px solid var(--line);border-radius:6px;padding:10px;max-height:360px;overflow:auto}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-auto-rows:minmax(260px,1fr);gap:12px;height:calc(100vh - 126px)}.wide{grid-column:1/-1;min-height:340px}.plot{position:relative;min-height:240px;border:1px solid var(--line);border-radius:8px;background:#111519;overflow:hidden}.plot-title{position:absolute;left:10px;top:8px;color:var(--muted);font-size:12px;z-index:1}canvas{display:block;width:100%;height:100%}@media(max-width:960px){.layout{grid-template-columns:1fr}.grid{grid-template-columns:1fr;height:auto}.plot{height:300px}}
</style>
</head>
<body><header><h1 id="title"></h1><div id="subtitle" class="sub"></div></header><div class="layout"><aside><div class="row"><button id="rawBtn" class="active">Raw</button><button id="se3Btn">SE(3)</button></div><div class="legend"><span><i class="dot" style="background:var(--gt)"></i>Robot TCP GT full trajectory</span><span><i class="dot" style="background:var(--vins)"></i>VINS-derived TCP full trajectory</span><span><i class="dot" style="background:var(--se3)"></i>VINS after global SE(3)</span></div><table id="stats"></table><pre id="inputs"></pre></aside><main><div class="grid"><div class="plot wide"><div class="plot-title">3D full trajectory overlay, drag to rotate, wheel to zoom</div><canvas id="view3d"></canvas></div><div class="plot"><div class="plot-title">XY plane</div><canvas id="xy"></canvas></div><div class="plot"><div class="plot-title">XZ plane</div><canvas id="xz"></canvas></div><div class="plot"><div class="plot-title">YZ plane</div><canvas id="yz"></canvas></div><div class="plot"><div class="plot-title">Normalized progress path length</div><canvas id="progress"></canvas></div></div></main></div>
<script>
const data=__VIEWER_DATA__;let mode="raw",yaw=-0.55,pitch=0.48,zoom=1,drag=false,last=[0,0];const css=getComputedStyle(document.documentElement);const colors={gt:css.getPropertyValue('--gt').trim(),vins:css.getPropertyValue('--vins').trim(),se3:css.getPropertyValue('--se3').trim(),grid:css.getPropertyValue('--grid').trim(),text:'#9aa7b2'};
function activeVins(){return mode==='se3'?data.vins_se3:data.vins_raw}function resize(c){const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=Math.max(1,Math.round(r.width*d));c.height=Math.max(1,Math.round(r.height*d));const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return{x,w:r.width,h:r.height}}function grid(ctx,w,h){ctx.clearRect(0,0,w,h);ctx.strokeStyle=colors.grid;ctx.lineWidth=1;for(let i=1;i<5;i++){const x=w*i/5,y=h*i/5;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}}function line(ctx,arr,pick,color,width=2){if(!arr.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();arr.forEach((p,i)=>{const q=pick(p);if(i===0)ctx.moveTo(q[0],q[1]);else ctx.lineTo(q[0],q[1])});ctx.stroke()}function allPoints(){return data.gt.concat(activeVins()).map(p=>p.p)}function bounds3d(){const pts=allPoints();let mn=[Infinity,Infinity,Infinity],mx=[-Infinity,-Infinity,-Infinity];for(const p of pts){for(let i=0;i<3;i++){mn[i]=Math.min(mn[i],p[i]);mx[i]=Math.max(mx[i],p[i])}}const c=mn.map((v,i)=>(v+mx[i])/2),s=Math.max(mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2],0.05);return{c,s}}function project(p,view,w,h){const x=p[0]-view.c[0],y=p[1]-view.c[1],z=p[2]-view.c[2],cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1;return[w/2+x1*view.scale,h/2-y1*view.scale]}function drawAxes(ctx,w,h,view){const o=view.c,len=view.s*.18,axes=[["X",'#ff6b6b',[o[0]+len,o[1],o[2]]],["Y",'#7bd88f',[o[0],o[1]+len,o[2]]],["Z",'#6aa9ff',[o[0],o[1],o[2]+len]]],op=project(o,view,w,h);for(const [label,color,end] of axes){const e=project(end,view,w,h);ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(op[0],op[1]);ctx.lineTo(e[0],e[1]);ctx.stroke();ctx.fillStyle=color;ctx.fillText(label,e[0]+4,e[1]+4)}}function draw3d(){const {x:ctx,w,h}=resize(document.getElementById('view3d'));grid(ctx,w,h);const b=bounds3d(),view={c:b.c,s:b.s,scale:Math.min(w,h)*.72/b.s*zoom};drawAxes(ctx,w,h,view);line(ctx,data.gt,p=>project(p.p,view,w,h),colors.gt,2.5);line(ctx,activeVins(),p=>project(p.p,view,w,h),mode==='se3'?colors.se3:colors.vins,2.1);drawEndpoint(ctx,data.gt[0].p,view,w,h,colors.gt,5);drawEndpoint(ctx,data.gt[data.gt.length-1].p,view,w,h,colors.gt,7);drawEndpoint(ctx,activeVins()[0].p,view,w,h,mode==='se3'?colors.se3:colors.vins,5);drawEndpoint(ctx,activeVins()[activeVins().length-1].p,view,w,h,mode==='se3'?colors.se3:colors.vins,7)}function drawEndpoint(ctx,p,view,w,h,color,r){const q=project(p,view,w,h);ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],r,0,Math.PI*2);ctx.fill()}function planeBounds(a,b){let mnA=Infinity,mxA=-Infinity,mnB=Infinity,mxB=-Infinity;for(const p of data.gt.concat(activeVins()).map(x=>x.p)){mnA=Math.min(mnA,p[a]);mxA=Math.max(mxA,p[a]);mnB=Math.min(mnB,p[b]);mxB=Math.max(mxB,p[b])}const pa=Math.max((mxA-mnA)*.08,.02),pb=Math.max((mxB-mnB)*.08,.02);return{mnA:mnA-pa,mxA:mxA+pa,mnB:mnB-pb,mxB:mxB+pb}}function drawPlane(id,a,b){const {x:ctx,w,h}=resize(document.getElementById(id));grid(ctx,w,h);const bb=planeBounds(a,b),sx=v=>26+(v-bb.mnA)/(bb.mxA-bb.mnA||1)*(w-52),sy=v=>h-26-(v-bb.mnB)/(bb.mxB-bb.mnB||1)*(h-52),pr=p=>[sx(p.p[a]),sy(p.p[b])];line(ctx,data.gt,pr,colors.gt,2.4);line(ctx,activeVins(),pr,mode==='se3'?colors.se3:colors.vins,2)}function drawProgress(){const {x:ctx,w,h}=resize(document.getElementById('progress'));grid(ctx,w,h);const series=[['GT',data.gt,colors.gt],['VINS',activeVins(),mode==='se3'?colors.se3:colors.vins]],maxLen=Math.max(...series.map(s=>s[1][s[1].length-1].s),.001);for(const [,arr,color] of series){line(ctx,arr,p=>[30+p.u*(w-58),h-28-(p.s/maxLen)*(h-56)],color,2)}ctx.fillStyle=colors.text;ctx.fillText(maxLen.toFixed(3)+' m',8,18)}function render(){draw3d();drawPlane('xy',0,1);drawPlane('xz',0,2);drawPlane('yz',1,2);drawProgress()}function setMode(m){mode=m;rawBtn.classList.toggle('active',m==='raw');se3Btn.classList.toggle('active',m==='se3');render()}title.textContent=data.title;subtitle.textContent=data.subtitle;inputs.textContent=JSON.stringify(data.inputs,null,2);stats.innerHTML=Object.entries(data.stats).map(([k,v])=>`<tr><td>${k}</td><td>${typeof v==='number'?v.toFixed(4):v}</td></tr>`).join('');rawBtn.onclick=()=>setMode('raw');se3Btn.onclick=()=>setMode('se3');const v=document.getElementById('view3d');v.addEventListener('mousedown',e=>{drag=true;last=[e.clientX,e.clientY]});window.addEventListener('mouseup',()=>drag=false);window.addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];last=[e.clientX,e.clientY];yaw+=dx*.006;pitch=Math.max(-1.45,Math.min(1.45,pitch+dy*.006));render()});v.addEventListener('wheel',e=>{e.preventDefault();zoom*=e.deltaY>0?.9:1.1;zoom=Math.max(.15,Math.min(20,zoom));render()},{passive:false});window.addEventListener('resize',render);render();
</script></body></html>"""


def normalize_timestamp(value: float) -> float:
    value = float(value)
    av = abs(value)
    if av > 1e17:
        return value * 1e-9
    if av > 1e13:
        return value * 1e-6
    if av > 1e10:
        return value * 1e-3
    return value


def normalize_quat(q: Sequence[float]) -> np.ndarray:
    arr = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("invalid quaternion")
    return arr / norm


def quat_xyzw_to_rot(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quat(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def transform_from_pose(position: Sequence[float], quaternion_xyzw: Sequence[float]) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, :3] = quat_xyzw_to_rot(quaternion_xyzw)
    out[:3, 3] = np.asarray(position, dtype=float)
    return out


def path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=float)
    if points.shape[0] == 1:
        return np.zeros(1, dtype=float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def load_gt(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples", data)
    rows = []
    for sample in samples:
        rows.append(
            (
                normalize_timestamp(float(sample["timestamp"])),
                np.asarray(sample["position_m"], dtype=float),
                quat_xyzw_to_rot(sample["quaternion_xyzw"]),
            )
        )
    rows.sort(key=lambda item: item[0])
    return np.asarray([r[0] for r in rows]), np.asarray([r[1] for r in rows]), np.asarray([r[2] for r in rows])


def first_existing(row: dict, names: Sequence[str]) -> str | None:
    lower = {str(k).strip().lower(): k for k in row.keys()}
    for name in names:
        key = lower.get(name.lower())
        if key is not None and row.get(key) not in (None, ""):
            return str(row[key])
    return None


def load_vins_tcp(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_left_camera_tcp = np.linalg.inv(T_TCP_LEFT_CAMERA)
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            ts = first_existing(row, ["Timestamp_us", "timestamp_us", "timestamp", "time", "t"])
            xyz = [first_existing(row, names) for names in (["X", "x"], ["Y", "y"], ["Z", "z"])]
            quat = [
                first_existing(row, ["Quat_X", "qx", "q_x"]),
                first_existing(row, ["Quat_Y", "qy", "q_y"]),
                first_existing(row, ["Quat_Z", "qz", "q_z"]),
                first_existing(row, ["Quat_W", "qw", "q_w"]),
            ]
            if any(v is None for v in xyz + quat):
                continue
            timestamp = normalize_timestamp(float(ts)) if ts is not None else float(idx)
            t_world_left_camera = transform_from_pose([float(v) for v in xyz], [float(v) for v in quat])
            t_world_tcp = t_world_left_camera @ t_left_camera_tcp
            rows.append((timestamp, t_world_tcp[:3, 3].copy(), t_world_tcp[:3, :3].copy()))
    rows.sort(key=lambda item: item[0])
    return np.asarray([r[0] for r in rows]), np.asarray([r[1] for r in rows]), np.asarray([r[2] for r in rows])


def resample_by_arclength(points: np.ndarray, count: int) -> np.ndarray:
    lengths = cumulative_lengths(points)
    total = float(lengths[-1]) if lengths.size else 0.0
    if total <= 1e-12:
        return np.repeat(points[:1], count, axis=0)
    src = lengths / total
    dst = np.linspace(0.0, 1.0, count)
    return np.column_stack([np.interp(dst, src, points[:, axis]) for axis in range(3)])


def align_umeyama(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("alignment requires paired arrays with at least 3 points")
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    xs = source - mu_s
    xt = target - mu_t
    cov = (xt.T @ xs) / source.shape[0]
    u, _, vt = np.linalg.svd(cov)
    d = np.ones(3, dtype=float)
    if np.linalg.det(u @ vt) < 0.0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt
    trans = mu_t - rot @ mu_s
    return rot, trans


def trajectory_payload(times: np.ndarray, points: np.ndarray, t0: float) -> List[dict]:
    lengths = cumulative_lengths(points)
    total = float(lengths[-1]) if lengths.size else 0.0
    denom = total if total > 1e-12 else 1.0
    return [
        {"t": float(t - t0), "p": points[i].tolist(), "s": float(lengths[i]), "u": float(lengths[i] / denom)}
        for i, t in enumerate(times)
    ]


def stats_for(times: np.ndarray, points: np.ndarray) -> dict:
    return {
        "samples": int(points.shape[0]),
        "duration_s": float(times[-1] - times[0]) if times.size > 1 else 0.0,
        "path_length_m": path_length(points),
        "endpoint_distance_m": float(np.linalg.norm(points[-1] - points[0])) if points.shape[0] > 1 else 0.0,
        "x_span_m": float(points[:, 0].ptp()),
        "y_span_m": float(points[:, 1].ptp()),
        "z_span_m": float(points[:, 2].ptp()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--vins", type=Path, default=DEFAULT_VINS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--align-samples", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gt_path = args.gt.expanduser().resolve()
    vins_path = args.vins.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_t, gt_pos, _ = load_gt(gt_path)
    vins_t, vins_pos, vins_rot = load_vins_tcp(vins_path)
    count = max(3, min(args.align_samples, gt_pos.shape[0], vins_pos.shape[0]))
    gt_fit = resample_by_arclength(gt_pos, count)
    vins_fit = resample_by_arclength(vins_pos, count)
    rot, trans = align_umeyama(vins_fit, gt_fit)
    vins_se3_pos = (rot @ vins_pos.T).T + trans
    vins_se3_rot = np.asarray([rot @ r for r in vins_rot], dtype=float)
    _ = vins_se3_rot  # Rotations are computed to document the frame transform; viewer draws positions only.

    raw_fit_rmse = float(math.sqrt(np.mean(np.sum((vins_fit - gt_fit) ** 2, axis=1))))
    se3_fit = (rot @ vins_fit.T).T + trans
    se3_fit_rmse = float(math.sqrt(np.mean(np.sum((se3_fit - gt_fit) ** 2, axis=1))))
    t0 = min(float(gt_t[0]), float(vins_t[0]))
    payload = {
        "title": "episode_20260617_0004 full trajectory overlay",
        "subtitle": "Full GT and VINS-derived TCP trajectories. No timestamp pairing; SE(3) uses arc-length-normalized shape alignment.",
        "inputs": {
            "ground_truth": str(gt_path),
            "vins_cam0_csv": str(vins_path),
            "vins_conversion": "T_world_tcp = T_world_cam0 @ inverse(T_tcp_left_camera)",
            "se3_alignment": f"Umeyama rigid fit on {count} arc-length-normalized samples; no timestamp pairing",
        },
        "stats": {
            "gt_samples": int(gt_pos.shape[0]),
            "gt_duration_s": stats_for(gt_t, gt_pos)["duration_s"],
            "gt_path_length_m": stats_for(gt_t, gt_pos)["path_length_m"],
            "gt_endpoint_distance_m": stats_for(gt_t, gt_pos)["endpoint_distance_m"],
            "vins_samples": int(vins_pos.shape[0]),
            "vins_duration_s": stats_for(vins_t, vins_pos)["duration_s"],
            "vins_path_length_m": stats_for(vins_t, vins_pos)["path_length_m"],
            "vins_endpoint_distance_m": stats_for(vins_t, vins_pos)["endpoint_distance_m"],
            "raw_shape_fit_rmse_m": raw_fit_rmse,
            "se3_shape_fit_rmse_m": se3_fit_rmse,
        },
        "se3_transform": {"rotation": rot.tolist(), "translation": trans.tolist()},
        "gt": trajectory_payload(gt_t, gt_pos, t0),
        "vins_raw": trajectory_payload(vins_t, vins_pos, t0),
        "vins_se3": trajectory_payload(vins_t, vins_se3_pos, t0),
    }
    data_path = output_dir / "viewer_data.json"
    html_path = output_dir / "index.html"
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(HTML.replace("__VIEWER_DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    print(f"[OK] wrote {html_path}")
    print(f"[OK] wrote {data_path}")
    print(f"[INFO] raw shape-fit RMSE: {raw_fit_rmse:.4f} m")
    print(f"[INFO] SE3 shape-fit RMSE: {se3_fit_rmse:.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
