#!/usr/bin/env python3
"""Close a verified first/last RGB stereo loop on a Basalt TUM trajectory.

The loop observation is estimated solely from RGB stereo geometry.  The device
head pose is deliberately read only after correction for APE/RPE evaluation.
"""
from __future__ import annotations

import argparse, json, subprocess
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation, Slerp

from check_sxr_csv_calibration import inverse, pose


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode-dir', type=Path, required=True)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    return p.parse_args()


def tum(path):
    rows = []
    for line in path.read_text().splitlines():
        v = line.split()
        if len(v) == 8: rows.append([float(x) for x in v])
    a = np.asarray(rows); out = []
    for r in a:
        T = np.eye(4); T[:3,:3] = Rotation.from_quat(r[4:8]).as_matrix(); T[:3,3] = r[1:4]
        out.append(T)
    return a[:,0], out


def loop_measurement(video, c0, c1):
    cap = cv2.VideoCapture(str(video)); images = {}
    index = 0
    while True:
        ok, image = cap.read()
        if not ok: break
        if index == 0: images[0] = image
        images[-1] = image
        index += 1
    cap.release()
    if len(images) != 2: raise RuntimeError('unable to decode RGB endpoint frames')
    l0 = cv2.cvtColor(images[0][:,:images[0].shape[1]//2], cv2.COLOR_BGR2GRAY)
    r0 = cv2.cvtColor(images[0][:,images[0].shape[1]//2:], cv2.COLOR_BGR2GRAY)
    ln = cv2.cvtColor(images[-1][:,:images[-1].shape[1]//2], cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=6000, fastThreshold=7); k0,d0=orb.detectAndCompute(l0,None); kr,dr=orb.detectAndCompute(r0,None); kn,dn=orb.detectAndCompute(ln,None)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    def matches(a,b): return [m for m,n in bf.knnMatch(a,b,k=2) if m.distance < .72*n.distance]
    stereo, temporal = matches(d0,dr), matches(d0,dn)
    right={m.queryIdx:m.trainIdx for m in stereo}; late={m.queryIdx:m.trainIdx for m in temporal}; ids=sorted(set(right)&set(late))
    if len(ids) < 20: raise RuntimeError(f'insufficient joined loop matches: {len(ids)}')
    K0=np.array([[c0['intrinsics']['focalX'],0,c0['intrinsics']['centerX']],[0,c0['intrinsics']['focalY'],c0['intrinsics']['centerY']],[0,0,1.]])
    K1=np.array([[c1['intrinsics']['focalX'],0,c1['intrinsics']['centerX']],[0,c1['intrinsics']['focalY'],c1['intrinsics']['centerY']],[0,0,1.]])
    D0=np.asarray(c0['intrinsics']['radialDistortion'][:4]); D1=np.asarray(c1['intrinsics']['radialDistortion'][:4])
    p0=np.float64([k0[i].pt for i in ids]).reshape(-1,1,2); pr=np.float64([kr[right[i]].pt for i in ids]).reshape(-1,1,2); pn=np.float64([kn[late[i]].pt for i in ids]).reshape(-1,1,2)
    u0=cv2.fisheye.undistortPoints(p0,K0,D0).reshape(-1,2); ur=cv2.fisheye.undistortPoints(pr,K1,D1).reshape(-1,2); un=cv2.fisheye.undistortPoints(pn,K0,D0).reshape(-1,2)
    T10=pose(c0,'wxyz') @ inverse(pose(c1,'wxyz'))
    X=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],T10[:3],u0.T,ur.T); X=(X[:3]/X[3]).T; Xr=(T10[:3,:3]@X.T+T10[:3,3:]).T
    keep=(X[:,2]>.08)&(X[:,2]<30)&(Xr[:,2]>.08)&np.isfinite(X).all(1)
    ok, rv, tv, inn = cv2.solvePnPRansac(X[keep],un[keep],np.eye(3),None,iterationsCount=2000,reprojectionError=.003,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or len(inn)<20: raise RuntimeError(f'loop PnP rejected: {0 if inn is None else len(inn)} inliers')
    R,_=cv2.Rodrigues(rv); T=np.eye(4); T[:3,:3]=R; T[:3,3]=tv.ravel()
    return T, {'decoded_pairs':index,'stereo_matches':len(stereo),'temporal_matches':len(temporal),'joined_matches':len(ids),'triangulated_points':int(keep.sum()),'pnp_inliers':int(len(inn))}


def main():
    a=args(); ep=a.episode_dir.resolve(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    cams=json.loads((ep/'camera_params_rgb.json').read_text())['cameras']; loop, info=loop_measurement(ep/'rgb.mp4',*cams)
    ts, poses=tum(a.trajectory); first,last=poses[0],poses[-1]
    # Desired world pose at the end comes from T_end_from_start measured by PnP.
    target_end = first @ inverse(loop); end_correction = target_end @ inverse(last)
    rot=Slerp([0,1],Rotation.from_matrix([np.eye(3),end_correction[:3,:3]])); corrected=[]
    for i,T in enumerate(poses):
        f=i/(len(poses)-1); G=np.eye(4); G[:3,:3]=rot([f]).as_matrix()[0]; G[:3,3]=f*end_correction[:3,3]; corrected.append(G@T)
    trajectory=out/'basalt_rgb_visual_loop.tum'
    with trajectory.open('w') as h:
        for t,T in zip(ts,corrected):
            q=Rotation.from_matrix(T[:3,:3]).as_quat(); h.write(f'{t:.9f} {T[0,3]:.9f} {T[1,3]:.9f} {T[2,3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n')
    ref=out/'head_pose_reference.tum'; ref.write_bytes((ep/'head_pose.csv').read_bytes()) if False else None
    from run_sxr_csv_vinsfusion import write_reference
    write_reference(ep/'head_pose.csv',ref)
    ev=out/'evaluation'; ev.mkdir(exist_ok=True)
    for name,cmd in {'ape_translation.log':['evo_ape','tum',str(ref),str(trajectory),'--align','--pose_relation','trans_part'], 'rpe_translation.log':['evo_rpe','tum',str(ref),str(trajectory),'--align','--pose_relation','trans_part','--delta','1','--delta_unit','f']}.items():
        r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False); (ev/name).write_text(r.stdout)
        if r.returncode: raise RuntimeError(r.stdout)
    info.update({'loop_translation_m':float(np.linalg.norm(loop[:3,3])),'loop_rotation_deg':float(Rotation.from_matrix(loop[:3,:3]).magnitude()*180/np.pi),'correction_end_translation_m':float(np.linalg.norm(end_correction[:3,3])),'source_trajectory':str(a.trajectory),'head_pose_note':'Used only after visual loop closure for evaluation.'})
    (out/'loop_provenance.json').write_text(json.dumps(info,indent=2)+'\n'); print(json.dumps(info,indent=2)); print(trajectory)

if __name__=='__main__': main()
