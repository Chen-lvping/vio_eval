# `vio_eval` Delivery SOP

## 1. Delivery Scope

This repository delivers one auditable capability: compare an estimated camera/IMU trajectory with an RM75 robot TCP trajectory after applying the documented calibration chain and rigid SE(3) alignment.

The supported delivery boundary is deliberately small:

| Layer | Delivery status | Entry point |
| --- | --- | --- |
| Minimal evaluator proof | required | `python3 script/vio_eval.py demo` |
| One recorded RM75 episode | required when customer data is supplied | `python3 script/vio_eval.py evaluate ...` |
| ORB-SLAM3 trajectory generation | optional integration | `script/setup_orbslam3_adaptive_vio.sh`, `script/run_orbslam3_tcp_eval.py` |
| Historical sweeps, SXR research, capture tools | reference only | `script/experiments/`, `script/capture/`, `script/diagnose/` |

Do not present archived experiment results as a fresh benchmark. The published 8-episode portfolio in `docs/results/rm75_8_episode_portfolio.csv` is evidence of the recorded configuration, not a held-out result.

## 2. Clean-Machine Acceptance

Run these commands from the repository root before handoff:

```bash
python3 -m pip install -r requirements-minimal.txt
python3 script/vio_eval.py demo --output-dir local/minimal_reproduction
python3 -m unittest discover -s tests -v
```

Acceptance criteria:

- command exits with code `0`;
- `local/minimal_reproduction/REPORT.md`, `metrics.json`, and `summary.csv` exist;
- `metrics.json` reports `matched_samples: 5`;
- `metrics.ape_translation_se3.rmse < 1e-9` mm and `metrics.ape_rotation_se3.rmse < 1e-9` deg;
- the regression test passes.

`demo` intentionally selects the built-in metric backend. It therefore works without `evo`, ORB-SLAM3, robot hardware, or proprietary recordings.

## 3. Customer Episode Evaluation

Prepare only these immutable inputs:

1. Estimated pose CSV containing timestamp, position (`x,y,z`), and quaternion (`qx,qy,qz,qw`).
2. Robot TCP JSON containing timestamp, position, and quaternion samples.
3. Hand-eye YAML containing `result.T_cam_to_gripper`.
4. For IMU or VINS `base_link` estimates, the episode `calibration.json` and its rig key.

Run a camera-pose trajectory as follows:

```bash
python3 script/vio_eval.py evaluate \
  --estimate /data/episode/pose_camera.csv \
  --ground-truth /data/ground_truth/episode.json \
  --handeye-yaml data/calibration/handeye_0615/handeye_result.yaml \
  --estimate-frame camera \
  --output-dir local/customer_episode
```

For an IMU pose CSV, add the calibration source and camera rig:

```bash
python3 script/vio_eval.py evaluate \
  --estimate /data/episode/pose_imu.csv \
  --ground-truth /data/ground_truth/episode.json \
  --calibration-json /data/episode/calibration.json \
  --camera-rig stereo_right \
  --estimate-frame imu \
  --output-dir local/customer_episode
```

The evaluator computes:

```text
T_world_tcp = T_world_camera @ inverse(T_tcp_camera)
```

or, for IMU input:

```text
T_world_tcp = T_world_imu @ inverse(T_camera_imu) @ inverse(T_tcp_camera)
```

It then estimates a rigid `SE(3)` transform between the estimated TCP trajectory and `T_base_tcp`; scale is never fitted for the primary metric.

## 4. Output Review and Sign-Off

Keep one output directory per run. Review these files together:

- `REPORT.md`: human-readable input paths, coordinate chain, time-association policy, and metrics.
- `metrics.json`: machine-readable provenance, transforms, matched count, metric backend, and metrics.
- `summary.csv`: compact metric table.
- `gt_tcp*.tum` and `vio_tcp*.tum`: trajectories used for alignment and review.

Sign off only when all of the following are true:

- `matched_samples` and `matched_duration_s` cover the intended motion;
- coordinate-frame selection matches the exported pose semantics;
- calibration path/version and time-association arguments are recorded in `metrics.json`;
- primary result is `ape_translation_se3` (millimetres), not Sim(3);
- any `evo` output is labelled with its chosen time-offset settings.

## 5. Optional ORB-SLAM3 Integration

ORB-SLAM3 is intentionally excluded from minimal delivery because its build dependencies and raw recordings are not part of the lightweight evaluator. The RM75 adaptive implementation is nevertheless reproducible from its public pinned upstream commit and versioned patch:

```bash
script/setup_orbslam3_adaptive_vio.sh --orb-root /opt/ORB_SLAM3_adaptive_vio --build
```

Then validate the integration entry point:

```bash
python3 script/run_orbslam3_tcp_eval.py --orb-root /opt/ORB_SLAM3_adaptive_vio --help
```

See `docs/ALGORITHM_REPRODUCTION.md` for adaptive initialization parameters and ablations. Feed the generated pose CSV into `script/vio_eval.py evaluate` for the stable delivery evaluation contract. Do not depend on hard-coded paths in research runners for customer acceptance.

## 6. Handoff Checklist

- [ ] Source tree and `requirements-minimal.txt` included.
- [ ] `examples/minimal_reproduction/` included unchanged.
- [ ] Clean-machine acceptance evidence attached.
- [ ] Customer input file hashes and calibration revision recorded.
- [ ] One evaluation output directory attached per accepted episode.
- [ ] Hardware, raw data, and optional ORB-SLAM3 dependencies listed separately from the core evaluator.
