# vio_eval

Independent workspace for:

- robot static-pose capture
- AprilGrid hand-eye calibration
- robot TCP trajectory recording
- VINS / DynaVINS trajectory accuracy evaluation
- orientation diagnostics and comparison viewers

## Layout

- `script/`
  Main entry scripts.
- `data/calibration/`
  Static poses, calibration images, camera intrinsics, hand-eye results.
- `data/ground_truth/`
  Robot trajectory ground truth and future recordings.
- `data/evaluation/core/`
  Current main TCP evaluation results and visualizations.
- `data/evaluation/camera_benchmark/`
  Right-cam0 comparison results and viewers.
- `data/evaluation/workbench/`
  Default output area for fresh reruns and temporary analysis.
- `data/archive/`
  Historical experiments, old captures, and no-longer-core outputs.

## Core Data

- `data/calibration/camera_cam0_640x400_intrinsics.yaml`
- `data/calibration/static_pose_samples_0615/`
- `data/calibration/image_0615/`
- `data/calibration/handeye_0615/`
- `data/ground_truth/trajectory_samples0617/`
- `data/evaluation/core/evo_vio_tcp_0616/`
- `data/evaluation/core/evo_dynavins_tcp_0616/`
- `data/evaluation/core/tcp_orientation_diagnostics_0616/`
- `data/evaluation/core/tcp_orientation_diagnostics_corrected_0616/`
- `data/evaluation/core/evo_tcp_visualization_0616/`

## Main Entry Points

- `script/capture_static_pose.py`
  Capture static robot poses for hand-eye calibration.
- `script/handeye_calibrate_aprilgrid.py`
  Solve hand-eye using AprilGrid images and robot poses.
- `script/record_trajectory.py`
  Record robot TCP ground-truth trajectories.
- `script/evaluate_vins_accuracy.py`
  Generic trajectory evaluation and local viewer.
- `script/evaluate_vio_tcp_camera_evo.py`
  TCP-chain evaluation using `evo`.
- `script/diagnose_tcp_orientation.py`
  Diagnose large attitude error after alignment.
- `script/visualize_tcp_trajectory_comparison.py`
  Static 3D TCP comparison viewer.
- `script/visualize_vins_comparison.py`
  Static VINS / DynaVINS comparison viewer.
- `script/scan_vio_tcp_conventions.py`
  Scan convention choices for VIO pose direction and transform meaning.

## Typical Workflow

1. Hand-eye data capture
   Run `script/capture_static_pose.py`
   Default output: `data/calibration/static_pose_samples/`

2. Hand-eye calibration
   Run `script/handeye_calibrate_aprilgrid.py`
   Defaults read from:
   `data/calibration/static_pose_samples_0615/`
   `data/calibration/image_0615/`
   `data/calibration/camera_cam0_640x400_intrinsics.yaml`

3. Robot trajectory recording
   Run `script/record_trajectory.py`
   Default output: `data/ground_truth/trajectory_samples/`
   Required trajectory template for all new recordings:
   - `meta.orientation_mode`
   - `meta.raw_pose_saved = true`
   - per-sample `orientation_raw`
   - per-sample `orientation_mode`
   - per-sample `raw_pose`
   New trajectory recordings should follow the 0617 raw-pose format and must keep
   these fields so robot orientation can be rebuilt and audited later.

4. TCP accuracy evaluation
   Run `script/evaluate_vio_tcp_camera_evo.py`
   Fresh outputs default to: `data/evaluation/workbench/evo_vio_tcp/`

5. Orientation diagnosis
   Run `script/diagnose_tcp_orientation.py`
   Fresh outputs default to: `data/evaluation/workbench/tcp_orientation_diagnostics/`

## Development Notes

- The main TCP workflow now uses the 0617 raw-pose episodes as the primary
  reference path, especially `episode_20260617_0005` and `episode_20260617_0006`.
- `episode_20260614_0239` remains useful as a diagnostic sample, but it is no
  longer the preferred evaluation baseline.
- Core results stay under `data/evaluation/core/`.
- New reruns should prefer `data/evaluation/workbench/` so core results remain clean.
- Historical and exploratory outputs should be moved into `data/archive/` after they stop being active.
- Some raw VIO estimate paths and generated VINS configs still depend on external datasets, so those scripts may still need explicit `--estimate`, `--dataset`, or `--config` arguments when running on a new episode.
- `data/ground_truth/trajectory_samples0617/trajectory_sync_rawpose_005.json`
  and `trajectory_sync_rawpose_006.json` are the reference examples for the
  current ground-truth recording format.
