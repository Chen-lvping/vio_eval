# Data Layout

`data/` is organized by role, not by script:

- `calibration/`
  Inputs and outputs for hand-eye calibration.
- `ground_truth/`
  Robot trajectory truth data and future recordings.
- `evaluation/core/`
  Main current evaluation results worth keeping visible.
- `evaluation/camera_benchmark/`
  Camera-stream comparison results such as right-cam0 VINS vs DynaVINS.
- `evaluation/workbench/`
  Default place for fresh reruns, temporary scans, and new diagnostics.
- `archive/`
  Historical data, alternative convention scans, old recordings, and outputs that are no longer the current reference.

Recommended rule:

- If a result is the current reference, keep it in `evaluation/core/`.
- If it is useful but secondary, keep it in `evaluation/camera_benchmark/`.
- If it is a fresh rerun or temporary experiment, write it into `evaluation/workbench/`.
- If it stops being active, move it to `archive/`.
