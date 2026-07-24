# Hanpu / YCTC SC233HGS SLAM Development Notes

This document records the current Hanpu/YCTC bring-up state and the repeatable
camera-quality and ORB-SLAM3 workflow. It is intentionally separate from the
RM75 evaluation mainline.

## Current Status

The device is usable for stereo capture and offline ORB-SLAM3 testing.

Verified on 2026-07-16:

- UVC node: `/dev/v4l/by-id/usb-YCTC_YCTC_SC233HGS_0152312181647-video-index0`
  resolving to `/dev/video0`.
- CDC ACM IMU node: `/dev/ttyACM0`.
- Current firmware: `hi3516cv610_YCTC_SC233HGS_NOR_202606271716_SEC_REL`.
- Video modes include `3840x1080` MJPEG/H.264. The correct split is two
  `1920x1080` images. Do not apply the old top-half `1920x540` crop when using
  the real calibration.
- Current output sync mode is `INTERNAL`.
- Current YCTC video metadata is a 48-byte V1 block. Older code accepted only
  the legacy 32-byte block; `run_hanpu_orbslam3_smoketest.py` now accepts both.
- The active calibration blob is available through UVC XU and has been saved
  under `data/hanpu/calibration/`.

The camera transport and synchronization pass basic checks. ORB can initialize
with the real calibration, but tracking is not yet stable enough for a final
SLAM or VINS accuracy claim.

## Important Files

- `data/hanpu/calibration/stereo_calibration.bin`
  Raw 964-byte calibration blob read from the device.
- `data/hanpu/calibration/stereo_calibration.json`
  Parsed manufacturer calibration.
- `data/hanpu/calibration/hanpu_stereo_real_1920x1080.yaml`
  ORB-SLAM3 stereo YAML generated from the manufacturer calibration.
- `script/capture/run_hanpu_manual_record_orb_plot.py`
  Manual recording. Press `q` to stop, then optionally export/run ORB/plot.
- `script/capture/report_hanpu_slam_run.py`
  Generates camera and SLAM metrics as JSON and Markdown.
- `script/capture/hanpu_calibration_to_orbslam3_yaml.py`
  Converts the parsed calibration JSON to ORB-SLAM3 YAML.
- `script/capture/set_yctc_exposure_gain.py`
  Applies a test exposure and ISP system gain while video is stopped.
- `script/capture/run_hanpu_live_orbslam3_stereo.sh`
  Real-time stereo preview and Pangolin viewer.
- `script/capture/run_hanpu_orbslam3_smoketest.py`
  Existing fixed-duration capture/export/ORB helper.

## Device Checks

Run these before every experiment:

```bash
cd /home/chenlvping/1_DM_work/vio_eval
ls -l /dev/v4l/by-id /dev/video* /dev/ttyACM*
fuser -v /dev/video0 /dev/ttyACM0
```

Only one process should own the video and serial nodes. Do not run the live
viewer and a recorder at the same time.

## Read Calibration

The protocol package is located at:

```text
YCTC_SC233HGS_protocol.tgz(1)/YCTC_SC233HGS_protocol/
```

Use `/dev/video0` for UVC XU access. `/dev/video1` is a data node and does not
report the YCTC XU controls on this device.

```bash
cd /home/chenlvping/1_DM_work/vio_eval/YCTC_SC233HGS_protocol.tgz\(1\)/YCTC_SC233HGS_protocol

python3 src/uvc_xu_protocol/yctc_uvc_xu_procotol.py \
  --device /dev/video0 \
  --out /home/chenlvping/1_DM_work/vio_eval/data/hanpu/calibration/stereo_calibration.bin \
  --json-out /home/chenlvping/1_DM_work/vio_eval/data/hanpu/calibration/stereo_calibration.json \
  --strict-parse \
  --validate-protocol
```

The current active calibration reports:

- Serial number: `YC202607090143`
- Stereo RMS: `0.373479 px`
- Baseline: `60.295440 mm`
- Calibration sample count: `23`

The blob contains stereo `K1/D1/K2/D2/R/T`. It does not contain the camera-IMU
extrinsic transform needed for a final stereo-inertial/VINS accuracy result.

## Manual Record Only

This is the preferred camera-quality test. It records until `q` is pressed and
does not run ORB yet.

```bash
cd /home/chenlvping/1_DM_work/vio_eval
python3 script/capture/run_hanpu_manual_record_orb_plot.py \
  --no-orb \
  --no-plot
```

The preview window is a reduced-size display. Press `q` or `Esc` in that window
to stop. `Ctrl-C` in the terminal is also supported.

The script writes a run directory like:

```text
data/hanpu/runs/hanpu_manual_YYYYMMDD_HHMMSS/
```

Important outputs:

```text
raw/stereo_raw.mjpg
raw/imu_raw.csv
euroc_export/mav0/cam0/
euroc_export/mav0/cam1/
euroc_export/mav0/imu0/data.csv
hanpu_stereo_real_1920x1080.yaml
```

## Generate the Camera/SLAM Report

```bash
RUN_ROOT="$(ls -dt /home/chenlvping/1_DM_work/vio_eval/data/hanpu/runs/hanpu_manual_* | head -n1)"
cd /home/chenlvping/1_DM_work/vio_eval
python3 script/capture/report_hanpu_slam_run.py "$RUN_ROOT"
cat "$RUN_ROOT/camera_slam_report.md"
```

The report measures:

- Video frame count, PTS duration, rate, and frame interval distribution.
- Left/right PTS equality.
- Left/right exposure metadata.
- IMU row count, timestamp rate, and interval distribution.
- Manufacturer calibration RMS and baseline.
- Image brightness, contrast, Laplacian sharpness, and ORB keypoint count.
- Rectified stereo match count and temporal match count.
- ORB log counts for local-map failure, low matches, lost frames, and map creation.
- Output trajectory and keyframe row counts.

The report files are:

```text
camera_slam_report.json
camera_slam_report.md
```

## ORB-SLAM3 Offline Run

Use the selected `RUN_ROOT` from the previous step.

```bash
ORB_ROOT=/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_clean
mkdir -p "$RUN_ROOT/orbslam3_real_run"
cd "$RUN_ROOT/orbslam3_real_run"

export LD_LIBRARY_PATH="$ORB_ROOT/lib:$ORB_ROOT/Thirdparty/DBoW2/lib:$ORB_ROOT/Thirdparty/g2o/lib:/home/chenlvping/1_DM_work/orbslam3_fresh/Pangolin_v06/install/lib:${LD_LIBRARY_PATH:-}"
set -o pipefail

ORB_SLAM3_ENABLE_VIEWER=0 \
  "$ORB_ROOT/Examples/Stereo/stereo_euroc" \
  "$ORB_ROOT/Vocabulary/ORBvoc.txt" \
  "$RUN_ROOT/hanpu_stereo_real_1920x1080.yaml" \
  "$RUN_ROOT/euroc_export" \
  "$RUN_ROOT/euroc_export/times.txt" \
  hanpu_real_stereo 2>&1 | tee orbslam3_real.log
```

Regenerate the report after ORB finishes:

```bash
cd /home/chenlvping/1_DM_work/vio_eval
python3 script/capture/report_hanpu_slam_run.py "$RUN_ROOT"
cat "$RUN_ROOT/camera_slam_report.md"
```

## Trajectory Visualization

Interactive plot:

```bash
evo_traj euroc \
  "$RUN_ROOT/orbslam3_real_run/f_hanpu_real_stereo.txt" \
  --plot --plot_mode xyz
```

Static PNG:

```bash
python3 script/visualize_orbslam3_trajectory.py \
  "$RUN_ROOT/orbslam3_real_run/f_hanpu_real_stereo.txt" \
  --output "$RUN_ROOT/trajectory.png"
xdg-open "$RUN_ROOT/trajectory.png"
```

## Exposure/Gain A/B Test

The current device reports a fixed `2000 us` exposure in the sample run. To
test whether imaging conditions are limiting tracking, stop all video/IMU
processes first and apply a conservative test setting:

```bash
cd /home/chenlvping/1_DM_work/vio_eval
python3 script/capture/set_yctc_exposure_gain.py \
  --exposure-us 10000 \
  --gain 0x1000
```

This requests `10 ms` exposure and `4x` system gain. The protocol only accepts
`PWM_EXPOSURE` while video is stopped. After applying it, record the same scene
and the same motion again, then compare the two Markdown reports.

Do not judge a setting from one arbitrary scene. Use:

1. A bright, static, textured scene.
2. Slow translation and rotation.
3. A second run with the same path and lighting.

## Current Example Result

Run:

```text
data/hanpu/runs/hanpu_manual_20260716_163616
```

Measured:

- `603` frames over `20.066 s` at `30.0003 Hz`.
- Left/right PTS equal.
- IMU `1967` rows at `98.69 Hz`.
- Stereo RMS `0.373 px`, baseline `60.295 mm`.
- ORB keypoints reached the configured `4200` per image.
- Mean stereo matches: `1537`.
- Mean temporal matches: `1436`.
- `10` local-map failures, `2` low-match events, `2` lost-frame events.
- `3` maps created, `599` trajectory rows, `43` keyframes.

Interpretation: the camera transport, timing, calibration and detector output
are usable. The remaining weakness is tracking stability under the tested scene
and motion, not a lack of raw ORB detections.

## Log Interpretation

Normal startup messages:

```text
No JPEG data found in image
QFontDatabase: Cannot find font directory
```

These are not sufficient evidence of camera failure if frames and YCTC data are
still exported.

Important tracking messages:

- `Less than 15 matches`: reference-keyframe matching has collapsed.
- `Fail to track local map`: local-map tracking failed.
- `Frames set to lost`: tracking entered LOST state.
- `New Map created`: ORB reset/reinitialized a map.
- `Relocalized`: tracking recovered after a loss.

The X Window `BadWindow` message at shutdown can occur after ORB has already
saved `f_hanpu_real_stereo.txt` and `kf_hanpu_real_stereo.txt`. Check the files
and the report before treating it as a run failure.

## Development Boundaries

Do not use the current Hanpu result as final VINS accuracy evidence yet.

The next sensible order is:

1. Repeat the camera-quality report on controlled bright/textured scenes.
2. Compare default exposure against the `10 ms / 4x` A/B run.
3. Keep the real `1920x1080` calibration and 48-byte YCTC parsing.
4. Make ORB tracking stable across repeated recordings.
5. Add and verify the camera-IMU extrinsic transform.
6. Only then integrate the 100 Hz IMU into a stereo-inertial ORB/VINS pipeline.

Avoid returning to RM75 evaluator, `Tlr`, or strict-sync conclusions when
debugging this Hanpu camera bring-up; those are separate chains.
