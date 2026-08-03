# Minimal Reproduction Fixture

This fixture is synthetic, small, and versioned. It proves the evaluation contract without a camera, robot, ORB-SLAM3 build, or proprietary recording.

It contains a camera-frame pose CSV, a matching robot TCP JSON trajectory, and an identity hand-eye calibration. The expected rigid SE(3) translation and rotation APE are both zero (within floating-point precision).

Run it from the repository root:

```bash
python3 -m pip install -r requirements-minimal.txt
python3 script/vio_eval.py demo --output-dir local/minimal_reproduction
```

Inspect `local/minimal_reproduction/metrics.json` and `REPORT.md`. Generated output is intentionally written to `local/`, which is excluded from Git.
