# Project Housekeeping

## Goals

- Keep algorithm/code history recoverable.
- Keep the repo root focused on code and core config.
- Push local outputs, experiments, logs, and scratch artifacts out of git's way.

## Current Rules

- Local artifacts live under `local/` or ignored output roots such as `data/evaluation/workbench/`.
- Core code stays in:
  - `script/`
  - `data/calibration/`
  - `README.md`
  - `docs/`
- Recovery anchors live in `docs/version_anchors/`.

## Best-Version Policy

- The current RM75 "best-known" ORB-SLAM3 variant is preserved via:
  - a git tag on the ORB base repo commit
  - a patch snapshot for the exact working-tree delta
  - a restore script
  - a metrics note pointing to the corresponding evaluation batch

## Cleanup Notes

- Root-level scratch such as `docx/`, `logs/`, `new_system*`, temporary trajectory dumps, `.codex/`, and `.deps/` should not stay in the repo root long-term.
- If new one-off outputs appear, prefer `local/<topic>/...`.
