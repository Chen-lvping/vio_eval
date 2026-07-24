#!/usr/bin/env bash
set -euo pipefail

ORB_ROOT="${ORB_ROOT:-/home/chenlvping/1_DM_work/orbslam3_fresh/ORB_SLAM3_clean}"
BASE_COMMIT="4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"
PATCH_FILE="/home/chenlvping/1_DM_work/vio_eval/docs/version_anchors/patches/orbslam3_clean_rm75_best_20260707.patch"

if [[ ! -d "$ORB_ROOT/.git" ]]; then
  echo "ORB repo not found: $ORB_ROOT" >&2
  exit 1
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "Patch snapshot not found: $PATCH_FILE" >&2
  exit 1
fi

cat <<EOF
About to restore the anchored RM75 ORB-SLAM3 version.

Repo:        $ORB_ROOT
Base commit: $BASE_COMMIT
Patch:       $PATCH_FILE

This will:
1. git reset --hard to the base commit
2. git apply the anchored patch

EOF

read -r -p "Continue? [y/N] " reply
if [[ "$reply" != "y" && "$reply" != "Y" ]]; then
  echo "Cancelled."
  exit 0
fi

git -C "$ORB_ROOT" reset --hard "$BASE_COMMIT"
git -C "$ORB_ROOT" apply "$PATCH_FILE"

echo "Restored anchored RM75 ORB-SLAM3 version."
