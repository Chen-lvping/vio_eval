#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_URL="https://github.com/UZ-SLAMLab/ORB_SLAM3.git"
UPSTREAM_COMMIT="4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"
PATCH_FILE="$ROOT/docs/version_anchors/patches/orbslam3_adaptive_vio_20260803.patch"
EXPECTED_PATCH_SHA256="7833a42bdeda3743e4fc89f1543aed37e5d6907626f42529aff96f694048d0e8"
ORB_ROOT=""
BUILD=0

usage() {
    cat <<'EOF'
Usage: script/setup_orbslam3_adaptive_vio.sh --orb-root PATH [--build]

Clones the pinned ORB-SLAM3 upstream commit into an empty PATH and applies the
versioned RM75 adaptive-VIO patch. --build runs the upstream build.sh afterward.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --orb-root) ORB_ROOT="${2:-}"; shift 2 ;;
        --build) BUILD=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$ORB_ROOT" ]]; then
    echo "--orb-root is required" >&2
    usage >&2
    exit 2
fi
if [[ ! -f "$PATCH_FILE" ]]; then
    echo "Patch not found: $PATCH_FILE" >&2
    exit 1
fi
actual_patch_sha256="$(sha256sum "$PATCH_FILE" | awk '{print $1}')"
if [[ "$actual_patch_sha256" != "$EXPECTED_PATCH_SHA256" ]]; then
    echo "Patch checksum mismatch: expected $EXPECTED_PATCH_SHA256, got $actual_patch_sha256" >&2
    exit 1
fi

ORB_ROOT="$(realpath -m "$ORB_ROOT")"
if [[ -e "$ORB_ROOT" ]]; then
    if [[ ! -d "$ORB_ROOT" ]] || [[ -n "$(find "$ORB_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Target must be a new or empty directory: $ORB_ROOT" >&2
        exit 1
    fi
else
    mkdir -p "$ORB_ROOT"
fi

git clone "$UPSTREAM_URL" "$ORB_ROOT"
git -C "$ORB_ROOT" checkout --detach "$UPSTREAM_COMMIT"
git -C "$ORB_ROOT" apply --check "$PATCH_FILE"
git -C "$ORB_ROOT" apply "$PATCH_FILE"
git -C "$ORB_ROOT" diff --check
echo "[DONE] adaptive ORB-SLAM3 prepared: $ORB_ROOT"
echo "[DONE] upstream: $UPSTREAM_COMMIT"

if [[ "$BUILD" -eq 1 ]]; then
    (cd "$ORB_ROOT" && ./build.sh)
    echo "[DONE] ORB-SLAM3 build completed"
fi
