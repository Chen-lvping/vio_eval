#!/usr/bin/env bash
set -euo pipefail

BOARD="${BOARD:-10.30.120.52}"
BOARD_USER="${BOARD_USER:-ubuntu}"
BOARD_PASSWORD="${BOARD_PASSWORD:-ubuntu}"
ARM_IP="${ARM_IP:-192.168.1.18}"
ARM_PORT="${ARM_PORT:-8080}"
TRAJ="${TRAJ:-/home/chenlvping/1_DM_work/ugripper/scripts/6_26.txt}"
DEST="${DEST:-/home/chenlvping/data_6_26}"
RECORD_SEC="${RECORD_SEC:-10}"
ARM_DURATION_SEC="${ARM_DURATION_SEC:-10}"
ARM_SPEED="${ARM_SPEED:-5}"
POSE_RATE_HZ="${POSE_RATE_HZ:-50}"
STRICT_HOST_KEY_CHECKING="${STRICT_HOST_KEY_CHECKING:-no}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPLAY_SCRIPT="${REPLAY_SCRIPT:-${SCRIPT_DIR}/smooth_reteach_trajectory.py}"
POSE_SCRIPT="${POSE_SCRIPT:-${SCRIPT_DIR}/get_rm75_end_pose.py}"

if [ ! -f "$REPLAY_SCRIPT" ] && [ -f "${SCRIPT_DIR}/experiments/smooth_reteach_trajectory.py" ]; then
  REPLAY_SCRIPT="${SCRIPT_DIR}/experiments/smooth_reteach_trajectory.py"
fi

SSH_OPTS=(
  -o "StrictHostKeyChecking=${STRICT_HOST_KEY_CHECKING}"
  -o UserKnownHostsFile=/dev/null
  -o ConnectTimeout=8
)

log() {
  printf '[record-replay] %s\n' "$*"
}

die() {
  printf '[record-replay][ERROR] %s\n' "$*" >&2
  exit 1
}

board_ssh() {
  sshpass -p "$BOARD_PASSWORD" ssh "${SSH_OPTS[@]}" "${BOARD_USER}@${BOARD}" "$@"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

tcp_open() {
  local host="$1"
  local port="$2"
  timeout 3 bash -c "</dev/tcp/${host}/${port}" >/dev/null 2>&1
}

send_stop() {
  log "sending STOP to ${BOARD}"
  board_ssh 'if [ -p /tmp/umi_record_control.pipe ]; then echo STOP > /tmp/umi_record_control.pipe; fi' >/dev/null 2>&1 || true
}

wait_for_episode() {
  local before_latest="$1"
  local episode=""

  for _ in $(seq 1 90); do
    episode="$(board_ssh "ls -dt /mnt/data_disk/*/data/episode_* 2>/dev/null | grep -v -- '-temp' | head -n1" || true)"
    if [ -n "$episode" ] && [ "$episode" != "$before_latest" ]; then
      if board_ssh "test -f '$episode/metadata.json'" >/dev/null 2>&1; then
        printf '%s\n' "$episode"
        return 0
      fi
    fi
    sleep 1
  done

  return 1
}

require_cmd sshpass
require_cmd rsync
require_cmd python3
require_cmd timeout

session_duration_sec="$(python3 - "$RECORD_SEC" "$ARM_DURATION_SEC" <<'PY'
import sys

record = float(sys.argv[1])
replay = float(sys.argv[2])
print(max(record, replay))
PY
)"

[ -f "$TRAJ" ] || die "trajectory file not found: $TRAJ"
[ -f "$REPLAY_SCRIPT" ] || die "replay script not found: $REPLAY_SCRIPT"
[ -f "$POSE_SCRIPT" ] || die "pose recorder script not found: $POSE_SCRIPT"

log "checking board ${BOARD}"
ping -c 1 -W 2 "$BOARD" >/dev/null || die "board is not reachable: $BOARD"
board_ssh 'test -p /tmp/umi_record_control.pipe' || die "record control pipe is not ready on board"
board_ssh 'test ! -f /tmp/umi_recording.lock' || die "board is already recording"

log "checking robot arm ${ARM_IP}:${ARM_PORT}"
ping -c 1 -W 2 "$ARM_IP" >/dev/null || die "robot arm is not reachable: $ARM_IP"
tcp_open "$ARM_IP" "$ARM_PORT" || die "robot arm TCP port is not open: ${ARM_IP}:${ARM_PORT}"

before_latest="$(board_ssh "ls -dt /mnt/data_disk/*/data/episode_* 2>/dev/null | grep -v -- '-temp' | head -n1" || true)"
mkdir -p "$DEST"

timestamp_tag="$(date +%Y%m%d_%H%M%S)"
run_root="${DEST%/}/record_replay_${timestamp_tag}"
mkdir -p "$run_root"
pose_tmp="${run_root}/rm75_end_pose.json"
pose_log="${run_root}/rm75_end_pose.log"

stop_sent=0
pose_stop_sent=0
pose_pid=""
pose_timer_pid=""
rec_timer_pid=""
cleanup() {
  if [ "$stop_sent" = "0" ]; then
    send_stop
    stop_sent=1
  fi
  if [ -n "${rec_timer_pid:-}" ] && kill -0 "$rec_timer_pid" >/dev/null 2>&1; then
    kill "$rec_timer_pid" >/dev/null 2>&1 || true
  fi
  if [ -n "${pose_timer_pid:-}" ] && kill -0 "$pose_timer_pid" >/dev/null 2>&1; then
    kill "$pose_timer_pid" >/dev/null 2>&1 || true
  fi
  if [ "$pose_stop_sent" = "0" ] && [ -n "${pose_pid:-}" ] && kill -0 "$pose_pid" >/dev/null 2>&1; then
    log "stopping local RM75 end-pose recording"
    kill -INT "$pose_pid" >/dev/null 2>&1 || true
    pose_stop_sent=1
  fi
}
trap cleanup EXIT INT TERM

log "record/replay session duration: ${session_duration_sec}s (board=${RECORD_SEC}s, replay=${ARM_DURATION_SEC}s)"
log "starting local RM75 end-pose recording -> ${pose_tmp}"
python3 "$POSE_SCRIPT" \
  --output "$pose_tmp" \
  --ip "$ARM_IP" \
  --port "$ARM_PORT" \
  --rate-hz "$POSE_RATE_HZ" \
  >"$pose_log" 2>&1 &
pose_pid=$!
sleep 1
if ! kill -0 "$pose_pid" >/dev/null 2>&1; then
  wait "$pose_pid" || true
  die "local pose recorder exited early; see ${pose_log}"
fi

(
  sleep "$session_duration_sec"
  if kill -0 "$pose_pid" >/dev/null 2>&1; then
    log "stopping local RM75 end-pose recording"
    kill -INT "$pose_pid" >/dev/null 2>&1 || true
    pose_stop_sent=1
  fi
) &
pose_timer_pid=$!

log "starting recording for ${session_duration_sec}s on board"
board_ssh 'echo START > /tmp/umi_record_control.pipe'

(
  sleep "$session_duration_sec"
  send_stop
  stop_sent=1
) &
rec_timer_pid=$!

log "replaying trajectory for ${ARM_DURATION_SEC}s"
python3 "$REPLAY_SCRIPT" \
  --input "$TRAJ" \
  --ip "$ARM_IP" \
  --port "$ARM_PORT" \
  --speed "$ARM_SPEED" \
  --duration-sec "$ARM_DURATION_SEC" \
  --control-api movej_canfd \
  --final-lock on

wait "$rec_timer_pid" || true
stop_sent=1

if [ -n "${pose_timer_pid:-}" ]; then
  wait "$pose_timer_pid" || true
fi
if [ -n "${pose_pid:-}" ]; then
  wait "$pose_pid" || die "local pose recorder failed; see ${pose_log}"
fi
pose_stop_sent=1
trap - EXIT INT TERM

log "waiting for finalized episode"
episode="$(wait_for_episode "$before_latest")" || die "timed out waiting for finalized episode"
episode_name="$(basename "$episode")"

log "pulling ${episode} -> ${DEST}/${episode_name}"
sshpass -p "$BOARD_PASSWORD" rsync -az --info=progress2 \
  -e "ssh -o StrictHostKeyChecking=${STRICT_HOST_KEY_CHECKING} -o UserKnownHostsFile=/dev/null" \
  "${BOARD_USER}@${BOARD}:${episode}/" "${DEST}/${episode_name}/"

if [ -f "$pose_tmp" ]; then
  cp "$pose_tmp" "${DEST}/${episode_name}/rm75_end_pose.json"
  cp "$pose_log" "${DEST}/${episode_name}/rm75_end_pose.log"
  log "saved RM75 end-pose trajectory: ${DEST}/${episode_name}/rm75_end_pose.json"
fi

log "saved episode: ${DEST}/${episode_name}"
