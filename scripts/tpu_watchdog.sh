#!/bin/bash
# Keep one spot pod alive and working through the run queue until the queue reports completion.
#   tpu_watchdog.sh NODE ZONE DATA_DIR CKPT_BASE [ACCELERATOR]
# Pods sharing one sweep claim runs in the bucket (see tpu_queue.sh); give them opposite seed orders:
#   SEEDS_124M="0 1 2 3 4" SEEDS_1_3B="0 1 2" tpu_watchdog.sh node3 us-east1-d gs://... gs://... v6e-16
#   SPOT=0 RUNTIME=tpu-ubuntu2204-base SEEDS_124M="4 3 2 1 0" SEEDS_1_3B="2 1 0" tpu_watchdog.sh node6 us-central2-b gs://... gs://... v4-32
# Run it in tmux on a login node; it needs gcloud auth and the repo's .wandbkey.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

NODE=${1:?usage: tpu_watchdog.sh NODE ZONE DATA_DIR CKPT_BASE [ACCELERATOR]}
ZONE=${2:?}
DATA_DIR=${3:?}
CKPT_BASE=${4:?}
ACCEL=${5:-v6e-32}
RUNTIME=${RUNTIME:-v2-alpha-tpuv6e}
PROJECT=${PROJECT:-epi-data-gen}
POLL=${POLL:-60}
SPOT=${SPOT:-1}   # SPOT=0 for an on-demand grant

# stopping the watchdog must also stop its queue + ssh polls, or an orphan queue keeps driving the pod
trap 'trap - TERM INT; kill -- -$$ 2>/dev/null; exit 143' TERM INT

[ -f "$HERE/../.wandbkey" ] || { echo "ERROR: .wandbkey missing in repo root; refusing to launch runs without W&B" >&2; exit 1; }

log() { echo "$(date '+%F %T') [$NODE] $*"; }
qr() { gcloud compute tpus queued-resources "$@" --zone="$ZONE" --project="$PROJECT"; }
# the queued resource can still say ACTIVE minutes after the node itself was preempted
state() {
  local s err; err=$(mktemp)
  s=$(qr describe "$NODE" --format='value(state.state)' 2>"$err") || {
    # only a real NOT_FOUND may trigger a create; auth/network errors must wait (gcloud reauth expires ~daily)
    if grep -q NOT_FOUND "$err"; then echo NOT_FOUND; else echo "ERROR:$(grep -m1 -oE 'Reauthentication failed|auth tokens|[A-Z_]{5,}' "$err" | head -1)"; fi
    rm -f "$err"; return; }
  rm -f "$err"
  if [ "$s" = "ACTIVE" ]; then
    local n; n=$(gcloud compute tpus tpu-vm describe "$NODE" --zone="$ZONE" --project="$PROJECT" --format='value(state)' 2>/dev/null)
    case "$n" in PREEMPTED|TERMINATED) s=PREEMPTED;; esac
  fi
  echo "$s"
}
created() { qr describe "$NODE" --format='value(createTime)' 2>/dev/null; }

request() {
  local flags=(--node-id="$NODE" --accelerator-type="$ACCEL" --runtime-version="$RUNTIME")
  [ "$SPOT" = "1" ] && flags+=(--spot)
  qr create "$NODE" "${flags[@]}" 2>&1 | tail -1
}

setup_for=""
while true; do
  s=$(state)
  case "$s" in
    NOT_FOUND)
      log "no request; creating $ACCEL"; request ;;
    FAILED|SUSPENDED|PREEMPTED)
      log "state $s ($(qr describe "$NODE" --format='value(state.failedData.error.message)' 2>/dev/null)); re-requesting"
      qr delete "$NODE" --force --quiet 2>&1 | tail -1; request ;;
    ACTIVE)
      c=$(created)
      if [ "$setup_for" != "$c" ]; then
        log "new node (created $c); running setup"
        # bounded: gcloud scp/ssh retry for ages against a node that died right after provisioning
        if timeout -k 30 "${SETUP_TIMEOUT:-900}" "$HERE/tpu_setup.sh" "$NODE" "$ZONE"; then setup_for=$c; else log "setup failed or timed out; retrying next tick"; sleep "$POLL"; continue; fi
      fi
      log "running queue"
      if "$HERE/tpu_queue.sh" "$DATA_DIR" "$CKPT_BASE" "$NODE" "$ZONE"; then log "queue complete; exiting"; exit 0; fi
      log "queue stopped early (state now $(state)); will resume" ;;
    ERROR:*)
      log "gcloud failed (${s#ERROR:}); waiting -- if this repeats, run 'gcloud auth login' on this machine" ;;
    *)
      log "state $s; waiting" ;;
  esac
  sleep "$POLL"
done
