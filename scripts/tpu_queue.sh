#!/bin/bash
# Run the whole sweep on one pod, one run at a time, skipping runs whose rundir has a DONE marker.
# Each run is launched detached on every worker and polled until its tmux session ends; a run that
# ends without DONE (crash, preemption) makes the queue exit 1 so a supervisor can re-run it -- the
# trainer resumes from its last checkpoint + loader state.
#
#   tpu_queue.sh DATA_DIR CKPT_BASE [NODE] [ZONE]
#   SIZES="124M 1.3B" METHODS="Natural ADO EpiSelect" SEEDS_124M="0 1 2 3 4" SEEDS_1_3B="0 1 2" POLL=120
# Order: size -> seed -> method, so every seed has all METHODS before the next seed starts. After the
# main methods of a size, its ablations run method by method (ABLATIONS_124M / ABLATIONS_1_3B, all seeds
# of the first ablation, then the next); ABLATIONS_124M="" skips them.
# Several pods can share one sweep: a run is claimed by writing the pod name to <rundir>/CLAIMED before it
# starts, and other pods skip runs claimed by someone else. Give pods opposite seed orders (SEEDS_124M="0 1 2 3 4"
# vs "4 3 2 1 0") so they meet in the middle. A pod that dies for good leaves stale claims:
# `gcloud storage rm <rundir>/CLAIMED` frees them.
set -uo pipefail
source "$(dirname "$0")/tpu_common.sh"

DATA_DIR=${1:?usage: tpu_queue.sh DATA_DIR CKPT_BASE [NODE] [ZONE]}
CKPT_BASE=${2:?}
NODE=${3:-node1}
ZONE=${4:-us-east1-d}
SIZES=${SIZES:-"124M 1.3B"}
METHODS=${METHODS:-"Natural ADO EpiSelect"}
SEEDS_124M=${SEEDS_124M:-"0 1 2 3 4"}
SEEDS_1_3B=${SEEDS_1_3B:-"0 1 2"}
ABLATIONS_124M=${ABLATIONS_124M:-"ADO_noprior EpiSelect_diag"}
ABLATIONS_1_3B=${ABLATIONS_1_3B:-""}
POLL=${POLL:-120}
STRAGGLER_LIMIT=${STRAGGLER_LIMIT:-900}  # seconds some workers may sit idle while others still run

seeds_for() { case "$1" in 124M) echo "$SEEDS_124M";; 1.3B) echo "$SEEDS_1_3B";; *) echo "unknown size $1" >&2; exit 2;; esac; }
ablations_for() { case "$1" in 124M) echo "$ABLATIONS_124M";; 1.3B) echo "$ABLATIONS_1_3B";; esac; }
is_done() { gcloud storage ls "$1/DONE" >/dev/null 2>&1; }
claimed_by() { gcloud storage cat "$1/CLAIMED" 2>/dev/null | tr -d '[:space:]'; }
# generation-match=0 makes the claim atomic: two pods that pass the owner check together cannot both win
claim() { echo "$NODE" | gcloud storage cp --if-generation-match=0 - "$1/CLAIMED" >/dev/null 2>&1; }

# prints RUNNING/IDLE per worker; exit 1 if the pod is unreachable
worker_states() {
  [ "$(tpu_node_state "$NODE" "$ZONE")" = "READY" ] || return 1
  tpu_ssh_quick "$NODE" "$ZONE" 'tmux has-session -t launch 2>/dev/null && echo RUNNING || echo IDLE' 2>/dev/null | grep -E '^(RUNNING|IDLE)$'
}

wait_for_run() {
  local straggler_since=""
  while true; do
    sleep "$POLL"
    local states=""; local try
    # a node that is not READY is gone: fail fast. Otherwise tolerate a few transient ssh/gcloud failures.
    for try in 1 2 3 4 5 6; do
      [ "$(tpu_node_state "$NODE" "$ZONE")" = "READY" ] || { echo "node not READY"; return 2; }
      states=$(worker_states) && break; sleep 30
    done
    [ -n "$states" ] || { echo "pod unreachable"; return 2; }
    local running idle; running=$(grep -c RUNNING <<<"$states"); idle=$(grep -c IDLE <<<"$states")
    if [ "$running" -eq 0 ]; then return 0; fi
    if [ "$idle" -gt 0 ]; then
      straggler_since=${straggler_since:-$(date +%s)}
      if [ $(( $(date +%s) - straggler_since )) -gt "$STRAGGLER_LIMIT" ]; then
        echo "$idle worker(s) exited while $running still run; killing the run"
        tpu_ssh "$NODE" "$ZONE" "$TPU_RESET" >/dev/null 2>&1
        return 1
      fi
    else
      straggler_since=""
    fi
  done
}

# launch one run and wait for it; exits the queue if the run does not reach DONE
run_one() {
  local exp=$1 seed=$2
  local rundir=$CKPT_BASE/$exp/seed$seed
  if is_done "$rundir"; then echo "done   $exp seed $seed"; return; fi
  local owner; owner=$(claimed_by "$rundir")
  if [ -n "$owner" ] && [ "$owner" != "$NODE" ]; then echo "taken  $exp seed $seed (claimed by $owner)"; return; fi
  if [ -z "$owner" ] && ! claim "$rundir"; then
    owner=$(claimed_by "$rundir")
    [ "$owner" = "$NODE" ] || { echo "taken  $exp seed $seed (claimed by ${owner:-?} while claiming)"; return; }
  fi
  echo "=== $(date '+%F %T') start $exp seed $seed -> $rundir"
  RUN=$(run_cmd "$exp" "$seed" "$DATA_DIR" "$rundir")
  SSH_TIMEOUT=600 tpu_ssh_quick "$NODE" "$ZONE" "$TPU_RESET; cd ~/EpiSelect && tmux new -d -s launch '$RUN' && echo launched on \$(hostname)" \
    || { echo "launch failed"; exit 1; }
  wait_for_run; rc=$?
  if [ "$rc" -ne 0 ] || ! is_done "$rundir"; then
    echo "=== $(date '+%F %T') $exp seed $seed did not finish (rc=$rc); it will resume on the next queue pass"
    exit 1
  fi
  echo "=== $(date '+%F %T') finished $exp seed $seed"
}

for size in $SIZES; do
  for seed in $(seeds_for "$size"); do
    for method in $METHODS; do run_one "$size/$method" "$seed"; done
  done
  for method in $(ablations_for "$size"); do
    for seed in $(seeds_for "$size"); do run_one "$size/$method" "$seed"; done
  done
done
echo "queue complete"
