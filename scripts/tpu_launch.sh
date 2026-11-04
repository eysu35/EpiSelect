#!/bin/bash
# Launch one run detached (tmux session "launch") on every worker. For the full sweep use tpu_queue.sh.
set -euo pipefail
source "$(dirname "$0")/tpu_common.sh"

EXPERIMENT=${1:?usage: tpu_launch.sh EXPERIMENT SEED DATA_DIR CKPT_BASE [NODE] [ZONE]}
SEED=${2:?}
DATA_DIR=${3:?}
CKPT_BASE=${4:?}
NODE=${5:-node1}
ZONE=${6:-us-east1-d}

# rundir layout: CKPT_BASE/<size>/<method>/seed<N>, resumed from the latest checkpoint if it exists
RUNDIR=$CKPT_BASE/$EXPERIMENT/seed$SEED
RUN=$(run_cmd "$EXPERIMENT" "$SEED" "$DATA_DIR" "$RUNDIR")

tpu_ssh "$NODE" "$ZONE" "
  set -euo pipefail
  cd ~/EpiSelect
  [ '${PULL:-0}' = '1' ] && git pull --ff-only
  $TPU_RESET
  tmux new -d -s launch '$RUN'
  echo \"launched on \$(hostname)\"
"

echo
echo "Started detached. Follow with: ./scripts/tpu_check.sh $NODE $ZONE"
