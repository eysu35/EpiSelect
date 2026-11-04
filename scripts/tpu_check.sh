#!/bin/bash
set -euo pipefail

NODE=${1:-node1}
ZONE=${2:-us-east1-d}
LINES=${3:-2000}

gcloud compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" --worker=all \
  --command="echo \"===== \$(hostname) =====\"; tmux capture-pane -pt launch -S -$LINES 2>/dev/null || echo 'no launch session'"
