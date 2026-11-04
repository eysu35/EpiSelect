#!/bin/bash
set -euo pipefail

NODE=${1:-node1}
ZONE=${2:-us-east1-d}

gcloud compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" --worker=all --command="
  tmux kill-session -t launch 2>/dev/null || true
  pkill -9 python || true
  pkill -9 wandb || true
  sudo rm -rf /tmp/tpu_logs
  sudo rm -f /tmp/libtpu_lockfile
  echo \"stopped on \$(hostname)\"
"
