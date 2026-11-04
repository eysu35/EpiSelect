#!/bin/bash
# Provision every worker of a pod: ship the local working tree (tracked files + .wandbkey), create the venv,
# install jax[tpu] + this package, and print the device count. Safe to re-run; only missing pieces are installed.
set -euo pipefail
source "$(dirname "$0")/tpu_common.sh"

NODE=${1:-node1}
ZONE=${2:-us-east1-d}
REPO=$(cd "$(dirname "$0")/.." && pwd)

KEY=.wandbkey
if [ ! -f "$REPO/$KEY" ]; then
  if [ "${REQUIRE_WANDB_KEY:-1}" = "1" ]; then
    echo "ERROR: $REPO/.wandbkey not found; write the W&B API key there so runs can log (REQUIRE_WANDB_KEY=0 to set up anyway)" >&2
    exit 1
  fi
  echo "WARNING: no .wandbkey; setting up without it (runs will not be able to log)" >&2
  KEY=""
fi

# code sync from here rather than git clone: the workers get exactly what is being tested locally
TAR=$(mktemp /tmp/episelect_src.XXXXXX.tgz)
(cd "$REPO" && git ls-files -z | tar czf "$TAR" --null -T - $KEY)
gcloud compute tpus tpu-vm scp "$TAR" "$NODE":~/episelect_src.tgz --zone="$ZONE" --worker=all
rm -f "$TAR"

tpu_ssh "$NODE" "$ZONE" "
  set -euo pipefail
  { command -v tmux && python3 -c 'import ensurepip'; } >/dev/null 2>&1 \
    || { sudo apt-get -o DPkg::Lock::Timeout=600 update -qq && sudo apt-get -o DPkg::Lock::Timeout=600 install -y -qq python3-venv tmux; }

  mkdir -p ~/EpiSelect
  rm -rf ~/EpiSelect/src ~/EpiSelect/scripts ~/EpiSelect/configs ~/EpiSelect/tests
  tar xzf ~/episelect_src.tgz -C ~/EpiSelect && rm ~/episelect_src.tgz

  [ -x ~/venv/bin/pip ] || { rm -rf ~/venv; python3 -m venv ~/venv; }
  source ~/venv/bin/activate
  python -c 'import jax, jaxlib' 2>/dev/null || pip install -q --upgrade pip
  pip install -q -e ~/EpiSelect'[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

  grep -q tpu-reset ~/.bashrc || echo 'alias tpu-reset=\"$TPU_RESET\"' >> ~/.bashrc

  if tmux has-session -t launch 2>/dev/null; then echo \"\$(hostname): run in progress, skipping device check\"; else
    pkill -9 python || true; sudo rm -f /tmp/libtpu_lockfile   # a device check cut off by a timeout must not hold the TPU
    timeout 300 python -c 'import jax; print(f\"host {jax.process_index()}: {jax.local_device_count()} local / {jax.device_count()} global devices\")'
  fi
"
echo "setup complete on $NODE ($ZONE)"
