# shared by tpu_launch.sh / tpu_queue.sh / tpu_watchdog.sh; source, don't run

# kills a stale run and frees the TPU on a worker
TPU_RESET='tmux kill-session -t launch 2>/dev/null || true; pkill -9 python || true; pkill -9 wandb || true; sudo rm -rf /tmp/tpu_logs; sudo rm -f /tmp/libtpu_lockfile'

# run_cmd EXPERIMENT SEED DATA_DIR RUNDIR [HYDRA_OVERRIDE...] -> the training command executed inside tmux on
# every worker. ${WANDB_PROJECT:-...} is left unexpanded on purpose; the worker's shell resolves it.
run_cmd() {
  local log="~/EpiSelect/logs/$(echo "$1" | tr / -)-seed$2.log"
  local exp=$1 seed=$2 data=$3 rundir=$4; shift 4
  echo "mkdir -p ~/EpiSelect/logs && cd ~/EpiSelect && source ~/venv/bin/activate && export WANDB_PROJECT=\${WANDB_PROJECT:-episelect-iclr26} && python -u launch.py +experiment=$exp seed=$seed loader.data_dir=$data rundir=$rundir multihost=true $* 2>&1 | tee -a $log"
}

# tpu_ssh NODE ZONE COMMAND -> runs COMMAND on every worker
tpu_ssh() {
  gcloud compute tpus tpu-vm ssh "$1" --zone="$2" --worker=all --command="$3"
}

# same, for short polls: ssh to a preempted node hangs forever, so bound it
tpu_ssh_quick() {
  timeout "${SSH_TIMEOUT:-120}" gcloud compute tpus tpu-vm ssh "$1" --zone="$2" --worker=all --command="$3"
}

# READY/PREEMPTED/... of the node itself (the queued resource can lag minutes behind)
tpu_node_state() {
  gcloud compute tpus tpu-vm describe "$1" --zone="$2" --format='value(state)' 2>/dev/null
}
