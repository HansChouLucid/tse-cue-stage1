#!/usr/bin/env bash
set -euo pipefail
BASE=/data/tse_cue_project
REPO=$BASE/repos/wesep-real-tse
PLAN=$BASE/experiments/normal_basic_single_cue/bsrnn_continue_a100_20260607
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
cd "$REPO"
export OMP_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

start_job() {
  local name=$1
  local gpu=$2
  local config=$3
  local log=$4
  if pgrep -af "${config}" >/dev/null; then
    echo "$name already running"
    return 0
  fi
  echo "Starting $name on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    wesep/bin/train.py --config "$config" > "$log" 2>&1 &
  echo $! > "${log%.log}.pid"
}

start_job tfmap 0 "$PLAN/configs/tfmap_continue.yaml" "$PLAN/logs/tfmap_continue.outer.log"
start_job usef 1 "$PLAN/configs/usef_continue.yaml" "$PLAN/logs/usef_continue.outer.log"
