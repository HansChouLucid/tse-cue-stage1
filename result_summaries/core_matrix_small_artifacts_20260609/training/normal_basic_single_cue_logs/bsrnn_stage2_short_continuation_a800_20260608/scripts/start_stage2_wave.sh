#!/usr/bin/env bash
set -euo pipefail
BASE=/data/tse_cue_project
REPO=$BASE/repos/wesep-real-tse
PLAN=$BASE/experiments/normal_basic_single_cue/bsrnn_stage2_short_continuation_a800_20260608
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
cd "$REPO"
export OMP_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$PLAN/logs"
start_job() {
  local name=$1 gpu=$2 cfg=$3 log=$4
  if pgrep -af "wesep/bin/train.py --config $cfg" >/dev/null; then
    echo "$(date '+%F %T') $name already running"
    return 0
  fi
  echo "$(date '+%F %T') starting $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    wesep/bin/train.py --config "$cfg" > "$log" 2>&1 &
  echo $! > "${log%.log}.pid"
}
start_job tfmap_A 0 "$PLAN/configs/tfmap_A_lr3e6_5ep.yaml" "$PLAN/logs/tfmap_A.outer.log"
start_job usef_stabilize 1 "$PLAN/configs/usef_stabilize_lr2e6_3ep.yaml" "$PLAN/logs/usef_stabilize.outer.log"
nohup bash "$PLAN/scripts/start_context_after_usef.sh" > "$PLAN/logs/context_queue.outer.log" 2>&1 &
echo $! > "$PLAN/logs/context_queue.pid"
