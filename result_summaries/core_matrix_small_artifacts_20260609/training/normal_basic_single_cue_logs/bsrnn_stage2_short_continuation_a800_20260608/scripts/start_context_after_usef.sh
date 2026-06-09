#!/usr/bin/env bash
set -euo pipefail
BASE=/data/tse_cue_project
REPO=$BASE/repos/wesep-real-tse
PLAN=$BASE/experiments/normal_basic_single_cue/bsrnn_stage2_short_continuation_a800_20260608
USEF_EXP=$BASE/experiments/normal_basic_single_cue/usef_only_bsrnn_stage2_stabilize_lr2e6_3ep_a800_gpu1_bs8_20260608
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
cd "$REPO"
export OMP_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date '+%F %T') waiting for USEF checkpoint_3.pt before Context-A" 
while true; do
  if [ -f "$USEF_EXP/models/checkpoint_3.pt" ]; then
    echo "$(date '+%F %T') USEF stage complete; starting Context-A on GPU1"
    break
  fi
  if ! pgrep -af "usef_stabilize_lr2e6_3ep.yaml" >/dev/null && [ -d "$USEF_EXP" ]; then
    echo "$(date '+%F %T') WARNING: USEF process not found and final ckpt missing; not starting Context" >&2
    exit 2
  fi
  sleep 300
done
CFG="$PLAN/configs/context_A_lr5e6_5ep.yaml"
LOG="$PLAN/logs/context_A.outer.log"
if pgrep -af "wesep/bin/train.py --config $CFG" >/dev/null; then
  echo "$(date '+%F %T') Context-A already running"
  exit 0
fi
CUDA_VISIBLE_DEVICES=1 nohup torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  wesep/bin/train.py --config "$CFG" > "$LOG" 2>&1 &
echo $! > "${LOG%.log}.pid"
