#!/usr/bin/env bash
set -euo pipefail
BASE=/data/tse_cue_project/experiments/normal_basic_single_cue
PLAN=$BASE/bsrnn_continue_a100_20260607
REPO=/data/tse_cue_project/repos/wesep-real-tse
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
cd "$REPO"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
check_and_start() {
  local name=$1 gpu=$2 cfg=$3 exp=$4 final_epoch=$5
  local latest="$exp/models/latest_checkpoint.pt"
  local final="$exp/models/checkpoint_${final_epoch}.pt"
  if [ -f "$final" ]; then
    echo "$(date '+%F %T') $name final checkpoint exists; no restart" >> "$PLAN/logs/watchdog.log"
    return 0
  fi
  if pgrep -af "$cfg" | grep -v grep >/dev/null; then
    echo "$(date '+%F %T') $name running" >> "$PLAN/logs/watchdog.log"
    return 0
  fi
  local log="$PLAN/logs/${name}_watchdog_restart_$(date +%Y%m%d_%H%M%S).log"
  echo "$(date '+%F %T') restarting $name on GPU $gpu" >> "$PLAN/logs/watchdog.log"
  if [ -f "$latest" ]; then
    CUDA_VISIBLE_DEVICES=$gpu nohup torchrun --standalone --nnodes=1 --nproc_per_node=1 \
      wesep/bin/train.py --config "$cfg" --checkpoint "$latest" > "$log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$gpu nohup torchrun --standalone --nnodes=1 --nproc_per_node=1 \
      wesep/bin/train.py --config "$cfg" > "$log" 2>&1 &
  fi
}
while true; do
  check_and_start tfmap 0 "$PLAN/configs/tfmap_continue_bs8.yaml" "$BASE/tfmap_only_bsrnn_continue_e18_plus40_a100_gpu0_bs8_20260607" 40
  check_and_start usef 1 "$PLAN/configs/usef_continue_bs8.yaml" "$BASE/usef_only_bsrnn_continue_e18_plus30_a100_gpu1_bs8_20260607" 30
  sleep 300
done
