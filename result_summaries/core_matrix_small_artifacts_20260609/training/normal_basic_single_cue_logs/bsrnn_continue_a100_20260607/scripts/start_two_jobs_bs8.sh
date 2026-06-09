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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nnodes=1 --nproc_per_node=1 wesep/bin/train.py --config "$PLAN/configs/tfmap_continue_bs8.yaml" > "$PLAN/logs/tfmap_continue_bs8.outer.log" 2>&1 &
echo $! > "$PLAN/logs/tfmap_continue_bs8.outer.pid"
CUDA_VISIBLE_DEVICES=1 nohup torchrun --standalone --nnodes=1 --nproc_per_node=1 wesep/bin/train.py --config "$PLAN/configs/usef_continue_bs8.yaml" > "$PLAN/logs/usef_continue_bs8.outer.log" 2>&1 &
echo $! > "$PLAN/logs/usef_continue_bs8.outer.pid"
