#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse

cd /data/tse_cue_project/repos/wesep-real-tse

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[START] $(date '+%F %T')"
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  wesep/bin/train.py \
  --config examples/audio/librimix/confs/tse_bsrnn_spk_usef_train100_5ep.yaml \
  --exp_dir /data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_5ep_dualA100_ddp2 \
  --gpus '[0,1]' \
  --num_avg 1 \
  --data_type raw \
  --train_data /data/tse_cue_project/manifests/librimix_basic_clean/train-100/raw.list \
  --train_cues /data/tse_cue_project/manifests/librimix_basic_clean/train-100/cues.yaml \
  --train_samples /data/tse_cue_project/manifests/librimix_basic_clean/train-100/samples.jsonl \
  --val_data /data/tse_cue_project/manifests/librimix_basic_clean/dev/raw.list \
  --val_cues /data/tse_cue_project/manifests/librimix_basic_clean/dev/cues.yaml \
  --val_samples /data/tse_cue_project/manifests/librimix_basic_clean/dev/samples.jsonl
status=$?
echo "[END] $(date '+%F %T') status=${status}"
exit ${status}
