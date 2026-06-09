#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
BASE=/data/tse_cue_project/diagnostics/pretrained_usef_tfgridnet_hard_simspk_v2_20260605
SRC=/data/tse_cue_project/diagnostics/pretrained_usef_tfgridnet_completion_20260604/src
mkdir -p $BASE/logs $BASE/speaker_diagnostic
run_diag() {
  local name=$1
  local cond=$2
  local bundle=$BASE/bundles/$name
  local out=$BASE/speaker_diagnostic/$cond
  rm -rf "$out"
  mkdir -p "$out"
  echo "[$cond] start $(date)"
  CUDA_VISIBLE_DEVICES=0 python $SRC/run_speaker_local_diagnostic.py \
    --metadata-jsonl $bundle/metadata.jsonl \
    --est-dir $bundle/full_infer/est_wavs \
    --inference-summary $bundle/full_infer/inference_summary.csv \
    --out-dir $out \
    --sample-rate 8000 --chunk-sec 1.0 --hop-ratio 0.5 --min-rms-db -45 --device cuda \
    > $BASE/logs/${cond}_speaker_diagnostic.log 2>&1
  echo "[$cond] done $(date)"
}
run_diag pretrained_usef_tfgridnet_hsimv2_sir0db hard_v2_sir0db
run_diag pretrained_usef_tfgridnet_hsimv2_sirm3db hard_v2_sirm3db
