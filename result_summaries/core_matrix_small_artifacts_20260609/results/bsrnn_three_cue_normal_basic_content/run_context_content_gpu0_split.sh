#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
BASE=/data/tse_cue_project/diagnostics/normal_basic_content_baseline_20260604
PY=/root/miniconda3/envs/tse/bin/python
cue=context_ft13
while [ ! -f "$BASE/bundles/$cue/DONE" ]; do sleep 20; done
CUDA_VISIBLE_DEVICES=0 $PY /data/tse_cue_project/repos/stage1-tools/run_similar_content_ssl_diagnostic.py \
  --metadata-jsonl "$BASE/bundles/$cue/metadata.jsonl" --est-dir "$BASE/bundles/$cue/est_wavs" --inference-summary "$BASE/bundles/$cue/inference_summary.csv" --out-dir "$BASE/content_ssl/$cue" --device cuda --batch-size 96
CUDA_VISIBLE_DEVICES=0 $PY /data/tse_cue_project/repos/stage1-tools/run_ssl_content_diagnostic_bundle.py \
  --metadata-jsonl "$BASE/bundles/$cue/metadata.jsonl" --est-dir "$BASE/bundles/$cue/est_wavs" --inference-summary "$BASE/bundles/$cue/inference_summary.csv" --out-dir "$BASE/content_hubert/$cue" --bundle HUBERT_BASE --layer-idx 6 --device cuda --batch-size 48
