#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
BASE=/data/tse_cue_project/diagnostics/normal_basic_content_baseline_20260604
PY=/root/miniconda3/envs/tse/bin/python
cue=context_ft13
while [ ! -f "$BASE/bundles/$cue/DONE" ]; do sleep 20; done
CUDA_VISIBLE_DEVICES=1 $PY /data/tse_cue_project/repos/stage1-tools/run_ssl_content_diagnostic_bundle.py \
  --metadata-jsonl "$BASE/bundles/$cue/metadata.jsonl" --est-dir "$BASE/bundles/$cue/est_wavs" --inference-summary "$BASE/bundles/$cue/inference_summary.csv" --out-dir "$BASE/content_wavlm/$cue" --bundle WAVLM_BASE_PLUS --layer-idx 9 --device cuda --batch-size 48
