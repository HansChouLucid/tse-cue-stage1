#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
name=$1
PHYS_GPU=${2:-1}
OUT=/data/tse_cue_project/diagnostics/similar_content_v3_quality_first_full_pipeline_20260609
TOOLS=/data/tse_cue_project/repos/stage1-tools
MERGE=/data/tse_cue_project/diagnostics/pretrained_usef_tfgridnet_completion_20260604/src/merge_content_2tri.py
bundle=$OUT/bundles/$name
inf=$bundle/full_infer/inference_summary.csv
est=$bundle/full_infer/est_wavs
meta=$bundle/metadata.jsonl
diag=$OUT/content_diagnostic/$name
mkdir -p "$diag" "$OUT/logs"
if [ -f "$diag/DONE" ]; then echo "skip $name DONE"; exit 0; fi
echo "[$(date '+%F %T')] start content diag $name" | tee -a "$OUT/logs/content_diag_gpu1_manager.log"
CUDA_VISIBLE_DEVICES=$PHYS_GPU python "$TOOLS/run_similar_content_ssl_diagnostic.py" --metadata-jsonl "$meta" --est-dir "$est" --inference-summary "$inf" --out-dir "$diag/content_ssl" --sample-rate 8000 --batch-size 96 --device cuda > "$OUT/logs/${name}_content_ssl.gpu1.log" 2>&1
CUDA_VISIBLE_DEVICES=$PHYS_GPU python "$TOOLS/run_ssl_content_diagnostic_bundle.py" --metadata-jsonl "$meta" --est-dir "$est" --inference-summary "$inf" --out-dir "$diag/content_wavlm" --bundle WAVLM_BASE_PLUS --layer-idx 9 --sample-rate 8000 --batch-size 48 --device cuda > "$OUT/logs/${name}_content_wavlm.gpu1.log" 2>&1
CUDA_VISIBLE_DEVICES=$PHYS_GPU python "$TOOLS/run_ssl_content_diagnostic_bundle.py" --metadata-jsonl "$meta" --est-dir "$est" --inference-summary "$inf" --out-dir "$diag/content_hubert" --bundle HUBERT_BASE --layer-idx 6 --sample-rate 8000 --batch-size 48 --device cuda > "$OUT/logs/${name}_content_hubert.gpu1.log" 2>&1
python "$MERGE" --w2v-chunk "$diag/content_ssl/ssl_content_per_chunk_1s.csv" --w2v-utt "$diag/content_ssl/ssl_content_per_utterance_1s.csv" --wavlm-chunk "$diag/content_wavlm/wavlm_base_plus_per_chunk_1s.csv" --wavlm-utt "$diag/content_wavlm/wavlm_base_plus_per_utterance_1s.csv" --hubert-chunk "$diag/content_hubert/hubert_base_per_chunk_1s.csv" --hubert-utt "$diag/content_hubert/hubert_base_per_utterance_1s.csv" --inference-summary "$inf" --out-dir "$diag/content_multiverifier"
date > "$diag/DONE"
echo "[$(date '+%F %T')] done content diag $name" | tee -a "$OUT/logs/content_diag_gpu1_manager.log"
