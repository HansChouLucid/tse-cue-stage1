#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
BASE=/data/tse_cue_project/diagnostics/bsrnn_ft13_hard_simspk_v3_20260605
SRC=/data/tse_cue_project/diagnostics/pretrained_usef_tfgridnet_completion_20260604/src/run_speaker_local_diagnostic.py
export CUDA_VISIBLE_DEVICES=0
run_one(){ cue=$1; cond=$2; echo "[RUN] gpu0 $cue $cond $(date '+%F %T')"; /root/miniconda3/envs/tse/bin/python $SRC --metadata-jsonl $BASE/bsrnn_hard_simspk_v3_${cond}_speaker_diag_metadata.jsonl --est-dir $BASE/est_by_key/${cue}_${cond} --inference-summary $BASE/bsrnn_hard_simspk_v3_${cue}_${cond}_traditional_metrics_per_utt.csv --out-dir $BASE/speaker_diag/${cue}_${cond} --sample-rate 16000 --chunk-sec 1.0 --hop-ratio 0.5 --device cuda; echo "[DONE] gpu0 $cue $cond $(date '+%F %T')"; }
run_one usef sir0db
run_one context sir0db
run_one tfmap sirm3db
run_one usef sirm5db
run_one context sirm5db
