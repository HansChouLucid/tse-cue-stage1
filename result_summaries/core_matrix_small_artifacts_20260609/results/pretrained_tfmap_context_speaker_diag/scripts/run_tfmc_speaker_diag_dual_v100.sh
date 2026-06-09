#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
BASE=/data/tse_cue_project/diagnostics/real_tse_tfmap_context_speaker_diag_20260608
SRC=/data/tse_cue_project/diagnostics/real_tse_tfmap_context_20260607
PY=$BASE/scripts/run_speaker_local_diagnostic_tfmc.py
MERGE=$BASE/scripts/merge_speaker_diag.py
mkdir -p "$BASE/logs" "$BASE/tmp" "$BASE/speaker_diagnostic"

split_meta(){
  local bundle=$1 cond=$2
  /root/miniconda3/envs/tse/bin/python - <<PY
from pathlib import Path
src=Path('$SRC/bundles/$bundle/metadata.jsonl')
rows=[x for x in src.read_text(encoding='utf-8').splitlines() if x.strip()]
for shard in [0,1]:
    out=Path('$BASE/tmp/${cond}_metadata_shard%d.jsonl' % shard)
    out.write_text('\n'.join(rows[shard::2])+'\n', encoding='utf-8')
    print(out, len(rows[shard::2]))
PY
}

run_diag(){
  local gpu=$1 bundle=$2 cond=$3
  local out=$BASE/speaker_diagnostic/$cond
  rm -rf "$out"; mkdir -p "$out"
  echo "[$cond] gpu=$gpu start $(date)"
  CUDA_VISIBLE_DEVICES=$gpu /root/miniconda3/envs/tse/bin/python "$PY" \
    --metadata-jsonl "$SRC/bundles/$bundle/metadata.jsonl" \
    --est-dir "$SRC/bundles/$bundle/full_infer/est_wavs" \
    --inference-summary "$SRC/bundles/$bundle/full_infer/inference_summary.csv" \
    --out-dir "$out" --sample-rate 16000 --chunk-sec 1.0 --hop-ratio 0.5 --min-rms-db -45 --device cuda \
    > "$BASE/logs/${cond}_speaker_diag.log" 2>&1
  echo "[$cond] gpu=$gpu done $(date)"
}

run_shard(){
  local gpu=$1 bundle=$2 cond=$3 shard=$4
  local out=$BASE/speaker_diagnostic/${cond}_shard${shard}
  rm -rf "$out"; mkdir -p "$out"
  echo "[$cond shard$shard] gpu=$gpu start $(date)"
  CUDA_VISIBLE_DEVICES=$gpu /root/miniconda3/envs/tse/bin/python "$PY" \
    --metadata-jsonl "$BASE/tmp/${cond}_metadata_shard${shard}.jsonl" \
    --est-dir "$SRC/bundles/$bundle/full_infer/est_wavs" \
    --inference-summary "$SRC/bundles/$bundle/full_infer/inference_summary.csv" \
    --out-dir "$out" --sample-rate 16000 --chunk-sec 1.0 --hop-ratio 0.5 --min-rms-db -45 --device cuda \
    > "$BASE/logs/${cond}_speaker_diag_shard${shard}.log" 2>&1
  echo "[$cond shard$shard] gpu=$gpu done $(date)"
}

# Stage 1: v3 0dB and -3dB in parallel.
run_diag 0 tfmap_context_hsimv3_sir0db tfmap_context_hsimv3_sir0db & p0=$!
run_diag 1 tfmap_context_hsimv3_sirm3db tfmap_context_hsimv3_sirm3db & p1=$!
wait $p0 $p1

# Stage 2: v3 -5dB split across both GPUs.
split_meta tfmap_context_hsimv3_sirm5db tfmap_context_hsimv3_sirm5db
run_shard 0 tfmap_context_hsimv3_sirm5db tfmap_context_hsimv3_sirm5db 0 & p0=$!
run_shard 1 tfmap_context_hsimv3_sirm5db tfmap_context_hsimv3_sirm5db 1 & p1=$!
wait $p0 $p1
/root/miniconda3/envs/tse/bin/python "$MERGE" "$BASE" tfmap_context_hsimv3_sirm5db > "$BASE/logs/tfmap_context_hsimv3_sirm5db_merge.log" 2>&1

# Stage 3: v2 0dB and -3dB in parallel.
run_diag 0 tfmap_context_hsimv2_sir0db tfmap_context_hsimv2_sir0db & p0=$!
run_diag 1 tfmap_context_hsimv2_sirm3db tfmap_context_hsimv2_sirm3db & p1=$!
wait $p0 $p1

/root/miniconda3/envs/tse/bin/python "$BASE/scripts/summarize_tfmc_speaker_diag.py"
echo "[ALL_DONE] $(date)"
