#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
cd /data/tse_cue_project/repos/wesep-real-tse
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
OUT_ROOT=/data/tse_cue_project/diagnostics/stage2_p0_20260604/normal_basic_multiverifier_protocol_1k
CACHE_DIR=/data/tse_cue_project/cache/verifier_cache
mkdir -p "$OUT_ROOT" "$CACHE_DIR"
run_one () {
  local name="$1"; local config="$2"; local ckpt="$3"; local outdir="$4"
  rm -rf "$outdir.tmp"; mkdir -p "$outdir.tmp"
  echo "[RUN] $name $(date '+%F %T')"
  python tools/run_model_output_mismatch_multiverifier.py \
    --config "$config" \
    --checkpoint "$ckpt" \
    --test-data /data/tse_cue_project/manifests/librimix_basic_clean/test/samples.jsonl \
    --test-cues /data/tse_cue_project/manifests/librimix_basic_clean/test/cues.yaml \
    --audio-json /data/tse_cue_project/manifests/librimix_basic_clean/test/cues/audio.json \
    --out-dir "$outdir.tmp" \
    --chunk-sec 1.0 2.0 \
    --max-samples 1000 \
    --cache-dir "$CACHE_DIR"
  rm -rf "$outdir"; mv "$outdir.tmp" "$outdir"
  echo "[DONE] $name $(date '+%F %T')"
}
run_one usef_ft13 examples/audio/librimix/confs/tse_bsrnn_spk_usef_train100_ft13_from5.yaml /data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt "$OUT_ROOT/usef_ft13"
run_one context_ft13 examples/audio/librimix/confs/tse_bsrnn_spk_context_train100_ft13_from5.yaml /data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt "$OUT_ROOT/context_ft13"
