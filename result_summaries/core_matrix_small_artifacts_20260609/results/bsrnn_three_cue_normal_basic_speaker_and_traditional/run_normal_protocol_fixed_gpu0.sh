#!/usr/bin/env bash
set -euo pipefail
# Wait for previous possibly-running USEF attempt, then run fixed-enroll USEF + Context on GPU0.
P0=/data/tse_cue_project/diagnostics/stage2_p0_20260604
if [ -f "$P0/logs/normal_protocol_gpu0.pid" ]; then
  while kill -0 "$(cat $P0/logs/normal_protocol_gpu0.pid)" 2>/dev/null; do sleep 15; done
fi
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
cd /data/tse_cue_project/repos/wesep-real-tse
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
OUT_ROOT=$P0/normal_basic_multiverifier_protocol_fixed_1k
CACHE_DIR=/data/tse_cue_project/cache/verifier_cache
AUDIO_JSON=$P0/basic_test_fixed_enroll_audio_1k.json
mkdir -p "$OUT_ROOT" "$CACHE_DIR"
run_one () {
  local name="$1" config="$2" ckpt="$3" outdir="$4"
  rm -rf "$outdir.tmp"; mkdir -p "$outdir.tmp"
  echo "[RUN] fixed $name $(date '+%F %T')"
  python tools/run_model_output_mismatch_multiverifier.py --config "$config" --checkpoint "$ckpt" \
    --test-data /data/tse_cue_project/manifests/librimix_basic_clean/test/samples.jsonl \
    --test-cues /data/tse_cue_project/manifests/librimix_basic_clean/test/cues.yaml \
    --audio-json "$AUDIO_JSON" --out-dir "$outdir.tmp" --chunk-sec 1.0 2.0 --max-samples 1000 --cache-dir "$CACHE_DIR"
  rm -rf "$outdir"; mv "$outdir.tmp" "$outdir"
  echo "[DONE] fixed $name $(date '+%F %T')"
}
run_one usef_ft13 examples/audio/librimix/confs/tse_bsrnn_spk_usef_train100_ft13_from5.yaml /data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt "$OUT_ROOT/usef_ft13"
run_one context_ft13 examples/audio/librimix/confs/tse_bsrnn_spk_context_train100_ft13_from5.yaml /data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt "$OUT_ROOT/context_ft13"
