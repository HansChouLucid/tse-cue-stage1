#!/usr/bin/env bash
set -euo pipefail
P0=/data/tse_cue_project/diagnostics/stage2_p0_20260604
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
cd /data/tse_cue_project/repos/wesep-real-tse
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
OUT_ROOT=$P0/normal_basic_multiverifier_protocol_fixed_1k
CACHE_DIR=/data/tse_cue_project/cache/verifier_cache
AUDIO_JSON=$P0/basic_test_fixed_enroll_audio_1k.json
mkdir -p "$OUT_ROOT" "$CACHE_DIR"
outdir="$OUT_ROOT/tfmap_ft13"
rm -rf "$outdir.tmp"; mkdir -p "$outdir.tmp"
echo "[RUN] fixed tfmap_ft13 $(date '+%F %T')"
python tools/run_model_output_mismatch_multiverifier.py --config examples/audio/librimix/confs/tse_bsrnn_spk_tfmap_train100_ft13_from5.yaml \
  --checkpoint /data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt \
  --test-data /data/tse_cue_project/manifests/librimix_basic_clean/test/samples.jsonl \
  --test-cues /data/tse_cue_project/manifests/librimix_basic_clean/test/cues.yaml \
  --audio-json "$AUDIO_JSON" --out-dir "$outdir.tmp" --chunk-sec 1.0 2.0 --max-samples 1000 --cache-dir "$CACHE_DIR"
rm -rf "$outdir"; mv "$outdir.tmp" "$outdir"
echo "[DONE] fixed tfmap_ft13 $(date '+%F %T')"
