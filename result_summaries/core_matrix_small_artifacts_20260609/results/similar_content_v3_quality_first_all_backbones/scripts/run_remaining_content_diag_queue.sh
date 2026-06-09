#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
OUT=/data/tse_cue_project/diagnostics/similar_content_v3_quality_first_full_pipeline_20260609
run_one(){
  local gpu=$1 name=$2
  if [ -f "$OUT/content_diagnostic/$name/DONE" ]; then echo "skip $name"; return; fi
  bash "$OUT/scripts/run_one_content_diag_gpu1.sh" "$name" "$gpu" > "$OUT/logs/${name}_content_diag_gpu${gpu}.queue.log" 2>&1
}
(
  run_one 0 pretrained_tfmap_context_scv3_quality
  run_one 0 bsrnn_tfmap_scv3_quality
) & p0=$!
(
  run_one 1 bsrnn_usef_scv3_quality
  run_one 1 bsrnn_context_scv3_quality
) & p1=$!
wait $p0 $p1
bash "$OUT/scripts/run_after_tts_full_pipeline.sh" > "$OUT/logs/final_controller_after_manual_diag.log" 2>&1 || true
