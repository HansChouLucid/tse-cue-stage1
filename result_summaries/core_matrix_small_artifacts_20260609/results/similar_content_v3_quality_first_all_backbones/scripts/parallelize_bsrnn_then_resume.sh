#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
ROOT=/data/tse_cue_project
OUT=$ROOT/diagnostics/similar_content_v3_quality_first_full_pipeline_20260609
BUNDLE=$OUT/bundles/quality_first_scv3_content
USEF_OUT=$OUT/bundles/bsrnn_usef_scv3_quality
TFMAP_OUT=$OUT/bundles/bsrnn_tfmap_scv3_quality
CONTEXT_OUT=$OUT/bundles/bsrnn_context_scv3_quality
PY=/root/miniconda3/envs/tse/bin/python
REPO=$ROOT/repos/wesep-real-tse
USEF_CFG=examples/audio/librimix/confs/tse_bsrnn_spk_usef_train100_ft13_from5.yaml
TFMAP_CFG=examples/audio/librimix/confs/tse_bsrnn_spk_tfmap_train100_ft13_from5.yaml
CONTEXT_CFG=examples/audio/librimix/confs/tse_bsrnn_spk_context_train100_ft13_from5.yaml
TFMAP_CKPT=$ROOT/experiments/normal_basic_single_cue/tfmap_only_bsrnn_stage2_A_lr3e6_5ep_a800_gpu0_bs8_20260608/models/checkpoint_4.pt
CONTEXT_CKPT=$ROOT/experiments/normal_basic_single_cue/context_only_bsrnn_continue_from_total17_plus10_lr1e5_dualA100_ddp2_bs4_20260608/models/checkpoint_9.pt
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OUT/logs/parallel_bsrnn_manager.log"; }
postprocess_one(){
  local out=$1
  if [ ! -f "$out/DONE" ]; then
    $PY "$OUT/scripts/postprocess_bsrnn_scv3_bundle.py" --cue-dir "$out" --manifest "$BUNDLE/bsrnn/samples.jsonl" --metadata-template "$BUNDLE/metadata.jsonl" >> "$OUT/logs/parallel_bsrnn_manager.log" 2>&1
  fi
}
run_one(){
  local cue=$1 cfg=$2 ckpt=$3 gpu=$4 out=$5
  if [ -f "$out/DONE" ]; then log "skip $cue DONE"; return; fi
  mkdir -p "$out/infer_run"
  log "start $cue on gpu$gpu"
  (cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu $PY wesep/bin/infer.py --config "$cfg" --fs 16k --gpus 0 \
    --exp_dir "$out/infer_run" --data_type raw --test_data "$BUNDLE/bsrnn/samples.jsonl" \
    --test_cues "$BUNDLE/bsrnn/cues.yaml" --save_wav true --checkpoint "$ckpt" \
    > "$OUT/logs/bsrnn_${cue}.infer.parallel.log" 2>&1)
  log "infer done $cue, postprocess"
  postprocess_one "$out"
  log "done $cue"
}
# Start TFMap immediately on GPU1 if not already running/done.
if [ ! -f "$TFMAP_OUT/DONE" ] && ! pgrep -f "bsrnn_tfmap_scv3_quality/infer_run" >/dev/null; then
  run_one tfmap "$TFMAP_CFG" "$TFMAP_CKPT" 1 "$TFMAP_OUT" &
  echo $! > "$OUT/bsrnn_tfmap_parallel.pid"
fi
# Wait for the original controller's USEF process to finish; if postprocess did not happen, do it here.
log "waiting for USEF inference/postprocess"
while pgrep -f "bsrnn_usef_scv3_quality/infer_run" >/dev/null; do sleep 20; done
sleep 5
postprocess_one "$USEF_OUT"
log "usef DONE ensured"
# Stop original sequential controller before it starts duplicate TFMap/Context.
if [ -f "$OUT/full_pipeline.pid" ] && kill -0 $(cat "$OUT/full_pipeline.pid") 2>/dev/null; then
  kill $(cat "$OUT/full_pipeline.pid") || true
  log "stopped sequential controller $(cat $OUT/full_pipeline.pid)"
fi
# Start Context on GPU0 while TFMap continues/finishes on GPU1.
if [ ! -f "$CONTEXT_OUT/DONE" ]; then
  run_one context "$CONTEXT_CFG" "$CONTEXT_CKPT" 0 "$CONTEXT_OUT" &
  echo $! > "$OUT/bsrnn_context_parallel.pid"
fi
# Wait for TFMap and Context managers.
for pidfile in "$OUT/bsrnn_tfmap_parallel.pid" "$OUT/bsrnn_context_parallel.pid"; do
  if [ -f "$pidfile" ]; then
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then wait "$pid" || exit 1; fi
  fi
done
postprocess_one "$TFMAP_OUT"
postprocess_one "$CONTEXT_OUT"
log "all BSRNN cues DONE; restart main controller for diagnostics"
nohup bash "$OUT/scripts/run_after_tts_full_pipeline.sh" > "$OUT/logs/full_pipeline.nohup.log" 2>&1 &
echo $! > "$OUT/full_pipeline.pid"
nohup $PY "$OUT/scripts/watch_scv3_pipeline.py" > "$OUT/logs/watchdog.nohup.log" 2>&1 &
echo $! > "$OUT/watchdog.pid"
log "restarted controller $(cat $OUT/full_pipeline.pid), watchdog $(cat $OUT/watchdog.pid)"
