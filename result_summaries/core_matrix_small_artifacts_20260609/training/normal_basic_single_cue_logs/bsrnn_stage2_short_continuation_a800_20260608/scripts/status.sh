#!/usr/bin/env bash
set -euo pipefail
BASE=/data/tse_cue_project/experiments/normal_basic_single_cue
PLAN=$BASE/bsrnn_stage2_short_continuation_a800_20260608
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
printf '\nProcesses:\n'
pgrep -af 'wesep/bin/train.py|torchrun|start_context_after_usef' || true
printf '\nLatest train logs:\n'
for d in \
  "$BASE/tfmap_only_bsrnn_stage2_A_lr3e6_5ep_a800_gpu0_bs8_20260608" \
  "$BASE/usef_only_bsrnn_stage2_stabilize_lr2e6_3ep_a800_gpu1_bs8_20260608" \
  "$BASE/context_only_bsrnn_stage2_A_lr5e6_5ep_a800_gpu1_bs4_20260608"; do
  echo "==== $d/train.log ===="
  tail -35 "$d/train.log" 2>/dev/null || true
  echo "---- models ----"
  ls -lh "$d/models" 2>/dev/null || true
done
printf '\nOuter logs:\n'
for f in "$PLAN"/logs/*.outer.log; do echo "==== $f ===="; tail -25 "$f" 2>/dev/null || true; done
