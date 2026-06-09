#!/usr/bin/env bash
set -euo pipefail
PLAN=/data/tse_cue_project/experiments/normal_basic_single_cue/bsrnn_continue_a100_20260607
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
printf '\nProcesses:\n'
pgrep -af 'wesep/bin/train.py|torchrun' || true
for f in "$PLAN"/logs/*.outer.log; do echo "==== $f ===="; tail -40 "$f" 2>/dev/null || true; done
for d in /data/tse_cue_project/experiments/normal_basic_single_cue/*continue_e18*20260607; do [ -d "$d" ] || continue; echo "==== $d/train.log ===="; tail -40 "$d/train.log" 2>/dev/null || true; done
