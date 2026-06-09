#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse

cd /data/tse_cue_project/repos/wesep-real-tse

LOG_DIR=/data/tse_cue_project/logs
mkdir -p "${LOG_DIR}"

echo "[QUEUE START] $(date '+%F %T')"

if ! test -f /data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_5ep_dualA100_ddp2/models/final.pt; then
  ./tools/run_tfmap_train_dualA100.sh |& tee "${LOG_DIR}/train_tfmap_bsrnn_5ep_dualA100_bg.log"
fi

if ! test -f /data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_5ep_dualA100_ddp2/models/final.pt; then
  ./tools/run_context_train_dualA100.sh |& tee "${LOG_DIR}/train_context_bsrnn_5ep_dualA100_bg.log"
fi

if ! test -f /data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_5ep_dualA100_ddp2/models/final.pt; then
  ./tools/run_usef_train_dualA100.sh |& tee "${LOG_DIR}/train_usef_bsrnn_5ep_dualA100_bg.log"
fi

echo "[QUEUE END] $(date '+%F %T')"
