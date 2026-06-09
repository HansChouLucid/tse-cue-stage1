#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse

cd /data/tse_cue_project/repos/wesep-real-tse

LOG_DIR=/data/tse_cue_project/logs
mkdir -p "${LOG_DIR}"

run_with_retry() {
  local name="$1"
  local done_file="$2"
  local log_file="$3"
  shift 3

  if test -f "${done_file}"; then
    echo "[SKIP] ${name} already finished at $(date '+%F %T')" | tee -a "${log_file}"
    return 0
  fi

  local attempt=1
  local max_attempts=2
  while [ "${attempt}" -le "${max_attempts}" ]; do
    echo "[RUN] ${name} attempt ${attempt} at $(date '+%F %T')" | tee -a "${log_file}"
    if "$@" |& tee -a "${log_file}"; then
      if test -f "${done_file}"; then
        echo "[DONE] ${name} at $(date '+%F %T')" | tee -a "${log_file}"
        return 0
      fi
    fi
    echo "[RETRY] ${name} attempt ${attempt} failed at $(date '+%F %T')" | tee -a "${log_file}"
    attempt=$((attempt + 1))
    sleep 30
  done

  echo "[FAIL] ${name} exhausted retries at $(date '+%F %T')" | tee -a "${log_file}"
  return 1
}

echo "[QUEUE START] $(date '+%F %T')" | tee -a "${LOG_DIR}/stage1_backbone_ft13_queue.log"

run_with_retry \
  "usef_ft13_from5" \
  "/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt" \
  "${LOG_DIR}/train_usef_bsrnn_ft13_from5_dualA100_bg.log" \
  ./tools/run_usef_train_ft13_from5_dualA100.sh

run_with_retry \
  "context_ft13_from5" \
  "/data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt" \
  "${LOG_DIR}/train_context_bsrnn_ft13_from5_dualA100_bg.log" \
  ./tools/run_context_train_ft13_from5_dualA100.sh

run_with_retry \
  "tfmap_ft13_from5" \
  "/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt" \
  "${LOG_DIR}/train_tfmap_bsrnn_ft13_from5_dualA100_bg.log" \
  ./tools/run_tfmap_train_ft13_from5_dualA100.sh

echo "[QUEUE END] $(date '+%F %T')" | tee -a "${LOG_DIR}/stage1_backbone_ft13_queue.log"
