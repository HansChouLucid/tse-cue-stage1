#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
cd /data/tse_cue_project/repos/wesep-real-tse
export CUDA_VISIBLE_DEVICES=1
BASE=/data/tse_cue_project/diagnostics/bsrnn_ft13_hard_simspk_v3_20260605
run_infer () {
  local cond="$1" cue="$2" config="$3" ckpt="$4"
  local part="part1"
  local cuefile="/data/tse_cue_project/manifests/librimix_similar_speaker_hard_v3_${cond}/test/cues.yaml"
  local exp="$BASE/${cue}_${cond}_${part}"
  mkdir -p "$exp"
  echo "[RUN] gpu1 $cue $cond $part $(date '+%F %T')"
  python wesep/bin/infer.py --config "$config" \
    --fs 16k --gpus 0 --exp_dir "$exp" --data_type raw \
    --test_data "$BASE/splits/${cond}_${part}.jsonl" \
    --test_cues "$cuefile" \
    --save_wav false --checkpoint "$ckpt" 2>&1 | tee "$BASE/logs/${cue}_${cond}_${part}.log"
  echo "[DONE] gpu1 $cue $cond $part $(date '+%F %T')"
}
USEF_CFG=examples/audio/librimix/confs/tse_bsrnn_spk_usef_train100_ft13_from5.yaml
TFMAP_CFG=examples/audio/librimix/confs/tse_bsrnn_spk_tfmap_train100_ft13_from5.yaml
CONTEXT_CFG=examples/audio/librimix/confs/tse_bsrnn_spk_context_train100_ft13_from5.yaml
USEF_CKPT=/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt
TFMAP_CKPT=/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt
CONTEXT_CKPT=/data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt
for cond in sir0db sirm3db sirm5db; do
  run_infer "$cond" usef "$USEF_CFG" "$USEF_CKPT"
  run_infer "$cond" tfmap "$TFMAP_CFG" "$TFMAP_CKPT"
  run_infer "$cond" context "$CONTEXT_CFG" "$CONTEXT_CKPT"
done
