#!/usr/bin/env bash
set -euo pipefail
P0=/data/tse_cue_project/diagnostics/stage2_p0_20260604
if [ -f "$P0/logs/normal_protocol_gpu1.pid" ]; then
  while kill -0 "$(cat $P0/logs/normal_protocol_gpu1.pid)" 2>/dev/null; do sleep 30; done
fi
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
cd /data/tse_cue_project/repos/wesep-real-tse
export CUDA_VISIBLE_DEVICES=1
run_infer () {
  local cue="$1" config="$2" ckpt="$3" part="$4"
  local exp="/data/tse_cue_project/diagnostics/stage2_p0_20260604/simspk_sisnri_full/${cue}_${part}"
  mkdir -p "$exp"
  echo "[RUN] $cue $part $(date '+%F %T')"
  python wesep/bin/infer.py --config "$config" \
    --fs 16k --gpus 0 --exp_dir "$exp" --data_type raw \
    --test_data "/data/tse_cue_project/diagnostics/stage2_p0_20260604/simspk_sisnri_full/splits/samples_${part}.jsonl" \
    --test_cues /data/tse_cue_project/manifests/librimix_similar_speaker_clean/test/cues.yaml \
    --save_wav false --checkpoint "$ckpt"
  echo "[DONE] $cue $part $(date '+%F %T')"
}
run_infer usef examples/audio/librimix/confs/tse_bsrnn_spk_usef_train100_ft13_from5.yaml /data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt part1
run_infer tfmap examples/audio/librimix/confs/tse_bsrnn_spk_tfmap_train100_ft13_from5.yaml /data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt part1
run_infer context examples/audio/librimix/confs/tse_bsrnn_spk_context_train100_ft13_from5.yaml /data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt part1
run_infer tfmap examples/audio/librimix/confs/tse_bsrnn_spk_tfmap_train100_ft13_from5.yaml /data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt part0
