#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
BASE=/data/tse_cue_project/diagnostics/pretrained_usef_tfgridnet_hard_simspk_v2_20260605
SRC=/data/tse_cue_project/diagnostics/pretrained_usef_tfgridnet_completion_20260604/src
CONFIG=/data/tse_cue_project/diagnostics/pretrained_usef_tfgridnet_completion_20260604/hf/chkpt/USEF-TFGridNet/config.yaml
CKPT=/data/tse_cue_project/diagnostics/pretrained_usef_tfgridnet_completion_20260604/hf/chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar
mkdir -p $BASE/logs
run_bundle() {
  local name=$1
  local bundle=$BASE/bundles/$name
  local out=$bundle/full_infer
  rm -rf "$out"
  mkdir -p "$out"
  echo "[$name] start $(date)"
  (
    CUDA_VISIBLE_DEVICES=0 python $SRC/run_usef_tfgridnet_bundle.py --config $CONFIG --checkpoint $CKPT --scp-dir $bundle/scp --out-dir $out --device cuda:0 --shard-index 0 --num-shards 2
  ) > $BASE/logs/${name}_shard0.log 2>&1 &
  p0=$!
  (
    CUDA_VISIBLE_DEVICES=1 python $SRC/run_usef_tfgridnet_bundle.py --config $CONFIG --checkpoint $CKPT --scp-dir $bundle/scp --out-dir $out --device cuda:0 --shard-index 1 --num-shards 2
  ) > $BASE/logs/${name}_shard1.log 2>&1 &
  p1=$!
  wait $p0 $p1
  python - <<PY
from pathlib import Path
import pandas as pd
base=Path('$out')
df=pd.concat([pd.read_csv(base/f'inference_summary_shard{i}.csv') for i in [0,1]], ignore_index=True).sort_values('key')
df.to_csv(base/'inference_summary.csv', index=False)
print('$name merged', len(df), 'sisnri mean', df.sisnr_i.mean(), 'median', df.sisnr_i.median(), 'lt0', (df.sisnr_i<0).mean())
PY
  echo "[$name] done $(date)"
}
run_bundle pretrained_usef_tfgridnet_hsimv2_sir0db
run_bundle pretrained_usef_tfgridnet_hsimv2_sirm3db
