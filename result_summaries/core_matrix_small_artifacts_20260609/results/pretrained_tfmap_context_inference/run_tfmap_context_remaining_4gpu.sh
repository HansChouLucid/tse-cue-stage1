#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
BASE=/data/tse_cue_project/diagnostics/real_tse_tfmap_context_20260607
SCRIPT=$BASE/run_real_tse_scp_inference.py
MODEL=/data/tse_cue_project/pretrained/real_tse_baselines/extracted/tfmap_context_100
mkdir -p "$BASE/logs_4gpu"

run_condition_4gpu() {
  local name=$1
  local scp=$2
  local out=$BASE/bundles/$name/full_infer
  if [ -f "$out/metric_summary.json" ] && [ -f "$out/inference_summary.csv" ]; then
    echo "[$name] already complete, skip"
    return 0
  fi
  rm -rf "$out"
  mkdir -p "$out"
  echo "[$name] start 4gpu $(date) scp=$scp"
  pids=()
  for i in 0 1 2 3; do
    (
      CUDA_VISIBLE_DEVICES=$i python "$SCRIPT" --model-dir "$MODEL" --scp-dir "$scp" --out-dir "$out" --device cuda:0 --shard-index $i --num-shards 4
    ) > "$BASE/logs_4gpu/${name}_shard${i}.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
  python - <<PY
from pathlib import Path
import pandas as pd, json
base=Path('$out')
parts=[pd.read_csv(base/f'inference_summary_shard{i}.csv') for i in range(4)]
df=pd.concat(parts, ignore_index=True).sort_values('key')
df.to_csv(base/'inference_summary.csv', index=False)
summary={'condition':'$name','n':int(len(df)),'errors':int(df['error'].fillna('').astype(str).ne('').sum())}
for c in ['si_snr_est','si_snri','si_sdr_est','si_sdri']:
    s=pd.to_numeric(df[c], errors='coerce').dropna()
    summary[c+'_mean']=float(s.mean()) if len(s) else None
    summary[c+'_median']=float(s.median()) if len(s) else None
    summary[c+'_p10']=float(s.quantile(0.10)) if len(s) else None
    summary[c+'_p90']=float(s.quantile(0.90)) if len(s) else None
    summary[c+'_lt0_rate']=float((s<0).mean()) if len(s) else None
(base/'metric_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary, indent=2))
PY
  echo "[$name] done 4gpu $(date)"
}

run_condition_4gpu tfmap_context_hsimv2_sirm3db /data/tse_cue_project/diagnostics/pretrained_usef_tfgridnet_hard_simspk_v2_20260605/bundles/pretrained_usef_tfgridnet_hsimv2_sirm3db/scp
run_condition_4gpu tfmap_context_hsimv3_sir0db /data/tse_cue_project/diagnostics/pretrained_usef_tfgridnet_hard_simspk_v3_20260605/bundles/pretrained_usef_tfgridnet_hsimv3_sir0db/scp
run_condition_4gpu tfmap_context_hsimv3_sirm3db /data/tse_cue_project/diagnostics/pretrained_usef_tfgridnet_hard_simspk_v3_20260605/bundles/pretrained_usef_tfgridnet_hsimv3_sirm3db/scp
run_condition_4gpu tfmap_context_hsimv3_sirm5db /data/tse_cue_project/diagnostics/pretrained_usef_tfgridnet_hard_simspk_v3_20260605/bundles/pretrained_usef_tfgridnet_hsimv3_sirm5db/scp

python - <<'PY'
from pathlib import Path
import json, pandas as pd
base=Path('/data/tse_cue_project/diagnostics/real_tse_tfmap_context_20260607/bundles')
rows=[]
for p in sorted(base.glob('*/full_infer/metric_summary.json')):
    rows.append(json.loads(p.read_text()))
df=pd.DataFrame(rows)
out=Path('/data/tse_cue_project/diagnostics/real_tse_tfmap_context_20260607/tfmap_context_traditional_metrics_summary.csv')
df.to_csv(out,index=False)
print(df.to_string(index=False))
PY
