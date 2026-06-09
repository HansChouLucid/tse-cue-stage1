#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse
ROOT=/data/tse_cue_project
OUT=$ROOT/diagnostics/similar_content_v3_quality_first_full_pipeline_20260609
WORK=$ROOT/diagnostics/similar_content_v3_tts_enroll_conflict_quality_first_20260609
IN=$OUT/quality_tts_audio_manifest.asr_input.csv
ASR=$OUT/full_whisper_large_v3
MODEL=$ROOT/pretrained/faster_whisper_large_v3_aria2
mkdir -p "$ASR" "$OUT/logs"
python - "$IN" "$ASR" <<'PY'
import csv, sys
from pathlib import Path
inp=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
with inp.open(encoding='utf-8', newline='') as f:
    rows=list(csv.DictReader(f)); fields=list(rows[0].keys())
for shard in [0,1]:
    shard_rows=[r for i,r in enumerate(rows) if i % 2 == shard]
    p=out/f'asr_input_shard{shard}.csv'
    with p.open('w', encoding='utf-8', newline='') as g:
        w=csv.DictWriter(g, fieldnames=fields); w.writeheader(); w.writerows(shard_rows)
    print(p, len(shard_rows), flush=True)
PY
CUDA_VISIBLE_DEVICES=0 python "$WORK/scripts/whisper_smoke_csv.py" --csv "$ASR/asr_input_shard0.csv" --model "$MODEL" --out-dir "$ASR/shard0" --device cuda --compute-type float16 --beam-size 5 > "$OUT/logs/full_whisper_large_v3_shard0.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python "$WORK/scripts/whisper_smoke_csv.py" --csv "$ASR/asr_input_shard1.csv" --model "$MODEL" --out-dir "$ASR/shard1" --device cuda --compute-type float16 --beam-size 5 > "$OUT/logs/full_whisper_large_v3_shard1.log" 2>&1 &
p1=$!
wait $p0 $p1
python - "$ASR" <<'PY'
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
base=Path(sys.argv[1])
parts=[]
for shard in [0,1]:
    p=base/f'shard{shard}/quality_smoke40_whisper.csv'
    parts.append(pd.read_csv(p))
df=pd.concat(parts, ignore_index=True).sort_values('key')
df.to_csv(base/'quality_smoke40_whisper.csv', index=False)
def summ(xs):
    vals=[]
    for x in xs:
        try:
            v=float(x)
            if not math.isnan(v): vals.append(v)
        except Exception:
            pass
    arr=np.asarray(vals, dtype=float)
    return {'n':int(len(arr)), 'mean':float(arr.mean()), 'median':float(np.median(arr)), 'p25':float(np.percentile(arr,25)), 'p75':float(np.percentile(arr,75)), 'min':float(arr.min()), 'max':float(arr.max())} if len(arr) else {'n':0}
summary={'n':int(len(df)), 'wer_to_gen':summ(df['wer_to_gen']), 'cer_to_gen':summ(df['cer_to_gen']), 'token_f1_to_gen':summ(df['token_f1_to_gen']), 'pass_asr_rate':float(df['pass_asr'].mean()), 'pass_audio_rate':float(df['pass_audio'].astype(str).str.lower().eq('true').mean()), 'pass_audio_and_asr_rate':float((df['pass_audio'].astype(str).str.lower().eq('true') & df['pass_asr']).mean())}
(base/'quality_smoke40_whisper_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary, indent=2), flush=True)
PY
