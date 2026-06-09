import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
base=Path(sys.argv[1]); cond=sys.argv[2]
out=base/'speaker_diagnostic'/cond
out.mkdir(parents=True, exist_ok=True)
chunk_parts=[]; utt_parts=[]
for shard in [0,1]:
    sd=base/'speaker_diagnostic'/f'{cond}_shard{shard}'
    chunk_parts.append(pd.read_csv(sd/'speaker_multiverifier_per_chunk_1s.csv'))
    utt_parts.append(pd.read_csv(sd/'speaker_multiverifier_per_utterance_1s.csv'))
chunk=pd.concat(chunk_parts, ignore_index=True).sort_values(['key','chunk_idx'])
utt=pd.concat(utt_parts, ignore_index=True).sort_values('key')
chunk.to_csv(out/'speaker_multiverifier_per_chunk_1s.csv', index=False)
utt.to_csv(out/'speaker_multiverifier_per_utterance_1s.csv', index=False)
def summarize(vals):
    arr=np.asarray(pd.Series(vals).dropna(), dtype=float)
    if arr.size==0: return {'n':0}
    return {'n':int(arr.size),'mean':float(arr.mean()),'median':float(np.median(arr)),'p10':float(np.percentile(arr,10)),'p25':float(np.percentile(arr,25)),'p75':float(np.percentile(arr,75)),'p90':float(np.percentile(arr,90)),'min':float(arr.min()),'max':float(arr.max())}
def bmean(vals):
    return float(pd.Series(vals).astype(bool).mean()) if len(vals) else math.nan
summary={
  'num_utterances': int(len(utt)),
  'num_chunks': int(len(chunk)),
  'chunk_rates': {k:bmean(chunk[k]) for k in ['ecapa_mismatch','xvector_mismatch','mv_speaker_mismatch','waveform_prefers_interferer','mv_true_drift']},
  'utterance_rate_distributions': {c:summarize(utt[c]) for c in ['ecapa_mismatch_rate','xvector_mismatch_rate','mv_speaker_mismatch_rate','waveform_interferer_rate','mv_true_drift_rate','mean_local_target_interferer_si_sdr_gap']},
  'case_level_rates': {
    'any_mv_speaker_mismatch': bmean(utt['mv_speaker_mismatch_rate']>0),
    'any_mv_true_drift': bmean(utt['mv_true_drift_rate']>0),
    'mv_speaker_mismatch_ge20pct': bmean(utt['mv_speaker_mismatch_rate']>=0.2),
    'mv_true_drift_ge20pct': bmean(utt['mv_true_drift_rate']>=0.2),
  },
  'merge_note': 'Merged from two metadata shards; definitions match run_speaker_local_diagnostic_tfmc.py.'
}
(out/'speaker_multiverifier_summary_1s.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, indent=2, ensure_ascii=False))
