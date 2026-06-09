import json
from pathlib import Path
import pandas as pd
BASE=Path('/data/tse_cue_project/diagnostics/real_tse_tfmap_context_speaker_diag_20260608')
conds=['tfmap_context_hsimv2_sir0db','tfmap_context_hsimv2_sirm3db','tfmap_context_hsimv3_sir0db','tfmap_context_hsimv3_sirm3db','tfmap_context_hsimv3_sirm5db']
rows=[]
for cond in conds:
    p=BASE/'speaker_diagnostic'/cond/'speaker_multiverifier_summary_1s.json'
    if not p.exists():
        continue
    s=json.loads(p.read_text())
    rows.append({
        'condition':cond,
        'num_utterances':s.get('num_utterances'),
        'num_chunks':s.get('num_chunks'),
        'speaker_mismatch_1s':s['chunk_rates'].get('mv_speaker_mismatch'),
        'speaker_confirmed_drift_1s':s['chunk_rates'].get('mv_true_drift'),
        'waveform_prefers_interferer_1s':s['chunk_rates'].get('waveform_prefers_interferer'),
        'any_speaker_mismatch_case':s['case_level_rates'].get('any_mv_speaker_mismatch'),
        'any_confirmed_drift_case':s['case_level_rates'].get('any_mv_true_drift'),
        'persistent_speaker_mismatch_case_ge20':s['case_level_rates'].get('mv_speaker_mismatch_ge20pct'),
        'persistent_confirmed_drift_case_ge20':s['case_level_rates'].get('mv_true_drift_ge20pct'),
        'mean_target_interferer_si_sdr_gap':s['utterance_rate_distributions']['mean_local_target_interferer_si_sdr_gap'].get('mean'),
    })
df=pd.DataFrame(rows)
out=BASE/'tfmap_context_speaker_local_diagnostic_summary.csv'
df.to_csv(out,index=False)
print(df.to_string(index=False))
