#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

def summarize(s):
    arr=np.asarray(pd.Series(s).dropna(), dtype=float)
    if arr.size==0: return {'n':0}
    return {'n':int(arr.size),'mean':float(arr.mean()),'median':float(np.median(arr)),'p10':float(np.percentile(arr,10)),'p25':float(np.percentile(arr,25)),'p75':float(np.percentile(arr,75)),'p90':float(np.percentile(arr,90)),'min':float(arr.min()),'max':float(arr.max())}

def bmean(s): return float(pd.Series(s).astype(bool).mean()) if len(s) else math.nan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--w2v-chunk',type=Path,required=True); ap.add_argument('--w2v-utt',type=Path,required=True)
    ap.add_argument('--wavlm-chunk',type=Path,required=True); ap.add_argument('--wavlm-utt',type=Path,required=True)
    ap.add_argument('--hubert-chunk',type=Path,required=True); ap.add_argument('--hubert-utt',type=Path,required=True)
    ap.add_argument('--inference-summary',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    w2v=pd.read_csv(args.w2v_chunk); wavlm=pd.read_csv(args.wavlm_chunk); hubert=pd.read_csv(args.hubert_chunk)
    keep=['key','chunk_idx','target_spk','interferer_spk','start_sec','end_sec','content_gap','content_mismatch','local_si_sdr_gap','waveform_prefers_interferer']
    chunk=w2v[keep].merge(wavlm[['key','chunk_idx','content_gap','content_mismatch']],on=['key','chunk_idx'],suffixes=('_w2v','_wavlm'))
    chunk=chunk.merge(hubert[['key','chunk_idx','content_gap','content_mismatch']],on=['key','chunk_idx']).rename(columns={'content_gap':'content_gap_hubert','content_mismatch':'content_mismatch_hubert'})
    for c in ['content_mismatch_w2v','content_mismatch_wavlm','content_mismatch_hubert','waveform_prefers_interferer']:
        chunk[c]=chunk[c].astype(bool)
    chunk['two_verifier_content_mismatch']=chunk['content_mismatch_w2v'] & chunk['content_mismatch_wavlm']
    chunk['two_verifier_true_drift']=chunk['two_verifier_content_mismatch'] & chunk['waveform_prefers_interferer']
    chunk['tri_verifier_content_mismatch']=chunk['two_verifier_content_mismatch'] & chunk['content_mismatch_hubert']
    chunk['tri_verifier_true_drift']=chunk['tri_verifier_content_mismatch'] & chunk['waveform_prefers_interferer']
    chunk.to_csv(args.out_dir/'content_multiverifier_per_chunk_1s.csv',index=False)
    rows=[]
    for key,sub in chunk.groupby('key'):
        rows.append({
            'key':key,'target_spk':sub['target_spk'].iloc[0],'interferer_spk':sub['interferer_spk'].iloc[0],'num_chunks':int(len(sub)),
            'waveform_interferer_rate':float(sub['waveform_prefers_interferer'].mean()),
            'wav2vec2_content_mismatch_rate':float(sub['content_mismatch_w2v'].mean()),
            'wavlm_content_mismatch_rate':float(sub['content_mismatch_wavlm'].mean()),
            'hubert_content_mismatch_rate':float(sub['content_mismatch_hubert'].mean()),
            'two_verifier_content_mismatch_rate':float(sub['two_verifier_content_mismatch'].mean()),
            'two_verifier_true_drift_rate':float(sub['two_verifier_true_drift'].mean()),
            'tri_verifier_content_mismatch_rate':float(sub['tri_verifier_content_mismatch'].mean()),
            'tri_verifier_true_drift_rate':float(sub['tri_verifier_true_drift'].mean()),
            'mean_local_target_interferer_si_sdr_gap':float(sub['local_si_sdr_gap'].mean()),
        })
    utt=pd.DataFrame(rows).merge(pd.read_csv(args.inference_summary)[['key','sisnr_i']],on='key',how='left')
    utt.to_csv(args.out_dir/'content_multiverifier_per_utterance_1s.csv',index=False)
    summary={
        'num_utterances':int(len(utt)),'num_chunks':int(len(chunk)),
        'chunk_rates':{
            'wav2vec2_content_mismatch':bmean(chunk['content_mismatch_w2v']),
            'wavlm_content_mismatch':bmean(chunk['content_mismatch_wavlm']),
            'hubert_content_mismatch':bmean(chunk['content_mismatch_hubert']),
            'waveform_prefers_interferer':bmean(chunk['waveform_prefers_interferer']),
            'two_verifier_content_mismatch':bmean(chunk['two_verifier_content_mismatch']),
            'two_verifier_si_sdr_confirmed_drift':bmean(chunk['two_verifier_true_drift']),
            'tri_verifier_content_mismatch':bmean(chunk['tri_verifier_content_mismatch']),
            'tri_verifier_si_sdr_confirmed_drift':bmean(chunk['tri_verifier_true_drift']),
        },
        'utterance_rate_distributions':{c:summarize(utt[c]) for c in ['waveform_interferer_rate','wav2vec2_content_mismatch_rate','wavlm_content_mismatch_rate','hubert_content_mismatch_rate','two_verifier_content_mismatch_rate','two_verifier_true_drift_rate','tri_verifier_content_mismatch_rate','tri_verifier_true_drift_rate','mean_local_target_interferer_si_sdr_gap']},
        'case_level_rates':{
            'any_two_verifier_content_mismatch':bmean(utt['two_verifier_content_mismatch_rate']>0),
            'any_two_verifier_si_sdr_confirmed_drift':bmean(utt['two_verifier_true_drift_rate']>0),
            'any_tri_verifier_content_mismatch':bmean(utt['tri_verifier_content_mismatch_rate']>0),
            'any_tri_verifier_si_sdr_confirmed_drift':bmean(utt['tri_verifier_true_drift_rate']>0),
            'two_verifier_content_mismatch_ge20pct':bmean(utt['two_verifier_content_mismatch_rate']>=0.2),
            'two_verifier_si_sdr_confirmed_drift_ge20pct':bmean(utt['two_verifier_true_drift_rate']>=0.2),
            'tri_verifier_content_mismatch_ge20pct':bmean(utt['tri_verifier_content_mismatch_rate']>=0.2),
            'tri_verifier_si_sdr_confirmed_drift_ge20pct':bmean(utt['tri_verifier_true_drift_rate']>=0.2),
        }
    }
    (args.out_dir/'content_multiverifier_summary_1s.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
