#!/usr/bin/env python3
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

BASE=Path('/data/tse_cue_project/diagnostics/stage2_p0_20260604/simspk_sisnri_full')
OUTDIR=Path('/data/tse_cue_project/diagnostics/stage2_p0_20260604')

def si_snr(x, s):
    x=np.asarray(x,dtype=np.float64); s=np.asarray(s,dtype=np.float64)
    n=min(x.size,s.size); x=x[:n]; s=s[:n]
    x=x-np.mean(x); s=s-np.mean(s)
    t=np.sum(x*s)*s/(np.sum(s*s)+1e-12)
    e=x-t
    return float(20*np.log10((np.linalg.norm(t)+1e-12)/(np.linalg.norm(e)+1e-12)))

def load(path, sr=16000):
    x, r=sf.read(path, dtype='float32', always_2d=False)
    if getattr(x,'ndim',1)==2: x=x.mean(axis=1)
    if r != sr:
        x=librosa.resample(x, orig_sr=r, target_sr=sr)
    return x

def summarize(df, cue):
    r={'cue':cue,'n':len(df),'unique_keys':df.key.nunique()}
    for col,label in [('mix_sisnr','mixture_si_snr'),('est_sisnr','output_si_snr'),('sisnr_i','si_snri')]:
        s=df[col]
        r[f'{label}_mean']=float(s.mean()); r[f'{label}_median']=float(s.median())
        r[f'{label}_p10']=float(s.quantile(.1)); r[f'{label}_p25']=float(s.quantile(.25)); r[f'{label}_p75']=float(s.quantile(.75))
    r['si_snri_lt0_rate']=float((df.sisnr_i<0).mean())
    r['output_si_snr_lt0_rate']=float((df.est_sisnr<0).mean())
    return r

rows_by_cue={}
for cue in ['usef','tfmap','context']:
    all_rows=[]
    for part in [0,1]:
        samples=[json.loads(l) for l in open(BASE/f'splits/samples_part{part}.jsonl', encoding='utf-8') if l.strip()]
        audio_dir=BASE/f'{cue}_part{part}/audio'
        for i,r in enumerate(samples):
            key=r['key']; target_spk=str(r['spk'][0])
            mix_path=r['mix']['default'][0]
            target_path=r['src'][target_spk][0]
            matches=list(audio_dir.glob(f'*-{key}-T{target_spk}.wav'))
            if len(matches)!=1:
                matches=list(audio_dir.glob(f'*{key}*T{target_spk}.wav'))
            if len(matches)!=1:
                raise RuntimeError(f'{cue} part{part} key {key}: expected one target est, got {len(matches)}')
            est_path=str(matches[0])
            mix=load(mix_path); target=load(target_path); est=load(est_path)
            n=min(len(mix),len(target),len(est)); mix=mix[:n]; target=target[:n]; est=est[:n]
            mix_s=si_snr(mix,target); est_s=si_snr(est,target)
            all_rows.append({'cue':cue.upper() if cue!='tfmap' else 'TFMap','key':key,'target_spk':target_spk,'mix_path':mix_path,'target_path':target_path,'est_path':est_path,'mix_sisnr':mix_s,'est_sisnr':est_s,'sisnr_i':est_s-mix_s})
            if (len(all_rows)%2000)==0:
                print(cue, 'computed', len(all_rows), flush=True)
    df=pd.DataFrame(all_rows)
    df.to_csv(OUTDIR/f'bsrnn_simspk_{cue}_traditional_metrics_per_utt.csv', index=False)
    rows_by_cue[cue]=summarize(df, cue.upper() if cue!='tfmap' else 'TFMap')
summary=pd.DataFrame(list(rows_by_cue.values()))
summary.to_csv(OUTDIR/'bsrnn_simspk_traditional_metrics_summary.csv', index=False)
print(summary.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
