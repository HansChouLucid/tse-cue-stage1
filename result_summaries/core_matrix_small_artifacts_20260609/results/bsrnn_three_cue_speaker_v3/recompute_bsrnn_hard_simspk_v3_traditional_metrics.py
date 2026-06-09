#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

BASE=Path('/data/tse_cue_project/diagnostics/bsrnn_ft13_hard_simspk_v3_20260605')
OUTDIR=BASE
CONDS=['sir0db','sirm3db','sirm5db']
CUES=['usef','tfmap','context']

def si_snr(x, s):
    x=np.asarray(x,dtype=np.float64); s=np.asarray(s,dtype=np.float64)
    n=min(x.size,s.size); x=x[:n]; s=s[:n]
    x=x-np.mean(x); s=s-np.mean(s)
    t=np.sum(x*s)*s/(np.sum(s*s)+1e-12)
    e=x-t
    return float(20*np.log10((np.linalg.norm(t)+1e-12)/(np.linalg.norm(e)+1e-12)))

def load(path, sr=16000):
    x, r=sf.read(path, dtype='float32', always_2d=False)
    if getattr(x,'ndim',1)==2:
        x=x.mean(axis=1)
    if r != sr:
        x=librosa.resample(x, orig_sr=r, target_sr=sr)
    return x

def summarize(df, cue, cond):
    r={'cue':cue,'condition':cond,'n':len(df),'unique_keys':df.key.nunique()}
    for col,label in [('mix_sisnr','mixture_si_snr'),('est_sisnr','output_si_snr'),('sisnr_i','si_snri')]:
        s=df[col]
        r[f'{label}_mean']=float(s.mean()); r[f'{label}_median']=float(s.median())
        r[f'{label}_p10']=float(s.quantile(.1)); r[f'{label}_p25']=float(s.quantile(.25)); r[f'{label}_p75']=float(s.quantile(.75)); r[f'{label}_p90']=float(s.quantile(.9))
    r['si_snri_lt0_rate']=float((df.sisnr_i<0).mean())
    r['output_si_snr_lt0_rate']=float((df.est_sisnr<0).mean())
    return r

summary=[]
for cond in CONDS:
  for cue in CUES:
    rows=[]
    for part in ['part0','part1']:
      split=BASE/'splits'/f'{cond}_{part}.jsonl'
      audio_dir=BASE/f'{cue}_{cond}_{part}'/'audio'
      if not audio_dir.exists():
        raise FileNotFoundError(audio_dir)
      samples=[json.loads(l) for l in split.read_text().splitlines() if l.strip()]
      for r in samples:
        key=r['key']; target_spk=str(r['spk'][0])
        mix_path=r['mix']['default'][0]
        target_path=r['src'][target_spk][0]
        matches=list(audio_dir.glob(f'*-{key}-T{target_spk}.wav'))
        if len(matches)!=1:
          matches=list(audio_dir.glob(f'*{key}*T{target_spk}.wav'))
        if len(matches)!=1:
          raise RuntimeError(f'{cue} {cond} {part} key {key}: expected one target est, got {len(matches)}')
        mix=load(mix_path); target=load(target_path); est=load(str(matches[0]))
        n=min(len(mix),len(target),len(est)); mix=mix[:n]; target=target[:n]; est=est[:n]
        mix_s=si_snr(mix,target); est_s=si_snr(est,target)
        rows.append({'condition':cond,'cue':cue.upper() if cue!='tfmap' else 'TFMap','key':key,'target_spk':target_spk,'mix_path':mix_path,'target_path':target_path,'est_path':str(matches[0]),'mix_sisnr':mix_s,'est_sisnr':est_s,'sisnr_i':est_s-mix_s})
        if len(rows)%2000==0:
          print(cond, cue, 'computed', len(rows), flush=True)
    df=pd.DataFrame(rows)
    out=OUTDIR/f'bsrnn_hard_simspk_v3_{cue}_{cond}_traditional_metrics_per_utt.csv'
    df.to_csv(out,index=False)
    summary.append(summarize(df, cue.upper() if cue!='tfmap' else 'TFMap', cond))
summary=pd.DataFrame(summary)
summary.to_csv(OUTDIR/'bsrnn_hard_simspk_v3_traditional_metrics_summary.csv', index=False)
print(summary.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
