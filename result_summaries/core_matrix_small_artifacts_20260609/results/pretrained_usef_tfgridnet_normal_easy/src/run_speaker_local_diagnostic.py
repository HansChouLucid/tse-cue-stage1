#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier


def load_wav(path: str | Path, sr: int) -> np.ndarray:
    import librosa
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return wav.astype(np.float32)

def rms_db(x):
    return 20.0 * math.log10(float(np.sqrt(np.mean(np.square(x)) + 1e-10)) + 1e-10)

def si_sdr(est, ref):
    n=min(len(est),len(ref)); est=est[:n].astype(np.float64); ref=ref[:n].astype(np.float64)
    est=est-est.mean(); ref=ref-ref.mean(); proj=np.sum(est*ref)*ref/(np.sum(ref*ref)+1e-8); noise=est-proj
    return float(10*np.log10((np.sum(proj*proj)+1e-8)/(np.sum(noise*noise)+1e-8)))

def cosine(a,b):
    return float(np.dot(a,b)/((np.linalg.norm(a)+1e-8)*(np.linalg.norm(b)+1e-8)))

def chunks(n,sr,chunk_sec,hop_ratio):
    c=int(round(chunk_sec*sr)); h=int(round(c*hop_ratio))
    return [] if n<c else [(s,s+c) for s in range(0,n-c+1,h)]

@torch.no_grad()
def embed(classifier, wav, device):
    wt=torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)
    lens=torch.ones(1, device=device)
    e=classifier.encode_batch(wt,lens).squeeze().detach().float().cpu().numpy()
    return e/(np.linalg.norm(e)+1e-8)

def summarize(vals):
    arr=np.asarray(pd.Series(vals).dropna(),dtype=float)
    if arr.size==0: return {'n':0}
    return {'n':int(arr.size),'mean':float(arr.mean()),'median':float(np.median(arr)),'p10':float(np.percentile(arr,10)),'p25':float(np.percentile(arr,25)),'p75':float(np.percentile(arr,75)),'p90':float(np.percentile(arr,90)),'min':float(arr.min()),'max':float(arr.max())}

def bmean(vals):
    return float(pd.Series(vals).astype(bool).mean()) if len(vals) else math.nan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--metadata-jsonl',type=Path,required=True); ap.add_argument('--est-dir',type=Path,required=True)
    ap.add_argument('--inference-summary',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--sample-rate',type=int,default=8000); ap.add_argument('--chunk-sec',type=float,default=1.0); ap.add_argument('--hop-ratio',type=float,default=0.5)
    ap.add_argument('--min-rms-db',type=float,default=-45.0); ap.add_argument('--device',default='cuda'); ap.add_argument('--max-samples',type=int,default=0)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    ecapa=EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir=str(args.out_dir/'sb_ecapa_cache'), run_opts={'device':str(device)})
    xvec=EncoderClassifier.from_hparams(source='speechbrain/spkrec-xvect-voxceleb', savedir=str(args.out_dir/'sb_xvector_cache'), run_opts={'device':str(device)})
    rows=[]
    with args.metadata_jsonl.open('r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
            if args.max_samples and len(rows)>=args.max_samples: break
    inf=pd.read_csv(args.inference_summary)
    chunk_f=(args.out_dir/f'speaker_multiverifier_per_chunk_{args.chunk_sec:g}s.csv').open('w',newline='',encoding='utf-8')
    fields=['key','chunk_idx','target_spk','interferer_spk','start_sec','end_sec','ecapa_gap','xvector_gap','ecapa_mismatch','xvector_mismatch','mv_speaker_mismatch','local_si_sdr_gap','waveform_prefers_interferer','mv_true_drift']
    wr=csv.DictWriter(chunk_f,fieldnames=fields); wr.writeheader()
    utt_rows=[]; chunk_flags={k:[] for k in ['ecapa_mismatch','xvector_mismatch','mv_speaker_mismatch','waveform_prefers_interferer','mv_true_drift']}; gap_vals=[]
    for idx,s in enumerate(rows,1):
        key=s['key']; est_path=args.est_dir/f'{key}.wav'
        est=load_wav(est_path,args.sample_rate); tar=load_wav(s['target_ref'],args.sample_rate); inter=load_wav(s['interferer_ref'],args.sample_rate)
        n=min(len(est),len(tar),len(inter)); est=est[:n]; tar=tar[:n]; inter=inter[:n]
        tar_ec=embed(ecapa,tar,device); int_ec=embed(ecapa,inter,device); tar_x=embed(xvec,tar,device); int_x=embed(xvec,inter,device)
        cr=[]
        for ci,(a,b) in enumerate(chunks(n,args.sample_rate,args.chunk_sec,args.hop_ratio)):
            est_c=est[a:b]; tar_c=tar[a:b]; int_c=inter[a:b]
            if rms_db(tar_c)<args.min_rms_db or rms_db(int_c)<args.min_rms_db: continue
            ec=embed(ecapa,est_c,device); xv=embed(xvec,est_c,device)
            ec_gap=cosine(ec,tar_ec)-cosine(ec,int_ec); xv_gap=cosine(xv,tar_x)-cosine(xv,int_x)
            sis_gap=si_sdr(est_c,tar_c)-si_sdr(est_c,int_c)
            row={'key':key,'chunk_idx':ci,'target_spk':s.get('target_spk',''),'interferer_spk':s.get('interferer_spk',''),'start_sec':a/args.sample_rate,'end_sec':b/args.sample_rate,'ecapa_gap':ec_gap,'xvector_gap':xv_gap,'ecapa_mismatch':ec_gap<0,'xvector_mismatch':xv_gap<0,'mv_speaker_mismatch':ec_gap<0 and xv_gap<0,'local_si_sdr_gap':sis_gap,'waveform_prefers_interferer':sis_gap<0,'mv_true_drift':ec_gap<0 and xv_gap<0 and sis_gap<0}
            wr.writerow(row); cr.append(row)
            for k in chunk_flags: chunk_flags[k].append(row[k])
            gap_vals.append(sis_gap)
        if cr:
            utt_rows.append({'key':key,'target_spk':s.get('target_spk',''),'interferer_spk':s.get('interferer_spk',''),'num_chunks':len(cr),'ecapa_mismatch_rate':float(np.mean([r['ecapa_mismatch'] for r in cr])),'xvector_mismatch_rate':float(np.mean([r['xvector_mismatch'] for r in cr])),'mv_speaker_mismatch_rate':float(np.mean([r['mv_speaker_mismatch'] for r in cr])),'waveform_interferer_rate':float(np.mean([r['waveform_prefers_interferer'] for r in cr])),'mv_true_drift_rate':float(np.mean([r['mv_true_drift'] for r in cr])),'mean_local_target_interferer_si_sdr_gap':float(np.mean([r['local_si_sdr_gap'] for r in cr]))})
        if idx%250==0: print(f'processed {idx}/{len(rows)}', flush=True)
    chunk_f.close()
    utt=pd.DataFrame(utt_rows).merge(inf[['key','sisnr_i']],on='key',how='left')
    utt.to_csv(args.out_dir/f'speaker_multiverifier_per_utterance_{args.chunk_sec:g}s.csv',index=False)
    summary={'num_utterances':int(len(utt)),'num_chunks':int(sum(utt['num_chunks']) if len(utt) else 0),'chunk_rates':{k:bmean(v) for k,v in chunk_flags.items()},'utterance_rate_distributions':{c:summarize(utt[c]) for c in ['ecapa_mismatch_rate','xvector_mismatch_rate','mv_speaker_mismatch_rate','waveform_interferer_rate','mv_true_drift_rate','mean_local_target_interferer_si_sdr_gap']},'case_level_rates':{'any_mv_speaker_mismatch':bmean(utt['mv_speaker_mismatch_rate']>0),'any_mv_true_drift':bmean(utt['mv_true_drift_rate']>0),'mv_speaker_mismatch_ge20pct':bmean(utt['mv_speaker_mismatch_rate']>=0.2),'mv_true_drift_ge20pct':bmean(utt['mv_true_drift_rate']>=0.2)}}
    (args.out_dir/f'speaker_multiverifier_summary_{args.chunk_sec:g}s.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
