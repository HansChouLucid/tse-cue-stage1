#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, re, shutil
from pathlib import Path
import numpy as np
import soundfile as sf

def first_path(obj):
    if isinstance(obj, dict): obj=obj.get('default')
    if isinstance(obj, list): return obj[0]
    return obj

def load_manifest(path: Path):
    items={}
    for line in path.open(encoding='utf-8'):
        if line.strip():
            item=json.loads(line); items[item['key']]=item
    return items

def load_mono(path):
    x,sr=sf.read(path, dtype='float32', always_2d=False)
    if x.ndim==2: x=x.mean(axis=1)
    return x.astype(np.float64), sr

def si_sdr(est, ref, eps=1e-8):
    n=min(len(est),len(ref)); est=est[:n].astype(np.float64); ref=ref[:n].astype(np.float64)
    est-=est.mean(); ref-=ref.mean()
    proj=np.sum(est*ref)*ref/(np.sum(ref*ref)+eps); noise=est-proj
    return float(10*np.log10((np.sum(proj*proj)+eps)/(np.sum(noise*noise)+eps)))

def parse_log(log: Path):
    out={}
    pat=re.compile(r'Utt=(.*?) \| Target speaker=(.*?) \| SI-SNR=([-0-9.]+) \| SI-SNRi=([-0-9.]+)')
    for m in pat.finditer(log.read_text(errors='ignore') if log.exists() else ''):
        out[(m.group(1), str(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    return out

def read_scp(audio_dir: Path):
    rows=[]
    for scp in sorted(audio_dir.glob('*.scp')):
        for line in scp.read_text(errors='ignore').splitlines():
            if not line.strip(): continue
            src_rel, wav_path=line.split(maxsplit=1)
            key=Path(src_rel).stem
            m=re.search(r'-T([^-/\\]+)\.wav$', wav_path)
            if not m: continue
            rows.append((key, m.group(1), Path(wav_path)))
    # de-duplicate scp entries if any
    seen={}
    for key,spk,path in rows:
        seen[(key,spk)]=path
    return [(k,s,p) for (k,s),p in sorted(seen.items())]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cue-dir', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    args=ap.parse_args()
    items=load_manifest(args.manifest)
    audio_dir=args.cue_dir/'infer_run'/'audio'
    est_dir=args.cue_dir/'est_wavs'
    if est_dir.exists(): shutil.rmtree(est_dir)
    est_dir.mkdir(parents=True, exist_ok=True)
    log_vals=parse_log(args.cue_dir/'infer_run'/'infer.log')
    scp_rows=read_scp(audio_dir)
    if not scp_rows: raise RuntimeError(f'no scp rows in {audio_dir}')
    meta_path=args.cue_dir/'metadata.jsonl'
    inf_path=args.cue_dir/'inference_summary.csv'
    n_ok=0
    with meta_path.open('w', encoding='utf-8') as mf, inf_path.open('w', newline='', encoding='utf-8') as cf:
        writer=csv.DictWriter(cf, fieldnames=['key','target_spk','interferer_spk','mix_sisnr','est_sisnr','sisnr_i','est_path'])
        writer.writeheader()
        for key, spk, wav_path in scp_rows:
            item=items.get(key)
            if item is None: continue
            spks=[str(x) for x in item['spk']]
            if spk not in spks: continue
            inter=spks[1-spks.index(spk)]
            target_ref=first_path(item['src'][spk]); inter_ref=first_path(item['src'][inter]); mix_path=first_path(item['mix'])
            side=f'T{spks.index(spk)+1}'
            task_key=f'{key}__{side}__NBASIC__{spk}'
            out_wav=est_dir/f'{task_key}.wav'
            shutil.copy2(wav_path,out_wav)
            mix,_=load_mono(mix_path); src,_=load_mono(target_ref); est,_=load_mono(out_wav)
            mix_s=si_sdr(mix,src); est_s=si_sdr(est,src); delta=est_s-mix_s
            if (key,spk) in log_vals:
                est_s,delta=log_vals[(key,spk)]
            meta={'key':task_key,'base_key':key,'target_spk':spk,'interferer_spk':inter,'target_ref':target_ref,'interferer_ref':inter_ref,'mix':{'default':[mix_path]},'src':{spk:[target_ref],inter:[inter_ref]},'content_similarity':'normal_basic'}
            mf.write(json.dumps(meta, ensure_ascii=False)+'\n')
            writer.writerow({'key':task_key,'target_spk':spk,'interferer_spk':inter,'mix_sisnr':f'{mix_s:.6f}','est_sisnr':f'{est_s:.6f}','sisnr_i':f'{delta:.6f}','est_path':str(out_wav)})
            n_ok+=1
    (args.cue_dir/'DONE').write_text(f'done {n_ok}\n')
    print(args.cue_dir.name, 'postprocessed', n_ok)
if __name__=='__main__': main()
