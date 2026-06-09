#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import soundfile as sf

CONFIGS={
 'usef_ft13': ('examples/audio/librimix/confs/tse_bsrnn_spk_usef_train100_ft13_from5.yaml','/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt'),
 'tfmap_ft13': ('examples/audio/librimix/confs/tse_bsrnn_spk_tfmap_train100_ft13_from5.yaml','/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt'),
 'context_ft13': ('examples/audio/librimix/confs/tse_bsrnn_spk_context_train100_ft13_from5.yaml','/data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/final_checkpoint.pt'),
}

def load_manifest(path: Path):
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def first_path(obj):
    if isinstance(obj, dict): obj=obj.get('default')
    if isinstance(obj, list): return obj[0]
    return obj

def load_mono(path):
    x,sr=sf.read(path, dtype='float32', always_2d=False)
    if x.ndim==2: x=x.mean(axis=1)
    return x.astype(np.float64), sr

def si_sdr(est, ref, eps=1e-8):
    n=min(len(est), len(ref)); est=est[:n].astype(np.float64); ref=ref[:n].astype(np.float64)
    est=est-est.mean(); ref=ref-ref.mean(); den=np.sum(ref*ref)+eps
    proj=np.sum(est*ref)*ref/den; noise=est-proj
    return float(10*np.log10((np.sum(proj*proj)+eps)/(np.sum(noise*noise)+eps)))

def parse_log(log: Path):
    vals={}
    pat=re.compile(r'Num=(\d+) \| Utt=(.*?) \| Target speaker=(.*?) \| SI-SNR=([-0-9.]+) \| SI-SNRi=([-0-9.]+)')
    txt=log.read_text(errors='ignore') if log.exists() else ''
    for m in pat.finditer(txt):
        vals[(int(m.group(1)), m.group(2), str(m.group(3)))] = (float(m.group(4)), float(m.group(5)))
    return vals

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cue', required=True, choices=CONFIGS)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    args=ap.parse_args()
    repo=Path('/data/tse_cue_project/repos/wesep-real-tse')
    config,ckpt=CONFIGS[args.cue]
    cue_dir=args.base/'bundles'/args.cue
    infer_dir=cue_dir/'infer_run'
    est_dir=cue_dir/'est_wavs'
    if (cue_dir/'DONE').exists():
        print(f'[SKIP] {args.cue} bundle already done')
        return
    if infer_dir.exists(): shutil.rmtree(infer_dir)
    infer_dir.mkdir(parents=True, exist_ok=True)
    cmd=[sys.executable,'wesep/bin/infer.py','--config',config,'--fs','16k','--gpus','0','--exp_dir',str(infer_dir),'--data_type','raw','--test_data',str(args.manifest),'--test_cues','/data/tse_cue_project/manifests/librimix_basic_clean/test/cues.yaml','--save_wav','true','--checkpoint',ckpt]
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=args.gpu
    print('[RUN]', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo, env=env, check=True)
    items=load_manifest(args.manifest)
    if est_dir.exists(): shutil.rmtree(est_dir)
    est_dir.mkdir(parents=True, exist_ok=True)
    log_vals=parse_log(infer_dir/'infer.log')
    meta_path=cue_dir/'metadata.jsonl'
    inf_path=cue_dir/'inference_summary.csv'
    with meta_path.open('w', encoding='utf-8') as mf, inf_path.open('w', newline='', encoding='utf-8') as cf:
        writer=csv.DictWriter(cf, fieldnames=['key','target_spk','interferer_spk','mix_sisnr','est_sisnr','sisnr_i','est_path'])
        writer.writeheader()
        for idx,item in enumerate(items):
            key=item['key']; spks=[str(s) for s in item['spk']]
            mix_path=first_path(item['mix']); mix,_=load_mono(mix_path)
            for ti,spk in enumerate(spks):
                num=2*idx+ti+1; inter=spks[1-ti]
                task_key=f'{key}__T{ti+1}__NBASIC__{spk}'
                src_path=first_path(item['src'][spk]); inter_path=first_path(item['src'][inter])
                src,_=load_mono(src_path)
                expected=infer_dir/'audio'/f'Utt{num}-{key}-T{spk}.wav'
                if not expected.exists():
                    matches=list((infer_dir/'audio').glob(f'Utt{num}-*-T{spk}.wav'))
                    if not matches: raise FileNotFoundError(expected)
                    expected=matches[0]
                out_wav=est_dir/f'{task_key}.wav'
                shutil.copy2(expected, out_wav)
                est,_=load_mono(out_wav)
                mix_s=si_sdr(mix,src); est_s=si_sdr(est,src); delta=est_s-mix_s
                log_pair=log_vals.get((num,key,spk))
                if log_pair is not None:
                    est_s_log,delta_log=log_pair
                    est_s,delta=est_s_log,delta_log
                meta={
                    'key':task_key, 'base_key':key, 'target_spk':spk, 'interferer_spk':inter,
                    'target_ref':src_path, 'interferer_ref':inter_path, 'mix':{'default':[mix_path]},
                    'src':{spk:[src_path], inter:[inter_path]}, 'content_similarity':'normal_basic'
                }
                mf.write(json.dumps(meta, ensure_ascii=False)+'\n')
                writer.writerow({'key':task_key,'target_spk':spk,'interferer_spk':inter,'mix_sisnr':f'{mix_s:.6f}','est_sisnr':f'{est_s:.6f}','sisnr_i':f'{delta:.6f}','est_path':str(out_wav)})
    (cue_dir/'DONE').write_text('done\n')
    print('[DONE]', args.cue, cue_dir)
if __name__=='__main__': main()
