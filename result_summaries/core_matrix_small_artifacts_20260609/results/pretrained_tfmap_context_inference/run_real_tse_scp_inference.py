#!/usr/bin/env python3
import argparse, csv, json, math, statistics, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

REPO = Path('/data/tse_cue_project/repos/wesep-real-tse')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from wesep.cli.extractor import load_model_local


def read_scp(path):
    rows=[]
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                k,v=line.rstrip('\n').split(maxsplit=1)
                rows.append((k,v))
    return rows


def load_audio(path, sr):
    wav, in_sr = torchaudio.load(path)
    wav = wav.float()
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if in_sr != sr:
        wav = torchaudio.transforms.Resample(in_sr, sr)(wav)
    return wav.squeeze(0).numpy()


def si_snr(est, ref, remove_dc=True):
    est=np.asarray(est,dtype=np.float64); ref=np.asarray(ref,dtype=np.float64)
    n=min(est.size, ref.size); est=est[:n]; ref=ref[:n]
    if n == 0: return float('nan')
    if remove_dc:
        est=est-np.mean(est); ref=ref-np.mean(ref)
    target=np.sum(est*ref)*ref/(np.sum(ref*ref)+1e-12)
    noise=est-target
    return float(20*np.log10((np.linalg.norm(target)+1e-12)/(np.linalg.norm(noise)+1e-12)))


def si_sdr(est, ref):
    # Same scale-invariant projection as SI-SDR convention; kept separately for reporting consistency.
    return si_snr(est, ref, remove_dc=True)


def summarize(vals):
    vals=[float(v) for v in vals if not math.isnan(float(v))]
    if not vals:
        return {'n':0}
    arr=np.asarray(vals, dtype=np.float64)
    return {
        'n': int(arr.size),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'p10': float(np.percentile(arr,10)),
        'p90': float(np.percentile(arr,90)),
        'lt0_rate': float((arr<0).mean()),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--scp-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no-output-norm', action='store_true')
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    est_dir=out/'est_wavs'; est_dir.mkdir(parents=True, exist_ok=True)
    model=load_model_local(args.model_dir)
    model.set_device(args.device)
    model.set_vad(False)
    model.set_output_norm(not args.no_output_norm)
    sr=model.resample_rate

    scp=Path(args.scp_dir)
    mix=read_scp(scp/'mix.scp')
    ref=dict(read_scp(scp/'ref.scp'))
    aux=dict(read_scp(scp/'aux.scp'))
    interferer=dict(read_scp(scp/'interferer.scp')) if (scp/'interferer.scp').exists() else {}
    sel=[(k,p) for i,(k,p) in enumerate(mix) if i % args.num_shards == args.shard_index]
    if args.limit:
        sel=sel[:args.limit]

    fields=['key','mix_path','target_path','interferer_path','aux_path','est_path','sample_rate','num_samples',
            'si_snr_mix','si_snr_est','si_snri','si_sdr_mix','si_sdr_est','si_sdri','error']
    rows=[]
    csv_path=out/f'inference_summary_shard{args.shard_index}.csv'
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader()
        for idx,(key,mix_path) in enumerate(sel,1):
            row={name:'' for name in fields}
            row.update({'key':key,'mix_path':mix_path,'target_path':ref.get(key,''),'interferer_path':interferer.get(key,''),'aux_path':aux.get(key,''),'sample_rate':sr})
            try:
                speech=model.extract_speech(mix_path, aux[key])
                if speech is None:
                    raise RuntimeError('extract_speech returned None')
                est=speech[0].detach().cpu().numpy()
                mix_w=load_audio(mix_path, sr); tar_w=load_audio(ref[key], sr)
                n=min(est.size, mix_w.size, tar_w.size)
                est=est[:n]; mix_c=mix_w[:n]; tar_c=tar_w[:n]
                est_path=est_dir/f'{key}.wav'
                sf.write(str(est_path), est, sr)
                snr_mix=si_snr(mix_c, tar_c); snr_est=si_snr(est, tar_c)
                sdr_mix=si_sdr(mix_c, tar_c); sdr_est=si_sdr(est, tar_c)
                row.update({'est_path':str(est_path),'num_samples':n,
                            'si_snr_mix':snr_mix,'si_snr_est':snr_est,'si_snri':snr_est-snr_mix,
                            'si_sdr_mix':sdr_mix,'si_sdr_est':sdr_est,'si_sdri':sdr_est-sdr_mix,'error':''})
            except Exception as e:
                row.update({'est_path':'','num_samples':0,'error':repr(e)})
            wr.writerow(row); f.flush(); rows.append(row)
            if idx % 25 == 0:
                print(f'done {idx}/{len(sel)} shard={args.shard_index}', flush=True)
    metric_summary={}
    for col in ['si_snr_est','si_snri','si_sdr_est','si_sdri']:
        metric_summary[col]=summarize([r[col] for r in rows if r.get(col) not in ('', None)])
    metric_summary['errors']=sum(1 for r in rows if r.get('error'))
    metric_summary['total_rows']=len(rows)
    (out/f'metric_summary_shard{args.shard_index}.json').write_text(json.dumps(metric_summary,indent=2),encoding='utf-8')
    print(json.dumps(metric_summary, indent=2), flush=True)

if __name__ == '__main__':
    main()

