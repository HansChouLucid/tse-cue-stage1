#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, sys
from collections import OrderedDict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))


def si_snr(x, s, remove_dc=True):
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    end = min(x.size, s.size)
    x = x[:end]
    s = s[:end]
    if remove_dc:
        x = x - np.mean(x)
        s = s - np.mean(s)
    t = np.sum(x * s) * s / (np.sum(s * s) + 1e-12)
    n = x - t
    return float(20 * np.log10((np.linalg.norm(t) + 1e-12) / (np.linalg.norm(n) + 1e-12)))


def read_scp(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            key, wav = line.rstrip('\n').split(maxsplit=1)
            rows.append((key, wav))
    return rows


def load_model(config_path: Path, ckpt_path: Path, device: str):
    with config_path.open('r', encoding='utf-8') as f:
        config = load_hyperpyyaml(f.read())
    model = config['modules']['masknet']
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    sd = OrderedDict((k.replace('module.', '').replace('convolution_', 'convolution_module.'), v) for k, v in ckpt['model_state_dict'].items())
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, int(config.get('sample_rate', 8000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--scp-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    est_dir = args.out_dir / 'est_wavs'
    est_dir.mkdir(exist_ok=True)
    model, fs = load_model(args.config, args.checkpoint, args.device)
    mix = read_scp(args.scp_dir / 'mix.scp')
    ref = dict(read_scp(args.scp_dir / 'ref.scp'))
    aux = dict(read_scp(args.scp_dir / 'aux.scp'))
    selected = [(k, p) for i, (k, p) in enumerate(mix) if i % args.num_shards == args.shard_index]
    if args.limit:
        selected = selected[:args.limit]
    summary_path = args.out_dir / f'inference_summary_shard{args.shard_index}.csv'
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        fieldnames = ['key', 'mix_path', 'target_path', 'aux_path', 'est_path', 'sample_rate', 'num_samples', 'sisnr_mix', 'sisnr_est', 'sisnr_i']
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        with torch.no_grad():
            for key, mix_path in tqdm(selected, desc=f'shard{args.shard_index}'):
                target_path = ref[key]
                aux_path = aux[key]
                mix_wav, _ = librosa.load(mix_path, sr=fs, mono=True)
                tar_wav, _ = librosa.load(target_path, sr=fs, mono=True)
                aux_wav, _ = librosa.load(aux_path, sr=fs, mono=True)
                mt = torch.from_numpy(mix_wav).float().unsqueeze(0).to(args.device)
                at = torch.from_numpy(aux_wav).float().unsqueeze(0).to(args.device)
                est = model(mt, at).squeeze(0).detach().cpu().numpy()
                end = min(est.size, mix_wav.size, tar_wav.size)
                est = est[:end]
                mix_cut = mix_wav[:end]
                tar_cut = tar_wav[:end]
                out_path = est_dir / f'{key}.wav'
                sf.write(str(out_path), est, fs)
                sis_m = si_snr(mix_cut, tar_cut)
                sis_e = si_snr(est, tar_cut)
                wr.writerow({'key': key, 'mix_path': mix_path, 'target_path': target_path, 'aux_path': aux_path, 'est_path': str(out_path), 'sample_rate': fs, 'num_samples': end, 'sisnr_mix': sis_m, 'sisnr_est': sis_e, 'sisnr_i': sis_e - sis_m})
                f.flush()
    print('wrote', summary_path, 'n', len(selected))


if __name__ == '__main__':
    main()
