#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from wesep.dataset.collate import AUX_KEY_MAP, BASE_COLLECT_KEYS, build_collect_keys, tse_collate_fn
from wesep.dataset.dataset import Dataset
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.file_utils import load_yaml
from wesep.utils.utils import parse_config_or_kwargs, set_seed


def si_sdr(est, ref):
    n = min(len(est), len(ref))
    est = est[:n].astype(np.float64)
    ref = ref[:n].astype(np.float64)
    est = est - est.mean()
    ref = ref - ref.mean()
    ref_energy = np.sum(ref * ref) + 1e-8
    proj = np.sum(est * ref) * ref / ref_energy
    noise = est - proj
    return float(10.0 * np.log10((np.sum(proj * proj) + 1e-8) / (np.sum(noise * noise) + 1e-8)))


def extract_model_inputs(batch, device):
    mix = batch["wav_mix"].float().to(device)
    target = batch["wav_target"].float().to(device)
    cues = []
    for k in list(AUX_KEY_MAP.values()):
        if k in batch and batch[k] is not None:
            cues.append(batch[k].float().to(device))
    return mix, cues if cues else None, target


def load_metadata(path):
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                meta[row["key"]] = row
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--test-data", type=Path, required=True)
    ap.add_argument("--test-cues", type=Path, required=True)
    ap.add_argument("--metadata-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-rate", type=int, default=16000)
    args = ap.parse_args()

    est_dir = args.out_dir / "est_wav"
    est_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    configs = parse_config_or_kwargs(str(args.config))
    configs["checkpoint"] = str(args.checkpoint)
    configs["test_data"] = str(args.test_data)
    configs["test_cues"] = str(args.test_cues)
    configs["dataset_args"]["whole_utt"] = True
    configs["dataset_args"]["shuffle"] = False
    set_seed(configs["seed"])

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = get_model(configs["model"]["tse_model"])(configs["model_args"]["tse_model"])
    load_pretrained_model(model, str(args.checkpoint))
    model = model.to(device).eval()

    dataset = Dataset(configs["data_type"], str(args.test_data), configs["dataset_args"], state="test", cues_yaml=str(args.test_cues))
    collect_keys = build_collect_keys(load_yaml(str(args.test_cues)), configs["dataset_args"], BASE_COLLECT_KEYS)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: tse_collate_fn(batch, collect_keys))
    meta = load_metadata(args.metadata_jsonl)

    summary_path = args.out_dir / "inference_summary.csv"
    fields = ["key", "mix_path", "ref_path", "aux_path", "est_path", "sample_rate", "num_samples", "mix_sisnr", "est_sisnr", "sisnr_i"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        with torch.no_grad():
            for sample_idx, batch in enumerate(loader):
                if args.max_samples and sample_idx >= args.max_samples:
                    break
                mix, cues, target = extract_model_inputs(batch, device)
                outputs = model(mix, cues) if cues is not None else model(mix)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                if outputs.dim() == 3 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                if target.dim() == 3 and target.size(1) == 1:
                    target = target.squeeze(1)
                if mix.dim() == 3 and mix.size(1) == 1:
                    mix = mix.squeeze(1)

                key = batch["key"][0]
                est = outputs[0].detach().float().cpu().numpy()
                ref = target[0].detach().float().cpu().numpy()
                mix_np = mix[0].detach().float().cpu().numpy() if mix.dim() > 1 else mix.detach().float().cpu().numpy()
                n = min(len(est), len(ref), len(mix_np))
                est, ref, mix_np = est[:n], ref[:n], mix_np[:n]
                out_path = est_dir / f"{key}.wav"
                sf.write(out_path, est, args.sample_rate)
                est_sisnr = si_sdr(est, ref)
                mix_sisnr = si_sdr(mix_np, ref)
                m = meta.get(key, {})
                writer.writerow({
                    "key": key,
                    "mix_path": m.get("mix", ""),
                    "ref_path": m.get("target_ref", ""),
                    "aux_path": m.get("enrollment", ""),
                    "est_path": str(out_path),
                    "sample_rate": args.sample_rate,
                    "num_samples": n,
                    "mix_sisnr": mix_sisnr,
                    "est_sisnr": est_sisnr,
                    "sisnr_i": est_sisnr - mix_sisnr,
                })
                if (sample_idx + 1) % 250 == 0:
                    print(f"processed {sample_idx + 1}", flush=True)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
