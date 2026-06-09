#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wesep.dataset.collate import AUX_KEY_MAP, BASE_COLLECT_KEYS, build_collect_keys, tse_collate_fn
from wesep.dataset.dataset import Dataset
from wesep.models import get_model
from wesep.modules.speaker.encoder import Fbank_kaldi, SpeakerEncoder
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.file_utils import load_yaml
from wesep.utils.utils import parse_config_or_kwargs, set_seed


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def rms_db(x):
    return 20.0 * math.log10(float(np.sqrt(np.mean(np.square(x)) + 1e-10)) + 1e-10)


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


def chunk_bounds(n, sr, chunk_sec, hop_sec):
    chunk = int(round(chunk_sec * sr))
    hop = int(round(hop_sec * sr))
    if n < chunk:
        return []
    return [(s, s + chunk) for s in range(0, n - chunk + 1, hop)]


def speaker_config(configs):
    return configs["model_args"]["tse_model"]["speaker"]["speaker_model"]


def build_wesep_encoder(spk_conf, device):
    fbank = Fbank_kaldi(**spk_conf["fbank"]).to(device).eval()
    encoder = SpeakerEncoder(spk_conf["speaker_encoder"]).to(device).eval()
    return fbank, encoder


@torch.no_grad()
def embed_wav_array_wesep(wav, fbank, encoder, device):
    wav = wav.astype(np.float32)
    wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
    feat = fbank(wav_t)
    emb = encoder(feat)
    if isinstance(emb, (tuple, list)):
        emb = emb[-1]
    emb = emb.squeeze(0).detach().float().cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-8)


class SpeechBrainBackend:
    def __init__(self, name, source, savedir, device):
        from speechbrain.inference.speaker import EncoderClassifier

        self.name = name
        self.device = device
        if isinstance(device, torch.device) and device.type == "cuda":
            sb_device = f"cuda:{device.index if device.index is not None else 0}"
        else:
            sb_device = str(device)
        self.classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=str(savedir),
            run_opts={"device": sb_device},
        )

    @torch.no_grad()
    def embed_wav_array(self, wav):
        wav = wav.astype(np.float32)
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        wav_lens = torch.ones(1, device=self.device)
        emb = self.classifier.encode_batch(wav_t, wav_lens)
        emb = emb.squeeze().detach().float().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)


def load_audio_resource(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mono_sf(path):
    import soundfile as sf

    wav, _ = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav


def build_proto(audio_resource, sample_key, spk, max_items, backend_name, embed_fn):
    key = f"{sample_key}::{spk}"
    embs = []
    for item in audio_resource.get(key, [])[:max_items]:
        wav = load_mono_sf(item["path"])
        if len(wav) >= 8000:
            embs.append(embed_fn(wav))
    if not embs:
        return None
    proto = np.mean(np.stack(embs, axis=0), axis=0)
    return proto / (np.linalg.norm(proto) + 1e-8)


def extract_model_inputs(batch, device):
    mix = batch["wav_mix"].float().to(device)
    target = batch["wav_target"].float().to(device)
    cues = []
    for k in list(AUX_KEY_MAP.values()):
        if k in batch and batch[k] is not None:
            cues.append(batch[k].float().to(device))
    return mix, cues if cues else None, target


def summarize(vals):
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def count_rate(rows, predicate):
    flags = [1.0 if predicate(r) else 0.0 for r in rows]
    return float(np.mean(flags)) if flags else 0.0


def build_backends(configs, device, cache_dir):
    spk_conf = speaker_config(configs)
    fbank, encoder = build_wesep_encoder(spk_conf, device)
    backends = {
        "wesep_ecapa": lambda wav: embed_wav_array_wesep(wav, fbank, encoder, device),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    sb_ecapa = SpeechBrainBackend(
        "sb_ecapa",
        "speechbrain/spkrec-ecapa-voxceleb",
        cache_dir / "sb_ecapa",
        device,
    )
    sb_xvector = SpeechBrainBackend(
        "sb_xvector",
        "speechbrain/spkrec-xvect-voxceleb",
        cache_dir / "sb_xvector",
        device,
    )
    backends["sb_ecapa"] = sb_ecapa.embed_wav_array
    backends["sb_xvector"] = sb_xvector.embed_wav_array
    return backends


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--test-cues", type=Path, required=True)
    parser.add_argument("--audio-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-sec", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--hop-ratio", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-proto-utts", type=int, default=3)
    parser.add_argument("--min-rms-db", type=float, default=-45.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=Path("exp/verifier_cache"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = parse_config_or_kwargs(str(args.config))
    configs["checkpoint"] = str(args.checkpoint)
    configs["test_data"] = str(args.test_data)
    configs["test_cues"] = str(args.test_cues)
    configs.setdefault("data_type", "raw")
    configs["dataset_args"]["whole_utt"] = True
    configs["dataset_args"]["shuffle"] = False
    set_seed(configs["seed"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = get_model(configs["model"]["tse_model"])(configs["model_args"]["tse_model"])
    load_pretrained_model(model, str(args.checkpoint))
    model = model.to(device).eval()

    backends = build_backends(configs, device, args.cache_dir)
    audio_resource = load_audio_resource(args.audio_json)

    dataset = Dataset(configs["data_type"], str(args.test_data), configs["dataset_args"], state="test", cues_yaml=str(args.test_cues))
    collect_keys = build_collect_keys(load_yaml(str(args.test_cues)), configs["dataset_args"], BASE_COLLECT_KEYS)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: tse_collate_fn(batch, collect_keys))

    per_chunk_path = args.out_dir / "per_chunk_metrics.jsonl"
    per_utt_path = args.out_dir / "per_utterance_summary.csv"
    per_utt_rows = []
    aggregate = {str(c): {
        "local_sisdr_gap": [],
        "ecapa_mismatch_rate": [],
        "mv_mismatch_rate": [],
        "mv_true_swap_rate": [],
        "verifier_instability_rate": [],
    } for c in args.chunk_sec}

    with open(per_chunk_path, "w", encoding="utf-8") as chunk_f:
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

                outputs_np = outputs.detach().float().cpu().numpy()
                target_np = target.detach().float().cpu().numpy()
                keys = batch["key"]
                spks = batch["spk"]
                sample_key = keys[0]

                for i, spk in enumerate(spks):
                    j = 1 - i
                    est = outputs_np[i]
                    tar = target_np[i]
                    inter = target_np[j]
                    n = min(len(est), len(tar), len(inter))
                    est, tar, inter = est[:n], tar[:n], inter[:n]
                    sr = 16000

                    proto_map = {}
                    for backend_name, embed_fn in backends.items():
                        target_proto = build_proto(audio_resource, sample_key, spk, args.max_proto_utts, backend_name, embed_fn)
                        inter_proto = build_proto(audio_resource, sample_key, spks[j], args.max_proto_utts, backend_name, embed_fn)
                        if target_proto is None or inter_proto is None:
                            proto_map = None
                            break
                        proto_map[backend_name] = (target_proto, inter_proto, embed_fn)
                    if proto_map is None:
                        continue

                    utt_si_target = si_sdr(est, tar)
                    utt_si_inter = si_sdr(est, inter)
                    utt_row = {
                        "key": sample_key,
                        "target_spk": spk,
                        "interferer_spk": spks[j],
                        "utterance_si_sdr_target": utt_si_target,
                        "utterance_si_sdr_interferer": utt_si_inter,
                        "utterance_si_sdr_gap": utt_si_target - utt_si_inter,
                    }

                    for chunk_sec in args.chunk_sec:
                        hop_sec = chunk_sec * args.hop_ratio
                        rows = []
                        for ci, (s, e) in enumerate(chunk_bounds(n, sr, chunk_sec, hop_sec)):
                            est_c, tar_c, int_c = est[s:e], tar[s:e], inter[s:e]
                            if rms_db(tar_c) < args.min_rms_db or rms_db(int_c) < args.min_rms_db:
                                continue

                            speaker_scores = {}
                            for backend_name, (target_proto, inter_proto, embed_fn) in proto_map.items():
                                est_emb = embed_fn(est_c)
                                sim_tar = cosine(est_emb, target_proto)
                                sim_int = cosine(est_emb, inter_proto)
                                speaker_scores[backend_name] = {
                                    "target": sim_tar,
                                    "interferer": sim_int,
                                    "gap": sim_tar - sim_int,
                                    "prefer_interferer": sim_tar - sim_int < 0.0,
                                }

                            local_target = si_sdr(est_c, tar_c)
                            local_inter = si_sdr(est_c, int_c)
                            local_gap = local_target - local_inter

                            ecapa_gap = speaker_scores["wesep_ecapa"]["gap"]
                            mv_mismatch = all(v["prefer_interferer"] for v in speaker_scores.values())
                            any_mismatch = any(v["prefer_interferer"] for v in speaker_scores.values())
                            verifier_instability = any_mismatch and (not mv_mismatch)
                            mv_true_swap = mv_mismatch and (local_gap < 0.0)

                            row = {
                                "key": sample_key,
                                "target_spk": spk,
                                "interferer_spk": spks[j],
                                "chunk_sec": chunk_sec,
                                "chunk_idx": ci,
                                "start_sec": s / sr,
                                "end_sec": e / sr,
                                "ecapa_speaker_gap": ecapa_gap,
                                "ecapa_speaker_mismatch": ecapa_gap < 0.0,
                                "mv_speaker_mismatch": mv_mismatch,
                                "verifier_instability": verifier_instability,
                                "local_si_sdr_target": local_target,
                                "local_si_sdr_interferer": local_inter,
                                "local_si_sdr_gap": local_gap,
                                "mv_true_swap": mv_true_swap,
                                "speaker_similarity_detail": speaker_scores,
                            }
                            chunk_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            rows.append(row)

                        if rows:
                            sis = [r["local_si_sdr_gap"] for r in rows]
                            ecapa_rate = count_rate(rows, lambda r: r["ecapa_speaker_mismatch"])
                            mv_rate = count_rate(rows, lambda r: r["mv_speaker_mismatch"])
                            true_swap_rate = count_rate(rows, lambda r: r["mv_true_swap"])
                            instability_rate = count_rate(rows, lambda r: r["verifier_instability"])

                            aggregate[str(chunk_sec)]["local_sisdr_gap"].extend(sis)
                            aggregate[str(chunk_sec)]["ecapa_mismatch_rate"].append(ecapa_rate)
                            aggregate[str(chunk_sec)]["mv_mismatch_rate"].append(mv_rate)
                            aggregate[str(chunk_sec)]["mv_true_swap_rate"].append(true_swap_rate)
                            aggregate[str(chunk_sec)]["verifier_instability_rate"].append(instability_rate)

                            utt_row[f"chunks_{chunk_sec:g}s"] = len(rows)
                            utt_row[f"local_sisdr_gap_mean_{chunk_sec:g}s"] = float(np.mean(sis))
                            utt_row[f"ecapa_mismatch_rate_{chunk_sec:g}s"] = ecapa_rate
                            utt_row[f"mv_mismatch_rate_{chunk_sec:g}s"] = mv_rate
                            utt_row[f"mv_true_swap_rate_{chunk_sec:g}s"] = true_swap_rate
                            utt_row[f"verifier_instability_rate_{chunk_sec:g}s"] = instability_rate

                    per_utt_rows.append(utt_row)

    fieldnames = sorted({k for row in per_utt_rows for k in row})
    with open(per_utt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_utt_rows)

    report = {}
    for c, vals in aggregate.items():
        report[c] = {k: summarize(v) for k, v in vals.items()}
    with open(args.out_dir / "aggregate_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
