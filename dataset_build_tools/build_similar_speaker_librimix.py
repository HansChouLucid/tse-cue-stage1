#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from wesep.modules.speaker.encoder import Fbank_kaldi, SpeakerEncoder


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_mono(path, sample_rate):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        raise ValueError(f"sample-rate mismatch: {path}, got {sr}, want {sample_rate}")
    return wav


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-8))


def mix_at_sir(target, interferer, sir_db):
    n = min(len(target), len(interferer))
    target = target[:n]
    interferer = interferer[:n]
    scale = rms(target) / (rms(interferer) * (10.0 ** (sir_db / 20.0)))
    interferer = interferer * scale
    mix = target + interferer
    peak = max(float(np.max(np.abs(mix))), 1e-8)
    if peak > 0.99:
        gain = 0.99 / peak
        mix = mix * gain
        target = target * gain
        interferer = interferer * gain
    return mix.astype(np.float32), target.astype(np.float32), interferer.astype(np.float32)


def speaker_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    spk_conf = conf["model_args"]["tse_model"]["speaker"]["speaker_model"]
    return spk_conf


def build_encoder(spk_conf, device):
    fbank = Fbank_kaldi(**spk_conf["fbank"]).to(device).eval()
    encoder = SpeakerEncoder(spk_conf["speaker_encoder"]).to(device).eval()
    return fbank, encoder


@torch.no_grad()
def embed_wav(path, fbank, encoder, sample_rate, device):
    wav = load_mono(path, sample_rate)
    wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
    feat = fbank(wav_t)
    emb = encoder(feat)
    if isinstance(emb, (tuple, list)):
        emb = emb[-1]
    emb = emb.squeeze(0).detach().float().cpu().numpy()
    norm = np.linalg.norm(emb) + 1e-8
    return emb / norm


def build_prototypes(audio_resource, fbank, encoder, sample_rate, device, max_utts):
    utterance_embeddings = {}
    prototypes = {}
    for spk, items in sorted(audio_resource.items()):
        embs = []
        for item in items[:max_utts]:
            emb = embed_wav(item["path"], fbank, encoder, sample_rate, device)
            embs.append(emb)
            utterance_embeddings.setdefault(spk, []).append({
                "utt_id": item.get("utt_id", Path(item["path"]).stem),
                "path": item["path"],
                "embedding": emb.tolist(),
            })
        if embs:
            proto = np.mean(np.stack(embs, axis=0), axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-8)
            prototypes[spk] = proto
    return utterance_embeddings, prototypes


def build_topk(prototypes, topk):
    speakers = sorted(prototypes)
    mat = np.stack([prototypes[s] for s in speakers], axis=0)
    sim = mat @ mat.T
    rows = {}
    for i, spk in enumerate(speakers):
        order = np.argsort(-sim[i])
        rows[spk] = []
        for j in order:
            other = speakers[j]
            if other == spk:
                continue
            rows[spk].append({"speaker": other, "score": float(sim[i, j])})
            if len(rows[spk]) >= topk:
                break
    return rows


def choose_similar(spk, topk, audio_resource, rng):
    candidates = [x["speaker"] for x in topk.get(spk, []) if x["speaker"] in audio_resource]
    if not candidates:
        return None
    return candidates[0]


def choose_audio(spk, audio_resource, rng):
    items = audio_resource[spk]
    return rng.choice(items)


def write_cues_yaml(path, audio_json):
    text = {
        "cues": {
            "audio": {
                "type": "raw",
                "guaranteed": True,
                "scope": "speaker",
                "policy": {
                    "type": "random",
                    "key": "spk_id",
                    "resource": str(audio_json),
                },
            }
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(text, f, sort_keys=False, allow_unicode=True)


def construct_split(args, split, audio_resource, topk):
    rng = random.Random(args.seed + hash(split) % 100000)
    src_split = args.manifest_root / split
    out_manifest = args.out_manifest_root / split
    out_data = args.out_dataset_root / split
    for sub in ["mix_clean", "s1", "s2"]:
        (out_data / sub).mkdir(parents=True, exist_ok=True)
    (out_manifest / "cues").mkdir(parents=True, exist_ok=True)

    new_audio_resource = {}
    samples = []
    stats = {"split": split, "created": 0, "skipped": 0, "sir_db": args.sir_db}

    for sample in read_jsonl(src_split / "samples.jsonl"):
        for target_spk in sample["spk"]:
            sim_spk = choose_similar(target_spk, topk, audio_resource, rng)
            if sim_spk is None:
                stats["skipped"] += 1
                continue
            target_path = sample["src"][target_spk][0]
            interferer_item = choose_audio(sim_spk, audio_resource, rng)
            interferer_path = interferer_item["path"]

            target = load_mono(target_path, args.sample_rate)
            interferer = load_mono(interferer_path, args.sample_rate)
            mix, target_out, interferer_out = mix_at_sir(target, interferer, args.sir_db)

            target_utt = Path(target_path).stem
            int_utt = Path(interferer_path).stem
            key = f"{target_spk}-{target_utt}__SIM__{sim_spk}-{int_utt}"
            key = key.replace("/", "_")
            mix_path = out_data / "mix_clean" / f"{key}.wav"
            s1_path = out_data / "s1" / f"{key}.wav"
            s2_path = out_data / "s2" / f"{key}.wav"
            sf.write(mix_path, mix, args.sample_rate)
            sf.write(s1_path, target_out, args.sample_rate)
            sf.write(s2_path, interferer_out, args.sample_rate)

            samples.append({
                "key": key,
                "condition": "similar_speaker",
                "speaker_similarity": topk[target_spk][0]["score"],
                "spk": [target_spk, sim_spk],
                "mix": {"default": [str(mix_path)]},
                "src": {
                    target_spk: [str(s1_path)],
                    sim_spk: [str(s2_path)],
                },
                "source_meta": {
                    "target_source": target_path,
                    "interferer_source": interferer_path,
                    "sir_db": args.sir_db,
                },
            })
            new_audio_resource.setdefault(target_spk, []).append({
                "utt_id": target_utt,
                "path": str(s1_path),
            })
            new_audio_resource.setdefault(sim_spk, []).append({
                "utt_id": int_utt,
                "path": str(s2_path),
            })
            stats["created"] += 1

    with open(out_manifest / "samples.jsonl", "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    raw_list = out_manifest / "raw.list"
    if raw_list.exists() or raw_list.is_symlink():
        raw_list.unlink()
    raw_list.symlink_to("samples.jsonl")
    audio_json = out_manifest / "cues" / "audio.json"
    write_json(audio_json, new_audio_resource)
    write_cues_yaml(out_manifest / "cues.yaml", audio_json)
    write_json(out_manifest / "build_stats.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--out-manifest-root", type=Path, required=True)
    parser.add_argument("--out-dataset-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train-100", "dev", "test"])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-utts-per-speaker", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--sir-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spk_conf = speaker_config(args.config)
    fbank, encoder = build_encoder(spk_conf, device)

    args.feature_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in args.splits:
        audio_json = args.manifest_root / split / "cues" / "audio.json"
        audio_resource = read_json(audio_json)
        utt_embs, prototypes = build_prototypes(
            audio_resource,
            fbank,
            encoder,
            args.sample_rate,
            device,
            args.max_utts_per_speaker,
        )
        topk = build_topk(prototypes, args.topk)
        split_feature_root = args.feature_root / split
        write_json(split_feature_root / "utterance_embeddings.json", utt_embs)
        write_json(
            split_feature_root / "speaker_prototypes.json",
            {k: v.tolist() for k, v in prototypes.items()},
        )
        write_json(split_feature_root / "speaker_similarity_topk.json", topk)
        stats = construct_split(args, split, audio_resource, topk)
        summary[split] = stats
        print(json.dumps(stats, ensure_ascii=False))
    write_json(args.out_manifest_root / "build_summary.json", summary)


if __name__ == "__main__":
    main()
