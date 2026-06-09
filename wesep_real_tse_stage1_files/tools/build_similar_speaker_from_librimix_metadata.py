#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml


SPLIT_TO_CSV = {
    "train-100": "libri2mix_train-clean-100.csv",
    "train-360": "libri2mix_train-clean-360.csv",
    "train-other-500": "libri2mix_train-other-500.csv",
    "dev": "libri2mix_dev-clean.csv",
    "test": "libri2mix_test-clean.csv",
}

SPLIT_SEED_OFFSET = {
    "train-100": 100,
    "train-360": 360,
    "train-other-500": 500,
    "dev": 200,
    "test": 300,
}

DEFAULT_SPEECHBRAIN_SOURCES = {
    "sb_ecapa": "speechbrain/spkrec-ecapa-voxceleb",
    "sb_xvector": "speechbrain/spkrec-xvect-voxceleb",
}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def utt_from_relpath(relpath):
    return Path(relpath).stem


def build_spk2utt(rows, librispeech_root):
    spk2utt = defaultdict(dict)
    missing = 0
    for row in rows:
        for idx in (1, 2):
            rel = row[f"source_{idx}_path"]
            spk = rel.split("/")[1]
            utt_id = utt_from_relpath(rel)
            path = str(librispeech_root / rel)
            if not Path(path).is_file():
                missing += 1
                continue
            spk2utt[spk][utt_id] = {"utt_id": utt_id, "path": path}
    return {spk: list(items.values()) for spk, items in spk2utt.items()}, missing


def load_mono(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav, sr


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
        target = target * gain
        interferer = interferer * gain
        mix = mix * gain
    return mix.astype(np.float32), target.astype(np.float32), interferer.astype(np.float32)


def speaker_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    return conf["model_args"]["tse_model"]["speaker"]["speaker_model"]


class WeSepECAPABackend:
    name = "wesep_ecapa"

    def __init__(self, config_path, device):
        if config_path is None:
            raise ValueError("wesep_ecapa backend requires --config")
        from wesep.modules.speaker.encoder import Fbank_kaldi, SpeakerEncoder

        spk_conf = speaker_config(config_path)
        self.fbank = Fbank_kaldi(**spk_conf["fbank"]).to(device).eval()
        self.encoder = SpeakerEncoder(spk_conf["speaker_encoder"]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed_path(self, path):
        wav, _ = load_mono(path)
        wav_t = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        feat = self.fbank(wav_t)
        emb = self.encoder(feat)
        if isinstance(emb, (tuple, list)):
            emb = emb[-1]
        emb = emb.squeeze(0).detach().float().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)


class SpeechBrainBackend:
    def __init__(self, name, source, savedir, device):
        from speechbrain.inference.speaker import EncoderClassifier

        self.name = name
        self.device = device
        self.classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=str(savedir),
            run_opts={"device": str(device)},
        )

    @torch.no_grad()
    def embed_path(self, path):
        wav, _ = load_mono(path)
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        wav_lens = torch.ones(1, device=self.device)
        emb = self.classifier.encode_batch(wav_t, wav_lens)
        emb = emb.squeeze().detach().float().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)


def build_backends(args, device):
    backends = []
    cache_root = args.feature_root / "_verifier_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    for name in args.speaker_backends:
        if name == "wesep_ecapa":
            backends.append(WeSepECAPABackend(args.config, device))
        elif name in DEFAULT_SPEECHBRAIN_SOURCES:
            backends.append(
                SpeechBrainBackend(
                    name=name,
                    source=DEFAULT_SPEECHBRAIN_SOURCES[name],
                    savedir=cache_root / name,
                    device=device,
                )
            )
        else:
            raise ValueError(f"unknown speaker backend: {name}")
    return backends


def build_backend_prototypes(spk2utt, backend, max_utts):
    prototypes = {}
    embedded = {}
    failed = 0
    for spk in sorted(spk2utt):
        embs = []
        for item in spk2utt[spk]:
            if len(embs) >= max_utts:
                break
            try:
                emb = backend.embed_path(item["path"])
            except Exception:
                failed += 1
                continue
            embs.append(emb)
            embedded.setdefault(spk, []).append(
                {
                    "utt_id": item["utt_id"],
                    "path": item["path"],
                    "embedding": emb.tolist(),
                }
            )
        if not embs:
            continue
        proto = np.mean(np.stack(embs, axis=0), axis=0)
        prototypes[spk] = proto / (np.linalg.norm(proto) + 1e-8)
    return embedded, prototypes, failed


def build_backend_topk(prototypes, topk):
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


def build_fused_topk(prototypes_by_backend, topk):
    backend_names = sorted(prototypes_by_backend)
    common_speakers = None
    for name in backend_names:
        cur = set(prototypes_by_backend[name])
        common_speakers = cur if common_speakers is None else (common_speakers & cur)
    common_speakers = sorted(common_speakers or [])
    rows = {}
    for spk in common_speakers:
        candidates = []
        for other in common_speakers:
            if other == spk:
                continue
            backend_scores = {
                name: float(np.dot(prototypes_by_backend[name][spk], prototypes_by_backend[name][other]))
                for name in backend_names
            }
            score = float(np.mean(list(backend_scores.values())))
            candidates.append(
                {
                    "speaker": other,
                    "score": score,
                    "backend_scores": backend_scores,
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        rows[spk] = candidates[:topk]
    return rows, common_speakers


def choose_interferer(target_spk, topk_rows, spk2utt, rng):
    for item in topk_rows[target_spk]:
        spk = item["speaker"]
        if spk in spk2utt and spk2utt[spk]:
            return spk, item["score"], item["backend_scores"], rng.choice(spk2utt[spk])
    raise RuntimeError(f"no similar interferer found for {target_spk}")


def write_cues_yaml(path, audio_json, key_field="mix_spk_id"):
    conf = {
        "cues": {
            "audio": {
                "type": "raw",
                "guaranteed": True,
                "scope": "speaker",
                "policy": {
                    "type": "random",
                    "key": key_field,
                    "resource": str(audio_json),
                },
            }
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(conf, f, sort_keys=False, allow_unicode=True)


def construct_split(args, split, rows, spk2utt, topk):
    rng = random.Random(args.seed + SPLIT_SEED_OFFSET[split])
    out_manifest = args.out_manifest_root / split
    out_data = args.out_dataset_root / split
    for sub in ("mix_clean", "s1", "s2"):
        (out_data / sub).mkdir(parents=True, exist_ok=True)
    (out_manifest / "cues").mkdir(parents=True, exist_ok=True)

    samples_path = out_manifest / "samples.jsonl"
    audio_resource = {}
    created = 0
    skipped = 0
    enrollment_fallback_self = 0

    def cue_items_for_sample(sample_key, spk, exclude_utt):
        nonlocal enrollment_fallback_self
        items = [item for item in spk2utt.get(spk, []) if item["utt_id"] != exclude_utt]
        if not items:
            items = list(spk2utt.get(spk, []))
            enrollment_fallback_self += 1
        audio_resource[f"{sample_key}::{spk}"] = items

    with open(samples_path, "w", encoding="utf-8") as f:
        for row in rows:
            for idx in (1, 2):
                target_rel = row[f"source_{idx}_path"]
                target_spk = target_rel.split("/")[1]
                target_utt = utt_from_relpath(target_rel)
                target_path = args.librispeech_root / target_rel
                if not target_path.is_file() or target_spk not in topk:
                    skipped += 1
                    continue
                try:
                    int_spk, sim_score, backend_scores, int_item = choose_interferer(
                        target_spk, topk, spk2utt, rng
                    )
                except RuntimeError:
                    skipped += 1
                    continue
                int_path = Path(int_item["path"])
                int_utt = int_item["utt_id"]

                try:
                    target, sr1 = load_mono(target_path)
                    interferer, sr2 = load_mono(int_path)
                except Exception:
                    skipped += 1
                    continue
                if sr1 != args.sample_rate or sr2 != args.sample_rate:
                    raise ValueError(f"sample-rate mismatch in {target_path} / {int_path}")
                mix, target_out, int_out = mix_at_sir(target, interferer, args.sir_db)

                key = f"{target_spk}-{target_utt}__SIM__{int_spk}-{int_utt}"
                if args.add_row_id:
                    key = f"{row['mixture_ID']}__T{idx}__{key}"
                cue_items_for_sample(key, target_spk, target_utt)
                cue_items_for_sample(key, int_spk, int_utt)
                mix_path = out_data / "mix_clean" / f"{key}.wav"
                s1_path = out_data / "s1" / f"{key}.wav"
                s2_path = out_data / "s2" / f"{key}.wav"
                sf.write(mix_path, mix, args.sample_rate)
                sf.write(s1_path, target_out, args.sample_rate)
                sf.write(s2_path, int_out, args.sample_rate)

                sample = {
                    "key": key,
                    "condition": "similar_speaker",
                    "speaker_similarity": sim_score,
                    "speaker_similarity_detail": backend_scores,
                    "speaker_similarity_verifiers": list(args.speaker_backends),
                    "spk": [target_spk, int_spk],
                    "mix": {"default": [str(mix_path)]},
                    "src": {
                        target_spk: [str(s1_path)],
                        int_spk: [str(s2_path)],
                    },
                    "source_meta": {
                        "metadata_mixture_ID": row["mixture_ID"],
                        "target_source": str(target_path),
                        "interferer_source": str(int_path),
                        "sir_db": args.sir_db,
                    },
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                created += 1
                if created % args.log_interval == 0:
                    print(f"{split}: created {created}", flush=True)

    raw_list = out_manifest / "raw.list"
    if raw_list.exists() or raw_list.is_symlink():
        raw_list.unlink()
    raw_list.symlink_to("samples.jsonl")
    audio_json = out_manifest / "cues" / "audio.json"
    write_json(audio_json, audio_resource)
    write_cues_yaml(out_manifest / "cues.yaml", audio_json)
    stats = {
        "split": split,
        "created": created,
        "skipped": skipped,
        "sir_db": args.sir_db,
        "cue_key_field": "mix_spk_id",
        "enrollment_fallback_self": enrollment_fallback_self,
        "speaker_similarity_verifiers": list(args.speaker_backends),
    }
    write_json(out_manifest / "build_stats.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--out-manifest-root", type=Path, required=True)
    parser.add_argument("--out-dataset-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--speaker-backends", nargs="+", default=["sb_ecapa", "sb_xvector"])
    parser.add_argument("--splits", nargs="+", default=["train-100", "dev", "test"])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-utts-per-speaker", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--sir-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--add-row-id", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    backends = build_backends(args, device)

    summary = {
        "speaker_similarity_verifiers": list(args.speaker_backends),
    }
    for split in args.splits:
        rows = load_rows(args.metadata_root / SPLIT_TO_CSV[split])
        spk2utt, missing_sources = build_spk2utt(rows, args.librispeech_root)

        split_feature_root = args.feature_root / split
        split_feature_root.mkdir(parents=True, exist_ok=True)
        write_json(split_feature_root / "speaker_inventory.json", spk2utt)

        prototypes_by_backend = {}
        failed_embedding_sources = {}
        for backend in backends:
            embedded, prototypes, failed = build_backend_prototypes(
                spk2utt, backend, args.max_utts_per_speaker
            )
            prototypes_by_backend[backend.name] = prototypes
            failed_embedding_sources[backend.name] = failed
            write_json(split_feature_root / f"utterance_embeddings_{backend.name}.json", embedded)
            write_json(
                split_feature_root / f"speaker_prototypes_{backend.name}.json",
                {k: v.tolist() for k, v in prototypes.items()},
            )
            write_json(
                split_feature_root / f"speaker_similarity_topk_{backend.name}.json",
                build_backend_topk(prototypes, args.topk),
            )

        topk, common_speakers = build_fused_topk(prototypes_by_backend, args.topk)
        write_json(split_feature_root / "speaker_similarity_topk.json", topk)
        write_json(split_feature_root / "speaker_prototypes_common_speakers.json", common_speakers)

        stats = construct_split(args, split, rows, spk2utt, topk)
        stats["missing_source_entries"] = missing_sources
        stats["failed_embedding_sources"] = failed_embedding_sources
        stats["common_speaker_count"] = len(common_speakers)
        write_json(args.out_manifest_root / split / "build_stats.json", stats)
        summary[split] = stats
        print(json.dumps(stats, ensure_ascii=False), flush=True)
    write_json(args.out_manifest_root / "build_summary.json", summary)


if __name__ == "__main__":
    main()
