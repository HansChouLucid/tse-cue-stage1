#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


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

STOPWORDS = {
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BUT", "BY", "FOR",
    "FROM", "HAD", "HAS", "HAVE", "HE", "HER", "HIS", "I", "IN", "IS",
    "IT", "ITS", "ME", "MY", "NOT", "OF", "ON", "OR", "SHE", "THAT",
    "THE", "THEIR", "THEM", "THERE", "THEY", "THIS", "TO", "WAS", "WE",
    "WERE", "WITH", "YOU",
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


def spk_from_relpath(relpath):
    return relpath.split("/")[1]


def normalize_tokens(text):
    tokens = re.findall(r"[A-Z0-9']+", text.upper())
    return [t for t in tokens if t not in STOPWORDS]


def parse_transcripts(librispeech_root):
    utt2text = {}
    for trans_path in librispeech_root.rglob("*.trans.txt"):
        with open(trans_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, _, text = line.partition(" ")
                utt2text[utt_id] = text
    return utt2text


def build_inventory(rows, librispeech_root, utt2text):
    by_utt = {}
    missing = 0
    for row in rows:
        for idx in (1, 2):
            rel = row[f"source_{idx}_path"]
            utt_id = utt_from_relpath(rel)
            path = librispeech_root / rel
            text = utt2text.get(utt_id)
            if not path.is_file() or not text:
                missing += 1
                continue
            tokens = normalize_tokens(text)
            if not tokens:
                missing += 1
                continue
            by_utt[utt_id] = {
                "utt_id": utt_id,
                "spk": spk_from_relpath(rel),
                "path": str(path),
                "text": text,
                "tokens": tokens,
                "token_set": set(tokens),
            }
    return list(by_utt.values()), missing


def build_index(inventory):
    df = Counter()
    for item in inventory:
        df.update(item["token_set"])
    inv = defaultdict(list)
    n = len(inventory)
    for i, item in enumerate(inventory):
        for tok in item["token_set"]:
            # Very common words are weak evidence and make candidate pools huge.
            if df[tok] <= max(50, int(0.05 * n)):
                inv[tok].append(i)
    return inv, df


def text_similarity(a, b, idf):
    set_a = a["token_set"]
    set_b = b["token_set"]
    inter = set_a & set_b
    if not inter:
        return 0.0
    union = set_a | set_b
    jaccard = len(inter) / max(1, len(union))
    weighted_inter = sum(idf[t] for t in inter)
    weighted_union = sum(idf[t] for t in union)
    weighted_jaccard = weighted_inter / max(1e-8, weighted_union)
    length_penalty = min(len(a["tokens"]), len(b["tokens"])) / max(
        len(a["tokens"]), len(b["tokens"]))
    return float((0.35 * jaccard + 0.65 * weighted_jaccard) * length_penalty)


def choose_interferer(target, inventory, inv, idf, rng, candidate_limit):
    counts = Counter()
    for tok in target["token_set"]:
        for idx in inv.get(tok, []):
            cand = inventory[idx]
            if cand["utt_id"] != target["utt_id"] and cand["spk"] != target["spk"]:
                counts[idx] += 1
    if not counts:
        return None
    candidates = [idx for idx, _ in counts.most_common(candidate_limit)]
    scored = []
    for idx in candidates:
        cand = inventory[idx]
        score = text_similarity(target, cand, idf)
        if score > 0:
            scored.append((score, rng.random(), cand))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2], scored[0][0]


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
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(conf, f, sort_keys=False, allow_unicode=True)


def construct_split(args, split, rows, inventory, inv, idf):
    rng = random.Random(args.seed + SPLIT_SEED_OFFSET[split])
    out_manifest = args.out_manifest_root / split
    out_data = args.out_dataset_root / split
    for sub in ("mix_clean", "s1", "s2"):
        (out_data / sub).mkdir(parents=True, exist_ok=True)
    (out_manifest / "cues").mkdir(parents=True, exist_ok=True)

    by_utt = {item["utt_id"]: item for item in inventory}
    spk2utt = defaultdict(list)
    for item in inventory:
        spk2utt[item["spk"]].append({
            "utt_id": item["utt_id"],
            "path": item["path"],
            "text": item["text"],
        })
    audio_resource = {}

    created = 0
    skipped = 0
    enrollment_fallback_self = 0

    def cue_items_for_sample(sample_key, spk, exclude_utt):
        nonlocal enrollment_fallback_self
        items = [item for item in spk2utt.get(spk, [])
                 if item["utt_id"] != exclude_utt]
        if not items:
            items = list(spk2utt.get(spk, []))
            enrollment_fallback_self += 1
        audio_resource[f"{sample_key}::{spk}"] = items

    samples_path = out_manifest / "samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for row_i, row in enumerate(rows):
            if args.max_rows and row_i >= args.max_rows:
                break
            for idx in (1, 2):
                target_rel = row[f"source_{idx}_path"]
                target_utt = utt_from_relpath(target_rel)
                target = by_utt.get(target_utt)
                if not target:
                    skipped += 1
                    continue
                chosen = choose_interferer(
                    target, inventory, inv, idf, rng, args.candidate_limit)
                if not chosen:
                    skipped += 1
                    continue
                interferer, content_score = chosen
                try:
                    target_wav, sr1 = load_mono(target["path"])
                    int_wav, sr2 = load_mono(interferer["path"])
                except Exception:
                    skipped += 1
                    continue
                if sr1 != args.sample_rate or sr2 != args.sample_rate:
                    raise ValueError(f"sample-rate mismatch in {target['path']} / {interferer['path']}")
                mix, target_out, int_out = mix_at_sir(target_wav, int_wav, args.sir_db)

                key = (
                    f"{row['mixture_ID']}__T{idx}__"
                    f"{target['spk']}-{target['utt_id']}__CONTENT__"
                    f"{interferer['spk']}-{interferer['utt_id']}"
                )
                cue_items_for_sample(key, target["spk"], target["utt_id"])
                cue_items_for_sample(key, interferer["spk"],
                                     interferer["utt_id"])
                mix_path = out_data / "mix_clean" / f"{key}.wav"
                s1_path = out_data / "s1" / f"{key}.wav"
                s2_path = out_data / "s2" / f"{key}.wav"
                sf.write(mix_path, mix, args.sample_rate)
                sf.write(s1_path, target_out, args.sample_rate)
                sf.write(s2_path, int_out, args.sample_rate)

                sample = {
                    "key": key,
                    "condition": "similar_content",
                    "content_similarity": content_score,
                    "spk": [target["spk"], interferer["spk"]],
                    "mix": {"default": [str(mix_path)]},
                    "src": {
                        target["spk"]: [str(s1_path)],
                        interferer["spk"]: [str(s2_path)],
                    },
                    "source_meta": {
                        "metadata_mixture_ID": row["mixture_ID"],
                        "target_source": target["path"],
                        "interferer_source": interferer["path"],
                        "target_text": target["text"],
                        "interferer_text": interferer["text"],
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
    write_json(audio_json, dict(audio_resource))
    write_cues_yaml(out_manifest / "cues.yaml", audio_json)
    stats = {
        "split": split,
        "created": created,
        "skipped": skipped,
        "sir_db": args.sir_db,
        "candidate_limit": args.candidate_limit,
        "inventory_size": len(inventory),
        "cue_key_field": "mix_spk_id",
        "enrollment_fallback_self": enrollment_fallback_self,
    }
    write_json(out_manifest / "build_stats.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--out-manifest-root", type=Path, required=True)
    parser.add_argument("--out-dataset-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train-100", "dev", "test"])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--sir-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-limit", type=int, default=300)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1000)
    args = parser.parse_args()

    utt2text = parse_transcripts(args.librispeech_root)
    summary = {}
    for split in args.splits:
        rows = load_rows(args.metadata_root / SPLIT_TO_CSV[split])
        inventory, missing_sources = build_inventory(rows, args.librispeech_root, utt2text)
        inv, df = build_index(inventory)
        n = max(1, len(inventory))
        idf = {tok: float(np.log((n + 1) / (freq + 1)) + 1.0)
               for tok, freq in df.items()}
        stats = construct_split(args, split, rows, inventory, inv, idf)
        stats["missing_source_entries"] = missing_sources
        write_json(args.out_manifest_root / split / "build_stats.json", stats)
        summary[split] = stats
        print(json.dumps(stats, ensure_ascii=False), flush=True)
    write_json(args.out_manifest_root / "build_summary.json", summary)


if __name__ == "__main__":
    main()
