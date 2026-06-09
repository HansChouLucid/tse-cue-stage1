#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


SPLIT_TO_CSV = {
    "train-100": "libri2mix_train-clean-100.csv",
    "train-360": "libri2mix_train-clean-360.csv",
    "dev": "libri2mix_dev-clean.csv",
    "test": "libri2mix_test-clean.csv",
}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_mono(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav, sr


def apply_gain(wav, gain):
    return wav * float(gain)


def ensure_peak(*wavs):
    peak = max(float(np.max(np.abs(w))) for w in wavs)
    if peak > 0.99:
        scale = 0.99 / peak
        return [w * scale for w in wavs]
    return list(wavs)


def utt_id_from_rel(relpath):
    return Path(relpath).stem


def spk_id_from_rel(relpath):
    return relpath.split("/")[1]


def write_random_cues_yaml(path, audio_json):
    conf = {
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
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(conf, f, sort_keys=False, allow_unicode=True)


def write_fixed_cues_yaml(path, fixed_json):
    conf = {
        "cues": {
            "audio": {
                "type": "raw",
                "guaranteed": True,
                "scope": "speaker",
                "policy": {
                    "type": "fixed",
                    "key": "mix_spk_id",
                    "resource": str(fixed_json),
                },
            }
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(conf, f, sort_keys=False, allow_unicode=True)


def build_split(args, split):
    rows = load_rows(args.metadata_root / SPLIT_TO_CSV[split])
    out_manifest = args.out_manifest_root / split
    out_data = args.out_dataset_root / split
    for sub in ("mix_clean", "s1", "s2"):
        (out_data / sub).mkdir(parents=True, exist_ok=True)
    (out_manifest / "cues").mkdir(parents=True, exist_ok=True)

    samples_path = out_manifest / "samples.jsonl"
    spk2utt = defaultdict(list)
    created = 0
    skipped = 0

    with open(samples_path, "w", encoding="utf-8") as f:
        for row in rows:
            key = row["mixture_ID"]
            src1_rel = row["source_1_path"]
            src2_rel = row["source_2_path"]
            spk1 = spk_id_from_rel(src1_rel)
            spk2 = spk_id_from_rel(src2_rel)
            utt1 = utt_id_from_rel(src1_rel)
            utt2 = utt_id_from_rel(src2_rel)
            src1_path = args.librispeech_root / src1_rel
            src2_path = args.librispeech_root / src2_rel
            if not src1_path.is_file() or not src2_path.is_file():
                skipped += 1
                continue
            try:
                wav1, sr1 = load_mono(src1_path)
                wav2, sr2 = load_mono(src2_path)
            except Exception:
                skipped += 1
                continue
            if sr1 != args.sample_rate or sr2 != args.sample_rate:
                raise ValueError(f"sample-rate mismatch: {src1_path} / {src2_path}")

            n = min(len(wav1), len(wav2))
            wav1 = apply_gain(wav1[:n], row["source_1_gain"])
            wav2 = apply_gain(wav2[:n], row["source_2_gain"])
            mix, wav1, wav2 = ensure_peak(wav1 + wav2, wav1, wav2)

            mix_path = out_data / "mix_clean" / f"{key}.wav"
            s1_path = out_data / "s1" / f"{key}.wav"
            s2_path = out_data / "s2" / f"{key}.wav"
            sf.write(mix_path, mix.astype(np.float32), args.sample_rate)
            sf.write(s1_path, wav1.astype(np.float32), args.sample_rate)
            sf.write(s2_path, wav2.astype(np.float32), args.sample_rate)

            sample = {
                "key": key,
                "spk": [spk1, spk2],
                "mix": {"default": [str(mix_path)]},
                "src": {
                    spk1: [str(s1_path)],
                    spk2: [str(s2_path)],
                },
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            spk2utt[spk1].append({"utt_id": utt1, "path": str(s1_path)})
            spk2utt[spk2].append({"utt_id": utt2, "path": str(s2_path)})
            created += 1
            if created % args.log_interval == 0:
                print(f"{split}: created {created}", flush=True)

    raw_list = out_manifest / "raw.list"
    if raw_list.exists() or raw_list.is_symlink():
        raw_list.unlink()
    raw_list.symlink_to("samples.jsonl")

    audio_json = out_manifest / "cues" / "audio.json"
    write_json(audio_json, dict(spk2utt))

    if split == "train-100":
        write_random_cues_yaml(out_manifest / "cues.yaml", audio_json)
    else:
        fixed = {}
        for spk, items in spk2utt.items():
            by_utt = {item["utt_id"]: item for item in items}
            for item in items:
                mix_key = Path(item["path"]).stem
                others = [x for x in items if x["utt_id"] != item["utt_id"]]
                if not others:
                    others = [item]
                fixed[f"{mix_key}::{spk}"] = [others[0]]
        fixed_json = out_manifest / "cues" / "fixed_enroll.json"
        write_json(fixed_json, fixed)
        write_fixed_cues_yaml(out_manifest / "cues.yaml", fixed_json)

    stats = {
        "split": split,
        "created": created,
        "skipped": skipped,
        "num_speakers": len(spk2utt),
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
    parser.add_argument("--log-interval", type=int, default=1000)
    args = parser.parse_args()

    summary = {}
    for split in args.splits:
        stats = build_split(args, split)
        summary[split] = stats
        print(json.dumps(stats, ensure_ascii=False), flush=True)
    write_json(args.out_manifest_root / "build_summary.json", summary)


if __name__ == "__main__":
    main()
