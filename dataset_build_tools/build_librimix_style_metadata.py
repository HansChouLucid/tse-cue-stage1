#!/usr/bin/env python3
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def load_librispeech_utts(librispeech_root, subset):
    rows = []
    subset_root = librispeech_root / subset
    for path in sorted(subset_root.rglob("*.flac")):
        utt_id = path.stem
        spk = path.parts[-3]
        rows.append({
            "utt_id": utt_id,
            "spk": spk,
            "relpath": str(path.relative_to(librispeech_root)),
        })
    return rows


def make_pairs(utts, num_pairs, seed):
    rng = random.Random(seed)
    by_spk = defaultdict(list)
    for item in utts:
        by_spk[item["spk"]].append(item)
    speakers = sorted(by_spk)
    pairs = []
    seen = set()
    attempts = 0
    max_attempts = max(10000, num_pairs * 20)
    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        spk1, spk2 = rng.sample(speakers, 2)
        u1 = rng.choice(by_spk[spk1])
        u2 = rng.choice(by_spk[spk2])
        key = tuple(sorted((u1["utt_id"], u2["utt_id"])))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((u1, u2))
    if len(pairs) < num_pairs:
        raise RuntimeError(f"only built {len(pairs)} pairs, requested {num_pairs}")
    return pairs


def write_metadata(pairs, out_csv, out_info_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mixture_ID",
                "source_1_path",
                "source_1_gain",
                "source_2_path",
                "source_2_gain",
                "noise_path",
                "noise_gain",
            ],
        )
        writer.writeheader()
        for u1, u2 in pairs:
            writer.writerow({
                "mixture_ID": f"{u1['utt_id']}_{u2['utt_id']}",
                "source_1_path": u1["relpath"],
                "source_1_gain": 1.0,
                "source_2_path": u2["relpath"],
                "source_2_gain": 1.0,
                "noise_path": "",
                "noise_gain": 0.0,
            })
    with out_info_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mixture_ID",
                "speaker_1_ID",
                "speaker_1_sex",
                "speaker_2_ID",
                "speaker_2_sex",
            ],
        )
        writer.writeheader()
        for u1, u2 in pairs:
            writer.writerow({
                "mixture_ID": f"{u1['utt_id']}_{u2['utt_id']}",
                "speaker_1_ID": u1["spk"],
                "speaker_1_sex": "",
                "speaker_2_ID": u2["spk"],
                "speaker_2_sex": "",
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-root", type=Path, required=True)
    parser.add_argument("--subset", default="train-other-500")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-info-csv", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=74800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    utts = load_librispeech_utts(args.librispeech_root, args.subset)
    if not utts:
        raise RuntimeError(f"no utterances found under {args.librispeech_root / args.subset}")
    pairs = make_pairs(utts, args.num_pairs, args.seed)
    write_metadata(pairs, args.out_csv, args.out_info_csv)
    print({
        "subset": args.subset,
        "utterances": len(utts),
        "pairs": len(pairs),
        "out_csv": str(args.out_csv),
        "out_info_csv": str(args.out_info_csv),
    })


if __name__ == "__main__":
    main()
