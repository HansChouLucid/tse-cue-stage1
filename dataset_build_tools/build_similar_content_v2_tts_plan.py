#!/usr/bin/env python3
"""Build a transcript-based similar-content v2 TTS generation plan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SPLIT_TO_CSV = {
    "train-100": "libri2mix_train-clean-100.csv",
    "train-360": "libri2mix_train-clean-360.csv",
    "train-other-500": "libri2mix_train-other-500.csv",
    "dev": "libri2mix_dev-clean.csv",
    "test": "libri2mix_test-clean.csv",
}

SPLIT_TO_LIBRISPEECH = {
    "train-100": "train-clean-100",
    "train-360": "train-clean-360",
    "train-other-500": "train-other-500",
    "dev": "dev-clean",
    "test": "test-clean",
}


def utt_id(rel: str) -> str:
    return Path(rel).stem


def spk_id(rel: str) -> str:
    parts = Path(rel).parts
    return parts[1] if len(parts) > 1 and parts[0].startswith(("train", "dev", "test")) else parts[0]


def load_transcripts(root: Path) -> dict[str, str]:
    out = {}
    for trans in root.rglob("*.trans.txt"):
        with trans.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, _, text = line.partition(" ")
                out[key] = text.strip()
    return out


def collect_spk_utts(root: Path, split: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted((root / split).glob("*/*/*.flac")):
        spk = path.name.split("-")[0]
        out.setdefault(spk, []).append({"utt_id": path.stem, "path": str(path)})
    return out


def pick_prompt(spk_utts: dict[str, list[dict]], spk: str, exclude: str) -> dict | None:
    for item in spk_utts.get(spk, []):
        if item["utt_id"] != exclude:
            return item
    items = spk_utts.get(spk, [])
    return items[0] if items else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--librispeech-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--librispeech-split", default="")
    parser.add_argument("--limit-mixtures", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    synth_dir = args.out_dir / "synthetic_interferer"
    synth_dir.mkdir(parents=True, exist_ok=True)
    librispeech_split = args.librispeech_split or SPLIT_TO_LIBRISPEECH[args.split]
    transcripts = load_transcripts(args.librispeech_root / librispeech_split)
    spk_utts = collect_spk_utts(args.librispeech_root, librispeech_split)

    plan_path = args.out_dir / "tts_plan.jsonl"
    stats = {
        "split": args.split,
        "created": 0,
        "skipped": 0,
        "missing_text": 0,
        "missing_prompt": 0,
        "limit_mixtures": args.limit_mixtures,
    }
    csv_path = args.metadata_root / SPLIT_TO_CSV[args.split]
    with csv_path.open("r", encoding="utf-8") as f, plan_path.open("w", encoding="utf-8") as out:
        for mix_i, row in enumerate(csv.DictReader(f)):
            if args.limit_mixtures and mix_i >= args.limit_mixtures:
                break
            mixture_id = row["mixture_ID"]
            for idx in (1, 2):
                target_rel = row[f"source_{idx}_path"]
                other_idx = 2 if idx == 1 else 1
                inter_rel = row[f"source_{other_idx}_path"]
                target_utt = utt_id(target_rel)
                inter_utt = utt_id(inter_rel)
                text = transcripts.get(target_utt)
                if not text:
                    stats["missing_text"] += 1
                    stats["skipped"] += 1
                    continue
                inter_spk = spk_id(inter_rel)
                prompt = pick_prompt(spk_utts, inter_spk, inter_utt)
                if not prompt:
                    stats["missing_prompt"] += 1
                    stats["skipped"] += 1
                    continue
                target_spk = spk_id(target_rel)
                key = f"{mixture_id}__T{idx}__SCV2TTS__{target_spk}-{target_utt}__VOICE__{inter_spk}"
                synth_path = synth_dir / f"{key}.wav"
                prompt_text = transcripts.get(prompt["utt_id"], "")
                record = {
                    "key": key,
                    "mixture_id": mixture_id,
                    "target_source": str(args.librispeech_root / target_rel),
                    "target_spk": target_spk,
                    "target_utt": target_utt,
                    "target_text": text,
                    "original_interferer_source": str(args.librispeech_root / inter_rel),
                    "interferer_spk": inter_spk,
                    "interferer_utt": inter_utt,
                    "prompt_audio": prompt["path"],
                    "prompt_utt": prompt["utt_id"],
                    "prompt_text": prompt_text,
                    "synthetic_interferer": str(synth_path),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["created"] += 1

    with (args.out_dir / "build_plan_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
