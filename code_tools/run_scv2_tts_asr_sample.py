#!/usr/bin/env python3
"""ASR text sanity for similar-content v2 TTS samples."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r = normalize(ref).split()
    h = normalize(hyp).split()
    if not r:
        return float("nan")
    return edit_distance(r, h) / len(r)


def cer(ref: str, hyp: str) -> float:
    r = list(normalize(ref).replace(" ", ""))
    h = list(normalize(hyp).replace(" ", ""))
    if not r:
        return float("nan")
    return edit_distance(r, h) / len(r)


def load_wav(path: str | Path, sr: int) -> torch.Tensor:
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return torch.from_numpy(wav.astype(np.float32))


class GreedyCTC:
    def __init__(self, labels: list[str]):
        self.labels = labels

    def __call__(self, emission: torch.Tensor) -> str:
        ids = torch.argmax(emission, dim=-1).cpu().numpy().tolist()
        prev = None
        toks = []
        for idx in ids:
            if idx != prev and idx != 0:
                toks.append(self.labels[idx])
            prev = idx
        return "".join(toks).replace("|", " ").strip()


def read_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                out[obj["key"]] = obj
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--inference-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--low-count", type=int, default=80)
    parser.add_argument("--good-count", type=int, default=80)
    parser.add_argument("--random-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = read_jsonl(args.metadata_jsonl)
    df = pd.read_csv(args.inference_summary)
    low = df[df["sisnr_i"] < 0].sort_values("sisnr_i").head(args.low_count)
    good = df[df["sisnr_i"] > 15].sample(
        n=min(args.good_count, int((df["sisnr_i"] > 15).sum())),
        random_state=args.seed,
    )
    random = df.sample(n=min(args.random_count, len(df)), random_state=args.seed + 1)
    sample = pd.concat(
        [
            low.assign(group="low_sisnri"),
            good.assign(group="good_sisnri"),
            random.assign(group="random"),
        ],
        ignore_index=True,
    ).drop_duplicates("key")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(args.device).eval()
    decoder = GreedyCTC(bundle.get_labels())
    sr = bundle.sample_rate

    rows = []
    with torch.no_grad():
        for idx, row in sample.iterrows():
            key = row["key"]
            m = meta[key]
            ref_text = m["target_text"]
            synth_path = m["source_meta"]["synthetic_interferer"]
            target_path = m["target_ref"]
            for audio_type, path in [("target_clean", target_path), ("synthetic_interferer", synth_path)]:
                wav = load_wav(path, sr).to(args.device)
                emission, _ = model(wav.unsqueeze(0))
                hyp = decoder(emission[0])
                rows.append(
                    {
                        "key": key,
                        "group": row["group"],
                        "sisnr_i": float(row["sisnr_i"]),
                        "audio_type": audio_type,
                        "reference_text": ref_text,
                        "asr_text": hyp,
                        "wer": wer(ref_text, hyp),
                        "cer": cer(ref_text, hyp),
                    }
                )
            if (idx + 1) % 25 == 0:
                print(f"processed {idx + 1}/{len(sample)}", flush=True)

    out_csv = args.out_dir / "asr_text_sanity_sample.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rdf = pd.DataFrame(rows)
    summary = {}
    for (group, audio_type), g in rdf.groupby(["group", "audio_type"]):
        summary[f"{group}/{audio_type}"] = {
            "n": int(len(g)),
            "wer_mean": float(g["wer"].mean()),
            "wer_median": float(g["wer"].median()),
            "cer_mean": float(g["cer"].mean()),
            "cer_median": float(g["cer"].median()),
            "wer_gt_0p5": float((g["wer"] > 0.5).mean()),
            "wer_gt_1p0": float((g["wer"] > 1.0).mean()),
        }
    with (args.out_dir / "asr_text_sanity_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
