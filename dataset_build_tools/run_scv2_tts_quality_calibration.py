#!/usr/bin/env python3
"""Quality calibration for similar-content v2 TTS data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_mono(path: str | Path, sr: int) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return wav.astype(np.float32)


def audio_stat(path: str | Path) -> dict:
    info = sf.info(str(path))
    wav, _ = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    abs_wav = np.abs(wav)
    return {
        "path": str(path),
        "sr": int(info.samplerate),
        "duration": float(info.frames / info.samplerate),
        "frames": int(info.frames),
        "rms": float(np.sqrt(np.mean(wav * wav) + 1e-12)),
        "peak": float(abs_wav.max(initial=0.0)),
        "clip_rate_0p99": float(np.mean(abs_wav >= 0.99)),
        "silence_rate_1e_4": float(np.mean(abs_wav < 1e-4)),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def embed_one(classifier: EncoderClassifier, path: str | Path, sr: int, device: str) -> np.ndarray:
    wav = load_mono(path, sr)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    with torch.no_grad():
        emb = classifier.encode_batch(wav_t, normalize=True)
    return emb.squeeze().detach().cpu().numpy().astype(np.float32)


def summarize(xs: list[float]) -> dict:
    arr = np.asarray([x for x in xs if not math.isnan(float(x))], dtype=np.float64)
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def normalize_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--inference-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--speaker-source", default="speechbrain/spkrec-xvect-voxceleb")
    parser.add_argument("--speaker-savedir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-speaker-samples", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.metadata_jsonl)
    if args.max_speaker_samples:
        rows = rows[: args.max_speaker_samples]
    sisnr = pd.read_csv(args.inference_summary).set_index("key")["sisnr_i"].to_dict()

    audio_rows = []
    speaker_rows = []

    print("loading speaker verifier", flush=True)
    classifier = EncoderClassifier.from_hparams(
        source=args.speaker_source,
        savedir=str(args.speaker_savedir),
        run_opts={"device": args.device},
    )
    emb_cache: dict[str, np.ndarray] = {}

    def emb(path: str | Path) -> np.ndarray:
        p = str(path)
        if p not in emb_cache:
            emb_cache[p] = embed_one(classifier, p, args.sample_rate, args.device)
        return emb_cache[p]

    for idx, row in enumerate(rows, 1):
        key = row["key"]
        src = row["source_meta"]
        target_ref = row["target_ref"]
        inter_ref = row["interferer_ref"]
        mix = row["mix"]
        synth = src["synthetic_interferer"]
        target_enroll = row["enrollment"]
        inter_prompt = src["prompt_audio"]

        ts = audio_stat(target_ref)
        ins = audio_stat(inter_ref)
        ms = audio_stat(mix)
        ss = audio_stat(synth)
        audio_rows.append(
            {
                "key": key,
                "sisnr_i": float(sisnr.get(key, math.nan)),
                "target_duration": ts["duration"],
                "interferer_duration": ins["duration"],
                "mix_duration": ms["duration"],
                "synth_raw_duration": ss["duration"],
                "target_rms": ts["rms"],
                "interferer_rms": ins["rms"],
                "mix_rms": ms["rms"],
                "synth_raw_rms": ss["rms"],
                "target_peak": ts["peak"],
                "interferer_peak": ins["peak"],
                "mix_peak": ms["peak"],
                "synth_raw_peak": ss["peak"],
                "mix_clip_rate_0p99": ms["clip_rate_0p99"],
                "synth_clip_rate_0p99": ss["clip_rate_0p99"],
                "final_duration_ratio_inter_target": ins["duration"] / max(ts["duration"], 1e-8),
                "raw_synth_duration_ratio_target": ss["duration"] / max(ts["duration"], 1e-8),
                "final_inter_target_rms_ratio": ins["rms"] / max(ts["rms"], 1e-8),
            }
        )

        synth_emb = emb(inter_ref)
        target_emb = emb(target_enroll)
        inter_emb = emb(inter_prompt)
        sim_inter = cosine(synth_emb, inter_emb)
        sim_target = cosine(synth_emb, target_emb)
        speaker_rows.append(
            {
                "key": key,
                "sisnr_i": float(sisnr.get(key, math.nan)),
                "target_spk": row["target_spk"],
                "interferer_spk": row["interferer_spk"],
                "sim_synth_to_interferer_prompt": sim_inter,
                "sim_synth_to_target_enrollment": sim_target,
                "speaker_voice_gap_inter_minus_target": sim_inter - sim_target,
                "synth_prefers_interferer_voice": sim_inter > sim_target,
            }
        )
        if idx % 250 == 0:
            print(f"processed {idx}/{len(rows)}", flush=True)

    audio_path = args.out_dir / "audio_quality_per_utterance.csv"
    speaker_path = args.out_dir / "speaker_quality_per_utterance.csv"
    with audio_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audio_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audio_rows)
    with speaker_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(speaker_rows[0].keys()))
        writer.writeheader()
        writer.writerows(speaker_rows)

    adf = pd.DataFrame(audio_rows)
    sdf = pd.DataFrame(speaker_rows)
    low = sdf["sisnr_i"] < 0
    good = sdf["sisnr_i"] > 15
    summary = {
        "num_samples": int(len(rows)),
        "audio": {
            "target_duration": summarize(adf["target_duration"].tolist()),
            "raw_synth_duration_ratio_target": summarize(adf["raw_synth_duration_ratio_target"].tolist()),
            "final_inter_target_rms_ratio": summarize(adf["final_inter_target_rms_ratio"].tolist()),
            "mix_peak": summarize(adf["mix_peak"].tolist()),
            "synth_raw_peak": summarize(adf["synth_raw_peak"].tolist()),
            "num_mix_clip_rate_gt_0": int((adf["mix_clip_rate_0p99"] > 0).sum()),
            "num_synth_clip_rate_gt_0": int((adf["synth_clip_rate_0p99"] > 0).sum()),
        },
        "speaker": {
            "voice_gap_inter_minus_target": summarize(sdf["speaker_voice_gap_inter_minus_target"].tolist()),
            "synth_prefers_interferer_voice_rate": float(sdf["synth_prefers_interferer_voice"].mean()),
            "low_sisnri_voice_gap": summarize(sdf.loc[low, "speaker_voice_gap_inter_minus_target"].tolist()),
            "good_sisnri_voice_gap": summarize(sdf.loc[good, "speaker_voice_gap_inter_minus_target"].tolist()),
            "low_sisnri_prefers_interferer_rate": float(sdf.loc[low, "synth_prefers_interferer_voice"].mean()),
            "good_sisnri_prefers_interferer_rate": float(sdf.loc[good, "synth_prefers_interferer_voice"].mean()),
        },
    }
    with (args.out_dir / "tts_quality_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
