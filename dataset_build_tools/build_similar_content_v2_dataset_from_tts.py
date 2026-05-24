#!/usr/bin/env python3
"""Build similar-content v2 mixtures from target speech and TTS interferers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_mono(path: str | Path, sr: int) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return wav.astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-8))


def mix_at_sir(target: np.ndarray, interferer: np.ndarray, sir_db: float):
    n = min(len(target), len(interferer))
    target = target[:n]
    interferer = interferer[:n]
    scale = rms(target) / (rms(interferer) * (10.0 ** (sir_db / 20.0)))
    interferer = interferer * scale
    mix = target + interferer
    peak = float(np.max(np.abs(mix)) + 1e-8)
    if peak > 0.99:
        gain = 0.99 / peak
        target = target * gain
        interferer = interferer * gain
        mix = mix * gain
    return mix.astype(np.float32), target.astype(np.float32), interferer.astype(np.float32)


def infer_librispeech_root(target_source: str | Path, target_spk: str) -> Path:
    path = Path(target_source).resolve()
    parts = list(path.parts)
    if target_spk in parts:
        return Path(*parts[: parts.index(target_spk)])
    return path.parent.parent.parent


def choose_target_enrollment(row: dict) -> str:
    """Use a target-speaker enrollment, avoiding the current target utterance."""
    target_source = Path(row["target_source"]).resolve()
    target_spk = str(row["target_spk"])
    target_utt = str(row.get("target_utt", target_source.stem))
    root = infer_librispeech_root(target_source, target_spk)
    spk_dir = root / target_spk
    candidates = sorted(spk_dir.glob("*/*.flac"))
    for cand in candidates:
        if cand.stem != target_utt:
            return str(cand)
    return str(target_source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--sir-db", type=float, default=0.0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    args = parser.parse_args()

    wav_root = args.out_dir / "wav16k" / "min" / "test"
    dirs = {
        "mix": wav_root / "mix_clean",
        "s1": wav_root / "s1",
        "s2": wav_root / "s2",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    scp_dir = args.out_dir / "scp"
    scp_dir.mkdir(parents=True, exist_ok=True)

    handles = {
        "mix": (scp_dir / "mix.scp").open("w", encoding="utf-8"),
        "ref": (scp_dir / "ref.scp").open("w", encoding="utf-8"),
        "aux": (scp_dir / "aux.scp").open("w", encoding="utf-8"),
        "inter": (scp_dir / "interferer.scp").open("w", encoding="utf-8"),
        "meta": (args.out_dir / "metadata.jsonl").open("w", encoding="utf-8"),
    }
    stats = {"created": 0, "skipped": 0, "missing_tts": 0, "too_short": 0}
    with args.plan_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if args.limit and stats["created"] >= args.limit:
                break
            row = json.loads(line)
            synth = Path(row["synthetic_interferer"])
            if not synth.exists() or synth.stat().st_size < 1024:
                stats["missing_tts"] += 1
                stats["skipped"] += 1
                continue
            target = load_mono(row["target_source"], args.sample_rate)
            inter = load_mono(synth, args.sample_rate)
            if min(len(target), len(inter)) < int(args.min_duration * args.sample_rate):
                stats["too_short"] += 1
                stats["skipped"] += 1
                continue
            mix, target_out, inter_out = mix_at_sir(target, inter, args.sir_db)
            key = row["key"]
            mix_path = dirs["mix"] / f"{key}.wav"
            s1_path = dirs["s1"] / f"{key}.wav"
            s2_path = dirs["s2"] / f"{key}.wav"
            sf.write(mix_path, mix, args.sample_rate)
            sf.write(s1_path, target_out, args.sample_rate)
            sf.write(s2_path, inter_out, args.sample_rate)
            enrollment = choose_target_enrollment(row)

            handles["mix"].write(f"{key} {mix_path}\n")
            handles["ref"].write(f"{key} {s1_path}\n")
            handles["aux"].write(f"{key} {enrollment}\n")
            handles["inter"].write(f"{key} {s2_path}\n")
            meta = {
                "key": key,
                "condition": "similar_content_v2_tts",
                "mix": str(mix_path),
                "target_ref": str(s1_path),
                "interferer_ref": str(s2_path),
                "enrollment": enrollment,
                "target_spk": row["target_spk"],
                "interferer_spk": row["interferer_spk"],
                "target_text": row["target_text"],
                "source_meta": row,
                "sir_db": args.sir_db,
            }
            handles["meta"].write(json.dumps(meta, ensure_ascii=False) + "\n")
            stats["created"] += 1

    for h in handles.values():
        h.close()
    with (args.out_dir / "build_dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
