#!/usr/bin/env python3
"""Build a normal/easy Libri2Mix-style USEF reference set from metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def spk_from_rel(rel: str) -> str:
    return Path(rel).name.split("-")[0]


def collect_speaker_utts(librispeech_root: Path, split: str) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    split_root = librispeech_root / split
    for path in sorted(split_root.glob("*/*/*.flac")):
        out.setdefault(path.name.split("-")[0], []).append(path)
    return out


def load_gain(path: Path, gain: float, sr: int) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return (wav.astype(np.float32) * float(gain)).astype(np.float32)


def pick_enrollment(spk_utts: dict[str, list[Path]], spk: str, current: Path) -> Path:
    current = current.resolve()
    for cand in spk_utts.get(spk, []):
        if cand.resolve() != current:
            return cand
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--librispeech-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test-clean")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--limit-mixtures", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wav_root = args.out_dir / "wav16k" / "min" / "test"
    mix_dir = wav_root / "mix_clean"
    s1_dir = wav_root / "s1"
    s2_dir = wav_root / "s2"
    for d in [mix_dir, s1_dir, s2_dir]:
        d.mkdir(parents=True, exist_ok=True)
    scp_dir = args.out_dir / "scp"
    scp_dir.mkdir(exist_ok=True)

    spk_utts = collect_speaker_utts(args.librispeech_root, args.split)

    mix_scp = (scp_dir / "mix.scp").open("w", encoding="utf-8")
    ref_scp = (scp_dir / "ref.scp").open("w", encoding="utf-8")
    aux_scp = (scp_dir / "aux.scp").open("w", encoding="utf-8")
    inter_scp = (scp_dir / "interferer.scp").open("w", encoding="utf-8")
    meta_f = (args.out_dir / "metadata.jsonl").open("w", encoding="utf-8")

    count_mix = 0
    count_task = 0
    with args.metadata_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if args.limit_mixtures and count_mix >= args.limit_mixtures:
                break
            mixture_id = row["mixture_ID"]
            src1_rel = row["source_1_path"]
            src2_rel = row["source_2_path"]
            src1_path = args.librispeech_root / src1_rel
            src2_path = args.librispeech_root / src2_rel
            spk1 = spk_from_rel(src1_rel)
            spk2 = spk_from_rel(src2_rel)
            s1 = load_gain(src1_path, float(row["source_1_gain"]), args.sample_rate)
            s2 = load_gain(src2_path, float(row["source_2_gain"]), args.sample_rate)
            n = min(len(s1), len(s2))
            s1, s2 = s1[:n], s2[:n]
            mix = s1 + s2
            peak = float(np.max(np.abs(mix)) + 1e-8)
            if peak > 0.99:
                scale = 0.99 / peak
                mix = mix * scale
                s1 = s1 * scale
                s2 = s2 * scale
            mix_path = mix_dir / f"{mixture_id}.wav"
            s1_path = s1_dir / f"{mixture_id}.wav"
            s2_path = s2_dir / f"{mixture_id}.wav"
            sf.write(mix_path, mix, args.sample_rate)
            sf.write(s1_path, s1, args.sample_rate)
            sf.write(s2_path, s2, args.sample_rate)
            tasks = [
                ("T1", spk1, spk2, s1_path, s2_path, src1_path),
                ("T2", spk2, spk1, s2_path, s1_path, src2_path),
            ]
            for tag, target_spk, inter_spk, ref_path, inter_path, source_path in tasks:
                key = f"{mixture_id}__{tag}__NORMAL__{target_spk}"
                enrollment = pick_enrollment(spk_utts, target_spk, source_path)
                mix_scp.write(f"{key} {mix_path}\n")
                ref_scp.write(f"{key} {ref_path}\n")
                aux_scp.write(f"{key} {enrollment}\n")
                inter_scp.write(f"{key} {inter_path}\n")
                meta_f.write(
                    json.dumps(
                        {
                            "key": key,
                            "condition": "normal_librimix",
                            "mix": str(mix_path),
                            "target_ref": str(ref_path),
                            "interferer_ref": str(inter_path),
                            "enrollment": str(enrollment),
                            "target_spk": target_spk,
                            "interferer_spk": inter_spk,
                            "source_meta": row,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count_task += 1
            count_mix += 1

    for h in [mix_scp, ref_scp, aux_scp, inter_scp, meta_f]:
        h.close()
    print(f"wrote {count_mix} mixtures and {count_task} target tasks to {args.out_dir}")


if __name__ == "__main__":
    main()
