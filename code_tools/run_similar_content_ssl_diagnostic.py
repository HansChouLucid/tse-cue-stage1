#!/usr/bin/env python3
"""Local content diagnostic for similar-content TSE outputs.

The diagnostic compares SSL features from the separated output against target
and interferer references in short chunks. A negative content gap means the
output chunk is closer to the interferer content than to the target content.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio


def load_jsonl(path: Path, max_samples: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples and len(rows) >= max_samples:
                    break
    return rows


def load_mono(path: str | Path, sr: int) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return wav.astype(np.float32)


def chunk_bounds(n: int, sr: int, chunk_sec: float, hop_sec: float) -> list[tuple[int, int]]:
    chunk = int(round(chunk_sec * sr))
    hop = int(round(hop_sec * sr))
    if n < chunk:
        return []
    return [(s, s + chunk) for s in range(0, n - chunk + 1, hop)]


def rms_db(x: np.ndarray) -> float:
    return 20.0 * math.log10(float(np.sqrt(np.mean(np.square(x)) + 1e-10)) + 1e-10)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    est = est[:n].astype(np.float64)
    ref = ref[:n].astype(np.float64)
    est = est - est.mean()
    ref = ref - ref.mean()
    ref_energy = np.sum(ref * ref) + 1e-8
    proj = np.sum(est * ref) * ref / ref_energy
    noise = est - proj
    return float(10.0 * np.log10((np.sum(proj * proj) + 1e-8) / (np.sum(noise * noise) + 1e-8)))


def summarize(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


@torch.no_grad()
def embed_batch(model: torch.nn.Module, wavs: list[np.ndarray], device: str) -> np.ndarray:
    wav_t = torch.from_numpy(np.stack(wavs)).to(device)
    feats, _ = model.extract_features(wav_t)
    # Middle layers tend to retain phonetic/content information more directly
    # than the final layer, while being stable enough for cosine comparisons.
    layer = feats[min(6, len(feats) - 1)]
    emb = layer.mean(dim=1).detach().float().cpu().numpy()
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)


def mismatch_segments(mask: list[bool], hop_sec: float, min_consecutive: int = 2) -> dict:
    segs: list[tuple[int, int]] = []
    start = None
    for i, val in enumerate(mask + [False]):
        if val and start is None:
            start = i
        if not val and start is not None:
            if i - start >= min_consecutive:
                segs.append((start, i))
            start = None
    durs = [(e - s) * hop_sec for s, e in segs]
    return {
        "content_segment_count": len(segs),
        "content_max_duration": max(durs) if durs else 0.0,
        "content_mean_duration": float(np.mean(durs)) if durs else 0.0,
        "content_first_position": segs[0][0] * hop_sec if segs else -1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--est-dir", type=Path, required=True)
    parser.add_argument("--inference-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-sec", type=float, default=1.0)
    parser.add_argument("--hop-ratio", type=float, default=0.5)
    parser.add_argument("--min-rms-db", type=float, default=-45.0)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().to(device).eval()

    sisnr_by_key = {}
    with args.inference_summary.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sisnr_by_key[row["key"]] = float(row["sisnr_i"])

    samples = load_jsonl(args.metadata_jsonl, args.max_samples)
    chunk_path = args.out_dir / f"ssl_content_per_chunk_{args.chunk_sec:g}s.csv"
    utt_path = args.out_dir / f"ssl_content_per_utterance_{args.chunk_sec:g}s.csv"
    summary_path = args.out_dir / f"ssl_content_summary_{args.chunk_sec:g}s.json"

    chunk_fields = [
        "key",
        "target_spk",
        "interferer_spk",
        "content_similarity",
        "chunk_idx",
        "start_sec",
        "end_sec",
        "content_gap",
        "content_mismatch",
        "local_si_sdr_gap",
        "waveform_prefers_interferer",
    ]
    utt_fields = [
        "key",
        "target_spk",
        "interferer_spk",
        "content_similarity",
        "sisnr_i",
        "num_chunks",
        "content_mismatch_rate",
        "waveform_interferer_rate",
        "joint_content_waveform_rate",
        "mean_content_gap",
        "min_content_gap",
        "mean_local_si_sdr_gap",
        "content_segment_count",
        "content_max_duration",
        "content_mean_duration",
        "content_first_position",
    ]

    utt_rows: list[dict] = []
    all_content_gap: list[float] = []
    all_content_mismatch: list[float] = []
    all_waveform_interferer: list[float] = []
    all_joint: list[float] = []

    with chunk_path.open("w", newline="", encoding="utf-8") as chunk_f:
        writer = csv.DictWriter(chunk_f, fieldnames=chunk_fields)
        writer.writeheader()
        for idx, sample in enumerate(samples, start=1):
            key = sample["key"]
            est_path = args.est_dir / f"{key}.wav"
            if not est_path.exists():
                continue
            est = load_mono(est_path, args.sample_rate)
            target = load_mono(sample["target_ref"], args.sample_rate)
            interferer = load_mono(sample["interferer_ref"], args.sample_rate)
            n = min(len(est), len(target), len(interferer))
            est, target, interferer = est[:n], target[:n], interferer[:n]
            bounds = chunk_bounds(n, args.sample_rate, args.chunk_sec, args.chunk_sec * args.hop_ratio)

            chunk_items: list[tuple[int, int, int]] = []
            wavs: list[np.ndarray] = []
            local_gaps: list[float] = []
            for ci, (s, e) in enumerate(bounds):
                tar_c, int_c = target[s:e], interferer[s:e]
                if rms_db(tar_c) < args.min_rms_db or rms_db(int_c) < args.min_rms_db:
                    continue
                est_c = est[s:e]
                chunk_items.append((ci, s, e))
                wavs.extend([est_c, tar_c, int_c])
                local_gaps.append(si_sdr(est_c, tar_c) - si_sdr(est_c, int_c))

            if not chunk_items:
                continue

            embs = []
            for start in range(0, len(wavs), args.batch_size):
                embs.append(embed_batch(model, wavs[start : start + args.batch_size], device))
            embs_np = np.concatenate(embs, axis=0)

            content_gaps: list[float] = []
            content_mask: list[bool] = []
            waveform_mask: list[bool] = []
            joint_mask: list[bool] = []
            for j, (ci, s, e) in enumerate(chunk_items):
                est_emb, tar_emb, int_emb = embs_np[3 * j], embs_np[3 * j + 1], embs_np[3 * j + 2]
                content_gap = cosine(est_emb, tar_emb) - cosine(est_emb, int_emb)
                local_gap = local_gaps[j]
                content_mismatch = content_gap < 0.0
                waveform_prefers_interferer = local_gap < 0.0
                joint = content_mismatch and waveform_prefers_interferer
                content_gaps.append(content_gap)
                content_mask.append(content_mismatch)
                waveform_mask.append(waveform_prefers_interferer)
                joint_mask.append(joint)
                writer.writerow(
                    {
                        "key": key,
                        "target_spk": sample["target_spk"],
                        "interferer_spk": sample["interferer_spk"],
                        "content_similarity": sample.get("content_similarity", ""),
                        "chunk_idx": ci,
                        "start_sec": s / args.sample_rate,
                        "end_sec": e / args.sample_rate,
                        "content_gap": content_gap,
                        "content_mismatch": int(content_mismatch),
                        "local_si_sdr_gap": local_gap,
                        "waveform_prefers_interferer": int(waveform_prefers_interferer),
                    }
                )

            seg = mismatch_segments(content_mask, args.chunk_sec * args.hop_ratio)
            row = {
                "key": key,
                "target_spk": sample["target_spk"],
                "interferer_spk": sample["interferer_spk"],
                "content_similarity": sample.get("content_similarity", ""),
                "sisnr_i": sisnr_by_key.get(key, math.nan),
                "num_chunks": len(content_gaps),
                "content_mismatch_rate": float(np.mean(content_mask)),
                "waveform_interferer_rate": float(np.mean(waveform_mask)),
                "joint_content_waveform_rate": float(np.mean(joint_mask)),
                "mean_content_gap": float(np.mean(content_gaps)),
                "min_content_gap": float(np.min(content_gaps)),
                "mean_local_si_sdr_gap": float(np.mean(local_gaps)),
                **seg,
            }
            utt_rows.append(row)
            all_content_gap.extend(content_gaps)
            all_content_mismatch.append(row["content_mismatch_rate"])
            all_waveform_interferer.append(row["waveform_interferer_rate"])
            all_joint.append(row["joint_content_waveform_rate"])

            if idx % 250 == 0:
                print(f"processed {idx}/{len(samples)}", flush=True)

    with utt_path.open("w", newline="", encoding="utf-8") as utt_f:
        writer = csv.DictWriter(utt_f, fieldnames=utt_fields)
        writer.writeheader()
        writer.writerows(utt_rows)

    def frac(pred) -> float:
        return float(np.mean([pred(r) for r in utt_rows])) if utt_rows else math.nan

    summary = {
        "num_utterances": len(utt_rows),
        "num_chunks": len(all_content_gap),
        "content_gap": summarize(all_content_gap),
        "content_mismatch_rate_utterance": summarize(all_content_mismatch),
        "waveform_interferer_rate_utterance": summarize(all_waveform_interferer),
        "joint_content_waveform_rate_utterance": summarize(all_joint),
        "utterance_any_content_mismatch": frac(lambda r: r["content_mismatch_rate"] > 0.0),
        "utterance_content_rate_gt_0p1": frac(lambda r: r["content_mismatch_rate"] > 0.1),
        "utterance_content_rate_gt_0p2": frac(lambda r: r["content_mismatch_rate"] > 0.2),
        "utterance_any_joint": frac(lambda r: r["joint_content_waveform_rate"] > 0.0),
        "utterance_joint_rate_gt_0p1": frac(lambda r: r["joint_content_waveform_rate"] > 0.1),
        "low_sisnri_content_rate": summarize([r["content_mismatch_rate"] for r in utt_rows if r["sisnr_i"] < 0]),
        "good_sisnri_content_rate": summarize([r["content_mismatch_rate"] for r in utt_rows if r["sisnr_i"] > 15]),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
