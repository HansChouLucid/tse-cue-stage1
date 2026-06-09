#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def audio_sanity(path: Path) -> dict:
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        abs_wav = np.abs(wav)
        duration = float(len(wav) / sr) if sr else 0.0
        rms = float(np.sqrt(np.mean(np.square(wav)))) if len(wav) else 0.0
        peak = float(abs_wav.max()) if len(wav) else 0.0
        silence_rate = float(np.mean(abs_wav < 1e-4)) if len(wav) else 1.0
        clip_rate_0p99 = float(np.mean(abs_wav >= 0.99)) if len(wav) else 0.0
        ok = duration >= 1.0 and silence_rate <= 0.35 and rms >= 0.005 and peak < 0.99 and clip_rate_0p99 == 0.0
        return {
            "pass_audio": ok,
            "sample_rate": sr,
            "duration": duration,
            "rms": rms,
            "peak": peak,
            "silence_rate": silence_rate,
            "clip_rate_0p99": clip_rate_0p99,
            "audio_error": "",
        }
    except Exception as exc:
        return {
            "pass_audio": False,
            "sample_rate": "",
            "duration": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "silence_rate": 1.0,
            "clip_rate_0p99": 1.0,
            "audio_error": repr(exc),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    plan_path = args.manifest_dir / "clean/test/tts_plan_naturalized.jsonl"
    plan = {r["key"]: r for r in read_jsonl(plan_path)}
    logs = {}
    for log_path in sorted((args.work_dir / "logs").glob("f5tts_quality_naturalized_shard*_gpu*.jsonl")):
        for row in read_jsonl(log_path):
            logs[row["key"]] = row

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in sorted(plan):
        p = plan[key]
        l = logs.get(key, {})
        path = Path(l.get("path") or p.get("synthetic_enrollment", ""))
        row = {
            "key": key,
            "path": str(path),
            "status": l.get("status", "missing"),
            "gen_text": p.get("gen_text") or l.get("gen_text", ""),
            "gen_text_for_tts": p.get("gen_text_for_tts") or l.get("gen_text_for_tts", ""),
            "target_text": p.get("target_text") or l.get("target_text", ""),
            "interferer_text": p.get("interferer_text") or l.get("interferer_text", ""),
            "target_spk": p.get("target_spk", ""),
            "interferer_spk": p.get("interferer_spk", ""),
        }
        row.update(audio_sanity(path))
        rows.append(row)

    fields = list(rows[0].keys()) if rows else []
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(args.out_csv, "rows", len(rows), "audio_pass", sum(1 for r in rows if r["pass_audio"]))


if __name__ == "__main__":
    main()
