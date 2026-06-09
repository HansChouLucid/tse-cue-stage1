#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


def load_jsonl_by_base(path: Path):
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            base_key = row.get("source_meta", {}).get("mixture_id") or row["key"].split("__")[0]
            target_spk = str(row.get("target_spk"))
            out[(base_key, target_spk)] = row
            out[row["key"]] = row
    return out


def load_jsonl_by_key(path: Path):
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["key"]] = row
    return out


def read_manifest(path: Path):
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["key"]] = row
    return out


def first_path(obj):
    if isinstance(obj, dict):
        obj = obj.get("default")
    if isinstance(obj, list):
        return obj[0]
    return obj


def load_mono(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float64), sr


def si_sdr(est, ref, eps=1e-8):
    n = min(len(est), len(ref))
    est = est[:n].astype(np.float64)
    ref = ref[:n].astype(np.float64)
    est -= est.mean()
    ref -= ref.mean()
    proj = np.sum(est * ref) * ref / (np.sum(ref * ref) + eps)
    noise = est - proj
    return float(10 * np.log10((np.sum(proj * proj) + eps) / (np.sum(noise * noise) + eps)))


def parse_log(log: Path):
    vals = {}
    txt = log.read_text(errors="ignore") if log.exists() else ""
    pat = re.compile(r"Utt=(.*?) \| Target speaker=(.*?) \| SI-SNR=([-0-9.]+) \| SI-SNRi=([-0-9.]+)")
    for m in pat.finditer(txt):
        vals[(m.group(1), str(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    return vals


def infer_audio_rows(audio_dir: Path):
    rows = []
    for wav in sorted(audio_dir.rglob("*.wav")):
        m = re.search(r"Utt(\d+)-(.+)-T([^-/\\]+)\.wav$", wav.name)
        if not m:
            continue
        rows.append((int(m.group(1)), m.group(2), str(m.group(3)), wav))
    if rows:
        return rows
    for scp in sorted(audio_dir.glob("*.scp")):
        for line in scp.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            _, wav_path = line.split(maxsplit=1)
            wav = Path(wav_path)
            m = re.search(r"Utt(\d+)-(.+)-T([^-/\\]+)\.wav$", wav.name)
            if m:
                rows.append((int(m.group(1)), m.group(2), str(m.group(3)), wav))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cue-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--metadata-template", type=Path, required=True)
    args = ap.parse_args()

    manifest = read_manifest(args.manifest)
    meta_by_base = load_jsonl_by_base(args.metadata_template)
    meta_by_key = load_jsonl_by_key(args.metadata_template)
    audio_dir = args.cue_dir / "infer_run/audio"
    est_dir = args.cue_dir / "full_infer/est_wavs"
    est_dir.mkdir(parents=True, exist_ok=True)
    rows = infer_audio_rows(audio_dir)
    if not rows:
        raise RuntimeError(f"no BSRNN wav rows found under {audio_dir}")
    log_vals = parse_log(args.cue_dir / "infer_run/infer.log")

    meta_out = args.cue_dir / "metadata.jsonl"
    inf_out = args.cue_dir / "full_infer/inference_summary.csv"
    inf_out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with meta_out.open("w", encoding="utf-8") as mf, inf_out.open("w", newline="", encoding="utf-8") as cf:
        fields = [
            "key", "mix_path", "target_path", "interferer_path", "aux_path", "est_path",
            "sample_rate", "num_samples", "si_snr_mix", "si_snr_est", "si_snri",
            "si_sdr_mix", "si_sdr_est", "si_sdri", "error",
        ]
        wr = csv.DictWriter(cf, fieldnames=fields)
        wr.writeheader()
        for _, base_key, spk, wav_path in rows:
            item = manifest.get(base_key)
            if item is None:
                continue
            meta = meta_by_base.get((base_key, spk))
            if meta is None:
                meta = meta_by_key.get(base_key)
            if meta is None:
                continue
            if str(meta.get("target_spk")) != str(spk):
                continue
            inter = str(meta["interferer_spk"])
            task_key = meta["key"]
            out_wav = est_dir / f"{task_key}.wav"
            shutil.copy2(wav_path, out_wav)
            mix_path = first_path(item["mix"])
            target_path = meta["target_ref"]
            inter_path = meta["interferer_ref"]
            mix, _ = load_mono(mix_path)
            target, sr = load_mono(target_path)
            est, _ = load_mono(out_wav)
            n = min(len(est), len(target), len(mix))
            mix_s = si_sdr(mix, target)
            est_s = si_sdr(est, target)
            delta = est_s - mix_s
            if (base_key, spk) in log_vals:
                est_s, delta = log_vals[(base_key, spk)]
            wr.writerow({
                "key": task_key,
                "mix_path": mix_path,
                "target_path": target_path,
                "interferer_path": inter_path,
                "aux_path": meta.get("enrollment", ""),
                "est_path": str(out_wav),
                "sample_rate": sr,
                "num_samples": n,
                "si_snr_mix": mix_s,
                "si_snr_est": est_s,
                "si_snri": delta,
                "si_sdr_mix": mix_s,
                "si_sdr_est": est_s,
                "si_sdri": delta,
                "error": "",
            })
            meta = dict(meta)
            meta["est_path"] = str(out_wav)
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            n_ok += 1
    (args.cue_dir / "DONE").write_text(f"done {n_ok}\n", encoding="utf-8")
    print(args.cue_dir, "postprocessed", n_ok)


if __name__ == "__main__":
    main()
