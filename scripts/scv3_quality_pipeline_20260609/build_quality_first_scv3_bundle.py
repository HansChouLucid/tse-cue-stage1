#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
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


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def audio_sanity(path: Path) -> dict:
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        duration = float(len(wav) / sr) if sr else 0.0
        abs_wav = np.abs(wav)
        rms = float(np.sqrt(np.mean(np.square(wav)))) if len(wav) else 0.0
        peak = float(abs_wav.max()) if len(wav) else 0.0
        silence_rate = float(np.mean(abs_wav < 1e-4)) if len(wav) else 1.0
        clip_rate_0p99 = float(np.mean(abs_wav >= 0.99)) if len(wav) else 0.0
        ok = duration >= 1.0 and silence_rate <= 0.35 and rms >= 0.005 and peak < 0.99 and clip_rate_0p99 == 0.0
        return {
            "audio_ok": ok,
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
            "audio_ok": False,
            "sample_rate": "",
            "duration": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "silence_rate": 1.0,
            "clip_rate_0p99": 1.0,
            "audio_error": repr(exc),
        }


def load_asr(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["key"]] = row
    return out


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def metric_float(row: dict, key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def write_scp(path: Path, rows: list[tuple[str, str]]):
    with path.open("w", encoding="utf-8") as f:
        for key, wav in rows:
            f.write(f"{key} {wav}\n")


def side_paths(basic: Path, mixture_id: str, target_spk: str) -> tuple[Path, Path]:
    s1 = basic / "s1" / f"{mixture_id}.wav"
    s2 = basic / "s2" / f"{mixture_id}.wav"
    first_spk = mixture_id.split("_", 1)[0].split("-", 1)[0]
    if str(target_spk) == first_spk:
        return s1, s2
    return s2, s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path("/data/tse_cue_project"))
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--out-base", type=Path, required=True)
    ap.add_argument("--asr-csv", type=Path, default=None)
    ap.add_argument("--copy-final-enrollment", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    plan_path = args.manifest_dir / "clean/test/tts_plan_naturalized.jsonl"
    logs = sorted((args.work_dir / "logs").glob("f5tts_quality_naturalized_shard*_gpu*.jsonl"))
    plan_rows = {r["key"]: r for r in read_jsonl(plan_path)}
    log_rows = {}
    for log in logs:
        for row in read_jsonl(log):
            log_rows[row["key"]] = row
    asr_rows = load_asr(args.asr_csv)

    bundle = args.out_base / "bundles" / "quality_first_scv3_content"
    scp_dir = bundle / "scp"
    bsrnn_dir = bundle / "bsrnn"
    final_enroll = bundle / "final_enrollment"
    for d in [scp_dir, bsrnn_dir, final_enroll]:
        d.mkdir(parents=True, exist_ok=True)

    basic = args.project_root / "datasets/Libri2Mix_basic_clean/wav16k/min/test"
    metadata_rows = []
    quality_rows = []
    scp_mix, scp_ref, scp_inter, scp_aux = [], [], [], []
    bsrnn_items = []
    fixed_enroll = {}

    for key in sorted(plan_rows):
        if args.max_rows and len(metadata_rows) >= args.max_rows:
            break
        plan = plan_rows[key]
        log = log_rows.get(key, {})
        generated = Path(log.get("path") or plan.get("synthetic_enrollment", ""))
        sanity = audio_sanity(generated)
        asr = asr_rows.get(key, {})
        asr_pass = True if not asr_rows else (
            metric_float(asr, "wer_to_gen", 999.0) <= 0.70 or metric_float(asr, "token_f1_to_gen", 0.0) >= 0.35
        )
        pass_final = bool(log.get("status") == "ok" and generated.exists() and sanity["audio_ok"] and asr_pass)

        q = {
            "key": key,
            "status": log.get("status", "missing"),
            "generated_path": str(generated),
            "pass_audio": sanity["audio_ok"],
            "pass_asr": asr_pass,
            "pass_final": pass_final,
            "target_spk": plan.get("target_spk"),
            "interferer_spk": plan.get("interferer_spk"),
            "target_text": plan.get("target_text"),
            "interferer_text": plan.get("interferer_text"),
            "gen_text_for_tts": plan.get("gen_text_for_tts"),
        }
        q.update(sanity)
        if asr:
            for k in ["wer_to_gen", "cer_to_gen", "token_f1_to_gen", "whisper_text"]:
                q[k] = asr.get(k, "")
        quality_rows.append(q)
        if not pass_final:
            continue

        mixture_id = plan["mixture_id"]
        target_spk = str(plan["target_spk"])
        interferer_spk = str(plan["interferer_spk"])
        mix = basic / "mix_clean" / f"{mixture_id}.wav"
        ref, inter = side_paths(basic, mixture_id, target_spk)
        if not mix.exists() or not ref.exists() or not inter.exists():
            continue

        aux = generated
        if args.copy_final_enrollment:
            aux = final_enroll / f"{key}.wav"
            if not aux.exists():
                shutil.copy2(generated, aux)

        source_meta = dict(plan)
        source_meta.update({"tts_log": log, "quality": q})
        meta = {
            "key": key,
            "condition": "similar_content_v3_tts_quality_first",
            "mix": str(mix),
            "target_ref": str(ref),
            "interferer_ref": str(inter),
            "enrollment": str(aux),
            "target_spk": target_spk,
            "interferer_spk": interferer_spk,
            "target_text": plan.get("target_text"),
            "interferer_text": plan.get("interferer_text"),
            "enrollment_text": plan.get("gen_text"),
            "source_meta": source_meta,
        }
        metadata_rows.append(meta)
        scp_mix.append((key, str(mix)))
        scp_ref.append((key, str(ref)))
        scp_inter.append((key, str(inter)))
        scp_aux.append((key, str(aux)))
        bsrnn_items.append({
            "key": key,
            "spk": [target_spk, interferer_spk],
            "spk1": target_spk,
            "spk2": interferer_spk,
            "mix": {"default": [str(mix)]},
            "src": {target_spk: [str(ref)], interferer_spk: [str(inter)]},
        })
        fixed_enroll[f"{key}::{target_spk}"] = [{"path": str(aux)}]
        fixed_enroll[f"{key}::{interferer_spk}"] = [{"path": str(inter)}]

    write_scp(scp_dir / "mix.scp", scp_mix)
    write_scp(scp_dir / "ref.scp", scp_ref)
    write_scp(scp_dir / "interferer.scp", scp_inter)
    write_scp(scp_dir / "aux.scp", scp_aux)
    with (bundle / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (bundle / "quality_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for row in quality_rows for k in row})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(quality_rows)
    with (bsrnn_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in bsrnn_items:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    cue_resource = bsrnn_dir / "fixed_enroll.json"
    cue_resource.write_text(json.dumps(fixed_enroll, indent=2, ensure_ascii=False), encoding="utf-8")
    (bsrnn_dir / "cues.yaml").write_text(
        "cues:\n"
        "  audio:\n"
        "    type: raw\n"
        "    guaranteed: true\n"
        "    scope: speaker\n"
        "    policy:\n"
        "      type: fixed\n"
        "      key: mix_spk_id\n"
        f"      resource: {cue_resource}\n",
        encoding="utf-8",
    )
    summary = {
        "plan_rows": len(plan_rows),
        "log_rows": len(log_rows),
        "quality_rows": len(quality_rows),
        "final_rows": len(metadata_rows),
        "asr_csv": str(args.asr_csv) if args.asr_csv else None,
    }
    (bundle / "bundle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
