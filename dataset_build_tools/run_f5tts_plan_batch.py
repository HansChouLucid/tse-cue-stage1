#!/usr/bin/env python3
"""Run F5-TTS generation from a JSONL plan while keeping model/vocoder loaded."""

from __future__ import annotations

import argparse
import json
import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    cfg_strength as default_cfg_strength,
    cross_fade_duration as default_cross_fade_duration,
    infer_process,
    load_model,
    load_vocoder,
    nfe_step as default_nfe_step,
    preprocess_ref_audio_text,
    sway_sampling_coef as default_sway_sampling_coef,
    target_rms as default_target_rms,
)


def load_rows(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def build_f5(model_name: str, vocoder_name: str, device: str):
    vocoder = load_vocoder(vocoder_name=vocoder_name, device=device)
    model_cfg_path = str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
    model_cfg = OmegaConf.load(model_cfg_path)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
    if model_name == "F5TTS_Base":
        ckpt_step = 1200000
    elif model_name == "E2TTS_Base":
        repo_name = "E2-TTS"
        ckpt_step = 1200000
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model_name}/model_{ckpt_step}.{ckpt_type}"))
    model = load_model(model_cls, model_arc, ckpt_file, mel_spec_type=vocoder_name, device=device)
    return model, vocoder, vocoder_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument("--log-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default="F5TTS_v1_Base")
    parser.add_argument("--vocoder-name", default="vocos")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--nfe-step", type=int, default=default_nfe_step)
    args = parser.parse_args()

    args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.plan_jsonl, args.limit)
    model, vocoder, mel_spec_type = build_f5(args.model, args.vocoder_name, args.device)
    done = 0
    failed = 0
    prompt_cache: dict[tuple[str, str], tuple[str, str]] = {}

    def quiet(*_args, **_kwargs):
        return None

    with args.log_jsonl.open("a", encoding="utf-8") as log_f:
        for idx, row in enumerate(rows):
            if idx < args.start_index:
                continue
            out_path = Path(row["synthetic_interferer"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and out_path.stat().st_size > 1024 and not args.overwrite:
                status = {"index": idx, "key": row["key"], "status": "skip_exists", "path": str(out_path)}
                log_f.write(json.dumps(status, ensure_ascii=False) + "\n")
                log_f.flush()
                done += 1
                continue
            try:
                cache_key = (row["prompt_audio"], row.get("prompt_text", ""))
                if cache_key not in prompt_cache:
                    prompt_cache[cache_key] = preprocess_ref_audio_text(
                        row["prompt_audio"], row.get("prompt_text", ""), show_info=quiet
                    )
                ref_audio, ref_text = prompt_cache[cache_key]
                audio_segment, final_sample_rate, _ = infer_process(
                    ref_audio,
                    ref_text,
                    row["target_text"],
                    model,
                    vocoder,
                    mel_spec_type=mel_spec_type,
                    show_info=quiet,
                    target_rms=default_target_rms,
                    cross_fade_duration=default_cross_fade_duration,
                    nfe_step=args.nfe_step,
                    cfg_strength=default_cfg_strength,
                    sway_sampling_coef=default_sway_sampling_coef,
                    speed=args.speed,
                    device=args.device,
                )
                if audio_segment is None:
                    raise RuntimeError("F5-TTS returned no audio")
                sf.write(out_path, np.asarray(audio_segment), final_sample_rate)
                status = {
                    "index": idx,
                    "key": row["key"],
                    "status": "ok",
                    "path": str(out_path),
                    "sample_rate": int(final_sample_rate),
                    "num_samples": int(len(audio_segment)),
                }
                done += 1
            except Exception as exc:
                status = {
                    "index": idx,
                    "key": row["key"],
                    "status": "failed",
                    "path": str(out_path),
                    "error": repr(exc),
                }
                failed += 1
            log_f.write(json.dumps(status, ensure_ascii=False) + "\n")
            log_f.flush()
            if (done + failed) % 10 == 0:
                print(f"processed={done + failed} ok={done} failed={failed}", flush=True)
    print(json.dumps({"processed": done + failed, "ok": done, "failed": failed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
