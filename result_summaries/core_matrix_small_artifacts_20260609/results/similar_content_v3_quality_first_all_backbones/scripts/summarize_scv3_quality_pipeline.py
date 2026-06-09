#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize_infer(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {"n": int(len(df))}
    for col in ["sisnr_est", "sisnr_i", "si_snr_est", "si_snri", "si_sdr_est", "si_sdri"]:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s):
            out[col + "_mean"] = float(s.mean())
            out[col + "_median"] = float(s.median())
            out[col + "_p10"] = float(s.quantile(0.10))
            out[col + "_p90"] = float(s.quantile(0.90))
            out[col + "_lt0_rate"] = float((s < 0).mean())
    return out


def summarize_diag(path: Path) -> dict:
    summary_json = path / "content_multiverifier/content_multiverifier_summary_1s.json"
    if summary_json.exists():
        try:
            return json.loads(summary_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    out = {}
    utt = path / "content_multiverifier/content_multiverifier_per_utterance_1s.csv"
    if utt.exists():
        df = pd.read_csv(utt)
        out["n_utterance"] = int(len(df))
        for col in df.columns:
            if "drift" in col.lower() or "mismatch" in col.lower():
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s):
                    out[col + "_mean"] = float(s.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for bundle in sorted((args.base / "bundles").glob("*_scv3_quality")):
        infer = summarize_infer(bundle / "full_infer/inference_summary.csv")
        diag = summarize_diag(args.base / "content_diagnostic" / bundle.name)
        row = {"system": bundle.name}
        row.update({"infer_" + k: v for k, v in infer.items()})
        row.update({"local_" + k: v for k, v in diag.items() if isinstance(v, (int, float, str))})
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        out = args.base / "scv3_quality_first_summary.csv"
        df.to_csv(out, index=False)
        print(out)
        print(df.to_string(index=False))
    bundle_summary = args.base / "bundles/quality_first_scv3_content/bundle_summary.json"
    if bundle_summary.exists():
        print(bundle_summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
