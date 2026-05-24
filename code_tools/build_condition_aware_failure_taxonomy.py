#!/usr/bin/env python3
"""Build a condition-aware failure taxonomy for USEF diagnostics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/data/tse_cue_project/experiments/fine_grained_cue/usef_tse_tfgridnet")
SIM_SPK = ROOT / "similar_speaker_test_full/usef_tfgridnet_wsj0_2mix"
SIM_CONT = ROOT / "similar_content_test_full/usef_tfgridnet_wsj0_2mix"
OUT = ROOT / "condition_aware_failure_taxonomy"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict, key: str, default: float = math.nan) -> float:
    try:
        val = row.get(key, "")
        if val == "":
            return default
        return float(val)
    except Exception:
        return default


def summarize(vals: list[float]) -> dict:
    arr = np.asarray([v for v in vals if not math.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def pct(rows: list[dict], pred) -> float:
    return float(np.mean([pred(r) for r in rows])) if rows else math.nan


def rel_band(sisnri: float, p10: float, median: float, p75: float) -> str:
    if sisnri < 0:
        return "absolute_failure_lt0"
    if sisnri < 5:
        return "absolute_weak_0_5"
    if sisnri <= p10:
        return "relative_tail_le_p10"
    if sisnri < median:
        return "below_median"
    if sisnri >= p75:
        return "strong_upper_quartile"
    return "normal_mid"


def analyze_similar_speaker() -> tuple[list[dict], dict]:
    inf = {r["key"]: r for r in read_csv(SIM_SPK / "inference_summary.csv")}
    tax = {r["key"]: r for r in read_csv(SIM_SPK / "failure_taxonomy_v2/utterance_taxonomy_1s.csv")}
    sb = {r["key"]: r for r in read_csv(SIM_SPK / "speechbrain_xvector_full/speechbrain_full_per_utterance_1s.csv")}
    keys = sorted(set(inf) & set(tax) & set(sb))
    sis = [f(inf[k], "sisnr_i") for k in keys]
    p10, med, p75 = np.percentile(sis, [10, 50, 75])
    rows = []
    for key in keys:
        i, t, s = inf[key], tax[key], sb[key]
        sisnri = f(i, "sisnr_i")
        mv_true = f(s, "mv_true_swap_rate_targetref")
        mv_identity = f(s, "mv_mismatch_rate_targetref")
        ecapa_true = f(s, "ecapa_true_swap_rate")
        ecapa_mis = f(s, "ecapa_mismatch_rate")
        local_wave_gap = f(t, "local_si_sdr_gap_mean")
        confirmed_identity = mv_identity > 0.0
        confirmed_swap = mv_true > 0.0
        strong_swap = mv_true > 0.2
        if confirmed_swap and sisnri < 5:
            cause = "mismatch_explained_extreme_failure"
        elif confirmed_swap:
            cause = "confirmed_local_identity_mismatch"
        elif confirmed_identity and local_wave_gap > 0:
            cause = "identity_disagreement_without_waveform_swap"
        elif sisnri < 5:
            cause = "non_mismatch_or_unconfirmed_failure"
        elif ecapa_mis > 0.2 and mv_identity == 0:
            cause = "single_verifier_instability"
        else:
            cause = "no_confirmed_mismatch"
        rows.append(
            {
                "condition": "similar_speaker",
                "key": key,
                "sisnr_i": sisnri,
                "relative_band": rel_band(sisnri, p10, med, p75),
                "condition_p10": float(p10),
                "condition_median": float(med),
                "condition_p75": float(p75),
                "failure_cause": cause,
                "primary_mismatch_rate": mv_true,
                "secondary_mismatch_rate": mv_identity,
                "ecapa_mismatch_rate": ecapa_mis,
                "ecapa_true_swap_rate": ecapa_true,
                "local_wave_gap_mean": local_wave_gap,
                "speaker_similarity": f(t, "speaker_similarity"),
                "target_spk": t.get("target_spk", ""),
                "interferer_spk": t.get("interferer_spk", ""),
            }
        )
    return rows, {"p10": float(p10), "median": float(med), "p75": float(p75), "n": len(rows)}


def analyze_similar_content() -> tuple[list[dict], dict]:
    inf = {r["key"]: r for r in read_csv(SIM_CONT / "inference_summary.csv")}
    ssl = {r["key"]: r for r in read_csv(SIM_CONT / "local_content_ssl/ssl_content_per_utterance_1s.csv")}
    ec = {r["key"]: r for r in read_csv(SIM_CONT / "local_mismatch_ecapa/per_utterance_summary.csv")}
    keys = sorted(set(inf) & set(ssl) & set(ec))
    sis = [f(inf[k], "sisnr_i") for k in keys]
    p10, med, p75 = np.percentile(sis, [10, 50, 75])
    rows = []
    for key in keys:
        i, c, e = inf[key], ssl[key], ec[key]
        sisnri = f(i, "sisnr_i")
        content = f(c, "content_mismatch_rate")
        wave = f(c, "waveform_interferer_rate")
        joint = f(c, "joint_content_waveform_rate")
        spk = f(e, "mismatch_rate_1s")
        if joint > 0.1 and sisnri < 5:
            cause = "mismatch_explained_extreme_failure"
        elif joint > 0.1:
            cause = "confirmed_local_content_mismatch"
        elif content > 0.2 and wave <= 0.1:
            cause = "content_only_unconfirmed_by_waveform"
        elif sisnri < 5:
            cause = "non_mismatch_or_unconfirmed_failure"
        elif spk > 0.2 and content <= 0.1:
            cause = "speaker_verifier_instability"
        else:
            cause = "no_confirmed_mismatch"
        rows.append(
            {
                "condition": "similar_content",
                "key": key,
                "sisnr_i": sisnri,
                "relative_band": rel_band(sisnri, p10, med, p75),
                "condition_p10": float(p10),
                "condition_median": float(med),
                "condition_p75": float(p75),
                "failure_cause": cause,
                "primary_mismatch_rate": joint,
                "secondary_mismatch_rate": content,
                "waveform_interferer_rate": wave,
                "ecapa_speaker_mismatch_rate": spk,
                "content_similarity": f(c, "content_similarity"),
                "target_spk": c.get("target_spk", ""),
                "interferer_spk": c.get("interferer_spk", ""),
            }
        )
    return rows, {"p10": float(p10), "median": float(med), "p75": float(p75), "n": len(rows)}


def group_counts(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return dict(sorted(out.items(), key=lambda x: (-x[1], x[0])))


def cause_summary(rows: list[dict]) -> dict:
    out = {}
    for cond in sorted(set(r["condition"] for r in rows)):
        cr = [r for r in rows if r["condition"] == cond]
        out[cond] = {
            "n": len(cr),
            "sisnr_i": summarize([f(r, "sisnr_i") for r in cr]),
            "relative_bands": group_counts(cr, "relative_band"),
            "failure_causes": group_counts(cr, "failure_cause"),
            "absolute_failure_lt0": {
                "n": sum(r["sisnr_i"] < 0 for r in cr),
                "cause_counts": group_counts([r for r in cr if r["sisnr_i"] < 0], "failure_cause"),
                "primary_mismatch_rate": summarize([f(r, "primary_mismatch_rate") for r in cr if r["sisnr_i"] < 0]),
            },
            "weak_or_failed_lt5": {
                "n": sum(r["sisnr_i"] < 5 for r in cr),
                "cause_counts": group_counts([r for r in cr if r["sisnr_i"] < 5], "failure_cause"),
                "primary_mismatch_rate": summarize([f(r, "primary_mismatch_rate") for r in cr if r["sisnr_i"] < 5]),
            },
            "relative_tail_le_p10": {
                "n": sum(r["relative_band"] in {"absolute_failure_lt0", "absolute_weak_0_5", "relative_tail_le_p10"} for r in cr),
                "cause_counts": group_counts(
                    [r for r in cr if r["relative_band"] in {"absolute_failure_lt0", "absolute_weak_0_5", "relative_tail_le_p10"}],
                    "failure_cause",
                ),
                "primary_mismatch_rate": summarize(
                    [
                        f(r, "primary_mismatch_rate")
                        for r in cr
                        if r["relative_band"] in {"absolute_failure_lt0", "absolute_weak_0_5", "relative_tail_le_p10"}
                    ]
                ),
            },
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spk_rows, spk_ref = analyze_similar_speaker()
    cont_rows, cont_ref = analyze_similar_content()
    all_rows = spk_rows + cont_rows
    write_csv(OUT / "condition_aware_utterance_taxonomy.csv", all_rows)
    write_csv(OUT / "top_extreme_failures.csv", sorted(all_rows, key=lambda r: r["sisnr_i"])[:200])
    write_csv(
        OUT / "top_relative_tail.csv",
        sorted(
            [r for r in all_rows if r["relative_band"] in {"absolute_failure_lt0", "absolute_weak_0_5", "relative_tail_le_p10"}],
            key=lambda r: (r["condition"], r["sisnr_i"]),
        ),
    )
    summary = {
        "reference_distribution": {
            "similar_speaker": spk_ref,
            "similar_content": cont_ref,
        },
        "summary": cause_summary(all_rows),
        "interpretation": {
            "primary_mismatch_rate": {
                "similar_speaker": "multi-verifier true-swap rate using ECAPA + SpeechBrain + waveform preference",
                "similar_content": "joint SSL content mismatch + waveform-interferer rate",
            },
            "relative_tail": "SI-SNRi <= condition p10, plus absolute failure/weak bands",
        },
    }
    with (OUT / "condition_aware_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
