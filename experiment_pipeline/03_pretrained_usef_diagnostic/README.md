# Stage 03 USEF-Only Diagnostic

## Purpose

This stage tests whether a strong pretrained fine-grained cue model exposes local mismatch under the hard conditions built in Stage 01.

```text
model: public pretrained USEF-TFGridNet
cue: USEF
backbone: TFGridNet
conditions: similar-speaker and similar-content
```

This stage is a diagnostic reference, not a final comparison between cue types.

## Relation To Earlier Stages

| Dependency | Why It Matters |
|---|---|
| Stage 01 data | provides hard similar-speaker and similar-content test sets |
| Stage 02 verifier sanity | shows ECAPA can separate clean chunks before applying it to model outputs |
| Stage 05 taxonomy | tests whether failures are actually mismatch-driven |

## Completed Experiments

| Experiment | Status | Main result |
|---|---|---|
| similar-speaker USEF inference | done | globally strong but has negative failures |
| similar-speaker ECAPA local diagnostic | done | many local identity disagreements |
| similar-speaker SpeechBrain recheck | done | many ECAPA-only disagreements are not confirmed |
| similar-content v1 USEF inference | done | globally strong but has negative failures |
| similar-content v2 TTS USEF inference | done | clear degradation under same-text synthetic interferer |
| similar-content SSL content diagnostic | done | negative failures are mostly content-attribution mismatch |
| normal/easy USEF reference | done | no broad hard-set average degradation found |
| condition-aware failure taxonomy | done | extreme failures are mostly mismatch-driven |

## Headline Global Results

| Condition | Utterances | SI-SNRi mean | SI-SNRi median | SI-SNRi < 0 | SI-SNRi < 5 |
|---|---:|---:|---:|---:|---:|
| similar-speaker | 6000 | 15.86 dB | 18.49 dB | 323 | 404 |
| similar-content v1 | 5989 | 16.84 dB | 18.81 dB | 215 | 252 |
| similar-content v2 TTS | 5984 | 13.18 dB | 15.81 dB | 343 | 397 |
| normal/easy reference | 6000 | 15.90 dB | 18.69 dB | 310 | 347 |

Interpretation:

```text
USEF-TFGridNet is not a weak model.
The useful question is why the hard-condition tail fails.
```

Reference-set caveat:

```text
The normal/easy reference is rebuilt from Libri2Mix test-clean metadata and LibriSpeech clean sources.
It is an internal USEF reference, not a byte-identical official Libri2Mix reproduction.
```

Relative to normal/easy:

| Condition | Mean difference | Median difference | p10 difference |
|---|---:|---:|---:|
| similar-speaker | -0.03 dB | -0.20 dB | -1.04 dB |
| similar-content v1 | +0.94 dB | +0.12 dB | +1.50 dB |
| similar-content v2 TTS | -2.72 dB | -2.88 dB | -2.64 dB |

Interpretation:

```text
similar-content v2 TTS changes the conclusion for content stress:
the stronger same-text interferer produces both broad degradation and tail failures.
The local diagnostic still matters because it tells whether the degradation is content-attribution mismatch or TTS/data artifacts.
```

## Local Mismatch Results

### Similar-Speaker

Strong identity evidence requires:

```text
ECAPA prefers interferer
and SpeechBrain prefers interferer
and waveform is closer to interferer
```

Key results:

| Metric | Value |
|---|---:|
| ECAPA mismatch, chunk-level | 28.60% |
| multi-verifier mismatch, chunk-level | 6.99% |
| ECAPA true swap, utterance has at least one chunk | 5.80% |
| multi-verifier true swap, utterance has at least one chunk | 4.57% |
| multi-verifier true swap rate > 0.2 | 3.70% |

Interpretation:

```text
ECAPA alone over-reports local identity disagreement.
Multi-verifier + waveform evidence gives a smaller but more reliable set of true identity mismatch.
```

### Similar-Content v1

Strong content evidence requires:

```text
SSL content gap prefers interferer
and local waveform gap prefers interferer
```

Key results:

| Metric | Value |
|---|---:|
| any content mismatch | 22.83% utterances |
| content mismatch rate > 20% | 5.03% |
| any joint content + waveform mismatch | 3.94% |
| joint content + waveform rate > 10% | 3.79% |

Interpretation:

```text
similar-content failures are better explained by content-attribution mismatch than by speaker mismatch.
```

### Similar-Content v2 TTS

v2 uses the same target text synthesized in the interferer voice. The corrected build uses target-speaker enrollment for USEF.

Key global results:

| Metric | Value |
|---|---:|
| SI-SNRi mean | 13.18 dB |
| SI-SNRi median | 15.81 dB |
| SI-SNRi p10 | 9.39 dB |
| SI-SNRi < 0 | 343 / 5984 |
| SI-SNRi < 5 | 397 / 5984 |

Key local content results:

| Metric | Value |
|---|---:|
| any content mismatch | 20.85% utterances |
| content mismatch rate > 20% | 7.83% utterances |
| any joint content + waveform mismatch | 6.14% utterances |
| joint content + waveform rate > 10% | 5.99% utterances |
| mean content mismatch rate when SI-SNRi < 0 | 0.919 |
| mean content mismatch rate when SI-SNRi > 15 | 0.017 |

Interpretation:

```text
v2 is substantially harder than natural similar-content v1.
The failures are not diffuse: the bad samples have very high local content mismatch,
while good samples almost never do.
```

## Failure Attribution

| Condition | Extreme failures | Mismatch-explained | Primary evidence |
|---|---:|---:|---|
| similar-speaker | 323 | 255 | multi-verifier identity true-swap + waveform |
| similar-content v1 | 215 | 207 | SSL content mismatch + waveform |
| similar-content v2 TTS | 343 | pending exact taxonomy count | SSL content mismatch strongly concentrated in failures |

Relative-tail samples:

| Condition | p10 tail count | Mismatch-explained |
|---|---:|---:|
| similar-speaker | 600 | 271 |
| similar-content v1 | 599 | 220 |

Interpretation:

```text
extreme failures are mostly mismatch-driven;
relative degradation is broader and includes non-mismatch or unconfirmed failures.
```

## Files

| File | Content |
|---|---|
| `01_usef_full_similar_speaker.md` | detailed similar-speaker USEF result |
| `02_usef_full_similar_content.md` | detailed similar-content USEF result |
| `03_usef_full_similar_content_v2_tts.md` | detailed TTS similar-content v2 USEF result |
| `../05_reports_and_figures/usef_similar_speaker_cases/` | case studies and second-verifier reports |
| `../05_reports_and_figures/usef_similar_content_diagnostic/` | content diagnostic report |
| `../05_reports_and_figures/condition_aware_failure_taxonomy/` | unified failure attribution |

## Supported Claim

This stage supports:

```text
USEF-style fine-grained cue extraction can fail locally under hard conditions.
The failure mode is condition-dependent:
similar-speaker -> identity mismatch
similar-content -> content-attribution mismatch
```

## Not Yet Supported

This stage does not yet prove:

```text
USEF is more or less robust than TFMap or Context;
TTS similar-content v2 is free of TTS artifacts;
SSL content verifier is fully calibrated as a retrieval test;
relative degradation against byte-identical official USEF conditions is fully quantified.
```
