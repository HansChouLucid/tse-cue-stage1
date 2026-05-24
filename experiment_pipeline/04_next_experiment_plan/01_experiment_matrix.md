# Stage 04 Next Experiment Matrix

## Current Position

Stages 01-03 have established a validation chain for USEF-only:

```text
hard data -> chunk verifier sanity -> USEF model outputs -> condition-aware failure taxonomy
```

The next step is to test whether the observed mismatch pattern is specific to USEF or shared by other fine-grained cue mechanisms.

## Required Before Strong Cue Comparison

Two validation gaps were addressed in this round:

| Gap | Needed Work |
|---|---|
| content verifier clean-source sanity | first SSL non-collapse sanity passed on clean similar-content chunks |
| normal/easy USEF reference | completed with pretrained USEF-TFGridNet on internal Libri2Mix-style reference |

Remaining caveats:

```text
The SSL sanity is not yet a hard retrieval-style content-verifier calibration.
The normal/easy reference is internal and Libri2Mix-style, not byte-identical official reproduction.
similar-speaker and similar-content v1 do not show broad hard-set average degradation; compare tail behavior there.
similar-content v2 TTS shows broad degradation. Quality calibration supports voice/energy/duration sanity,
but ASR text sanity shows the synthetic transcript is not clean enough for a strict same-text benchmark claim.
```

## Controlled Single-Cue Experiments

Use the same data and as similar a backbone as possible.

| Cue | Status | Purpose |
|---|---|---|
| USEF-only | completed diagnostic | current reference |
| TFMap-only | next | test whether time-frequency cue changes identity/content mismatch |
| Context-only | next | test whether contextual cue changes identity/content mismatch |

Temporarily excluded:

```text
listen cue
cue combinations
large backbone changes
weak global-spkemb baseline as final comparison
```

## Evaluation Conditions

| Condition | Main hard factor | Primary local evidence |
|---|---|---|
| similar-speaker | speaker identity similarity | multi-verifier identity + waveform |
| similar-content v1 | natural content similarity | SSL content + waveform |
| similar-content v2 TTS | TTS content stress in interferer voice | SSL content + waveform + TTS quality checks |

## Required Metrics

For each cue and condition:

| Metric | Purpose |
|---|---|
| SI-SNRi mean / median | global quality |
| SI-SNRi < 0 | extreme failure |
| SI-SNRi < 5 | weak or failed output |
| condition p10 tail | relative degradation |
| mismatch-explained extreme failures | causal attribution |
| verifier-instability category | metric reliability check |

## Decision Rule

A cue is more robust only if it improves both:

```text
global separation quality
and
condition-aware local mismatch attribution
```

Possible outcomes:

| Outcome | Interpretation |
|---|---|
| higher SI-SNRi and lower mismatch | stronger cue under hard condition |
| higher SI-SNRi but similar mismatch | globally better, still locally fragile |
| similar SI-SNRi but lower mismatch | cue may improve attribution robustness |
| lower SI-SNRi and lower mismatch | possible quality/robustness tradeoff |

## Recommended Execution Order

1. Decide whether to improve v2 TTS text fidelity or keep it as a content-stress diagnostic.
2. Prepare TFMap-only inference on similar-speaker, similar-content v1, and similar-content v2.
3. Prepare Context-only inference on the same sets.
4. Run the same condition-aware taxonomy used for USEF-only.
5. Compare cue mechanisms with identical tables.
6. Add stronger retrieval-style SSL/ASR calibration.
