# Local Mismatch Definition And Validation Logic

## Core Question

We are not only asking whether a TSE model has a lower utterance-level SI-SNRi.

The research question is:

```text
Does a fine-grained cue TSE model locally drift toward the interferer
under hard target/interferer conditions?
```

The word locally is essential. A model may be globally strong while still producing short segments that are attributed to the wrong speaker or wrong content.

## Main Hypotheses

| ID | Hypothesis | Why It Matters | Required Validation |
|---|---|---|---|
| H1 | local mismatch exists under hard conditions | motivates fine-grained diagnostic beyond global SI-SNRi | model-output local metrics |
| H2 | mismatch is condition-dependent | similar-speaker and similar-content may fail differently | separate verifier per condition |
| H3 | chunk-level verifier is stable enough to support local diagnosis | otherwise local mismatch metrics may be artifacts | clean-source verifier sanity |
| H4 | single-verifier evidence is insufficient | speaker verifier can be unstable on separated chunks | second verifier and waveform checks |
| H5 | extreme global failures are often mismatch-driven | links local mismatch to meaningful model failure | failure taxonomy over negative / weak samples |
| H6 | relative SI-SNRi decline can reveal additional failure | strong models may fail without becoming negative | p10-tail and relative-degradation analysis |

## Local Metrics

### Speaker Gap

```text
speaker_gap(t)
= sim(output_chunk_t, target_speaker)
- sim(output_chunk_t, interferer_speaker)
```

| Condition | Meaning |
|---|---|
| `speaker_gap(t) > 0` | output chunk is closer to target speaker |
| `speaker_gap(t) < 0` | output chunk is closer to interferer speaker |

### Local Waveform Gap

```text
local_sisdr_gap(t)
= SI-SDR(output_chunk_t, target_chunk_t)
- SI-SDR(output_chunk_t, interferer_chunk_t)
```

| Condition | Meaning |
|---|---|
| `local_sisdr_gap(t) > 0` | waveform is closer to target |
| `local_sisdr_gap(t) < 0` | waveform is closer to interferer |

### Content Gap

```text
content_gap(t)
= sim(SSL_content(output_chunk_t), SSL_content(target_chunk_t))
- sim(SSL_content(output_chunk_t), SSL_content(interferer_chunk_t))
```

| Condition | Meaning |
|---|---|
| `content_gap(t) > 0` | local content is closer to target |
| `content_gap(t) < 0` | local content is closer to interferer |

## Evidence Levels

### Similar-Speaker

| Evidence | Definition | Reliability |
|---|---|---|
| single speaker verifier | ECAPA or SpeechBrain alone says output is closer to interferer | weak |
| multi-verifier identity disagreement | ECAPA and SpeechBrain both prefer interferer | medium |
| waveform-confirmed identity swap | multi-verifier evidence plus `local_sisdr_gap < 0` | strong |

### Similar-Content

| Evidence | Definition | Reliability |
|---|---|---|
| content-only mismatch | SSL content gap prefers interferer | medium |
| waveform-confirmed content mismatch | content gap and local waveform gap both prefer interferer | strong |
| speaker verifier mismatch | ECAPA local speaker disagreement | side evidence only |

## Experiment Logic

The pipeline should be read as a validation chain.

| Stage | Question | Current Status |
|---|---|---|
| 01 Data preparation | Are hard-condition data and enrollment cues constructed without obvious leakage? | completed |
| 02 Verifier sanity | Can chunk-level verifier separate clean target/interferer sources? | completed for ECAPA on clean chunks |
| 03 USEF-only diagnostic | Does a strong pretrained USEF model show local mismatch on hard conditions? | completed |
| 05 Failure taxonomy | Are extreme and relative-tail failures explained by mismatch? | completed for USEF |
| 04 Next experiments | Do TFMap-only and Context-only change the mismatch pattern? | not yet done |

## Current Supported Claims

The current evidence supports:

```text
1. local mismatch can be measured with condition-aware metrics;
2. USEF-TFGridNet is globally strong but still has hard-condition failures;
3. similar-speaker failures are mostly identity-mismatch driven;
4. similar-content failures are mostly content-attribution driven;
5. ECAPA alone is not enough for final local mismatch claims.
```

## Current Logic Gaps

These are not resolved yet:

| Gap | Why It Matters | Needed Experiment |
|---|---|---|
| content verifier sanity on clean chunks is incomplete | SSL content gap was used on model outputs, but clean-source calibration is not yet as explicit as ECAPA sanity | run content verifier sanity on clean target/interferer chunks |
| ECAPA stability on separated outputs is weaker than on clean chunks | domain shift may create false local identity disagreement | compare ECAPA, SpeechBrain, and waveform on more controlled outputs |
| similar-content difficulty may be too mild | current content pairs often share words rather than near-identical utterances | build harder similar-content v2 |
| relative degradation is not tied to official USEF reference distribution | p10 tail is internal to our hard set | run normal/easy Libri2Mix reference or official-condition inference |
| TFMap-only and Context-only are not evaluated yet | cannot compare cue mechanisms | run controlled single-cue experiments |

These gaps should be treated as future validation work, not as contradictions of the current USEF-only findings.
