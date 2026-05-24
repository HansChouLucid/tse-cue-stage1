# Stage 02 Chunk-Level Verifier Sanity

## Purpose

This stage validates a key assumption:

```text
Before using a verifier on model outputs, check whether it can separate
clean target/interferer chunks at the same local time scale.
```

This is not outdated pilot work. It is the foundation for local mismatch diagnosis.

## What Was Tested

The first verifier sanity experiment uses clean source audio, not model outputs.

```text
verifier: ECAPA speaker verifier
chunk sizes: 1.0s and 2.0s
score: sim(chunk, correct speaker prototype) - sim(chunk, wrong speaker prototype)
```

## Similar-Speaker Clean-Source Sanity

Question:

```text
Even when target and interferer speakers are similar,
can ECAPA separate the two speakers at chunk level?
```

Setting:

| Item | Value |
|---|---|
| data | similar-speaker test |
| samples | 1000 manifest samples |
| chunks | 1.0s / 2.0s |
| hop | chunk / 2 |
| speaker anchor | up to 3 enrollment utterances averaged into prototype |

Result:

| chunk | AUC | accuracy | target margin mean | target margin p10 | interferer margin mean | interferer margin p10 |
|---|---:|---:|---:|---:|---:|---:|
| 1.0s | 0.9998 | 0.9944 | 0.3307 | 0.1794 | 0.3248 | 0.1775 |
| 2.0s | 0.99999 | 0.9977 | 0.3981 | 0.2642 | 0.3948 | 0.2640 |

Interpretation:

```text
ECAPA is stable on clean similar-speaker chunks.
Therefore, ECAPA failure on model outputs should not be dismissed as pure clean-source verifier incapability.
```

## Similar-Content Clean-Source Speaker Sanity

Question:

```text
When content is similar but speaker identity differs,
does ECAPA remain stable on clean chunks?
```

Setting:

| Item | Value |
|---|---|
| data | similar-content test |
| samples | 100 manifest samples |
| chunks | 1.0s / 2.0s |
| verifier | ECAPA |

Result:

| chunk | AUC | accuracy | target margin mean |
|---|---:|---:|---:|
| 1.0s | 0.99999 | 0.9991 | 0.5000 |
| 2.0s | 1.0000 | 1.0000 | 0.6017 |

Interpretation:

```text
Speaker identity is easy on clean similar-content chunks.
But this does not validate content mismatch detection.
Similar-content needs a content verifier, not only a speaker verifier.
```

## What This Stage Supports

This stage supports:

```text
1s chunks are usable for first-pass local speaker diagnostics;
2s chunks are more stable and useful for robustness checks;
ECAPA has clean-source speaker discrimination ability;
SSL content features do not collapse on clean similar-content source chunks.
```

## What This Stage Does Not Support

This stage does not prove:

```text
ECAPA is always stable on separated model outputs;
ECAPA alone is enough to confirm local mismatch;
content mismatch can be detected with speaker verification.
```

Those assumptions require Stage 03 and Stage 05 checks.

## Remote Outputs

```text
/data/tse_cue_project/experiments/local_mismatch_pilot/similar_speaker_test_1000
/data/tse_cue_project/experiments/local_mismatch_pilot/similar_content_test_100
```

Remote script:

```text
/data/tse_cue_project/code/wesep-real-tse/tools/run_local_mismatch_pilot.py
```

## Remaining Verifier Gap

The old missing validation was clean-source content-verifier sanity. A first version is now complete:

| Item | Value |
|---|---:|
| data | similar-content test |
| samples | 1000 |
| chunks | 10985 one-second chunks |
| target clean-gap mean | 0.4000 |
| target clean-gap p01 | 0.1855 |
| target clean-gap min | 0.0351 |
| target / interferer clean-source accuracy | 1.000 / 1.000 |

Interpretation:

```text
On clean target/interferer sources, the SSL content representation keeps a clear margin even under similar-content construction.
Therefore, similar-content output mismatches are unlikely to be caused only by clean-source SSL collapse.
```

Important caveat:

```text
This sanity compares aligned clean target/interferer chunks and therefore is easier than a retrieval-style content verifier test.
It validates non-collapse, not full content-verifier calibration.
```
