# Stage 01 Data Preparation

## Purpose

This stage prepares hard-condition data for local mismatch diagnosis.

The data must support two different probes:

```text
similar-speaker: stress identity attribution
similar-content: stress content attribution
```

It must also avoid cue leakage: enrollment should not be the current target utterance.

## Source Data

Remote LibriSpeech root:

```text
/data/tse_cue_project/imported_data/LibriSpeech960
```

Included splits:

```text
dev-clean
dev-other
test-clean
test-other
train-clean-100
train-clean-360
train-other-500
```

Libri2Mix-style metadata:

```text
/data/tse_cue_project/imported_data/Libri2Mix
```

Metadata coverage:

```text
train-clean-100
train-clean-360
train-other-500
dev-clean
test-clean
```

`train-other-500` was added through LibriMix-style pairing from LibriSpeech train-other-500.

## Similar-Speaker Data

Purpose:

```text
make target speaker and interferer speaker acoustically similar
```

Remote paths:

| Item | Path |
|---|---|
| manifest | `/data/tse_cue_project/manifests/librimix_similar_speaker_960h/clean` |
| audio | `/data/tse_cue_project/datasets/Libri2Mix_similar_speaker_960h/wav16k/min` |
| speaker features | `/data/tse_cue_project/features/similar_speaker/ecapa_librimix_960h` |

Sample counts:

| Split | Count |
|---|---:|
| train-100 | 27800 |
| train-360 | 101600 |
| train-other-500 | 149600 |
| dev | 6000 |
| test | 6000 |

Quality checks:

```text
skipped = 0
missing_source_entries = 0
failed_embedding_sources = 0
enrollment_fallback_self = 0
```

## Similar-Content Data

Purpose:

```text
make target content and interferer content partially similar
```

Remote paths:

| Item | Path |
|---|---|
| manifest | `/data/tse_cue_project/manifests/librimix_similar_content_960h/clean` |
| audio | `/data/tse_cue_project/datasets/Libri2Mix_similar_content_960h/wav16k/min` |

Sample counts:

| Split | Count |
|---|---:|
| train-100 | 27799 |
| train-360 | 101597 |
| train-other-500 | 149534 |
| dev | 5989 |
| test | 5989 |

Quality checks:

```text
train-100 skipped = 1
train-360 skipped = 3
train-other-500 skipped = 66
dev skipped = 11
test skipped = 11
enrollment_fallback_self = 0
```

Interpretation:

```text
similar-content currently uses natural LibriSpeech text overlap / similarity.
It is a useful first probe, but it is not a maximally hard content-control setting.
```

## Similar-Content v2 TTS Data

Purpose:

```text
force the interferer to speak the target transcript while keeping the interferer voice
```

Construction:

```text
target speech: original LibriSpeech target utterance
target text: LibriSpeech transcript
synthetic interferer: F5-TTS conditioned on interferer prompt audio and prompt text
mixture: target speech + synthetic interferer at 0 dB SIR
enrollment cue for USEF: target-speaker utterance different from the current target utterance
```

Remote paths:

| Item | Path |
|---|---|
| TTS plan | `/data/tse_cue_project/manifests/librimix_similar_content_v2_tts/clean/test/tts_plan.jsonl` |
| synthetic interferers | `/data/tse_cue_project/manifests/librimix_similar_content_v2_tts/clean/test/synthetic_interferer` |
| final dataset | `/data/tse_cue_project/datasets/Libri2Mix_similar_content_v2_tts/clean/test` |

Sample counts:

| Item | Count |
|---|---:|
| TTS plan entries | 6000 |
| synthetic interferers generated | 6000 |
| final mixtures | 5984 |
| skipped because final overlap was too short | 16 |

Important correction:

```text
An initial v2 build accidentally used the interferer prompt as USEF enrollment.
That produced artificial reverse-target separation and was archived as a wrong-aux run.
The corrected build uses target-speaker enrollment, and same-as-current-target enrollment count is 0.
```

Remaining caveat:

```text
v2 is a stronger content stress test, but it introduces TTS artifacts.
Strong causal claims need TTS quality calibration: text correctness, speaker preservation, duration, and energy checks.
```

## Enrollment Cue Policy

Both datasets use speaker-specific enrollment candidate sets:

```yaml
policy:
  type: random
  key: mix_spk_id
```

Purpose:

```text
avoid using the current target utterance as enrollment;
reduce cue leakage;
make local mismatch diagnosis more credible.
```

## Validation Status

| Assumption | Status |
|---|---|
| full 960h-based data exists | validated |
| similar-speaker test has 6000 samples | validated |
| similar-content test has 5989 samples | validated |
| similar-content v2 TTS test has 5984 corrected samples | validated |
| enrollment avoids current target/source utterance in USEF input preparation | validated for current USEF runs |
| similar-content v1 is sufficiently hard | partially validated |
| similar-content v2 TTS is harder than v1 | supported by corrected USEF result |

## Remaining Data Gap

The current similar-content data is not extremely difficult. Many pairs share words or short text fragments, but they are not forced to say the same sentence.

Future hard-content options:

```text
raise text overlap threshold
use phrase-level / phoneme-level overlap
use forced alignment for local word overlap
improve / calibrate TTS same-text multi-speaker stress tests
```
