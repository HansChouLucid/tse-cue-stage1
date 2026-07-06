# Stage-1 Current Status And Next Steps

Date: 2026-07-06

This note is the compact current-status companion to the restart guide.

## Current Valid Scope

Use only the current mainline:

- Speaker side:
  - normal/easy
  - similar-speaker v2
  - similar-speaker v3
- Content side:
  - normal/easy
  - quality-first similar-content v3
- Backbone groups:
  - pretrained USEF-TFGridNet
  - pretrained REAL-TSE TFMap+Context
  - BSRNN controlled backbone with USEF / TFMap / Context cue

Do not use old `similar-content v2 TTS` or old pre-rebuild `similar-content v3` as current formal evidence.

## Already Completed

- Current formal report cleaned to the quality-first v3 boundary.
- GitHub freeze package prepared with:
  - scripts
  - pruned result summaries
  - selected WeSep Stage-1 files
  - restart / artifact-boundary documentation
- Pretrained USEF-TFGridNet coverage completed for:
  - normal/easy
  - similar-speaker v2
  - similar-speaker v3
  - quality-first similar-content v3
- Pretrained TFMap+Context coverage completed for:
  - normal/easy
  - similar-speaker v2
  - similar-speaker v3
  - quality-first similar-content v3
- BSRNN controlled three-cue line available for:
  - normal/basic
  - similar-speaker v3
  - quality-first similar-content v3
- Quality-first similar-content v3 quality validation completed:
  - final rows: 5256
  - Whisper large-v3 ASR pass rate: 99.98%
  - WER mean: 0.033
  - CER mean: 0.015
  - token F1 mean: 0.971

## Current Interpretation Boundary

- The strongest content-side comparison is pretrained USEF-TFGridNet vs pretrained TFMap+Context on quality-first similar-content v3.
- The BSRNN three-cue line is useful for controlled cue comparison, but should still be treated as a weaker backbone line until continuation checkpoints are fully rerun.
- Speaker-side v3 is the cleanest cross-backbone hard stress test.

## Next Work

### P0

- Keep the local freeze package and the GitHub branch aligned to one current-valid scope.
- Ensure report snapshot, scripts, result summaries, and restart/status docs do not contradict one another.

### P1

- Run paired tail attribution for pretrained USEF-TFGridNet vs pretrained TFMap+Context on quality-first similar-content v3.
- Run content-side mechanism probes so cue ambiguity can be compared against confirmed content drift.

### P1

- Re-run BSRNN continuation checkpoints on:
  - normal/basic
  - similar-speaker v3
  - quality-first similar-content v3
- Check whether the weak 18-epoch three-cue ordering changes after stronger checkpoints.

### P2

- If resources allow, train matched-backbone dual-cue systems such as TFMap+Context or USEF+Context.
- Use them to test whether content-side gains come from cue composition rather than checkpoint/backbone mismatch.

## Archive Rule

Historical materials may remain for traceability, but they must be clearly treated as archive-only if they still mention:

- `similar-content v2 TTS`
- old pre-rebuild `similar-content v3`
- old invalid case-study or diagnostic tables
