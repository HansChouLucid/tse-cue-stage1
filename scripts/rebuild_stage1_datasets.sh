#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-/data/tse_cue_project}
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TOOLS="$ROOT/code/wesep-real-tse/tools"
mkdir -p "$TOOLS"
cp -av "$REPO_DIR"/dataset_build_tools/*.py "$TOOLS"/
cat <<MSG
Dataset build tools copied to: $TOOLS

Next steps:
1. Download/extract raw LibriSpeech and Libri2Mix metadata to:
   $ROOT/imported_data/LibriSpeech960
   $ROOT/imported_data/Libri2Mix
2. Activate env:
   source $ROOT/scripts/activate_tse.sh
3. Inspect each build command before running:
   cd $ROOT/code/wesep-real-tse
   python tools/build_similar_speaker_from_librimix_metadata.py --help
   python tools/build_similar_speaker_librimix.py --help
   python tools/build_similar_content_from_librimix_metadata.py --help
   python tools/build_similar_content_v2_tts_plan.py --help
   python tools/run_f5tts_plan_batch.py --help
   python tools/build_similar_content_v2_dataset_from_tts.py --help

See docs/DATA_RECONSTRUCTION.md for the full rebuild plan.
MSG
