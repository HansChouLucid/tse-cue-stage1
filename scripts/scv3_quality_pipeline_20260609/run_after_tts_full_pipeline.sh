#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate tse

ROOT=/data/tse_cue_project
WORK=$ROOT/diagnostics/similar_content_v3_tts_enroll_conflict_quality_first_20260609
MANIFEST=$ROOT/manifests/librimix_similar_content_v3_tts_enroll_conflict_quality_first_20260609
OUT=$ROOT/diagnostics/similar_content_v3_quality_first_full_pipeline_20260609
TOOLS=$ROOT/repos/stage1-tools
mkdir -p "$OUT/logs" "$OUT/scripts"

EXPECTED_PER_SHARD=2815
wait_for_tts() {
  while true; do
    n0=$(wc -l < "$WORK/logs/f5tts_quality_naturalized_shard0_gpu0.jsonl" 2>/dev/null || echo 0)
    n1=$(wc -l < "$WORK/logs/f5tts_quality_naturalized_shard1_gpu1.jsonl" 2>/dev/null || echo 0)
    echo "[wait_tts] $(date '+%F %T') shard0=$n0/$EXPECTED_PER_SHARD shard1=$n1/$EXPECTED_PER_SHARD"
    if [ "$n0" -ge "$EXPECTED_PER_SHARD" ] && [ "$n1" -ge "$EXPECTED_PER_SHARD" ]; then
      break
    fi
    sleep 120
  done
}

merge_inference() {
  local out_dir=$1
  python - "$out_dir" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
base=Path(sys.argv[1])
parts=sorted(base.glob('inference_summary_shard*.csv'))
if not parts:
    raise SystemExit(f'no shard summaries in {base}')
df=pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).sort_values('key')
df.to_csv(base/'inference_summary.csv', index=False)
summary={'n':int(len(df))}
for col in ['sisnr_est','sisnr_i','si_snr_est','si_snri','si_sdr_est','si_sdri']:
    if col in df.columns:
        s=pd.to_numeric(df[col], errors='coerce').dropna()
        if len(s):
            summary[col+'_mean']=float(s.mean())
            summary[col+'_median']=float(s.median())
            summary[col+'_p10']=float(s.quantile(0.10))
            summary[col+'_p90']=float(s.quantile(0.90))
            summary[col+'_lt0_rate']=float((s<0).mean())
(base/'metric_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
PY
}

run_content_diag() {
  local name=$1
  local bundle=$2
  local inf=$bundle/full_infer/inference_summary.csv
  local est=$bundle/full_infer/est_wavs
  local meta=$bundle/metadata.jsonl
  local out=$OUT/content_diagnostic/$name
  mkdir -p "$out"
  if [ -f "$out/DONE" ]; then
    echo "[skip] content diag $name"
    return
  fi
  echo "[diag] $name start $(date)"
  CUDA_VISIBLE_DEVICES=0 python "$TOOLS/run_similar_content_ssl_diagnostic.py" \
    --metadata-jsonl "$meta" --est-dir "$est" --inference-summary "$inf" \
    --out-dir "$out/content_ssl" --sample-rate 8000 --batch-size 96 --device cuda \
    > "$OUT/logs/${name}_content_ssl.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python "$TOOLS/run_ssl_content_diagnostic_bundle.py" \
    --metadata-jsonl "$meta" --est-dir "$est" --inference-summary "$inf" \
    --out-dir "$out/content_wavlm" --bundle WAVLM_BASE_PLUS --layer-idx 9 \
    --sample-rate 8000 --batch-size 48 --device cuda \
    > "$OUT/logs/${name}_content_wavlm.log" 2>&1 &
  p0=$!
  CUDA_VISIBLE_DEVICES=1 python "$TOOLS/run_ssl_content_diagnostic_bundle.py" \
    --metadata-jsonl "$meta" --est-dir "$est" --inference-summary "$inf" \
    --out-dir "$out/content_hubert" --bundle HUBERT_BASE --layer-idx 6 \
    --sample-rate 8000 --batch-size 48 --device cuda \
    > "$OUT/logs/${name}_content_hubert.log" 2>&1 &
  p1=$!
  wait $p0 $p1
  python "$ROOT/diagnostics/pretrained_usef_tfgridnet_completion_20260604/src/merge_content_2tri.py" \
    --w2v-chunk "$out/content_ssl/ssl_content_per_chunk_1s.csv" \
    --w2v-utt "$out/content_ssl/ssl_content_per_utterance_1s.csv" \
    --wavlm-chunk "$out/content_wavlm/wavlm_base_plus_per_chunk_1s.csv" \
    --wavlm-utt "$out/content_wavlm/wavlm_base_plus_per_utterance_1s.csv" \
    --hubert-chunk "$out/content_hubert/hubert_base_per_chunk_1s.csv" \
    --hubert-utt "$out/content_hubert/hubert_base_per_utterance_1s.csv" \
    --inference-summary "$inf" --out-dir "$out/content_multiverifier"
  date > "$out/DONE"
}

wait_for_tts

QUALITY_CSV=$OUT/quality_tts_audio_manifest.csv
ASR_INPUT_CSV=$OUT/quality_tts_audio_manifest.asr_input.csv
ASR_DIR=$OUT/full_whisper_large_v3
ASR_CSV=$ASR_DIR/quality_smoke40_whisper.csv
if [ ! -f "$QUALITY_CSV" ]; then
  python "$OUT/scripts/make_tts_quality_csv.py" \
    --work-dir "$WORK" --manifest-dir "$MANIFEST" --out-csv "$QUALITY_CSV" \
    > "$OUT/logs/make_tts_quality_csv.log" 2>&1
fi
if [ ! -f "$ASR_INPUT_CSV" ]; then
  python - "$QUALITY_CSV" "$ASR_INPUT_CSV" <<'PY'
import csv, sys
from pathlib import Path
inp=Path(sys.argv[1]); out=Path(sys.argv[2])
rows=[]
with inp.open(encoding='utf-8', newline='') as f:
    reader=csv.DictReader(f)
    fields=reader.fieldnames
    for row in reader:
        ok_audio=str(row.get('pass_audio','')).lower() == 'true'
        ok_status=row.get('status') == 'ok'
        ok_path=Path(row.get('path','')).exists()
        if ok_audio and ok_status and ok_path:
            rows.append(row)
with out.open('w', encoding='utf-8', newline='') as f:
    w=csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print('asr_input_rows', len(rows), 'from', inp)
PY
fi
if [ ! -f "$ASR_CSV" ]; then
  CUDA_VISIBLE_DEVICES=0 python "$WORK/scripts/whisper_smoke_csv.py" \
    --csv "$ASR_INPUT_CSV" \
    --model /root/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478 \
    --out-dir "$ASR_DIR" --device cuda --compute-type float16 --beam-size 5 \
    > "$OUT/logs/full_whisper_large_v3.log" 2>&1
fi

if [ ! -f "$OUT/bundles/quality_first_scv3_content/bundle_summary.json" ]; then
  python "$OUT/scripts/build_quality_first_scv3_bundle.py" \
    --project-root "$ROOT" --work-dir "$WORK" --manifest-dir "$MANIFEST" \
    --out-base "$OUT" --asr-csv "$ASR_CSV" --copy-final-enrollment \
    > "$OUT/logs/build_quality_first_bundle.log" 2>&1
fi

BUNDLE=$OUT/bundles/quality_first_scv3_content
SCP=$BUNDLE/scp

USEF_BASE=$ROOT/diagnostics/pretrained_usef_tfgridnet_completion_20260604
USEF_CONFIG=$USEF_BASE/hf/chkpt/USEF-TFGridNet/config.yaml
USEF_CKPT=$USEF_BASE/hf/chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar
USEF_OUT=$OUT/bundles/pretrained_usef_tfgridnet_scv3_quality/full_infer
if [ ! -f "$USEF_OUT/inference_summary.csv" ]; then
  rm -rf "$USEF_OUT"; mkdir -p "$USEF_OUT"
  CUDA_VISIBLE_DEVICES=0 python "$USEF_BASE/src/run_usef_tfgridnet_bundle.py" --config "$USEF_CONFIG" --checkpoint "$USEF_CKPT" --scp-dir "$SCP" --out-dir "$USEF_OUT" --device cuda:0 --shard-index 0 --num-shards 2 > "$OUT/logs/usef_tfgridnet_shard0.log" 2>&1 &
  p0=$!
  CUDA_VISIBLE_DEVICES=1 python "$USEF_BASE/src/run_usef_tfgridnet_bundle.py" --config "$USEF_CONFIG" --checkpoint "$USEF_CKPT" --scp-dir "$SCP" --out-dir "$USEF_OUT" --device cuda:0 --shard-index 1 --num-shards 2 > "$OUT/logs/usef_tfgridnet_shard1.log" 2>&1 &
  p1=$!
  wait $p0 $p1
  merge_inference "$USEF_OUT"
fi
cp "$BUNDLE/metadata.jsonl" "$OUT/bundles/pretrained_usef_tfgridnet_scv3_quality/metadata.jsonl"

TFMC_SCRIPT=$ROOT/diagnostics/real_tse_tfmap_context_20260607/run_real_tse_scp_inference.py
TFMC_MODEL=$ROOT/pretrained/real_tse_baselines/extracted/tfmap_context_100
TFMC_OUT=$OUT/bundles/pretrained_tfmap_context_scv3_quality/full_infer
if [ ! -f "$TFMC_OUT/inference_summary.csv" ]; then
  rm -rf "$TFMC_OUT"; mkdir -p "$TFMC_OUT"
  CUDA_VISIBLE_DEVICES=0 python "$TFMC_SCRIPT" --model-dir "$TFMC_MODEL" --scp-dir "$SCP" --out-dir "$TFMC_OUT" --device cuda:0 --shard-index 0 --num-shards 2 > "$OUT/logs/tfmap_context_shard0.log" 2>&1 &
  p0=$!
  CUDA_VISIBLE_DEVICES=1 python "$TFMC_SCRIPT" --model-dir "$TFMC_MODEL" --scp-dir "$SCP" --out-dir "$TFMC_OUT" --device cuda:0 --shard-index 1 --num-shards 2 > "$OUT/logs/tfmap_context_shard1.log" 2>&1 &
  p1=$!
  wait $p0 $p1
  merge_inference "$TFMC_OUT"
fi
cp "$BUNDLE/metadata.jsonl" "$OUT/bundles/pretrained_tfmap_context_scv3_quality/metadata.jsonl"

run_bsrnn_one() {
  local cue=$1
  local cfg=$2
  local ckpt=$3
  local gpu=$4
  local out=$OUT/bundles/bsrnn_${cue}_scv3_quality
  if [ -f "$out/DONE" ]; then
    echo "[skip] bsrnn $cue"
    return
  fi
  mkdir -p "$out/infer_run"
  cd "$ROOT/repos/wesep-real-tse"
  CUDA_VISIBLE_DEVICES=$gpu python wesep/bin/infer.py --config "$cfg" --fs 16k --gpus 0 \
    --exp_dir "$out/infer_run" --data_type raw \
    --test_data "$BUNDLE/bsrnn/samples.jsonl" --test_cues "$BUNDLE/bsrnn/cues.yaml" \
    --save_wav true --checkpoint "$ckpt" > "$OUT/logs/bsrnn_${cue}.infer.log" 2>&1
  python "$OUT/scripts/postprocess_bsrnn_scv3_bundle.py" \
    --cue-dir "$out" --manifest "$BUNDLE/bsrnn/samples.jsonl" --metadata-template "$BUNDLE/metadata.jsonl"
}

BSRNN_USEF_CKPT=$ROOT/experiments/normal_basic_single_cue/usef_only_bsrnn_stage2_stabilize_lr2e6_3ep_a800_gpu1_bs8_20260608/models/checkpoint_2.pt
BSRNN_TFMAP_CKPT=$ROOT/experiments/normal_basic_single_cue/tfmap_only_bsrnn_stage2_A_lr3e6_5ep_a800_gpu0_bs8_20260608/models/checkpoint_4.pt
BSRNN_CONTEXT_CKPT=$ROOT/experiments/normal_basic_single_cue/context_only_bsrnn_continue_from_total17_plus10_lr1e5_dualA100_ddp2_bs4_20260608/models/checkpoint_9.pt
USEF_CFG=examples/audio/librimix/confs/tse_bsrnn_spk_usef_train100_ft13_from5.yaml
TFMAP_CFG=examples/audio/librimix/confs/tse_bsrnn_spk_tfmap_train100_ft13_from5.yaml
CONTEXT_CFG=examples/audio/librimix/confs/tse_bsrnn_spk_context_train100_ft13_from5.yaml

run_bsrnn_one usef "$USEF_CFG" "$BSRNN_USEF_CKPT" 0
run_bsrnn_one tfmap "$TFMAP_CFG" "$BSRNN_TFMAP_CKPT" 1
run_bsrnn_one context "$CONTEXT_CFG" "$BSRNN_CONTEXT_CKPT" 0

run_content_diag pretrained_usef_tfgridnet_scv3_quality "$OUT/bundles/pretrained_usef_tfgridnet_scv3_quality"
run_content_diag pretrained_tfmap_context_scv3_quality "$OUT/bundles/pretrained_tfmap_context_scv3_quality"
run_content_diag bsrnn_usef_scv3_quality "$OUT/bundles/bsrnn_usef_scv3_quality"
run_content_diag bsrnn_tfmap_scv3_quality "$OUT/bundles/bsrnn_tfmap_scv3_quality"
run_content_diag bsrnn_context_scv3_quality "$OUT/bundles/bsrnn_context_scv3_quality"

python "$OUT/scripts/summarize_scv3_quality_pipeline.py" --base "$OUT" > "$OUT/logs/final_summary.log" 2>&1
date > "$OUT/DONE"
