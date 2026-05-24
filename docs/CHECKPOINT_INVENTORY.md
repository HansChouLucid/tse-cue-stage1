# Checkpoint Inventory

Large model checkpoints are not stored in this GitHub repository. This file records what existed on the remote machine and what must be redownloaded or retrained.

## Critical Controlled BSRNN Checkpoints

| Cue | Remote path | Size | Recovery plan |
|---|---|---:|---|
| TFMap-only | /data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_5ep_ddp2/models/checkpoint_5.pt | 245.91M | retrain from configs/normal_basic_single_cue or store externally |
| Context-only | /data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_5ep_ddp2_noamp/models/checkpoint_5.pt | 294.40M | retrain from configs/normal_basic_single_cue or store externally |
| USEF-only | /data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_5ep_ddp2_bs8/models/checkpoint_5.pt | 294.71M | retrain from configs/normal_basic_single_cue or store externally |
| Public USEF pretrained | /data/tse_cue_project/models/pretrained/modelscope_wesep_pretrained_models/extracted/avg_model.pt | 269.54M | redownload from public source / ModelScope if available |
| ECAPA/wespeaker | /data/tse_cue_project/code/wesep-real-tse/wespeaker_models/voxceleb_ECAPA512/avg_model.pt | 36.92M | redownload or restore with WeSpeaker assets |

## Full remote checkpoint listing snapshot

| Size | Path |
|---:|---|
| 294.71M | `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_5ep_ddp2_bs8/models/latest_checkpoint.pt` |
| 294.71M | `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_5ep_ddp2_bs8/models/final_checkpoint.pt` |
| 294.71M | `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_5ep_ddp2_bs8/models/checkpoint_5.pt` |
| 294.71M | `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_5ep_ddp2_bs8/models/checkpoint_4.pt` |
| 294.71M | `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_5ep_ddp2_bs8/models/checkpoint_3.pt` |
| 294.71M | `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_5ep_ddp2_bs8/models/checkpoint_2.pt` |
| 294.71M | `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_5ep_ddp2_bs8/models/checkpoint_1.pt` |
| 294.40M | `/data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_5ep_ddp2_noamp/models/latest_checkpoint.pt` |
| 294.40M | `/data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_5ep_ddp2_noamp/models/final_checkpoint.pt` |
| 294.40M | `/data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_5ep_ddp2_noamp/models/checkpoint_5.pt` |
| 294.40M | `/data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_5ep_ddp2_noamp/models/checkpoint_4.pt` |
| 294.40M | `/data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_5ep_ddp2_noamp/models/checkpoint_3.pt` |
| 294.40M | `/data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_5ep_ddp2_noamp/models/checkpoint_2.pt` |
| 294.40M | `/data/tse_cue_project/experiments/normal_basic_single_cue/context_only_bsrnn_normal_train100_5ep_ddp2_noamp/models/checkpoint_1.pt` |
| 269.54M | `/data/tse_cue_project/models/pretrained/modelscope_wesep_pretrained_models/extracted/avg_model.pt` |
| 245.91M | `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_5ep_ddp2/models/latest_checkpoint.pt` |
| 245.91M | `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_5ep_ddp2/models/final_checkpoint.pt` |
| 245.91M | `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_5ep_ddp2/models/checkpoint_5.pt` |
| 245.91M | `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_5ep_ddp2/models/checkpoint_4.pt` |
| 245.91M | `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_5ep_ddp2/models/checkpoint_3.pt` |
| 245.91M | `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_5ep_ddp2/models/checkpoint_2.pt` |
| 245.91M | `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_5ep_ddp2/models/checkpoint_1.pt` |
| 42.97M | `/data/tse_cue_project/code/wesep-real-tse/wespeaker_models/voxceleb_resnet34/avg_model.pt` |
| 36.92M | `/data/tse_cue_project/code/wesep-real-tse/wespeaker_models/voxceleb_ECAPA512/avg_model.pt` |
| 16.11M | `/data/tse_cue_project/models/pretrained/speechbrain_spkrec_xvect_voxceleb/embedding_model.ckpt` |
| 15.12M | `/data/tse_cue_project/models/pretrained/speechbrain_spkrec_xvect_voxceleb/classifier.ckpt` |
| 0.12M | `/data/tse_cue_project/models/pretrained/speechbrain_spkrec_xvect_voxceleb/label_encoder.ckpt` |
| 0.00M | `/data/tse_cue_project/models/pretrained/speechbrain_spkrec_xvect_voxceleb/mean_var_norm_emb.ckpt` |
