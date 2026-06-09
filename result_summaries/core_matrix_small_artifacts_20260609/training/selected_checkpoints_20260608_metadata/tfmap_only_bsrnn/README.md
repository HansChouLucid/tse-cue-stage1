# TFMap-only BSRNN checkpoint selection (2026-06-08)
## Selected checkpoint
- Stable path: `/data/tse_cue_project/experiments/normal_basic_single_cue/selected_checkpoints_20260608/tfmap_only_bsrnn/best_tfmap_bsrnn_train100.pt`
- Source checkpoint: `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_stabilize_from_total66_lr3e6_3ep_a100_gpu0_bs8_20260608/models/checkpoint_3.pt`
- Branch: `stabilize_from_total66_plus3`
- Total epoch: `69`; branch epoch: `3`
- Best validation loss: `-10.643738`
- Reason: lowest validation SISDR loss among completed TFMap-only BSRNN branches. The later small-LR stabilization branch did not beat this value.

## Branch summary
- `initial_5ep`: best total epoch `5`, val `-4.925864`, ckpt `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_5ep_dualA100_ddp2/models/checkpoint_5.pt`
- `ft13_from5_total18`: best total epoch `18`, val `-9.093913`, ckpt `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/checkpoint_13.pt`
- `continue_e18_plus40_total58`: best total epoch `58`, val `-10.567708`, ckpt `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_continue_e18_plus40_a100_gpu0_bs8_20260607/models/checkpoint_40.pt`
- `lowLR_from_total58_plus5`: best total epoch `62`, val `-10.636629`, ckpt `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_continue_e58_plus5_lr6e6_a100_gpu0_bs8_20260608/models/checkpoint_4.pt`
- `warmrestart_from_total62_plus5`: best total epoch `66`, val `-10.640062`, ckpt `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_warmrestart_from_total62_lr1e5_5ep_a100_gpu0_bs8_20260608/models/checkpoint_4.pt`
- `stabilize_from_total66_plus3`: best total epoch `69`, val `-10.643738`, ckpt `/data/tse_cue_project/experiments/normal_basic_single_cue/tfmap_only_bsrnn_stabilize_from_total66_lr3e6_3ep_a100_gpu0_bs8_20260608/models/checkpoint_3.pt`

## Recommended use for inference
Use `/data/tse_cue_project/experiments/normal_basic_single_cue/selected_checkpoints_20260608/tfmap_only_bsrnn/best_tfmap_bsrnn_train100.pt` as the TFMap-only BSRNN candidate unless lightweight screening shows an averaged checkpoint is more stable.
