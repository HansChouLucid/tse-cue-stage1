# USEF-only BSRNN checkpoint selection (2026-06-08)
## Selected checkpoint
- Stable path: `/data/tse_cue_project/experiments/normal_basic_single_cue/selected_checkpoints_20260608/usef_only_bsrnn/best_usef_bsrnn_train100.pt`
- Source checkpoint: `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_continue_e48_plus5_lr5e6_a100_gpu1_bs8_20260608/models/checkpoint_2.pt`
- Branch: `lowLR_from_total48_plus5`
- Total epoch: `50`; branch epoch: `2`
- Best validation loss: `-11.290254`
- Reason: lowest validation SISDR loss among completed USEF-only BSRNN branches. The later warm-restart branch did not beat this value.

## Branch summary
- `initial_5ep`: best total epoch `5`, val `-6.806921`, ckpt `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_5ep_dualA100_ddp2/models/checkpoint_5.pt`
- `ft13_from5_total18`: best total epoch `18`, val `-10.552886`, ckpt `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_normal_train100_ft13_from5_dualA100_ddp2/models/checkpoint_13.pt`
- `continue_e18_plus30_total48`: best total epoch `47`, val `-11.285524`, ckpt `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_continue_e18_plus30_a100_gpu1_bs8_20260607/models/checkpoint_29.pt`
- `lowLR_from_total48_plus5`: best total epoch `50`, val `-11.290254`, ckpt `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_continue_e48_plus5_lr5e6_a100_gpu1_bs8_20260608/models/checkpoint_2.pt`
- `warmrestart_from_total50_plus5`: best total epoch `53`, val `-11.283499`, ckpt `/data/tse_cue_project/experiments/normal_basic_single_cue/usef_only_bsrnn_warmrestart_from_total50_lr1e5_5ep_a100_gpu1_bs8_20260608/models/checkpoint_3.pt`

## Recommended use for inference
Use `/data/tse_cue_project/experiments/normal_basic_single_cue/selected_checkpoints_20260608/usef_only_bsrnn/best_usef_bsrnn_train100.pt` as the USEF-only BSRNN candidate for normal/easy, similar speaker, and similar content inference once BSRNN cue models are ready for screening.

## Averaged checkpoint candidates

Two averaged USEF-only BSRNN candidates were generated for stability checks:

- Same-branch average: `/data/tse_cue_project/experiments/normal_basic_single_cue/selected_checkpoints_20260608/usef_only_bsrnn/avg_usef_bsrnn_lowLR_ckpt2_4_5.pt`
  - Components: lowLR checkpoint 2 / 4 / 5, corresponding to total epochs 50 / 52 / 53.
  - Use this as the conservative averaged candidate because all components come from the same low-LR trajectory.

- Near-best cross-branch average: `/data/tse_cue_project/experiments/normal_basic_single_cue/selected_checkpoints_20260608/usef_only_bsrnn/avg_usef_bsrnn_top3_total47_50_53.pt`
  - Components: main checkpoint 29 plus lowLR checkpoint 2 and 5, corresponding to total epochs 47 / 50 / 53.
  - Use this as a secondary robustness candidate; it averages adjacent high-quality points across the continuation boundary.

Recommended next step: run lightweight inference screening for `best_usef_bsrnn_train100.pt`, `avg_usef_bsrnn_lowLR_ckpt2_4_5.pt`, and optionally `avg_usef_bsrnn_top3_total47_50_53.pt`. Choose the final checkpoint by normal/basic SI-SDR/SI-SDRi plus hard-set tail and drift, not by dev loss alone.
