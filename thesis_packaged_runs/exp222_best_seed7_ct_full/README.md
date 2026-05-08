# EXP222 Best Checkpoint Package (English)

## What this folder contains

- `best_model.pkl`: Best checkpoint selected from the 3-seed run.
- `cfg.yml`: Full training configuration used for that selected checkpoint.
- `README.md`: This summary document.

## Selected checkpoint

- Selected seed: **7**
- Selected run id in `exp 222`: **15**
- Best validation metric (`Mean IoU`): **0.685828314705711**
- Source run directory:
  - `/user/work/ig23200/sde_seg/logs/cityscapes_joint_222/2026-05-04_22-47-0915_tag=cityscapes_sel_ds_us_pad_transfer_dcompgt0030_ctcosinew0lam1p00_D372fixed_S7_sgdLr1E-021E-031E-061E-03stepx_clip10False_m1s1_crop512x512bs2_flip_dec6_lr5_fd2_crop512x512bs4_l9i7Trueos1_Unlab1_0depthcompFPLFalsejitblur`

## 3-seed comparison (same script submission)

- Seed 7 (run 15): `Mean IoU = 0.685828314705711`  **(best)**
- Seed 25 (run 16): `Mean IoU = 0.6718026583591076`
- Seed 42 (run 17): `Mean IoU = 0.6835764305503106`

## Model/loss setup used (from this run)

- Framework components:
  - **S (selection)**: ON (`restrict_to_subset.mode = fixed`, `n_subset = 372`)
  - **DX (DepthMix)**: ON (`mix_mask = depthcomp`, `depthmix_online_depth = true`)
  - **MTL (PAD)**: ON (`model.segmentation_name = mtl_pad`)

- Cross-task setup:
  - `training.cross_task_type = cosine`
  - `training.cross_task_lambda = 1.0`
  - `training.cross_task_warmup_iters = 0`
  - `model.segmentation_args.projection_mode = single`
    - mapping direction: **depth -> segmentation** (single projection)

- Other key training values:
  - `training.segmentation_lambda = 1`
  - `training.monodepth_lambda = 1`
  - `training.batch_size = 2`
  - `training.save_model = true`

## Notes

- This package is prepared for thesis/report submission purposes.
- If required by your supervisor, submit both `best_model.pkl` and `cfg.yml` together.
