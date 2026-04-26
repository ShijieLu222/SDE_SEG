# Exp 220 — 3-Seed Validation Summary

This file provides a compact English summary of Exp 220.

## Setup

- Dataset/setup: PAD + no-detach projection, N=372
- Training: 40k iterations, seeds 7/25/42
- Metric: best validation mIoU per seed, then mean ± population std
- Reference baseline: exp217 mean mIoU = 0.6269

## Key Results

| Setting | Seed 7 | Seed 25 | Seed 42 | Mean | Std |
|---|---:|---:|---:|---:|---:|
| single_cosine, w0, lambda=0.75 | 0.6308 | 0.6289 | 0.6248 | 0.6282 | 0.0025 |
| single_cosine, w0, lambda=1.00 | 0.6301 | 0.6337 | 0.6337 | 0.6325 | 0.0017 |
| single_cosine, w0, lambda=1.25 | 0.6244 | 0.6224 | 0.6433 | 0.6301 | 0.0094 |
| single_mse, w5k, lambda=1.00 | 0.6132 | 0.6292 | 0.6393 | 0.6272 | 0.0107 |
| single_mse, w0, lambda=1.00 | 0.6206 | 0.6291 | 0.6491 | 0.6329 | 0.0120 |
| dual_cosine, w0, lambda=0.50 | 0.6254 | 0.6291 | 0.6315 | 0.6287 | 0.0025 |
| dual_cosine, w0, lambda=0.75 | 0.6322 | 0.6291 | 0.6273 | 0.6295 | 0.0020 |
| dual_cosine, w0, lambda=1.00 | 0.6277 | 0.6181 | 0.6363 | 0.6274 | 0.0074 |
| dual_cosine, w0, lambda=1.25 | 0.6230 | 0.6255 | 0.6307 | 0.6264 | 0.0032 |
| dual_cosine, w5k, lambda=0.50 | 0.6213 | 0.6315 | 0.6158 | 0.6229 | 0.0065 |
| dual_cosine, w5k, lambda=0.75 | 0.6291 | 0.6228 | 0.6163 | 0.6227 | 0.0052 |
| dual_cosine, w5k, lambda=1.00 | 0.6239 | 0.6228 | 0.6143 | 0.6203 | 0.0043 |
| dual_cosine, w5k, lambda=1.25 | 0.6315 | 0.6236 | 0.6084 | 0.6212 | 0.0096 |

## Conclusions

1. Best mean in this batch: `single_mse w0 lambda=1.00` (0.6329).
2. Most stable cosine single setting: `single_cosine w0 lambda=1.00` (std 0.0017).
3. For dual cosine, `w0 lambda=0.75` is best (0.6295), while w5k variants are weaker.
4. Multiple settings improve over exp217 mean baseline (0.6269), with strongest gains from single projection settings.

## Logs

- Path pattern: `/user/work/ig23200/sde_seg/logs/exp220_*.out`
