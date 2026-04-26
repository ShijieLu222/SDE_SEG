# Full single_rev Results (exp 219 / 220 / 222)

> Updated: 2026-04-23  
> **single_rev**: project segmentation distillation features to depth space, then compute $\mathcal{L}_\text{ct}$ against depth features (seg->depth).  
> **Metric**: **best validation mIoU** over the full training run (Cityscapes val).  
> **Shared setup**: PAD-MTL, N=372, 40k iter, warmup **w0** (no CT warmup).

---

## Exp 219 — Phase I (seed 7 only)

Phase I is a single-seed sweep. The **single_rev** subset has **8** settings (runs **32-39**) with no seed 25 / 42.

| Run ID | CT | $\lambda_\text{ct}$ | Seed 7 best val mIoU |
|--------|-----|---------------------|----------------------|
| 32 | cosine | 0.25 | 0.6270 |
| 33 | cosine | 0.50 | 0.6255 |
| 34 | cosine | 0.75 | 0.6226 |
| 35 | cosine | 1.00 | 0.6280 |
| 36 | mse | 0.25 | 0.6208 |
| 37 | mse | 0.50 | 0.6264 |
| 38 | mse | 0.75 | 0.6283 |
| 39 | mse | 1.00 | 0.6308 |


---

## Exp 220 — Phase II (seeds 7 / 25 / 42)

**single_rev** lambda sweep includes **6** settings (runs **39-56**). Seed order is **7 -> 25 -> 42**. mean / std are arithmetic mean and population std (pstdev).

| CT | $\lambda_\text{ct}$ | Slurm JOBID | Run IDs | Seed 7 | Seed 25 | Seed 42 | Mean | Std |
|----|---------------------|-------------|---------|--------|---------|---------|------|-----|
| cosine | 0.50 | 16922872 | 39–41 | 0.6246 | 0.6239 | 0.6389 | 0.6291 | 0.0069 |
| cosine | 0.75 | 16922873 | 42–44 | 0.6292 | 0.6284 | 0.6410 | 0.6329 | 0.0071 |
| cosine | 1.00 | 16922874 | 45–47 | 0.6281 | 0.6361 | 0.6387 | 0.6343 | 0.0056 |
| mse | 0.50 | 16922875 | 48–50 | 0.6197 | 0.6246 | 0.6408 | 0.6284 | 0.0090 |
| mse | 0.75 | 16922876 | 51–53 | 0.6208 | 0.6288 | 0.6396 | 0.6297 | 0.0094 |
| mse | 1.00 | 16922877 | 54–56 | 0.6232 | 0.6272 | 0.6429 | 0.6311 | 0.0105 |


---

## Exp 222 — MTL Components × CT ($\lambda_\text{ct}=1.0$, w0)

Under **single_rev**, there are **6** settings (runs **18-35**). **mIoU is shown as percentages** (consistent with `RESULTS_single_rev.md`). mean / std are computed over three seeds.

| CT | Setting | Run IDs | Seed 7 | Seed 25 | Seed 42 | Mean | Std |
|----|---------|---------|--------|---------|---------|------|-----|
| mse | MTL+DX | 18,19,20 | 65.23% | 66.02% | 66.26% | 65.84% | ±0.44% |
| mse | MTL+Sel | 21,22,23 | 67.02% | 66.21% | 65.90% | 66.38% | ±0.47% |
| mse | MTL+DX+Sel | 24,25,26 | 68.37% | 67.29% | 67.91% | 67.86% | ±0.44% |
| cosine | MTL+DX | 27,28,29 | 64.47% | 65.55% | 67.79% | 65.93% | ±1.38% |
| cosine | MTL+Sel | 30,31,32 | 66.65% | 66.71% | 66.45% | 66.60% | ±0.11% |
| cosine | MTL+DX+Sel | 33,34,35 | 68.91% | 67.83% | 68.29% | 68.34% | ±0.55% |

---

