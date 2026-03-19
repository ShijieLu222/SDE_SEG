# Exp 219 — No-detach projection ablation (single vs dual, λ search)

- **固定**：no detach、seed=7、N=372、layer 7、PAD、40k iter、24h 每任务
- **变量**：projection_mode (single/dual)、ct_type (cosine/mse)、warmup_iters (5000/0)、lambda_ct (0.25, 0.5, 0.75, 1.0)
- **共 32 runs**，每个 run 一个脚本、1 seed

## Run ID 对应表

| Run ID | 脚本 | projection | ct_type | warmup | lambda |
|--------|------|------------|---------|--------|--------|
| 0 | train_single_cosine_w5k_lam0p25.slurm | single | cosine | 5000 | 0.25 |
| 1 | train_single_cosine_w5k_lam0p50.slurm | single | cosine | 5000 | 0.5 |
| 2 | train_single_cosine_w5k_lam0p75.slurm | single | cosine | 5000 | 0.75 |
| 3 | train_single_cosine_w5k_lam1p00.slurm | single | cosine | 5000 | 1.0 |
| 4–7 | single_cosine_w0_lam* | single | cosine | 0 | 0.25–1.0 |
| 8–11 | single_mse_w5k_lam* | single | mse | 5000 | 0.25–1.0 |
| 12–15 | single_mse_w0_lam* | single | mse | 0 | 0.25–1.0 |
| 16–19 | dual_cosine_w5k_lam* | dual | cosine | 5000 | 0.25–1.0 |
| 20–23 | dual_cosine_w0_lam* | dual | cosine | 0 | 0.25–1.0 |
| 24–27 | dual_mse_w5k_lam* | dual | mse | 5000 | 0.25–1.0 |
| 28–31 | dual_mse_w0_lam* | dual | mse | 0 | 0.25–1.0 |

提交示例：
- 单个：`sbatch slurm/exp219_projection/train_single_cosine_w5k_lam0p25.slurm`
- 全部 32 个：`bash slurm/exp219_projection/submit_all.sh`
