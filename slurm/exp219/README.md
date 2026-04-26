# Exp 219 — No-detach projection ablation (single vs dual vs single_rev, λ search)

- **固定**：no detach、seed=7、N=372、layer 7、PAD、40k iter、24h 每任务
- **变量**：projection_mode (single/dual/single_rev)、ct_type (cosine/mse)、warmup_iters (5000/0)、lambda_ct (0.25, 0.5, 0.75, 1.0)
- **共 40 runs**（原 32 + 新增 8 个 single_rev）

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
| 32 | train_single_rev_cosine_lam0p25.slurm | single_rev | cosine | 0 | 0.25 |
| 33 | train_single_rev_cosine_lam0p50.slurm | single_rev | cosine | 0 | 0.50 |
| 34 | train_single_rev_cosine_lam0p75.slurm | single_rev | cosine | 0 | 0.75 |
| 35 | train_single_rev_cosine_lam1p00.slurm | single_rev | cosine | 0 | 1.00 |
| 36 | train_single_rev_mse_lam0p25.slurm | single_rev | mse | 0 | 0.25 |
| 37 | train_single_rev_mse_lam0p50.slurm | single_rev | mse | 0 | 0.50 |
| 38 | train_single_rev_mse_lam0p75.slurm | single_rev | mse | 0 | 0.75 |
| 39 | train_single_rev_mse_lam1p00.slurm | single_rev | mse | 0 | 1.00 |

**single_rev 说明**：把 seg 特征（有 GT 监督）通过投影头映射到 depth 空间，以有 GT 的一侧为 anchor 指导无 GT 的 depth 分支对齐。全部 w0（无 warmup）。

## 结果汇总

- **主表（含 single / dual / single_rev）**：[`exp219_projection_results.md`](exp219_projection_results.md)（single_rev 见 **§10**）。

## 提交命令

```bash
# 单个（示例）
sbatch slurm/exp219/train_single_cosine_w5k_lam0p25.slurm

# 仅提交 single_rev 8 个（Run 32–39）
for f in slurm/exp219/train_single_rev_*.slurm; do sbatch "$f"; done

# 全部 40 个
for f in slurm/exp219/*.slurm; do sbatch "$f"; done
```
