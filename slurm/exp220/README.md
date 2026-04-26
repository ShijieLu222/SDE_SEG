# Exp 220 — 3-seeds 验证 exp219 候选配置（λ 细扫 + single_rev 方向消融）

- **固定**：no detach、N=372、layer 7、PAD、40k iter、**40h 每脚本**
- **变量**：见下表；每配置 **3 seeds (7, 25, 42)** 顺序跑
- **共 19 脚本 × 3 seeds = 57 runs**

## 配置与 Run ID 对应表

| Run IDs | 脚本 | projection | ct_type | warmup | λ |
|---------|------|------------|---------|--------|---|
| 0–2 | train_single_cosine_w0_lam0p75.slurm | single | cosine | w0 | 0.75 |
| 3–5 | train_single_cosine_w0_lam1p00.slurm | single | cosine | w0 | 1.0 |
| 6–8 | train_single_cosine_w0_lam1p25.slurm | single | cosine | w0 | 1.25 |
| 9–11 | train_single_mse_w5k_lam1p00.slurm | single | mse | w5k | 1.0 |
| 12–14 | train_single_mse_w0_lam1p00.slurm | single | mse | w0 | 1.0 |
| 15–17 | train_dual_cosine_w0_lam0p50.slurm | dual | cosine | w0 | 0.5 |
| 18–20 | train_dual_cosine_w0_lam0p75.slurm | dual | cosine | w0 | 0.75 |
| 21–23 | train_dual_cosine_w0_lam1p00.slurm | dual | cosine | w0 | 1.0 |
| 24–26 | train_dual_cosine_w0_lam1p25.slurm | dual | cosine | w0 | 1.25 |
| 27–29 | train_dual_cosine_w5k_lam0p50.slurm | dual | cosine | w5k | 0.5 |
| 30–32 | train_dual_cosine_w5k_lam0p75.slurm | dual | cosine | w5k | 0.75 |
| 33–35 | train_dual_cosine_w5k_lam1p00.slurm | dual | cosine | w5k | 1.0 |
| 36–38 | train_dual_cosine_w5k_lam1p25.slurm | dual | cosine | w5k | 1.25 |
| 39–41 | train_single_rev_cosine_lam0p50.slurm | single_rev | cosine | w0 | 0.50 |
| 42–44 | train_single_rev_cosine_lam0p75.slurm | single_rev | cosine | w0 | 0.75 |
| 45–47 | train_single_rev_cosine_lam1p00.slurm | single_rev | cosine | w0 | 1.00 |
| 48–50 | train_single_rev_mse_lam0p50.slurm | single_rev | mse | w0 | 0.50 |
| 51–53 | train_single_rev_mse_lam0p75.slurm | single_rev | mse | w0 | 0.75 |
| 54–56 | train_single_rev_mse_lam1p00.slurm | single_rev | mse | w0 | 1.00 |

**single_rev 说明**：把 seg 特征（有 GT 监督）投影到 depth 空间再对齐 depth。**三 seed 汇总表**见 [`exp220_results.md`](exp220_results.md) **§10**。

## 提交命令

```bash
# 单个（示例）
sbatch slurm/exp220/train_single_cosine_w0_lam1p00.slurm

# 仅提交 single_rev 6 个（等 exp219 结果确认后）
for f in slurm/exp220/train_single_rev_*.slurm; do sbatch "$f"; done

# 全部 19 个
for f in slurm/exp220/train_*.slurm; do sbatch "$f"; done
```

## 日志

`/user/work/ig23200/sde_seg/logs/exp220_<config>-%j.out`
