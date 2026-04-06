# Exp 220 — 3-seeds 验证 exp219 候选配置（λ 细扫）

- **固定**：no detach、N=372、layer 7、PAD、40k iter、**40h 每脚本**
- **变量**：见下表；每配置 **3 seeds (7, 25, 42)** 顺序跑
- **共 13 脚本 × 3 seeds = 39 runs**

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

## 提交

```bash
# 单个
sbatch slurm/exp220_projection/train_single_cosine_w0_lam1p00.slurm

# 全部 13 个（注意：会占用 13 个 GPU 槽位）
for f in slurm/exp220_projection/train_*.slurm; do sbatch "$f"; done
```

## 日志

`/user/work/ig23200/sde_seg/logs/exp220_<config>-%j.out`
