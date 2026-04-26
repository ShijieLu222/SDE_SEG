# Exp 222 — 完整模型：MTL + Cross-Task Projection Loss 消融

在 exp221 三组配置上加入 **cross-task projection loss**，
测试 single / single_rev 两种 projection 方向，MSE 和 Cosine 两种 loss，共 36 runs。

## Run ID 对应表

### 原始 single projection（runs 0–17）

| Run IDs | 脚本 | 配置 | CT Loss |
|---------|------|------|---------|
| 0, 1, 2 | `train_mse_mtl_dx_3seeds.slurm` | MTL + DX | MSE λ=1.0 |
| 3, 4, 5 | `train_mse_mtl_sel_3seeds.slurm` | MTL + Sel | MSE λ=1.0 |
| 6, 7, 8 | `train_mse_mtl_dx_sel_3seeds.slurm` | MTL + DX + Sel | MSE λ=1.0 |
| 9, 10, 11 | `train_cosine_mtl_dx_3seeds.slurm` | MTL + DX | Cosine λ=1.0 |
| 12, 13, 14 | `train_cosine_mtl_sel_3seeds.slurm` | MTL + Sel | Cosine λ=1.0 |
| 15, 16, 17 | `train_cosine_mtl_dx_sel_3seeds.slurm` | MTL + DX + Sel | Cosine λ=1.0 |

### single_rev projection（runs 18–35）— seg→depth，λ=1.0，w0

| Run IDs | 脚本 | 配置 | CT Loss |
|---------|------|------|---------|
| 18, 19, 20 | `train_srev_mse_dx_3seeds.slurm` | MTL + DX | MSE λ=1.0 |
| 21, 22, 23 | `train_srev_mse_sel_3seeds.slurm` | MTL + Sel | MSE λ=1.0 |
| 24, 25, 26 | `train_srev_mse_dx_sel_3seeds.slurm` | MTL + DX + Sel | MSE λ=1.0 |
| 27, 28, 29 | `train_srev_cosine_dx_3seeds.slurm` | MTL + DX | Cosine λ=1.0 |
| 30, 31, 32 | `train_srev_cosine_sel_3seeds.slurm` | MTL + Sel | Cosine λ=1.0 |
| 33, 34, 35 | `train_srev_cosine_dx_sel_3seeds.slurm` | MTL + DX + Sel | Cosine λ=1.0 |

> Seeds 顺序：7 → 25 → 42，每 seed 40k iter，每 1200 iter 验证 = 33 ckpt/seed
> 全部脚本：`--partition=mlcnu --gres=gpu:a100-sxm4-40gb:1 --time=42:00:00`

## 提交指令

```bash
# single_rev 新增 6 个
for f in slurm/exp222/train_srev_*.slurm; do sbatch "$f"; done

# 全部 12 个
for f in slurm/exp222/train_*.slurm; do sbatch "$f"; done
```

## 结果文件

| 投影方向 | 文档 |
|----------|------|
| **single**（depth→seg） | [`RESULTS.md`](RESULTS.md)（两批 job，两批汇总 **mean±std**） |
| **single_rev**（seg→depth） | [`RESULTS_single_rev.md`](RESULTS_single_rev.md) |

## 与 exp221 的对比关系（摘自上述结果文件）

| 配置 | exp221（无 CT loss） | exp222 single（MSE / Cosine 两批均值） | exp222 single_rev（**见 RESULTS_single_rev**） |
|------|---------------------|----------------------------------------|-----------------------------------------------|
| MTL + DX | 66.22% | 65.67% / 66.07% | MSE **65.84%** / Cos **65.93%** |
| MTL + Sel | 66.58% | 66.92% / 66.41% | MSE **66.38%** / Cos **66.60%** |
| MTL + DX + Sel | 67.74% | **68.18%** / 67.82% | MSE **67.69%** / Cos **68.14%** |
