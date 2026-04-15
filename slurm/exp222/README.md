# Exp 222 — 完整模型：MTL + Cross-Task Projection Loss 消融

在 exp221 三组配置上加入 **cross-task projection loss**（single, no warm-up），
测试两种 loss 类型（MSE 和 Cosine），共 18 runs（6 组 × 3 seeds）。

## Run ID 对应表

| Run IDs | 脚本 | 配置 | CT Loss |
|---------|------|------|---------|
| 0, 1, 2 | `train_mse_mtl_dx_3seeds.slurm` | MTL + DX | MSE λ=1.0 |
| 3, 4, 5 | `train_mse_mtl_sel_3seeds.slurm` | MTL + Sel | MSE λ=1.0 |
| 6, 7, 8 | `train_mse_mtl_dx_sel_3seeds.slurm` | MTL + DX + Sel | MSE λ=1.0 |
| 9, 10, 11 | `train_cosine_mtl_dx_3seeds.slurm` | MTL + DX | Cosine λ=1.0 |
| 12, 13, 14 | `train_cosine_mtl_sel_3seeds.slurm` | MTL + Sel | Cosine λ=1.0 |
| 15, 16, 17 | `train_cosine_mtl_dx_sel_3seeds.slurm` | MTL + DX + Sel | Cosine λ=1.0 |

> Seeds 顺序：7 → 25 → 42，每 seed 40k iter，每 1200 iter 验证 = 33 ckpt/seed

## GPU 要求

- **含 DX 的脚本**（runs 0-2, 6-8, 9-11, 15-17）：`#SBATCH --gres=gpu:rtx_3090:1`
- **不含 DX 的脚本**（runs 3-5, 12-14）：`#SBATCH --gres=gpu:1`

## 提交指令

```bash
# MSE 组
sbatch slurm/exp222/train_mse_mtl_dx_3seeds.slurm
sbatch slurm/exp222/train_mse_mtl_sel_3seeds.slurm
sbatch slurm/exp222/train_mse_mtl_dx_sel_3seeds.slurm

# Cosine 组
sbatch slurm/exp222/train_cosine_mtl_dx_3seeds.slurm
sbatch slurm/exp222/train_cosine_mtl_sel_3seeds.slurm
sbatch slurm/exp222/train_cosine_mtl_dx_sel_3seeds.slurm
```

## 与 exp221 的对比关系

| 配置 | exp221（无 CT loss） | exp222 MSE | exp222 Cosine |
|------|---------------------|------------|----------------|
| MTL + DX | 66.22% | TBD | TBD |
| MTL + Sel | 66.58% | TBD | TBD |
| MTL + DX + Sel | 67.74% | TBD | TBD |
