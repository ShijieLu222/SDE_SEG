# Exp 221 实验结果汇总

> 更新日期：2026-04-10
> 数据来源：BluePebble BP1 HPC 训练日志

---

## 任务状态总览

| Job ID | 描述 | Slurm 名称 | 节点 | 检查点数 | 状态 |
|--------|------|------------|------|----------|------|
| 16571334 | Baseline PAD-MTL+DX, seeds 7/25/42 (R4) | `bp1_base` | bp1-gpu003 | 99 (3×33) | **完成** |
| 16571335 | Baseline PAD-MTL+DX, seeds 7/25/42 (R5) | `bp1_base` | bp1-gpu007 | 99 (3×33) | **完成** |
| 16571337 | Exp221 MTL+Sel, seeds 7/25/42 (Trial 1) | `exp221_m` | bp1-gpu019 | 99 (3×33) | **完成** |
| 16578443 | Exp221 MTL+Sel, seeds 7/25/42 (Trial 2) | `exp221_m` | — | 99 (3×33) | **完成** |
| 16571336 | Exp221 MTL+DX, seeds 7/25/42 | `exp221_m` | — | 0 | **失败** |
| 16571338 | Exp221 MTL+DX+Sel, seeds 7/25/42 | `exp221_m` | — | 0 | **失败** |
| 16578444 | Exp221 MTL+DX+Sel, seeds 7/25/42 (重提) | `exp221_m` | — | 0 | **失败** |

> 每个 seed 训练 40k 次迭代，每 1200 iter 验证一次 = 33 个检查点/seed。

---

## 结果数据表

### 基准线：PAD-MTL + DepthMix（Exp 213, N=372, 随机子集）

配置：`pad_transfer_dcompgt0030`, D372random, exp 213 runs 6/7/8

| 运行 | Job ID | Seed 7 mIoU | Seed 25 mIoU | Seed 42 mIoU | **Mean** | **Std** |
|------|--------|-------------|--------------|--------------|----------|---------|
| R4   | 16571334 | 62.37% | 62.80% | 64.74% | **63.30%** | 1.27% |
| R5   | 16571335 | 61.99% | 62.93% | 63.97% | **62.96%** | 0.99% |

### Exp 221 MTL + Label Selection（N=372，固定预选子集，无 DX）

配置：`sel_ds_us_pad_transfer`, D372fixed, exp 221 runs 3/4/5

| 运行 | Job ID | Seed 7 mIoU | Seed 25 mIoU | Seed 42 mIoU | **Mean** | **Std** |
|------|--------|-------------|--------------|--------------|----------|---------|
| Trial 1 | 16571337 | 66.99% | 66.36% | 66.28% | **66.54%** | 0.39% |
| Trial 2 | 16578443 | 66.90% | 65.48% | 66.67% | **66.35%** | 0.76% |

### Exp 221 MTL + DepthMix（runs 0/1/2）

配置：`pad_transfer_dcompgt0030`, D372random

| 运行 | Job ID | Seed 7 | Seed 25 | Seed 42 | Mean | Std |
|------|--------|--------|---------|---------|------|-----|
| Trial 1 | 16571336 | — | — | — | **FAILED** | — |

> 需要重新提交：`sbatch slurm/exp221/train_mtl_dx_3seeds.slurm`

### Exp 221 MTL + DepthMix + Selection（runs 6/7/8）

配置：`sel_ds_us_pad_transfer_dcompgt0030`, D372fixed

| 运行 | Job ID | Seed 7 | Seed 25 | Seed 42 | Mean | Std |
|------|--------|--------|---------|---------|------|-----|
| Trial 1 | 16571338 | — | — | — | **FAILED** | — |
| Trial 2 | 16578444 | — | — | — | **FAILED** | — |

> 需要重新提交：`sbatch slurm/exp221/train_mtl_dx_sel_3seeds.slurm`

---

## 对比摘要

| 配置 | N | 子集策略 | DX | Mean mIoU | Std | vs Baseline (R4) |
|------|---|----------|----|-----------|-----|-----------------|
| Baseline PAD-MTL+DX (R4) | 372 | 随机 | ✓ | 63.30% | 1.27% | — |
| Baseline PAD-MTL+DX (R5) | 372 | 随机 | ✓ | 62.96% | 0.99% | −0.34% |
| **MTL+Sel Trial 1** | 372 | 固定预选 | ✗ | **66.54%** | **0.39%** | **+3.24%** |
| MTL+Sel Trial 2 | 372 | 固定预选 | ✗ | 66.35% | 0.76% | +3.05% |
| MTL+DX | 372 | 随机 | ✓ | pending | — | — |
| MTL+DX+Sel | 372 | 固定预选 | ✓ | pending | — | — |

**初步观察：**
- Label Selection 本身（无 DX）相比随机子集+DX 的基准线提升约 **+3.2 mIoU**（66.54% vs 63.30%）
- MTL+Sel 两次 trial 结果高度一致（66.35%–66.54%），std 也较小（0.39%–0.76%），说明稳定性好
- MTL+DX 和 MTL+DX+Sel 均因未知原因失败（0 个检查点），需要调查和重新提交

---

## 待完成事项

- [ ] 调查 MTL+DX 和 MTL+DX+Sel 失败原因（可能为 OOM 或节点故障）
- [ ] 重新提交 `train_mtl_dx_3seeds.slurm`
- [ ] 重新提交 `train_mtl_dx_sel_3seeds.slurm`
- [ ] 全部完成后更新此文件并纳入论文 Chapter 4
