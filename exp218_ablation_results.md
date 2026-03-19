# Exp 218 — Cross-Task Warmup & Detach Ablation

## 1. 实验目的

在 **MSE λ=0.30** 和 **Cosine λ=0.84** 两个最优配置下，对 L_ct 的两个设计做消融：

- **warmup**：前 5000 iter 是否不激活 L_ct
- **detach**：计算 L_ct 时是否对深度特征做 `detach()`（梯度不回传到 Depth Decoder）

共 4 种组合 × 2 种 loss × 3 seeds = **24 runs**。

---

## 2. 实验设置

| 项目 | 值 |
|------|-----|
| 数据集 | Cityscapes，标注子集 N=372 |
| Backbone | ResNet-101 |
| 解码器 | PAD (mtl_pad)，distillation_layer=7 |
| 训练迭代 | 40,000 iter |
| Seeds | 7 / 25 / 42 |
| ct_type × λ | MSE 0.30，Cosine 0.84 |
| 其它 | L_unlabeled 开启，与 exp217 一致 |

### 4 种消融组合

| 组合 | warmup_iters | detach_depth | 说明 |
|------|--------------|--------------|------|
| **full** | 5000 | True | 当前完整做法 |
| **no_warmup** | 0 | True | 关 warmup，保留 detach |
| **no_detach** | 5000 | False | 保留 warmup，关 detach |
| **no_warmup_no_detach** | 0 | False | 两者都关 |

---

## 3. Run ID 与配置对应表

| Run ID | ct_type | λ_ct | 消融组合 | seed |
|--------|---------|------|----------|------|
| 0 | mse | 0.30 | full | 7 |
| 1 | mse | 0.30 | full | 25 |
| 2 | mse | 0.30 | full | 42 |
| 3 | mse | 0.30 | no_warmup | 7 |
| 4 | mse | 0.30 | no_warmup | 25 |
| 5 | mse | 0.30 | no_warmup | 42 |
| 6 | mse | 0.30 | no_detach | 7 |
| 7 | mse | 0.30 | no_detach | 25 |
| 8 | mse | 0.30 | no_detach | 42 |
| 9 | mse | 0.30 | no_warmup_no_detach | 7 |
| 10 | mse | 0.30 | no_warmup_no_detach | 25 |
| 11 | mse | 0.30 | no_warmup_no_detach | 42 |
| 12 | cosine | 0.84 | full | 7 |
| 13 | cosine | 0.84 | full | 25 |
| 14 | cosine | 0.84 | full | 42 |
| 15 | cosine | 0.84 | no_warmup | 7 |
| 16 | cosine | 0.84 | no_warmup | 25 |
| 17 | cosine | 0.84 | no_warmup | 42 |
| 18 | cosine | 0.84 | no_detach | 7 |
| 19 | cosine | 0.84 | no_detach | 25 |
| 20 | cosine | 0.84 | no_detach | 42 |
| 21 | cosine | 0.84 | no_warmup_no_detach | 7 |
| 22 | cosine | 0.84 | no_warmup_no_detach | 25 |
| 23 | cosine | 0.84 | no_warmup_no_detach | 42 |

### SLURM 脚本与 Run 划分

| 脚本 | RUN_IDS | 对应配置 |
|------|---------|----------|
| train_exp218_ct_ablation_part1.slurm | 0,1,2 | mse_0p30_full × 3 seeds |
| train_exp218_ct_ablation_part2.slurm | 3,4,5 | mse_0p30_no_warmup × 3 seeds |
| train_exp218_ct_ablation_part3.slurm | 6,7,8 | mse_0p30_no_detach × 3 seeds |
| train_exp218_ct_ablation_part4.slurm | 9,10,11 | mse_0p30_no_warmup_no_detach × 3 seeds |
| train_exp218_ct_ablation_part5.slurm | 12,13,14 | cos_0p84_full × 3 seeds |
| train_exp218_ct_ablation_part6.slurm | 15,16,17 | cos_0p84_no_warmup × 3 seeds |
| train_exp218_ct_ablation_part7.slurm | 18,19,20 | cos_0p84_no_detach × 3 seeds |
| train_exp218_ct_ablation_part8.slurm | 21,22,23 | cos_0p84_no_warmup_no_detach × 3 seeds |

---

## 4. 训练日志路径

日志目录：`/scratch/u5hv/shijie.u5hv/sde_seg/logs/`

- `exp218_ct_ablation_p1-<job_id>.out` / `.err`（part1）
- `exp218_ct_ablation_p2-<job_id>.out` / `.err`（part2）
- … 至 part8

从 log 中可搜索 `Mean IoU` 或 `best mIoU` 得到各 run 的验证 mIoU。

---

## 5. 结果汇总（请从 log 中填入）

### 5.1 原始数据（每 run 的 best mIoU）

从各 part 的 `.out` 中按 Run ID 顺序解析出 best mIoU 后，填入下表（或替换为你的实际数值）：

| 配置 | seed=7 | seed=25 | seed=42 | mean | std |
|------|--------|---------|---------|------|-----|
| mse_0p30_full | 0.6228 | 0.6252 | 0.6313 | 0.6264 | 0.0036 |
| mse_0p30_no_warmup | 0.6088 | 0.6319 | 0.6371 | 0.6259 | 0.0123 |
| mse_0p30_no_detach | 0.6223 | 0.6312 | 0.6467 | 0.6334 | 0.0101 |
| mse_0p30_no_warmup_no_detach | 0.6219 | 0.6294 | 0.6365 | 0.6293 | 0.0060 |
| cos_0p84_full | 0.6101 | 0.6382 | 0.6311 | 0.6265 | 0.0119 |
| cos_0p84_no_warmup | 0.6334 | 0.6302 | 0.6362 | 0.6333 | 0.0025 |
| cos_0p84_no_detach | 0.6364 | 0.6309 | 0.6348 | 0.6341 | 0.0023 |
| cos_0p84_no_warmup_no_detach | 0.6265 | 0.6256 | 0.6398 | 0.6307 | 0.0065 |

### 5.2 与 baseline 的对比

若使用 exp217 的固定 baseline（如 Job 2629307，mean mIoU = 0.6269），可计算各配置的 **Δ (pp) = mean mIoU − 0.6269**：

| 配置 | mean mIoU | Δ (pp) vs 0.6269 |
|------|-----------|------------------|
| mse_0p30_full | 0.6264 | −0.05 |
| mse_0p30_no_warmup | 0.6259 | −0.10 |
| mse_0p30_no_detach | 0.6334 | **+0.65** |
| mse_0p30_no_warmup_no_detach | 0.6293 | +0.24 |
| cos_0p84_full | 0.6265 | −0.04 |
| cos_0p84_no_warmup | 0.6333 | **+0.64** |
| cos_0p84_no_detach | 0.6341 | **+0.72** |
| cos_0p84_no_warmup_no_detach | 0.6307 | +0.38 |

---

## 6. 结论（填完结果后简要写）

- **full（warmup + detach）** 相比各消融变体的表现：full 在 MSE 与 Cosine 下均与 baseline（0.6269）基本持平或略低（约 −0.04～−0.05 pp）。消融中 **去掉 detach** 或 **去掉 warmup** 的变体在平均 mIoU 上多数优于 full，尤其是 **no_detach**（MSE +0.65 pp、Cosine +0.72 pp）和 **no_warmup**（Cosine +0.64 pp）。
- **去掉 warmup** 或 **去掉 detach** 后，MSE / Cosine 各自变化：  
  - **MSE λ=0.30**：no_warmup 略降（−0.10 pp），no_detach 明显提升（+0.65 pp），no_warmup_no_detach 小幅提升（+0.24 pp）。  
  - **Cosine λ=0.84**：no_warmup 与 no_detach 均提升（+0.64 pp、+0.72 pp），no_warmup_no_detach 提升 +0.38 pp。
- **是否建议保留 warmup 与 detach**：从本消融看，**detach 关闭**（梯度回传到 Depth Decoder）在两种 loss 下都带来约 0.65～0.72 pp 的增益，且 no_detach 的 std 较小、更稳定；**warmup 关闭**在 Cosine 下有收益、在 MSE 下略差。若以分割 mIoU 为首要目标，更推荐尝试 **保留 warmup、关闭 detach**（no_detach）作为新默认；若希望进一步简化，Cosine 下 no_warmup 或 no_warmup_no_detach 也可考虑。当前 full 配置更偏保守、稳定，但略牺牲了约 0.6～0.7 pp 的潜在提升。

---

## 7. 如何从 log 里解析 best mIoU

每个 run 在 log 里会先打印当前 run 的 tag（如 `cityscapes_pad_ct_mse_0p30_full_D372_S7`），训练结束后会打印该 run 的验证 mIoU。可用：

```bash
grep -E "Mean IoU|best mIoU|mIoU" /scratch/u5hv/shijie.u5hv/sde_seg/logs/exp218_ct_ablation_p*.out
```

或按你现有的 exp216/217 解析脚本，对每个 part 的 `.out` 按 Run ID 顺序提取 best mIoU，再按上表对应到 (ct_type, 消融组合, seed)。
