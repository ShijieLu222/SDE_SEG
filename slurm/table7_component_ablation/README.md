# Table 7: 框架组件组合消融实验

复刻论文 **Table 7（Framework Component Ablation）** 的实验，比较 S (Data Selection)、DX (DepthMix)、MTL (SDE Multi-Task Learning) 三种组件的组合对 Cityscapes 语义分割性能的影响。

---

## 实验设置

| 组件 | 说明 |
|------|------|
| **S** | Data Selection（数据选择，预选标注） |
| **DX** | DepthMix（基于深度的数据混合） |
| **MTL** | SDE Multi-Task Learning（自监督深度估计多任务学习） |

### 8 种组合

| 实验 | S | DX | MTL | 372 Labels | 2975 Labels |
|------|---|---|-----|------------|-------------|
| Baseline | — | — | — | 59.14 ± 1.02 | 67.77 ± 0.13 |
| MTL only | — | — | ✓ | 61.25 ± 0.55 (+2.10) | 69.76 ± 0.39 (+1.99) |
| DX only | — | ✓ | — | 64.14 ± 1.34 (+5.00) | 69.83 ± 0.36 (+2.06) |
| S only | ✓ | — | — | 64.25 ± 0.18 (+5.11) | — |
| S + MTL | ✓ | — | ✓ | 65.35 ± 0.10 (+6.21) | — |
| S + DX | ✓ | ✓ | — | 66.48 ± 0.27 (+7.34) | — |
| DX + MTL | — | ✓ | ✓ | 66.66 ± 1.05 (+7.52) | 71.16 ± 0.16 (+3.40) |
| S + DX + MTL | ✓ | ✓ | ✓ | **68.01 ± 0.83 (+8.87)** | — |

- **标注量**：372 (1/8) 和 2975 (Full)
- **Seeds**：7, 25, 42（共 3 个）
- **共 36 个 config**：exp 213

---

## experiments.py 修改说明

在 `experiments.py` 中新增了 **exp 213**，用于生成 Table 7 的所有配置：

- **Baseline**：exp 210 风格，transfer（mono_pretrain 初始化），无 EMA/DepthMix/预选，纯监督学习
- **MTL only**：exp 212 风格，mtl_pad 架构，EMA=True（pseudo-labeling），但 mix_mask=None（无 DepthMix）
- **DX only**：exp 210 风格，transfer + DepthMix（EMA + mix_mask="depthcomp"），无预选
- **S only**：exp 210 风格，transfer + 预选标注，无 EMA/DepthMix
- **S + MTL**：exp 212 风格，mtl_pad + 预选 + EMA，无 DepthMix
- **S + DX**：exp 210 风格，transfer + DepthMix + 预选
- **DX + MTL**：exp 212 风格，mtl_pad + DepthMix + EMA
- **S + DX + MTL**：exp 212 风格，mtl_pad + DepthMix + 预选 + EMA

**关键修复**：
1. Baseline 使用 transfer（mono_pretrain）而非 scratch，与论文 Table 7 对齐
2. MTL-only 启用 EMA（semi-supervised），但 mix_mask=None（不启用 DepthMix）
3. mix_use_gt 仅在 use_dx=True 时设为 True

---

## Run ID 映射

### exp 213（36 configs）

循环顺序：`combo → n_subset → seed`

| 脚本 | 实验 | 标签数 | RUN_IDS | 对应 seeds |
|------|------|--------|---------|------------|
| `train_baseline_372.slurm` | Baseline | 372 | 0, 1, 2 | 7, 25, 42 |
| `train_baseline_2975.slurm` | Baseline | 2975 | 3, 4, 5 | 7, 25, 42 |
| `train_mtl_372.slurm` | MTL only | 372 | 6, 7, 8 | 7, 25, 42 |
| `train_mtl_2975.slurm` | MTL only | 2975 | 9, 10, 11 | 7, 25, 42 |
| `train_dx_372.slurm` | DX only | 372 | 12, 13, 14 | 7, 25, 42 |
| `train_dx_2975.slurm` | DX only | 2975 | 15, 16, 17 | 7, 25, 42 |
| `train_s_372.slurm` | S only | 372 | 18, 19, 20 | 7, 25, 42 |
| `train_s_mtl_372.slurm` | S + MTL | 372 | 21, 22, 23 | 7, 25, 42 |
| `train_s_dx_372.slurm` | S + DX | 372 | 24, 25, 26 | 7, 25, 42 |
| `train_dx_mtl_372.slurm` | DX + MTL | 372 | 27, 28, 29 | 7, 25, 42 |
| `train_dx_mtl_2975.slurm` | DX + MTL | 2975 | 30, 31, 32 | 7, 25, 42 |
| `train_s_dx_mtl_372.slurm` | S + DX + MTL | 372 | 33, 34, 35 | 7, 25, 42 |

---

## 运行方式

### 运行全部 12 个实验（推荐）

```bash
cd slurm/table7_component_ablation
bash submit_all.sh
```

一次性提交 12 个 job，每个 job 跑 3 个 seeds。

### 单独运行某个实验

```bash
cd slurm/table7_component_ablation

# 单个实验（3 seeds）
sbatch train_baseline_372.slurm
sbatch train_s_dx_mtl_372.slurm
# ...
```

### 命令行直跑（不用 SLURM）

```bash
# exp 213 - Baseline 372 (3 seeds)
python run_experiments.py --machine ws --exp 213 --run 0,1,2

# exp 213 - S+DX+MTL 372 (3 seeds)
python run_experiments.py --machine ws --exp 213 --run 33,34,35

# 跑完 exp 213 全部 36 个 config
python run_experiments.py --machine ws --exp 213 --run 0-36
```

---

## 约束说明

- **训练时间限制**：24 小时/脚本
- **预训练权重**：使用 `mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4`
- **Data Selection**：使用 ds_us 方法预选标注

---

## 常用命令

```bash
# 查看 job 状态
squeue -u $USER

# 查看日志
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/table7_baseline_372-<JOB_ID>.out
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/table7_*.out

# 取消所有 Table 7 jobs
scancel -n table7_
```
