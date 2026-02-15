# 实验 Run ID 映射表

---

## Table 3: 数据混合策略 (当前配置)

**选样方法**：固定 random  
**实验 ID**：`exp 210`  
**SLURM 脚本**：`slurm/table3_mixing_strategy/`

### 论文对齐（按 paper 作者设置）

| 实验 | 配置 |
|------|------|
| Baseline | scratch, 仅监督 |
| Pseudo-Labels | scratch_ema, mix_mask=None, only_unlabeled=True |
| ClassMix 372 | scratch_classmix, only_unlabeled=True |
| ClassMix 2975 | scratch_classmixgt, only_unlabeled=False, mix_use_gt=True |
| DepthMix | scratch_depthmixgt, only_unlabeled=False, mix_use_gt=True |

### Run ID 验证（3 种子: 7, 25, 42）

| Run ID | 实验 | n | 脚本 |
|--------|------|---|------|
| 0, 7, 14 | Baseline | 372 | train_baseline_372.slurm |
| 1, 8, 15 | Baseline | 2975 | train_baseline_2975.slurm |
| 2, 9, 16 | Pseudo-Labels | 372 | train_pseudo_labels_372.slurm |
| 3, 10, 17 | ClassMix | 372 | train_classmix_372.slurm |
| 4, 11, 18 | DepthMix | 372 | train_depthmix_372.slurm |
| 5, 12, 19 | DepthMix | 2975 | train_depthmix_2975.slurm |
| 6, 13, 20 | ClassMix-GT | 2975 | train_classmix_2975.slurm |

### 预期 mIoU

| Run ID | 实验 | 标签数量 | 预期 mIoU |
|--------|------|----------|-----------|
| 0 | Baseline | 372 | 59.14 ± 1.02 |
| 1 | Baseline | 2975 | 67.77 ± 0.13 |
| 2 | Pseudo-Labels | 372 | 62.39 ± 0.86 |
| 3 | ClassMix | 372 | 63.16 ± 0.89 |
| 4 | ClassMix | 2975 | 69.60 ± 0.32 |
| 5 | DepthMix | 372 | 64.14 ± 1.34 |
| 6 | DepthMix | 2975 | 69.83 ± 0.36 |

### 运行 Table 3

```bash
# 单个实验（1 个种子）
python run_experiments.py --machine ws --exp 210 --run 2

# 单个实验（3 个种子，取 mean±std）
python run_experiments.py --machine ws --exp 210 --run 2,9,16   # Pseudo-Labels 372

# 全部 21 个 config (Run 0-20)
python run_experiments.py --machine ws --exp 210 --run 0-21
```

### 3 个种子

已配置 `seed in [7, 25, 42]`，共 21 个 config。每 SLURM 脚本跑对应实验的 3 个种子。

### 代码与 Paper 对应关系

| 代码配置 | Paper | 说明 |
|----------|-------|------|
| `scratch` | Baseline | ema=False, mix_mask=None，纯监督 |
| `scratch_ema` | Pseudo-Labels | ema=True, mix_mask=None，均值教师 |
| `scratch_classmix` | ClassMix | ema=True, mix_mask="class", only_unlabeled=True |
| `scratch_depthmixgt` | DepthMix | ema=True, mix_mask="depthcomp", mix_use_gt=True |

---

## Table 1 实验的 Run ID 映射表

## 📋 完整的 Run ID 映射（Table 1 全部组合）

**重要**：experiments.py 的循环顺序是 **seed 在外层**，共 45 个 config（3 seeds × 15 组合）。  
每个 slurm 脚本跑 **3 个 Run ID**（对应 3 个 seed），用于计算 mIoU mean±std。

### experiments.py 生成顺序

```python
for seed in [7, 25, 42]:           # 外层：3 seeds
    for (name, ...), n_list in table1_exps:   # random, entropy, ours_us, ours_ds, ours_ds_us
        for n_subset in [100, 372, 744]:      # 内层
            # 生成 1 个 config
```

- Run 0–14: seed 7
- Run 15–29: seed 25
- Run 30–44: seed 42

### 完整映射表（每个脚本跑 3 个 Run ID = 3 seeds）

| 脚本文件 | 实验类型 | 标签数量 | RUN_IDS | 数据比例 | 说明 |
|----------|---------|---------|---------|---------|------|
| `train_random_100.slurm` | Random | 100 | **0,15,30** | 1/30 | Random (baseline) |
| `train_random_372.slurm` | Random | 372 | **1,16,31** | 1/8 | Random (baseline) |
| `train_random_744.slurm` | Random | 744 | **2,17,32** | 1/4 | Random (baseline) |
| `train_entropy_100.slurm` | Entropy | 100 | **3,18,33** | 1/30 | Entropy (segmentation uncertainty) |
| `train_entropy_372.slurm` | Entropy | 372 | **4,19,34** | 1/8 | Entropy (segmentation uncertainty) |
| `train_entropy_744.slurm` | Entropy | 744 | **5,20,35** | 1/4 | Entropy (segmentation uncertainty) |
| `train_us_100.slurm` | US | 100 | **6,21,36** | 1/30 | Uncertainty Sampling (depth) |
| `train_us_372.slurm` | US | 372 | **7,22,37** | 1/8 | Uncertainty Sampling (depth) |
| `train_us_744.slurm` | US | 744 | **8,23,38** | 1/4 | Uncertainty Sampling (depth) |
| `train_ds_100.slurm` | DS | 100 | **9,24,39** | 1/30 | Diversity Sampling (depth) |
| `train_ds_372.slurm` | DS | 372 | **10,25,40** | 1/8 | Diversity Sampling (depth) |
| `train_ds_744.slurm` | DS | 744 | **11,26,41** | 1/4 | Diversity Sampling (depth) |
| `train_ds_us_100.slurm` | DS+US | 100 | **12,27,42** | 1/30 | Diversity + Uncertainty (combined) |
| `train_ds_us_372.slurm` | DS+US | 372 | **13,28,43** | 1/8 | Diversity + Uncertainty (combined) |
| `train_ds_us_744.slurm` | DS+US | 744 | **14,29,44** | 1/4 | Diversity + Uncertainty (combined) |

### 按 Run ID 顺序（0–44 对应配置）

| Run ID | 实验类型 | 标签数量 | Seed |
|--------|---------|---------|------|
| 0, 15, 30 | Random | 100 | 7, 25, 42 |
| 1, 16, 31 | Random | 372 | 7, 25, 42 |
| 2, 17, 32 | Random | 744 | 7, 25, 42 |
| 3, 18, 33 | Entropy | 100 | 7, 25, 42 |
| ... | ... | ... | ... |
| 14, 29, 44 | DS+US | 744 | 7, 25, 42 |

## ✅ 关键实验

- **Random 100** (3 seeds): `--run 0,15,30`
- **Random 372** (3 seeds): `--run 1,16,31`
- **Random 744** (3 seeds): `--run 2,17,32`

## 📝 使用方法

### 使用 SLURM 脚本（推荐，每个脚本自动跑 3 seeds）

```bash
cd slurm/table1_data_selection

# 单个实验（自动跑 3 个 seed）
sbatch train_random_744.slurm

# 所有 15 个实验
./submit_all.sh
```

### 命令行运行

```bash
# Random 744（3 seeds -> mIoU mean±std）
python run_experiments.py --machine ws --exp 210 --run 2,17,32

# 全部 45 个 config
python run_experiments.py --machine ws --exp 210 --run 0-45
```

## 📊 日志输出信息

```
==========================================
Table 1 Experiment: Data Selection
==========================================
Experiment Type: Random
Number of Labels: 744
Run IDs: 2,17,32 (seeds 7, 25, 42)
==========================================
```

这样可以清楚地识别每个实验的配置。

---

**最后更新**：2026-02-15
**验证方法**：使用 `check_run_ids.py` 脚本确认
**所有脚本已更新**：包含详细的实验配置输出信息

