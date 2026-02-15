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

使用 `check_run_ids.py` 脚本确认的实验生成顺序：

### 完整映射表（按 Run ID 顺序）

| Run ID | 实验类型 | 标签数量 | 模式 | 脚本文件 | 数据比例 | 选择方法说明 |
|--------|---------|---------|------|----------|---------|-------------|
| **0** | Random | 100 | random | `train_random_100.slurm` | 1/30 | Random (baseline) |
| **1** | Entropy | 100 | fixed | `train_entropy_100.slurm` | 1/30 | Entropy (segmentation uncertainty) |
| **2** | US | 100 | fixed | `train_us_100.slurm` | 1/30 | Uncertainty Sampling (depth estimation error) |
| **3** | DS | 100 | fixed | `train_ds_100.slurm` | 1/30 | Diversity Sampling (depth feature diversity) |
| **4** | DS+US | 100 | fixed | `train_ds_us_100.slurm` | 1/30 | Diversity + Uncertainty Sampling (combined) |
| **5** | Random | 372 | random | `train_random_372.slurm` | 1/8 | Random (baseline) |
| **6** | Entropy | 372 | fixed | `train_entropy_372.slurm` | 1/8 | Entropy (segmentation uncertainty) |
| **7** | US | 372 | fixed | `train_us_372.slurm` | 1/8 | Uncertainty Sampling (depth estimation error) |
| **8** | DS | 372 | fixed | `train_ds_372.slurm` | 1/8 | Diversity Sampling (depth feature diversity) |
| **9** | DS+US | 372 | fixed | `train_ds_us_372.slurm` | 1/8 | Diversity + Uncertainty Sampling (combined) |
| **10** ⭐ | **Random** | **744** | **random** | `train_random_744.slurm` | **1/4** | **Random (baseline)** |
| **11** | Entropy | 744 | fixed | `train_entropy_744.slurm` | 1/4 | Entropy (segmentation uncertainty) |
| **12** | US | 744 | fixed | `train_us_744.slurm` | 1/4 | Uncertainty Sampling (depth estimation error) |
| **13** | DS | 744 | fixed | `train_ds_744.slurm` | 1/4 | Diversity Sampling (depth feature diversity) |
| **14** | DS+US | 744 | fixed | `train_ds_us_744.slurm` | 1/4 | Diversity + Uncertainty Sampling (combined) |

### 按实验类型分组

#### Random 基线实验
| Run ID | 标签数量 | 脚本文件 | 数据比例 |
|--------|---------|----------|---------|
| **0** | 100 | `train_random_100.slurm` | 1/30 |
| **5** | 372 | `train_random_372.slurm` | 1/8 |
| **10** ⭐ | **744** | `train_random_744.slurm` | **1/4** |

#### Entropy 熵值选择
| Run ID | 标签数量 | 脚本文件 | 数据比例 |
|--------|---------|----------|---------|
| **1** | 100 | `train_entropy_100.slurm` | 1/30 |
| **6** | 372 | `train_entropy_372.slurm` | 1/8 |
| **11** | 744 | `train_entropy_744.slurm` | 1/4 |

#### US 不确定性选择
| Run ID | 标签数量 | 脚本文件 | 数据比例 |
|--------|---------|----------|---------|
| **2** | 100 | `train_us_100.slurm` | 1/30 |
| **7** | 372 | `train_us_372.slurm` | 1/8 |
| **12** | 744 | `train_us_744.slurm` | 1/4 |

#### DS 多样性选择
| Run ID | 标签数量 | 脚本文件 | 数据比例 |
|--------|---------|----------|---------|
| **3** | 100 | `train_ds_100.slurm` | 1/30 |
| **8** | 372 | `train_ds_372.slurm` | 1/8 |
| **13** | 744 | `train_ds_744.slurm` | 1/4 |

#### DS+US 多样性+不确定性选择
| Run ID | 标签数量 | 脚本文件 | 数据比例 |
|--------|---------|----------|---------|
| **4** | 100 | `train_ds_us_100.slurm` | 1/30 |
| **9** | 372 | `train_ds_us_372.slurm` | 1/8 |
| **14** | 744 | `train_ds_us_744.slurm` | 1/4 |

## 🔍 实验生成顺序说明

实验生成顺序遵循以下逻辑：

```python
for n_subset in [100, 372, 744]:  # 外层循环：标签数量
    for pres_method in [None, "ent", "us", "ds", "ds_us"]:  # 内层循环：选择方法
        # 生成实验配置
```

**顺序**：
1. 先按标签数量分组（100 → 372 → 744）
2. 每组内按选择方法排序（Random → Entropy → US → DS → DS+US）

## ✅ 关键实验的 Run ID

- **Random 100**: Run 0
- **Random 372**: Run 5
- **Random 744**: Run 10 ⭐

## 🚨 常见错误

**错误**：使用 `--run 2` 运行 Random 744
- 实际运行的是：US 100
- 正确应该用：`--run 10`

## 📝 使用方法

### 运行单个实验

```bash
# Random 744
python run_experiments.py --machine ws --exp 210 --run 10

# Random 100
python run_experiments.py --machine ws --exp 210 --run 0

# Random 372
python run_experiments.py --machine ws --exp 210 --run 5
```

### 运行所有 Random 基线实验

```bash
python run_experiments.py --machine ws --exp 210 --run 0 5 10
```

### 运行所有实验

```bash
python run_experiments.py --machine ws --exp 210 --run all
```

### 使用 SLURM 脚本

```bash
# 单个实验
cd slurm/table1_data_selection
sbatch train_random_744.slurm

# 所有实验
./submit_all.sh
```

## 📊 日志输出信息

所有训练脚本现在都会在日志开头输出以下信息：

```
==========================================
Table 1 Experiment: Data Selection
==========================================
Experiment Type: Random
Number of Labels: 744
Selection Method: Random (baseline)
Data Fraction: 1/4
Run ID: 10
==========================================
```

这样可以清楚地识别每个实验的配置。

## 🔧 验证 Run ID

使用 `check_run_ids.py` 脚本验证：

```bash
source ~/fyp/venvs/sde_seg/bin/activate
python check_run_ids.py
```

---

**最后更新**：2026-02-12
**验证方法**：使用 `check_run_ids.py` 脚本确认
**所有脚本已更新**：包含详细的实验配置输出信息

