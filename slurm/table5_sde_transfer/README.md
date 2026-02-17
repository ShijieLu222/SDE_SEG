# Table 5: SDE 特征迁移方法研究

复刻论文 **Table 5（SDE Feature Transfer）** 的实验，验证多任务学习 (Multi-Task) 以及特征距离损失 (F) 对 Cityscapes 语义分割性能的影响。

---

## 实验设置

| 实验 | 说明 | SDE 权重 | 特征距离 (F) |
|------|------|----------|-------------|
| **Baseline** | 从头训练，不加载 SDE | 无 | — |
| **Transfer (no F)** | 加载 fd0 预训练权重，纯监督 | fd0 | — |
| **Transfer (F=✓)** | 加载 fd2 预训练权重，启用 feat_dist | fd2 | ✓ |
| **Multi-Task (F=✓)** | seg + depth 多任务，fd2 初始化 | fd2 | ✓ |

- **标注量**：372 (1/8) 和 2975 (Full)
- **Seeds**：7, 25, 42（共 3 个）
- **共 24 个 config**：4 设置 × 2 子集 × 3 seeds

---

## SDE 预训练权重说明

Table 5 复刻**必须使用论文作者提供的预训练权重**，区分方式看模型名前缀：

| 前缀 | 含义 | 来源 |
|------|------|------|
| **`mono_*`** | 论文作者预训练 | 从 Google Drive 自动/手动下载 |
| **`my_sde_*`** | 自己训练的 SDE | 本地训练得到 |

### Table 5 使用的模型

| 实验 | 使用模型 | 完整目录名 |
|------|----------|------------|
| Transfer (no F) | 作者 fd0 | `mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd0_crop512x512bs4` |
| Transfer (F=✓) / Multi-Task (F=✓) | 作者 fd2 | `mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4` |

**`my_sde_dec6` 是自训权重，Table 5 不用**。复刻论文请用 `mono_*` 模型。

### 模型存放路径

`MachineConfig.DOWNLOAD_MODEL_DIR`（如 `ws` 机器：`/scratch/u5hv/shijie.u5hv/sde_seg/models/`）

---

## Run ID 映射

### exp 210（18 configs）

循环顺序：`seed → n_subset → name`

| 脚本 | 实验 | 标签数 | RUN_IDS | 对应 seeds |
|------|------|--------|---------|------------|
| `train_baseline_372.slurm` | Baseline | 372 | 0, 6, 12 | 7, 25, 42 |
| `train_baseline_2975.slurm` | Baseline | 2975 | 3, 9, 15 | 7, 25, 42 |
| `train_transfer_noF_372.slurm` | Transfer (no F) | 372 | 1, 7, 13 | 7, 25, 42 |
| `train_transfer_noF_2975.slurm` | Transfer (no F) | 2975 | 4, 10, 16 | 7, 25, 42 |
| `train_transfer_F_372.slurm` | Transfer (F=✓) | 372 | 2, 8, 14 | 7, 25, 42 |
| `train_transfer_F_2975.slurm` | Transfer (F=✓) | 2975 | 5, 11, 17 | 7, 25, 42 |

### exp 212（6 configs）

| 脚本 | 实验 | 标签数 | RUN_IDS | 对应 seeds |
|------|------|--------|---------|------------|
| `train_multitask_372.slurm` | Multi-Task (F=✓) | 372 | 0, 2, 4 | 7, 25, 42 |
| `train_multitask_2975.slurm` | Multi-Task (F=✓) | 2975 | 1, 3, 5 | 7, 25, 42 |

---

## 预期 mIoU（val）

| 实验 | 372 Labels (1/8) | 2975 Labels (Full) |
|------|------------------|---------------------|
| Baseline | 59.14 ± 1.02 | 67.77 ± 0.13 |
| Transfer (no F) | 60.46 ± 0.64 (+1.31) | 69.00 ± 0.70 (+1.23) |
| Transfer (F=✓) | 60.80 ± 0.69 (+1.66) | 69.47 ± 0.38 (+1.71) |
| Multi-Task (F=✓) | **61.25 ± 0.55 (+2.10)** | **69.76 ± 0.39 (+1.99)** |

---

## 运行方式

### 运行全部 8 个实验（推荐）

```bash
cd slurm/table5_sde_transfer
bash submit_all.sh
```

一次性提交 8 个 job，每个 job 跑 3 个 seeds，共 24 个 config。

### 单独运行某个实验

```bash
cd slurm/table5_sde_transfer

# 单个实验（3 seeds）
sbatch train_baseline_372.slurm
sbatch train_transfer_F_2975.slurm
sbatch train_multitask_372.slurm
# ...
```

### 命令行直跑（不用 SLURM）

```bash
# exp 210 - Baseline 372 (3 seeds)
python run_experiments.py --machine ws --exp 210 --run 0,6,12

# exp 212 - Multi-Task 2975 (3 seeds)
python run_experiments.py --machine ws --exp 212 --run 1,3,5

# 跑完 exp 210 全部 18 个 config
python run_experiments.py --machine ws --exp 210 --run 0-18
```

---

## 常用命令

```bash
# 查看 job 状态
squeue -u $USER

# 查看日志
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/table5_baseline_372-<JOB_ID>.out
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/table5_*.out

# 取消所有 Table 5 jobs
scancel -n table5_
```

---

## 故障排除

### 1. Transfer (no F) 失败：fd0 模型下载错误

若看到 `ERROR: Downloaded file is not a valid zip file!` 或 `File header: b'<!DOCTYPE html>'`，说明 Google Drive 返回了 HTML 而非 zip，fd0 需手动下载：

```bash
# 方法 A：用 gdown（推荐，支持大文件）
pip install gdown
MODEL_DIR="/scratch/u5hv/shijie.u5hv/sde_seg/models"
mkdir -p ${MODEL_DIR}
cd ${MODEL_DIR}
gdown "https://drive.google.com/uc?id=1G7bDZ-0PsHeMSHK59EqJn5ncqMzWB1Js" -O mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd0_crop512x512bs4.zip
unzip -o mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd0_crop512x512bs4.zip -d mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd0_crop512x512bs4
# 若 zip 解压到当前目录，需把 encoder.pth depth.pth 等移入同名子目录
```

或从浏览器下载：https://drive.google.com/file/d/1G7bDZ-0PsHeMSHK59EqJn5ncqMzWB1Js/view

### 2. Transfer (F=✓) KeyError: ('cam_T_cam', 0, -1)

已修复：experiments.py 中 Transfer (F=✓) 现会设置 `disable_pose=False` 与 `pose_pretraining`，以启用 pose 网络生成 `cam_T_cam`。

---

## 约束说明

- 训练时间限制：24 小时/脚本
- 无 EMA、ClassMix、DepthMix、预选标注等额外方法
- 仅使用现有 exp id（210、212），未新增实验 id
