# 🎓 完整实验指南 - 毕设 Pipeline

## 📋 目录
- [实验概览](#实验概览)
- [已完成](#已完成-baseline)
- [当前位置](#当前位置)
- [下一步选择](#下一步选择)
- [快速决策树](#快速决策树)

---

## 实验概览

### 完整 Pipeline
```
1. Baseline (Scratch)          [已完成 ✅]
   └─ ImageNet → Segmentation
   
2. SDE Pretraining             [当前位置 📍]
   └─ Video Sequences → Depth Estimation
   
3. Transfer Learning           [下一步]
   └─ SDE Weights → Segmentation
   
4. DepthMix (可选)             [扩展]
   └─ Geometry-aware Augmentation
   
5. Multi-Task Learning (可选)  [扩展]
   └─ Joint Depth + Segmentation
```

### 论文贡献映射
```
Table 3 第1行 (Random): Baseline ✅
Table 3 第4-5行 (Ours): Transfer Learning + Data Selection
```

---

## 已完成: Baseline ✅

### 运行记录
- ✅ 100 labels (1/30) - baseline_100
- ✅ 372 labels (1/8) - baseline_372  
- ✅ 744 labels (1/4) - baseline_744

### 结果位置
```
/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_210/
    ├── <timestamp>_scratch_D100random_S42_.../
    ├── <timestamp>_scratch_D372random_S42_.../
    └── <timestamp>_scratch_D744random_S42_.../
```

### 下一步操作
1. ✅ 提取最终 mIoU 数值
2. ✅ 生成训练曲线图
3. ✅ 准备对比表格

---

## 当前位置 📍

你现在处于 **SDE 预训练** 阶段的决策点。

### 两个选择

#### 选项 A: 完整训练 SDE（学习价值高）
**时间成本:** ~5 天  
**文件:** `train_sde_dec5.slurm`, `train_sde_dec6.slurm`

```bash
# Phase 1: 300k iterations
sbatch slurm/train_sde_dec5.slurm

# Phase 2: 100k iterations (after Phase 1)
sbatch slurm/train_sde_dec6.slurm
```

**优点:**
- ✅ 完整理解整个 pipeline
- ✅ 可以调整参数做实验
- ✅ 训练经验对学术发展有价值

**缺点:**
- ⏰ 需要 4-5 天等待
- ⚠️ 可能遇到 24h 超时问题
- 💾 需要处理 checkpoint resume

**适合:** 时间充裕，想深入学习

---

#### 选项 B: 使用预训练模型（推荐 ⭐）
**时间成本:** 0 天（自动下载）

```python
# 代码自动从 Google Drive 下载
mono_pretrain = 'mono_cityscapes_1024x512_r101dil_aspp_dec6_...'
```

**优点:**
- ✅ 节省 5 天
- ✅ 官方模型，结果可靠
- ✅ 避免技术问题
- ✅ 更多时间写论文

**缺点:**
- ⚠️ 需要在论文中说明使用预训练模型

**适合:** 时间有限，专注结果和分析（**本科毕设推荐**）

---

## 下一步选择

### 如果选择 A（完整训练）

#### Step 1: 运行 SDE Phase 1
```bash
cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth
sbatch slurm/train_sde_dec5.slurm
```

**监控进度:**
```bash
# 查看任务状态
squeue -u $USER

# 查看日志
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/train_sde_dec5-*.out

# 检查迭代进度
grep "Train Iter" /scratch/.../train_sde_dec5-*.out | tail -5
```

**如果 24h 超时:**
```bash
# 1. 找到最后的 checkpoint
ls -lt /scratch/.../monodepth/cityscapes-monodepth-101aspp-dec5-crop/

# 2. 编辑配置文件添加 resume
nano configs/cityscapes_monodepth_highres_dec5_crop.yml
# 添加: resume: /path/to/checkpoint.pth

# 3. 重新提交
sbatch slurm/train_sde_dec5.slurm
```

#### Step 2: 运行 SDE Phase 2（等 Phase 1 完成）
```bash
# 确认 Phase 1 完成
grep "completed" /scratch/.../train_sde_dec5-*.out

# 提交 Phase 2
sbatch slurm/train_sde_dec6.slurm
```

#### Step 3: 创建 Transfer Learning 脚本
```bash
# 需要切换到 Agent 模式创建
# train_transfer_100.slurm
# train_transfer_372.slurm  
# train_transfer_744.slurm
```

**详细说明:** 见 `README_SDE_training.md`

---

### 如果选择 B（使用预训练模型）⭐

#### Step 1: 启用 Transfer 配置

编辑 `experiments.py` 第 170 行：

```python
# 取消注释这一行
('transfer', mono_pretrain, mono_pretrain, False, None, True, False, False, False),
```

#### Step 2: 创建 Transfer Learning 脚本

需要创建3个脚本（请切换到 Agent 模式）：
- `train_transfer_100.slurm`
- `train_transfer_372.slurm`
- `train_transfer_744.slurm`

#### Step 3: 直接运行 Transfer Learning

```bash
# 代码会自动下载 SDE 预训练模型
sbatch slurm/train_transfer_100.slurm
sbatch slurm/train_transfer_372.slurm
sbatch slurm/train_transfer_744.slurm
```

**预计时间:** 每个 ~12-24h

#### Step 4: 对比结果

```bash
# Baseline
Baseline 100: XX.XX%
Baseline 372: XX.XX%
Baseline 744: XX.XX%

# Transfer
Transfer 100: XX.XX%  (预期 +3~6%)
Transfer 372: XX.XX%  (预期 +3~5%)
Transfer 744: XX.XX%  (预期 +2~4%)
```

**详细对比:** 见 `SDE_vs_Baseline_comparison.md`

---

## 快速决策树

```
你有多少时间完成毕设？
    │
    ├─ 充裕 (>2个月) ───→ 选择 A (完整训练)
    │                    ├─ 阅读: README_SDE_training.md
    │                    └─ 运行: sbatch train_sde_dec5.slurm
    │
    └─ 紧张 (<2个月) ───→ 选择 B (预训练模型) ⭐
                         ├─ 阅读: SDE_vs_Baseline_comparison.md
                         ├─ 编辑: experiments.py (启用 transfer)
                         └─ 创建: transfer learning 脚本
```

---

## 📁 文件导航

### SLURM 脚本
```
slurm/
├── train_baseline_100.slurm       [已使用 ✅]
├── train_baseline_372.slurm       [已使用 ✅]
├── train_baseline_744.slurm       [已使用 ✅]
├── train_sde_dec5.slurm          [选项 A: SDE Phase 1]
├── train_sde_dec6.slurm          [选项 A: SDE Phase 2]
└── train_transfer_*.slurm        [待创建：Transfer Learning]
```

### 文档
```
slurm/
├── MASTER_GUIDE.md               [本文件：总体指南]
├── README_baseline_experiments.md [Baseline 详解]
├── README_SDE_training.md         [SDE 训练详解]
├── SDE_vs_Baseline_comparison.md  [对比分析]
└── QUICK_START.md                 [快速开始]
```

### 配置文件
```
configs/
├── cityscapes_joint.yml                        [Baseline 配置]
├── cityscapes_monodepth_highres_dec5_crop.yml [SDE Phase 1]
└── cityscapes_monodepth_highres_dec6_crop.yml [SDE Phase 2]
```

---

## 🎓 Dissertation 章节建议

### 你现在可以开始写的部分

#### 1. Background 章节
- ✅ 语义分割背景
- ✅ 半监督学习
- ✅ ImageNet 预训练
- 🔄 Self-supervised depth estimation（可以开始写原理）
- ⏳ Transfer learning from depth to segmentation

**参考:** `SDE_vs_Baseline_comparison.md` 中的 LaTeX 模板

#### 2. Project Execution - Baseline 部分
- ✅ 实验设置
- ✅ 训练配置
- ✅ Baseline 结果
- 🔄 可以开始准备 SDE 部分的框架

#### 3. Critical Evaluation - Baseline 分析
- ✅ 定量结果（mIoU 表格）
- ✅ 训练曲线分析
- ✅ 失败案例讨论
- ⏳ 等 Transfer 结果后做对比

### 论文结构建议

```latex
Chapter 3: Project Execution

3.1 Experimental Setup
    3.1.1 Dataset and Evaluation Metrics
    3.1.2 Implementation Details
    3.1.3 Hardware and Software

3.2 Baseline Experiments [可以写了 ✅]
    3.2.1 Configuration
    3.2.2 Training Procedure
    3.2.3 Results

3.3 Self-Supervised Depth Estimation [准备中 🔄]
    3.3.1 Motivation
    3.3.2 Architecture
    3.3.3 Two-Phase Training Strategy

3.4 Transfer Learning Experiments [下一步 ⏳]
    3.4.1 Transfer Strategy
    3.4.2 Configuration
    3.4.3 Results and Comparison

Chapter 4: Critical Evaluation

4.1 Quantitative Results
    4.1.1 Baseline Performance [有数据 ✅]
    4.1.2 Transfer Learning Impact [等实验 ⏳]
    4.1.3 Statistical Analysis

4.2 Qualitative Analysis
    4.2.1 Success Cases
    4.2.2 Failure Cases
    4.2.3 Boundary Quality Comparison

4.3 Discussion
    4.3.1 Why Transfer Learning Helps
    4.3.2 Limitations
    4.3.3 Future Improvements
```

---

## 💡 我的建议（基于本科毕设）

### 推荐路线 ⭐

1. **现在（第1周）**
   - ✅ 决定用预训练模型（选项 B）
   - ✅ 创建 transfer learning 脚本
   - ✅ 提交 transfer 实验
   - ✅ 开始写 Background 章节

2. **实验运行中（第2周）**
   - 📝 写 Background（SDE 原理、Transfer learning 理论）
   - 📝 写 Baseline 部分（Project Execution）
   - 📊 分析 Baseline 结果（图表、失败案例）

3. **Transfer 完成后（第3周）**
   - 📊 提取 Transfer 结果
   - 📈 制作对比图表
   - 📝 写 Transfer Learning 实验部分
   - 📝 写 Critical Evaluation

4. **完善阶段（第4周+）**
   - 📝 Introduction 和 Conclusion
   - 🎨 优化图表和表格
   - ✍️ 语言润色
   - 🔍 References 完善

### 时间分配建议

| 任务 | 时间 | 优先级 |
|-----|------|--------|
| Transfer 实验 | 3-5 天 | 高 |
| Background 章节 | 1 周 | 高 |
| Baseline 分析 | 3 天 | 高 |
| Transfer 分析 | 3 天 | 高 |
| 可视化制作 | 2 天 | 中 |
| Introduction | 2 天 | 中 |
| Conclusion | 1 天 | 中 |
| 润色修改 | 1 周 | 低 |

**总计:** 约 4-5 周完成

---

## 🆘 如果遇到问题

### 技术问题
- **SLURM 超时** → 见 `README_SDE_training.md` 的故障排除
- **GPU 内存不足** → 减小 batch_size
- **找不到预训练模型** → 检查网络连接和 Google Drive 访问

### 时间问题  
- **实验来不及** → 使用预训练模型（选项 B）
- **写作时间不够** → 先完成核心章节，砍掉可选部分

### 结果问题
- **性能不如预期** → 检查配置，对比论文，正常波动 ±2%
- **训练不收敛** → 检查学习率、数据加载、损失函数

---

## 📞 需要帮助？

根据你当前需求，我可以帮你：

1. **创建 Transfer Learning 脚本** ✨ [推荐：立即做]
2. **解释 SDE 原理** [如果选择 A]
3. **准备论文图表模板** [Background/Results]
4. **设置 TensorBoard 可视化**
5. **分析 Baseline 结果**

**建议下一步:** 告诉我你选择 A 还是 B，我会帮你创建相应的脚本和指导！

---

## 📌 快速命令参考

```bash
# 查看任务
squeue -u $USER

# 查看日志
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/train_*-<JOBID>.out

# 取消任务
scancel <JOBID>

# 检查 GPU
nvidia-smi

# 提取 mIoU
grep "Best Val" /scratch/.../train_*-*.out

# 提交任务
sbatch slurm/<script_name>.slurm
```

---

**更新日期:** 2026-01-31  
**版本:** 1.0  
**状态:** Baseline 完成，等待 SDE/Transfer 决策

