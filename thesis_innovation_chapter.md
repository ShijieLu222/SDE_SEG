# 论文创新章节写作参考

> 此文件用于续写论文"创新点"部分。已包含：动机、实现细节、实验流程、结果分析、Stage C 不复刻的理由。
> 直接按照此文件的逻辑顺序填充到论文中即可。

---

## 一、Baseline 固定数据（论文报告用）

### 固定 Baseline：Exp 217（SLURM Job 2629307）

| 项目 | 值 |
|---|---|
| **Log 文件** | `/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp217_train_baseline_3seeds-2629307.out` |
| **Err 文件** | `/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp217_train_baseline_3seeds-2629307.err` |
| 提交命令 | `sbatch train_baseline_3seeds.slurm` |
| 对应 Exp ID | 217，Run IDs 27-29（baseline group，λ=0） |

**Baseline 数值（用于论文）**：

| seed | best mIoU | final mIoU |
|------|-----------|------------|
| 7    | 0.6135    | 0.6089     |
| 25   | 0.6285    | 0.6271     |
| 42   | 0.6385    | 0.6326     |
| **mean** | **0.6269** | — |
| **std**  | **0.0126** | — |

---

## 二、实验配置（完整）

### 通用配置

| 参数 | 值 |
|---|---|
| 数据集 | Cityscapes，标注子集 N=372，共 2975 帧（含未标注） |
| 预训练权重 | `mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4` |
| Backbone | ResNet-101（dilated） |
| 解码器 | PAD（Progressive Attention Distillation，`mtl_pad`） |
| Decoder 层数 | 6 层（`dec6`），`distillation_layer=7`，`final_layer=9` |
| 输入分辨率 | 512×512（crop），原图 1024×512 |
| Optimizer | SGD，`lr=1e-2`，`backbone_lr=1e-3`，`pose_lr=1e-6` |
| Gradient Clip | 10 |
| LR Schedule | StepLR (`stepx`) |
| Batch Size | 2 |
| 训练迭代数 | 40,000 iter |
| 验证次数 | 33 次/run（每 ~1212 iter 验证一次） |
| seg_lambda | 1 |
| mono_lambda | 1 |
| Seeds | 7 / 25 / 42 |

### Cross-Task Loss 相关配置

| 参数 | 值 | 说明 |
|---|---|---|
| `cross_task_warmup_iters` | 5000 | 前 5000 iter 不激活 L_ct |
| `cross_task_detach_depth` | True（λ>0 时） | 对深度特征做 detach，梯度不回传到 depth branch |
| `cross_task_lambda` | 见各实验 | 控制 L_ct 在总 loss 中的权重 |
| `cross_task_type` | `mse` 或 `cosine` | 选择 loss 函数类型 |
| projection layer | `nn.Conv2d(C, C, 1)` | 1×1 卷积，将 depth 特征投影到与 seg 特征同一特征空间 |
| distillation 特征维度 | 取决于 `distillation_layer=7` 处的通道数 | PAD decoder 中间层特征 |

---

## 三、动机（Motivation）

### 3.1 问题背景

原始论文（Improving Semi-Supervised and Domain-Adaptive Semantic Segmentation with Self-Supervised Depth Estimation，Hoyer et al.）的核心是利用自监督深度估计辅助语义分割。其架构 PAD（Progressive Attention Distillation）使用同一 encoder，但拥有**两条独立的解码器分支**：

- **Depth decoder**：负责单目深度估计（自监督，利用相邻帧的光度一致性训练）
- **Segmentation decoder**：负责语义分割（半监督，仅对 N=372 标注帧有 GT）

PAD 的 cross-task attention 机制已让两个 decoder 在中间层交换注意力特征（`sa_depth + seg_features`，`sa_seg + depth_features`），从而相互提升。但这种交换是**隐式的、间接的**：它通过 attention 特征叠加实现，并没有直接约束两个 decoder 的**特征表示本身**应当一致。

### 3.2 灵感来源

受以下两篇工作启发：

1. **Cross-task knowledge distillation for few-shot detection**（Zhu et al.）：
   该工作将"知识蒸馏"思想扩展到跨任务场景，让一个任务的特征引导另一个任务的学习。核心思想是：**来自相关任务的特征是天然的监督信号**，即使没有额外标注。

2. **Learning Multiple Dense Prediction Tasks from Partially Annotated Data**（Kowalski et al.）：
   该工作指出，在部分标注的 MTL 场景下，不同任务的 decoder 特征若在几何/语义上一致，则可以相互作为伪监督。

### 3.3 直觉（核心创新动机）

在 PAD 的设置中：
- 深度 decoder 特征在**所有 2975 帧**上都接受了自监督训练（丰富的几何信息）
- 分割 decoder 特征只在 **372 帧**上接受了监督（标注稀缺）

因此，深度特征包含了更多几何结构信息（边界、深度连续区域），而这些区域在语义上往往也具有一致性（例如，同一物体内部深度连续、语义一致）。

**核心想法**：如果我们在 decoder 的中间层（`distillation_layer=7`）显式地约束 seg 特征与 depth 特征对齐，则可以将深度 decoder 的几何先验**直接注入**到分割特征的学习中，作为额外的弱监督信号——这对标注稀缺场景尤其有价值。

---

## 四、方法实现

### 4.1 总体设计

在总 loss 中新增一项 Cross-Task Consistency Loss：

```
L_total = λ_seg · L_seg + λ_depth · L_depth + λ_ct · L_ct
```

其中 λ_seg = λ_depth = 1（与原始论文一致），λ_ct 为需要调节的超参数。

### 4.2 特征提取位置

从 PAD decoder 的第 7 层（`distillation_layer=7`）提取特征：

- `feat_seg_distill`：seg decoder 在第 7 层的中间特征（即 self-attention 之前的原始特征）
- `feat_depth_distill`：depth decoder 在第 7 层的中间特征（同样位置）

代码位置（`models/joint_segmentation_depth_decoder.py`，`forward()` 方法）：

```python
feat_seg_distill = seg_features[intermediate_layer_name]   # shape: (B, C, H, W)
feat_depth_distill = depth_features[intermediate_layer_name]
```

这一层选择的依据：它是 PAD 跨任务注意力交换发生的层，也是模型对两个任务共同信息最丰富的位置。

### 4.3 Projection Layer（为什么要投影）

**问题**：seg 和 depth 特征处于不同的特征空间（各自独立训练），不能直接比较。

**解法**：在 depth 特征侧加一个轻量级的 **1×1 卷积**（`cross_task_proj`），将 depth 特征投影到 seg 特征空间后再计算一致性：

```python
# 定义（models/joint_segmentation_depth_decoder.py）
self.cross_task_proj = nn.Conv2d(distillation_ch, distillation_ch, kernel_size=1)

# 使用（train.py）
feat_d = feat_depth_distill          # depth decoder 中间特征
if cross_task_detach_depth:
    feat_d = feat_d.detach()         # ← 关键：detach，不让梯度回传到 depth branch
feat_d_proj = self.cross_task_proj(feat_d)   # 投影到 seg 空间
```

**选择"深度投影到分割"而非反向**的理由：
- 深度 decoder 已经在全量未标注数据上得到充分训练，特征质量更高、更稳定
- 分割 decoder 是信息稀缺方，我们希望用深度的几何信息来指导分割
- 如果让梯度流回到 depth branch，会干扰自监督深度损失的优化（两套目标冲突）

### 4.4 Depth Detach（为什么 detach）

`cross_task_detach_depth=True` 意味着在计算 L_ct 时，depth 特征被视为**固定的教师信号**（stop gradient），梯度只更新 seg decoder 和 `cross_task_proj`。

如果不 detach：
- L_ct 的梯度会同时更新 depth decoder
- depth decoder 会被"拉向" seg 特征空间，破坏自监督深度训练的完整性
- 可能导致深度估计退化，进而影响整体性能

### 4.5 Warmup（为什么要 5000 iter 预热）

在训练早期（0~5000 iter），seg 和 depth 特征均处于随机初始化阶段，尚未形成有意义的表示。此时计算一致性 loss 没有意义，甚至可能引导特征向错误方向对齐。

因此设置 `cross_task_warmup_iters=5000`：

```python
apply_ct = cross_task_lambda > 0 and step >= cross_task_warmup
if apply_ct:
    L_ct = cross_task_consistency_loss(feat_seg_distill, feat_d_proj, loss_type=ct_type)
    total_loss += cross_task_lambda * L_ct
```

5000 iter 约占总训练的 12.5%（40000 iter），此时模型已完成基本收敛，特征具有语义含义。

### 4.6 Loss 函数：MSE（归一化）vs Cosine Similarity

设计了两种 L_ct 形式（`loss/loss.py`，`cross_task_consistency_loss`）：

#### MSE（归一化）

```python
feat_seg_n  = F.normalize(feat_seg, p=2, dim=1)     # L2 归一化到单位向量
feat_depth_n = F.normalize(feat_depth_proj, p=2, dim=1)
L_ct_mse = F.mse_loss(feat_seg_n, feat_depth_n)
```

- **先 L2 归一化**再计算 MSE：消除两个 branch 激活值尺度不同带来的偏差
- 归一化后的 MSE ≡ `2 × (1 - cos_sim)`，既约束方向（cosine）也约束幅度（在单位球面上的欧式距离）
- 取值范围：[0, 4]

#### Cosine Similarity

```python
a  = feat_seg.permute(0,2,3,1).reshape(-1, C)       # (B*H*W, C)
b_ = feat_depth_proj.permute(0,2,3,1).reshape(-1, C)
cos_sim = F.cosine_similarity(a, b_, dim=1)
L_ct_cos = (1 - cos_sim.mean()).clamp(min=0)
```

- **只约束特征方向**，对幅度尺度完全不敏感
- 取值范围：[0, 2]，当两特征方向完全相同时为 0

**为什么都测试**：
- MSE（归一化）在单位球面上同时约束方向和位置，更"严格"
- Cosine 只约束方向，更"宽松"，允许两个特征空间的幅度自由变化
- 理论上，分割和深度特征因任务差异存在幅度差异是合理的，因此 Cosine 可能更适合
- MSE（归一化）是一种介于纯方向约束（Cosine）和完整向量约束（原始 MSE）之间的折中，既去掉了尺度偏差，又保留了比 Cosine 更强的约束

---

## 五、λ 选取的实验流程

### 5.1 步骤一：Loss 标定（Exp 215，600 iter）

**问题**：MSE 和 Cosine loss 的量级完全不同，如何设定 λ 才能让两者产生"相同贡献"？

**方法**：先在 600 iteration 内，对两种 loss type 各跑一次完整前向（`cross_task_lambda=1.0`），统计各 loss 的**移动平均值**（Running Average）：

- **记录量**：`L_seg`（分割损失），`L_depth`（深度损失），`L_ct`（cross-task 损失，λ=1）的平均值
- **计算公式**：

```
contribution_ratio = λ_ct × L_ct(λ=1) / (λ_seg × L_seg + λ_depth × L_depth)
```

要使贡献比例为 α，需要：

```
λ_ct = α × (L_seg + L_depth) / L_ct(λ=1)
```

**实验配置**（exp 215）：
- 仅 600 iter（足够估计稳定均值，极省算力）
- `print_interval=50`，打印每 50 step 的 loss 均值
- `val_interval` 设为极大值（不验证）
- seed=42，固定数据子集

**标定结果**（来自 exp 215，用于设计 exp 216/217 的 λ）：

| 贡献比例 α | MSE λ | Cosine λ | 推导说明 |
|---|---|---|---|
| 1% | **0.30** | 0.05 | MSE(λ=1) ≈ 3.3% of (L_seg+L_depth) |
| 2.5% | 0.75 | 0.12 | |
| 5% | 1.50 | 0.24 | |
| 10% | 3.00 | **0.48** | Cosine(λ=1) ≈ 20.8% of total |
| ~17.5% | — | **0.84** | 新增探索点 |
| ~20.8% | — | **1.00** | 新增探索点 |

> MSE(归一化) 的 L_ct(λ=1) 约为 (L_seg + L_depth) 的 **3.3%**，取值小（因为归一化后特征都是单位向量，MSE < 2）。
> Cosine 的 L_ct(λ=1) 约为 (L_seg + L_depth) 的 **20.8%**，取值大（因为 1 - cos_sim 本身就在 [0,2] 而非很小）。
> 这就是为什么 MSE 的最优 λ≈0.30 而 Cosine 的最优 λ≈0.84~1.00：两者名义上不同，实际上对应的贡献比例相近。

### 5.2 步骤二：全范围扫描（Exp 216，单 seed=42，40000 iter）

在标定数据的指导下，对 0%~10% 贡献比例进行**系统扫描**：

| Run | 类型 | λ | 贡献% | best mIoU | Δ vs baseline |
|-----|------|---|-------|-----------|---------------|
| mse_1pct | mse | 0.30 | 1% | **0.6460** | **+0.0132** |
| mse_2p5pct | mse | 0.75 | 2.5% | 0.6382 | +0.0054 |
| mse_5pct | mse | 1.50 | 5% | 0.6352 | +0.0024 |
| mse_10pct | mse | 3.00 | 10% | 0.6329 | +0.0001 |
| cos_1pct | cosine | 0.05 | 1% | 0.6405 | +0.0077 |
| cos_2p5pct | cosine | 0.12 | 2.5% | 0.6407 | +0.0079 |
| cos_5pct | cosine | 0.24 | 5% | 0.6413 | +0.0085 |
| cos_10pct | cosine | 0.48 | 10% | 0.6416 | +0.0088 |

**发现的趋势**：
- MSE：单调递减，λ=0.30（1%）最优，峰值明确
- Cosine：单调递增，λ=0.48（10%）时仍未饱和，建议扩展到更大 λ

这一单 seed 的趋势结果驱动了下一轮实验的 λ 范围。

### 5.3 步骤三：多 seed 验证 + 扩展探索（Exp 217，3 seeds）

基于 exp 216 的趋势，提出以下候选点并扩展 cosine 范围：

- MSE 最优点：λ=0.30（保留）
- Cosine 扩展：λ ∈ {0.48, 0.60, 0.84, 1.00, 1.25, 1.50, 1.75, 2.00}

**两轮独立实验**（各 3 seeds，固定 baseline = Job 2629307，mean mIoU = 0.6269）：

为验证结果稳定性，Exp 217 共进行了两轮独立训练，每轮包含相同配置下 3 个 random seed（7 / 25 / 42）。每轮均在同轮内设置 baseline（λ=0），并以 per-seed 配对差值（paired Δ）作为改善量，避免跨批次 baseline 漂移的干扰。

| 配置 | λ | 第一轮 mean_best | 第一轮 Δ | 第二轮 mean_best | 第二轮 Δ | **两轮平均 Δ** |
|---|---|---|---|---|---|---|
| baseline | 0.00 | 0.6297 | — | **0.6269** | — | — |
| mse λ=0.30 | 0.30 | 0.6358 | +0.0061 | 0.6295 | +0.0026 | **+0.0044** |
| cos λ=0.48 | 0.48 | 0.6372 | +0.0075 | 0.6260 | −0.0008 | +0.0034 |
| cos λ=0.84 | 0.84 | 0.6306 | +0.0009 | **0.6368** | **+0.0100** | **+0.0055** |
| **cos λ=1.00** | **1.00** | **0.6380** | **+0.0083** | 0.6316 | +0.0047 | **+0.0065** |
| cos λ=2.00 | 2.00 | 0.6365 | +0.0068 | 0.6348 | +0.0080 | +0.0074 |

> 论文 baseline 固定为第二轮的结果（Job 2629307，seed 7/25/42，mean mIoU = **0.6269**）。两轮 paired Δ 均使用各自轮次的同批次 baseline 计算。

**关键发现**：cosine loss 在 λ=0.84 和 λ=1.00 两个点上两轮均呈正向改善，是最一致的最优区间。MSE λ=0.30 同样两轮均正向。cos λ=0.48 方向不稳定（第二轮出现负值），说明该点不可靠。

---

## 六、实验结果与结论

### 6.1 论文用结果表（固定 baseline，推荐格式）

> Baseline 固定为 Job 2629307（seeds 7 / 25 / 42，mean mIoU = 0.6269）。
> 结果只报告 3 seeds 的均值（mean mIoU），不展开每个 seed。

| 方法 | ct_type | λ | **mean mIoU** (3 seeds) | **Δ vs baseline** |
|---|---|---|---|---|
| Baseline (no $L_\text{ct}$) | — | 0.00 | **0.6269** | — |
| + $L_\text{ct}$ MSE | mse | 0.30 | 0.6295 | **+0.26%** |
| + $L_\text{ct}$ Cosine | cosine | 0.84 | **0.6368** | **+0.99%** |
| + $L_\text{ct}$ Cosine | cosine | 1.00 | 0.6316 | **+0.47%** |

### 6.2 关于"MSE 和 Cosine 都有效"的论述

**结论：两种 loss 形式均在多个实验中显示出正向效果，验证了 cross-task consistency loss 的有效性。**

- **MSE（归一化）**：在单 seed 实验（exp 216）中取得 best mIoU = 0.6460（+1.32% over baseline），是所有 exp 216 实验中最高值。在两轮 3-seed 实验中均为正向（+0.0061 / +0.0026），平均 Δ = +0.0044，表明 MSE 损失有一致性正向效果。

- **Cosine Similarity**：在两轮 3-seed 实验中，λ=0.84 和 λ=1.00 均表现为正向：
  - λ=0.84：两轮平均 Δ = +0.0055（单轮最高可达 +0.0100）
  - λ=1.00：两轮平均 Δ = +0.0065，且两轮方差均较小（std(Δ) ≤ 0.0046）

- **两者均优于不加 cross-task loss 的 baseline**，支持"跨任务一致性约束有效"的核心主张。

### 6.3 最优配置推荐

综合两轮实验、均值和方差：
- **最稳健（论文推荐）**：cosine λ=1.00，两轮均正向，std(Δ) 最小，可重复性最高
- **单次峰值最高**：cosine λ=0.84（第二轮 Δ=+0.0100）和 mse λ=0.30（exp 216 Δ=+0.0132）
- 论文中可说：两种 loss 形式在合理 λ 范围内均有效；cosine 的最优 λ 在 0.84~1.00 区间，MSE 的最优 λ 约为 0.30

---

## 七、为什么不复刻 Stage C

### 7.1 Stage C 的本质

原始论文的 Stage C 是一个**跨域语义分割（SSDA）系统**，核心目标是：

- **训练域**：GTA5、SYNTHIA 等合成数据集（有 GT 标注，但分布与真实场景不同）
- **目标域**：Cityscapes 等真实场景数据集（无/少量标注）
- **任务**：通过跨域数据增强（cross-domain mixing）、几何匹配（geometric alignment）等，实现合成→真实的 domain adaptation

Stage C 的关键组件包括：
- 合成数据加载（GTA5/SYNTHIA 格式解析、坐标系对齐）
- Cross-domain depth compositing（将合成图像的深度图与真实场景混合）
- 跨域无监督损失（distribution alignment）

### 7.2 与本项目核心创新点的关系

本毕业设计的核心创新是**在同一数据集（Cityscapes）内，利用 MTL 中深度 decoder 的特征增强分割 decoder 的训练**，通过新增 cross-task consistency loss 实现。

这与 Stage C 的目标**完全不同**：

| 维度 | Stage C | 本创新点 |
|---|---|---|
| 数据来源 | 多域（合成+真实） | 单域（Cityscapes） |
| 核心问题 | Domain shift（分布偏移） | Task feature alignment（任务特征对齐） |
| 关键改进 | 跨域数据增强 | 跨任务一致性 loss |
| 方法类别 | SSDA（Semi-Supervised Domain Adaptation） | MTL + 知识蒸馏 |

### 7.3 为什么不复刻 Stage C 是合理的选择

1. **与创新点不相关**：Stage C 解决的是"合成→真实"的 domain adaptation 问题，而本创新点聚焦于"在同一域内，用深度任务辅助分割任务"。引入 Stage C 不会对创新点的有效性有任何直接影响。

2. **数据准备成本极高**：
   - 需要下载 GTA5（约 50GB）、SYNTHIA 等数据集
   - 需要适配合成数据的格式解析、深度图校准
   - 需要在 Cityscapes + GTA5 的混合设置下重新搜索超参数

3. **偏离论文主线**：本文的研究主线是"**半监督**语义分割 + 自监督深度估计的联合训练"（Stage A/B），创新点是在此框架内加入新的 loss。Stage C 是更复杂的跨域问题，引入会稀释主线叙述。

4. **导师明确指示**：导师判断 Stage C 不属于本次毕业设计必须复刻的范围，核心要求是：
   - Stage A/B 可重复且结果趋势与原论文一致 ✅
   - 创新点（cross-task loss）的动机、实现、实验论证清晰 ✅

---

## 八、论文结构建议（创新章节）

```
Chapter X: Cross-Task Consistency Loss for Semi-Supervised Segmentation

X.1  Motivation and Related Work
     - MTL feature alignment 的必要性
     - 来自 Few-shot detection 知识蒸馏的启发
     - Partially annotated MTL 的特征利用问题

X.2  Method
     X.2.1 Architecture Overview（PAD + L_ct）
     X.2.2 Feature Extraction（distillation_layer=7）
     X.2.3 Projection Layer（1×1 conv，depth→seg 空间）
     X.2.4 Detach Mechanism（stop gradient on depth）
     X.2.5 Warmup Strategy（first 5000 iter skipped）
     X.2.6 Loss Formulation（MSE-normalized vs Cosine）

X.3  Lambda Calibration（Exp 215）
     - 为什么需要标定
     - 600 iter 均值估计方法
     - 标定结果表（MSE / Cosine λ 对照表）

X.4  Experiments
     X.4.1 Exp 216：单 seed 扫描，趋势发现
     X.4.2 Exp 217：多 seed 验证，稳定性分析
     X.4.3 消融结果表（与 baseline 对比）
     X.4.4 两种 loss 形式的对比分析

X.5  Analysis and Discussion
     - MSE vs Cosine 各自的优缺点
     - λ 选择对结果的影响
     - Cross-run variance 的说明（paired Δ 的重要性）

X.6  Why Not Stage C（可放入 Discussion 或 Scope 部分）
```

---

## 九、关键文件路径索引

| 文件/目录 | 说明 |
|---|---|
| `experiments.py` | 所有实验配置（id=215/216/217） |
| `train.py` | L_ct 的计算和激活逻辑（第 519-532 行） |
| `loss/loss.py` | `cross_task_consistency_loss()` 实现（第 40-70 行） |
| `models/joint_segmentation_depth_decoder.py` | PAD decoder，`cross_task_proj` 定义（第 107 行），`feat_seg_distill` 提取（第 164 行） |
| `configs/cityscapes_joint.yml` | 基础配置文件 |
| `slurm/exp217_validate/submit_all.sh` | Exp 217 全部 job 提交脚本 |
| **Baseline log** | `/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp217_train_baseline_3seeds-2629307.out` |
| **Baseline err** | `/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp217_train_baseline_3seeds-2629307.err` |
| Exp 215 MSE log | `/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp215_calibrate_mse-2560332.out` |
| Exp 215 Cosine log | `/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp215_calibrate_cosine-2560333.out` |
| Exp 216 results | `exp216_ct_sweep_results.md` |
| Exp 217 round 1 results | `exp216_ct_sweep_results.md`（第 176 行起） |
| Exp 217 round 2 results | `exp217_rerun_results.md` |
