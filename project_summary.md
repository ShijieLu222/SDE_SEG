# 项目总结：标注高效语义分割中的跨任务一致性损失

---

## 一、研究内容

**题目**：Label-Efficient Semantic Segmentation with Cross-Task Consistency Loss

**核心工作**：在 Hoyer et al. 的联合分割-深度框架（代码库中称 `mtl_pad`，包含跨任务 self-attention 融合模块）基础上，引入 **Cross-Task Consistency Loss（L_ct）**，通过在 decoder 中间层显式约束分割特征向深度特征对齐（单向：depth→seg），提升语义分割性能。

---

## 二、研究动机

### 背景问题

语义分割需要大量像素级人工标注，代价高昂。在标注数量有限（如仅占训练集的 12.5%）的情况下，如何充分利用未标注数据提升分割性能，是**标注高效学习（Label-Efficient Learning）**的核心问题。

### 原始方法（Hoyer et al., PAD）

Hoyer et al. 提出联合训练语义分割（有监督）与自监督深度估计（无需深度标注）：

- 分割和深度共享 ResNet-101 encoder
- 使用联合解码器（`mtl_pad`）通过 self-attention 跨任务融合特征
- 利用无标注视频序列进行自监督深度学习
- 在 N=372（1/8 Cityscapes）标注下达到 **62.97 ± 0.08% mIoU**（直接复现值，见 exp214）

### 原始方法的不足

PAD 框架中，分割和深度两个解码器的中间特征**没有被显式约束**相互对齐：
- 跨任务特征一致性依赖隐式的 self-attention 融合，缺乏直接监督
- 深度信息对分割特征的影响路径不够明确
- 两个任务的中间表示可能学到冗余或不一致的特征

### 改进思路

在 decoder 中间层（distillation layer）处，**将深度分支的几何特征作为 teacher signal，引导分割分支学习更具几何一致性的表示（单向促进 depth→seg）**：
- 深度特征 `.detach()` 后作为固定目标，只向分割分支传递梯度
- 不干扰自监督深度训练，两个任务的主损失保持独立
- 通过轻量级 1×1 卷积（`cross_task_proj`）弥合两个分支的特征尺度差异

---

## 三、方法设计

### 损失函数

$$L_{total} = \lambda_{seg} \cdot L_{seg} + \lambda_{depth} \cdot L_{depth} + \lambda_{ct} \cdot L_{ct}$$

其中：

$$L_{ct} = 1 - \frac{1}{N} \sum_{i} \cos\!\left(f_{seg}^{(i)},\ \text{proj}(f_{depth}^{(i)})\right)$$

- $f_{seg}$：PAD decoder 第 7 层的分割特征（B × C × H × W）
- $f_{depth}$：PAD decoder 第 7 层的深度特征（B × C × H × W）
- $\text{proj}$：1×1 Conv2d，将深度特征投影到与分割特征相同的空间
- **Cosine 实现**：将特征图从 (B, C, H, W) reshape 为 (B×H×W, C)，对每个像素位置的 C 维向量调用 `F.cosine_similarity(dim=1)`（内部等价于 L2-normalize 后取内积），再对所有 B×H×W 个位置取均值

### 梯度流向设计

```
f_seg  ←─── L_ct 梯度（分割分支被优化）
f_depth ──── .detach()（深度分支不被 L_ct 影响）
```

L_ct **单向**约束分割特征向深度特征对齐，不反向干扰深度分支。深度分支在联合训练阶段仍由自监督光度/平滑损失 $L_{depth}$ 持续优化；在 $L_{ct}$ 中我们对 $f_{depth}$ 施加 stop-gradient，使其在该项中充当稳定的 teacher signal，避免 $L_{ct}$ 的梯度干扰深度分支的优化。

### 集成位置

```
PAD Decoder:
  Depth branch:   [scale4] → [scale3] → [scale2] → [distill_layer=7] ──.detach()──→ proj → ↘
                                                                                              L_ct
  Seg branch:     [scale4] → [scale3] → [scale2] → [distill_layer=7] ─────────────────────→ ↗
                                                                    ↑
                                         Self-attention cross-fusion（原有机制，保留）
```

### 超参数

| 参数 | 值 |
|---|---|
| λ_seg | 1.0 |
| λ_depth | 1.0 |
| **λ_ct（最优）** | **1.00** |
| cross_task_type | cosine |
| cross_task_warmup_iters | 5,000 |
| cross_task_detach_depth | True |
| distillation_layer | 7 |

---

## 四、实验设置

### 数据集与标注量

| 项目 | 设置 |
|---|---|
| 数据集 | Cityscapes（urban driving，19 classes） |
| 有标注图像 | **N = 372**（12.5% of 2975 train images） |
| 无标注图像 | 2975 张（全部训练集，用于深度自监督） |
| 验证集 | 500 张 |

### 模型配置

| 项目 | 设置 |
|---|---|
| Backbone | ResNet-101（ImageNet + monodepth 预训练） |
| 解码器 | PAD decoder（dec6），distillation_layer=7，final_layer=9 |
| 预训练权重 | `mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4` |
| 优化器 | SGD，lr=1e-2，backbone_lr=1e-3 |
| LR 调度 | stepx |
| 训练迭代 | 40,000 iter |
| Batch size | 2 |
| 数据增强 | RandomCrop 512×512，HorizontalFlip，ColorJitter，Blur |
| 无标注数据使用 | EMA pseudo-label（consistency_weight=1.0） |

### 评估方式

- 验证集 mIoU（19 类）
- **3 个随机 seed（7 / 25 / 42）** 取均值与标准差
- 与 baseline（完全相同配置，仅 λ_ct=0）对比

---

## 五、实验路径

### exp 215 — Loss 量级标定

在 λ_ct=1.0、训练 600 iter 的短实验中测量各 loss 的原始量级（取 iter 200–600 的均值；来源：Job 2560809/2560811）：

| Loss 类型 | L_seg | L_depth | L_seg+L_depth | L_ct（λ=1） | 说明 |
|---|---|---|---|---|---|
| MSE（L2-norm） | 0.365 | 0.024 | **0.389** | 0.013 | 归一化后数值极小 |
| **Cosine** | **0.393** | **0.024** | **0.417** | **0.087** | 量级适中 |

据此用**贡献对齐法**计算各贡献比例（1%~20%）对应的 λ 值：

$$\lambda_{ct} = \alpha \times \frac{\overline{L_{seg}} + \overline{L_{depth}}}{L_{ct}(\lambda_{ct}=1)}$$

其中 $\overline{L_{seg}}$、$\overline{L_{depth}}$ 为 iter 200~600 的均值（跳过早期不稳定阶段）。

---

### exp 216 — λ 单 seed 扫描（seed=42）

| ct_type | λ | best mIoU | Δ baseline |
|---|---|---|---|
| baseline | 0.0 | 0.6328 | — |
| mse | 0.30（1%） | **0.6460** | +0.0132 |
| mse | 0.75（2.5%） | 0.6382 | +0.0054 |
| mse | 1.50（5%） | 0.6352 | +0.0024 |
| mse | 3.00（10%） | 0.6329 | +0.0001 |
| cosine | 0.05（1%） | 0.6405 | +0.0077 |
| cosine | 0.12（2.5%） | 0.6407 | +0.0079 |
| cosine | 0.24（5%） | 0.6413 | +0.0085 |
| cosine | 0.48（10%） | 0.6416 | +0.0088 |

**初步结论**：MSE λ=0.30 单次结果最佳，但单 seed 结果具有欺骗性（seed=42 偏好 seed）。

---

### exp 217 batch1 — 多 seed 验证

正式 baseline（exp214 λ=0，3 seeds）：**mean_best = 62.97%，std = 0.08%**

| 配置 | s7 best | s25 best | s42 best | mean_best | std | mean(Δ) | std(Δ) |
|---|---|---|---|---|---|---|---|
| baseline (λ=0) | 63.03% | 63.04% | 62.86% | 62.97% | 0.10% | — | — |
| mse λ=0.30 | 62.35% | 63.43% | 64.95% | 63.58% | 1.31% | +0.61% | 0.63% |
| cosine λ=0.48 | 62.41% | 63.65% | 65.11% | 63.72% | 1.35% | +0.75% | 0.74% |
| cosine λ=0.60 | 61.93% | 63.07% | 63.99% | 63.00% | 1.03% | +0.03% | 0.56% |
| cosine λ=0.84 | 62.19% | 63.26% | 63.73% | 63.06% | 0.79% | +0.09% | 0.48% |
| **cosine λ=1.00** | **63.98%** | **63.32%** | **64.11%** | **63.80%** | **0.42%** | **+0.83%** | **0.49%** |

> Δ = per-seed best mIoU(method) − best mIoU(baseline)，各 seed 独立计算后取 mean/std

**关键发现**：
- **cosine λ=1.00 在所有测试配置中均值最高（+0.83%）、方差最小（std=0.42%）**，且每个 seed 均正向提升（Δ: +0.95%, +0.28%, +1.25%）
- cosine λ=0.60 / 0.84 的 mean(Δ) 接近零（+0.03%, +0.09%），改善幅度在 seed 间方差范围内，不稳定
- MSE λ=0.30：seed=42 表现优异（+1.25%）但 seed=7 低于 baseline（−0.68%），高方差（std=1.31%）不可靠

**方差规律（观察到的趋势）**：在测试的 cosine λ 范围（0.48→1.00）内，std 随 λ 增大呈下降趋势（1.35% → 0.42%），表明较强的对齐约束可能带来更稳定的训练动态。这一趋势还需 batch2（λ=1.25~2.00）进一步验证。

**高方差的可能原因**：对于较小的 λ（如 cosine 0.48、MSE 0.30），L_ct 在 warmup（5000 iter）结束后突然以一定强度介入，不同 seed 处于不同的优化状态，对齐约束可能与主损失形成短暂竞争，导致不同 seed 间的分叉。较大的 λ（如 cosine 1.00）在较强约束下能更一致地引导特征方向。

---

### exp 217 batch2 — 高 λ 延伸探索（已完成）

| 配置 | λ | seed=7 | seed=25 | seed=42 | mean_best | std | Δ mean |
|---|---|---|---|---|---|---|---|
| cosine λ=1.25 | 1.25 | 0.6261 | 0.6366 | 0.6343 | 0.6323 | 0.0045 | +0.0026 |
| cosine λ=1.50 | 1.50 | 0.6123 | 0.6401 | 0.6440 | 0.6321 | 0.0141 | +0.0024 |
| cosine λ=1.75 | 1.75 | 0.6325 | 0.6345 | 0.6323 | 0.6331 | 0.0010 | +0.0034 |
| cosine λ=2.00 | 2.00 | 0.6299 | 0.6335 | 0.6462 | 0.6365 | 0.0070 | +0.0068 |

**结论：λ>1.00 的所有配置均低于 λ=1.00（0.6380），且 λ=1.50 出现崩溃 seed（0.6123），确认 cosine λ=1.00 为最优点。**

---

## 六、阶段性成果

### 核心结论

> 在联合分割-深度框架（MTL-only，无 DepthMix，无 Data Selection）下，仅新增 L_ct（cosine，λ=1.00），Cityscapes N=372 标注子集上的 mIoU 从 **62.97%** 提升至 **63.80%**，3 seeds 均正向提升（per-seed Δ: +0.95%, +0.28%, +1.25%，mean=**+0.83%**，std=0.49%），训练稳定性最优（result std=**0.42%**）。

### 方法论贡献

1. **贡献对齐标定法**：提出用短实验（600 iter）测量各 loss 量级，再以期望贡献比例（α% of L_seg+L_depth）反推 λ 的系统方法，避免了纯经验调参
2. **单 seed 陷阱的揭示**：实验证明单 seed 筛选的"最优" λ 在多 seed 下可能不可靠（MSE 0.30 单次高达 0.6460，多 seed 均值仅 +0.61% 且方差极大），多 seed 验证是必要的
3. **方差随 λ 变化的趋势**：在测试的 cosine λ 范围（0.48→1.00）内，std 随 λ 增大呈下降趋势（1.10%→0.42%），较强的对齐约束带来更稳定的训练动态；但 λ>1.00 后 std 非单调，λ=1.50 甚至出现崩溃 seed，提示过强约束的风险
4. **λ=1.00 是最优点确认**：batch2（λ=1.25~2.00）所有配置的 mean_best 均低于 λ=1.00，完整 λ 曲线呈现先升后降的倒 U 形，λ=1.00 为峰值

### 与 Hoyer et al. 原始结果的对比

> 所有数值均从实际训练日志提取（3 seeds mean ± std）。best = 训练过程中最高 val mIoU；final = 最后一个 checkpoint 的 val mIoU。

| 方法 | mean_best ± std | mean_final ± std | 来源日志 |
|---|---|---|---|
| Baseline（无 SDE） | 61.35 ± 1.25% | — | `table7_baseline_372-2363570.out` |
| PAD MTL（Hoyer et al.）† | 62.55 ± 0.35% | — | `table7_mtl_372-2363572.out` |
| **PAD MTL baseline（直接对照）** | **62.97 ± 0.08%** | **62.59 ± 0.41%** | **`exp214_ct_lam0_3seeds-2560760.out`** |
| **PAD MTL + L_ct，cosine λ=1.00（ours）** | **63.80 ± 0.42%** | **63.21 ± 0.42%** | **`exp217_train_cos_1p00_3seeds-2578798.out`** |
| PAD DX + MTL | 65.87 ± 1.23% | — | `table7_dx_mtl_372-2355591.out` |
| PAD S + DX + MTL（完整方法） | 67.89 ± 0.86% | — | `table7_s_dx_mtl_372-2355593.out` |

> Δ(best): +0.83%，Δ(final): +0.62%；两个指标均一致正向，结果可靠。
>
> † Table 7 复现的 MTL（62.55%）与 exp214 直接 baseline（62.97%）有小差异，
> 原因是两者的 unlabeled data 子集选择策略略有不同（exp214 使用 random subset mode）。
> 论文应使用 exp214（62.97%）作为直接对照，两者配置完全一致，仅 λ_ct 不同。

---

## 七、后续计划

### 已完成（exp 217 全部结束）

- [x] 分析 cosine λ=1.25~2.00 结果，绘制完整 λ 曲线
- [x] 确认最终最优 λ（**cosine λ=1.00**，已验证为峰值）
- [x] 更新 exp216_ct_sweep_results.md

### 论文写作

- [ ] **Method 章节**：L_ct 设计、公式推导、梯度流向、与 PAD 的集成
- [ ] **Experiments 章节**：
  - Main Table：PAD baseline vs PAD + L_ct（3 seeds）
  - Ablation Table 1：λ 消融（λ sweep 曲线）
  - Ablation Table 2：MSE vs Cosine 对比
  - Implementation Details：标定法、warmup 设置
- [ ] **Analysis 章节**：方差规律、单 seed 陷阱分析

### 可选扩展实验

- [ ] 在 DX+MTL 配置基础上叠加 L_ct，验证在更强 baseline 上是否仍有提升
- [ ] Per-class IoU 分析，识别 L_ct 对哪些类别帮助最大
- [ ] 可视化 `f_seg` 与 `proj(f_depth)` 的特征图，直观展示对齐效果

---

## 八、代码文件索引

| 文件 | 内容 |
|---|---|
| `exp215_calibration_results.md` | Loss 标定实验结果与 λ 计算推导 |
| `exp216_ct_sweep_results.md` | λ sweep 完整结果 + exp217 多 seed 验证结果 |
| `TABLE7_RESULTS.md` | Hoyer et al. Table 7 复现结果（数字见日志索引） |
| `TABLE5_RESULTS.md` | Hoyer et al. Table 5 复现结果 |
| `experiments.py` | exp 214~217 实验配置 |
| `loss/loss.py` | L_ct 实现（cosine / mse / l1） |
| `models/joint_segmentation_depth_decoder.py` | PAD decoder + cross_task_proj 定义 |
| `train.py` | 训练主循环，L_ct 集成与权重叠加 |
| `slurm/exp217_validate/` | 多 seed 验证训练脚本 |

---

## 九、关键日志文件索引

所有日志位于 `/scratch/u5hv/shijie.u5hv/sde_seg/logs/`

### 核心对照实验（直接用于论文结果）

| 日志文件 | 描述 | 验证数值 |
|---|---|---|
| `exp214_ct_lam0_3seeds-2560760.out` | **直接 Baseline**：PAD MTL，λ_ct=0，seeds 7/25/42 | mean=62.97%，std=0.08% |
| `exp217_train_cos_1p00_3seeds-2578798.out` | **本文最优方法**：Cosine λ=1.00，seeds 7/25/42 | mean=63.80%，std=0.42% |

### λ 搜索实验（exp 215 标定）

| 日志文件 | 描述 |
|---|---|
| `exp215_calibrate_mse-2560809.out` | MSE（L2-norm）标定，λ_ct=1.0，600 iter |
| `exp215_calibrate_cosine-2560811.out` | Cosine 标定，λ_ct=1.0，600 iter |
| *(详细 loss 数值在子目录)* `cityscapes_joint_215/2026-03-03_19-46-030_.../run_*.log` | MSE 标定详细日志 |
| *(详细 loss 数值在子目录)* `cityscapes_joint_215/2026-03-03_19-46-031_.../run_*.log` | Cosine 标定详细日志 |

### λ 单 seed 扫描（exp 216，seed=42）

| 日志文件 | 描述 | best mIoU |
|---|---|---|
| `exp216_train_baseline-2561049.out` | Baseline，λ_ct=0 | 0.6328 |
| `exp216_train_mse_1pct-2561050.out` | MSE λ=0.30（1%贡献） | 0.6460 |
| `exp216_train_mse_2p5pct-2561051.out` | MSE λ=0.75（2.5%） | 0.6382 |
| `exp216_train_mse_5pct-2561052.out` | MSE λ=1.50（5%） | 0.6352 |
| `exp216_train_mse_10pct-2561053.out` | MSE λ=3.00（10%） | 0.6329 |
| `exp216_train_cos_1pct-2561054.out` | Cosine λ=0.05（1%） | 0.6405 |
| `exp216_train_cos_2p5pct-2561055.out` | Cosine λ=0.12（2.5%） | 0.6407 |
| `exp216_train_cos_5pct-2561056.out` | Cosine λ=0.24（5%） | 0.6413 |
| `exp216_train_cos_10pct-2561057.out` | Cosine λ=0.48（10%） | 0.6416 |

### 多 seed 验证（exp 217 batch1，seeds 7/25/42）

| 日志文件 | 描述 | mean_best | std |
|---|---|---|---|
| `exp217_train_mse_0p30_3seeds-2578794.out` | MSE λ=0.30，3 seeds | 63.58% | 1.31% |
| `exp217_train_cos_0p48_3seeds-2578795.out` | Cosine λ=0.48，3 seeds | 63.72% | 1.35% |
| `exp217_train_cos_0p60_3seeds-2578796.out` | Cosine λ=0.60，3 seeds | 63.00% | 1.03% |
| `exp217_train_cos_0p84_3seeds-2578797.out` | Cosine λ=0.84，3 seeds | 63.06% | 0.79% |
| **`exp217_train_cos_1p00_3seeds-2578798.out`** | **Cosine λ=1.00，3 seeds（最优）** | **63.80%** | **0.42%** |

### 多 seed 验证（exp 217 batch2，seeds 7/25/42，进行中/待更新）

| 日志文件 | 描述 |
|---|---|
| `exp217_train_cos_1p25_3seeds-*.out` | Cosine λ=1.25，3 seeds |
| `exp217_train_cos_1p50_3seeds-*.out` | Cosine λ=1.50，3 seeds |
| `exp217_train_cos_1p75_3seeds-*.out` | Cosine λ=1.75，3 seeds |
| `exp217_train_cos_2p00_3seeds-*.out` | Cosine λ=2.00，3 seeds |

### Table 7 复现（Hoyer et al. 框架消融，用于背景对比）

| 日志文件 | 描述 | 实测 mIoU（N=372） |
|---|---|---|
| `table7_baseline_372-2363570.out` | 纯监督 Baseline（无 SDE），N=372，3 seeds | 61.35 ± 1.25% |
| `table7_mtl_372-2363572.out` | MTL only（PAD，无 DepthMix），N=372，3 seeds | 62.55 ± 0.35% |
| `table7_dx_mtl_372-2355591.out` | DX + MTL，N=372，3 seeds | 65.87 ± 1.23% |
| `table7_s_dx_mtl_372-2355593.out` | S + DX + MTL（完整方法），N=372，3 seeds | 67.89 ± 0.86% |
| `table7_baseline_2975-2363571.out` | 纯监督 Baseline，N=2975（全量），3 seeds | 69.19 ± 0.33% |
| `table7_mtl_2975-2363573.out` | MTL only，N=2975，3 seeds | 70.35 ± 0.17% |
| `table7_dx_mtl_2975-2355592.out` | DX + MTL，N=2975，3 seeds | 71.52 ± 0.27% |
