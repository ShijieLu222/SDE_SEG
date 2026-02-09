# SDE (Self-Supervised Depth Estimation) 训练指南

## 📖 什么是 SDE？

SDE = Self-Supervised Depth Estimation（自监督单目深度估计）

**核心思想：** 利用**未标注的视频序列**，通过几何约束（光度一致性）学习深度预测，无需人工标注的深度标签。

## 🎯 为什么需要 SDE？

1. **学习场景几何结构** - 理解物体边界和空间关系
2. **提供有用的特征表示** - 深度预测任务学到的特征对分割任务有帮助
3. **利用无标注数据** - Cityscapes 有大量视频序列数据（无需分割标签）
4. **作为预训练** - SDE 权重可以用来初始化分割网络（Transfer Learning）

---

## 🔄 SDE 训练两阶段

### Phase 1: Decoder Training (dec5)
- **脚本**: `train_sde_dec5.slurm`
- **迭代次数**: 300,000
- **Encoder**: **冻结** ResNet-101（保持 ImageNet 预训练）
- **训练内容**: 只训练 Depth Decoder
- **预计时间**: ~48-72 小时（可能超过 24h 限制）
- **配置文件**: `configs/cityscapes_monodepth_highres_dec5_crop.yml`

**关键参数：**
```yaml
freeze_backbone: True          # 冻结 encoder
train_iters: 300000           # 300k iterations
batch_size: 4
monodepth_lambda: 1.0         # 深度损失权重
segmentation_lambda: 0.0      # 不训练分割
```

### Phase 2: Joint Fine-tuning with Feature Distance (dec6)
- **脚本**: `train_sde_dec6.slurm`
- **迭代次数**: 100,000
- **Encoder**: **解冻** ResNet-101
- **训练内容**: 微调整个网络
- **特殊损失**: ImageNet 特征距离损失（防止过度偏离）
- **预计时间**: ~24-36 小时
- **配置文件**: `configs/cityscapes_monodepth_highres_dec6_crop.yml`
- **依赖**: 需要 Phase 1 的 checkpoint

**关键参数：**
```yaml
freeze_backbone: False         # 解冻 encoder
train_iters: 100000           # 100k iterations
feat_dist_lambda: 1.0e-2      # 特征距离损失
depth_pretraining: mono_cityscapes_...dec5  # 加载 Phase 1 权重
enable_imnet_encoder: True    # 启用 ImageNet encoder 用于特征距离
```

---

## 🚀 运行 SDE 训练

### 方法1：顺序运行（推荐）

```bash
cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth

# Step 1: 提交 Phase 1
JOB1=$(sbatch slurm/train_sde_dec5.slurm | awk '{print $4}')
echo "Phase 1 Job ID: $JOB1"

# Step 2: 等 Phase 1 完成后，提交 Phase 2
# 可以用 --dependency 参数自动依赖
sbatch --dependency=afterok:$JOB1 slurm/train_sde_dec6.slurm
```

### 方法2：手动分步运行

```bash
# 先运行 Phase 1
sbatch slurm/train_sde_dec5.slurm

# 监控任务
squeue -u $USER

# 等 Phase 1 完成后（检查日志确认）
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/train_sde_dec5-*.out

# 再运行 Phase 2
sbatch slurm/train_sde_dec6.slurm
```

---

## ⚠️ 重要提醒：时间限制问题

### 问题
Phase 1 需要 **300k iterations**，可能需要 **48-96 小时**，但 BluePebble GPU 队列限制 **24 小时**。

### 解决方案

#### 选项 A: 使用 Checkpoint 恢复（推荐）

如果 Phase 1 超时，代码会自动保存 checkpoint。你可以修改配置文件添加 resume：

```yaml
# 编辑 configs/cityscapes_monodepth_highres_dec5_crop.yml
training:
  resume: /path/to/checkpoint.pth  # 添加这一行
```

然后重新提交任务，它会从 checkpoint 继续。

#### 选项 B: 分段训练

创建多个配置文件，每个训练 100k iterations：
- dec5_part1.yml: 0-100k
- dec5_part2.yml: 100k-200k (加 resume)
- dec5_part3.yml: 200k-300k (加 resume)

#### 选项 C: 使用预训练模型（最简单）

**跳过 SDE 训练**，直接使用论文作者提供的预训练模型：

```python
# 代码会自动从 Google Drive 下载
# 见 models/utils.py:108
mono_pretrain = 'mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4'
```

**这是最节省时间的选择！** 对于本科毕设，使用预训练模型是合理的。

---

## 📊 监控训练进度

### 查看任务状态
```bash
squeue -u $USER
```

### 查看实时日志
```bash
# SLURM 日志
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/train_sde_dec5-<JOBID>.out

# 训练进度（每 100 iterations 打印一次）
grep "Train Iter" /scratch/u5hv/shijie.u5hv/sde_seg/logs/train_sde_dec5-*.out | tail -20
```

### TensorBoard
```bash
# SSH 端口转发
ssh -L 6006:localhost:6006 u5hv@bluepebble.acrc.bris.ac.uk

# 在服务器上运行
source ~/fyp/venvs/sde_seg/bin/activate
tensorboard --logdir=/scratch/u5hv/shijie.u5hv/sde_seg/logs/monodepth/

# 浏览器打开 http://localhost:6006
```

---

## 📁 输出位置

### SLURM 日志
```
/scratch/u5hv/shijie.u5hv/sde_seg/logs/
    ├── train_sde_dec5-<jobid>.out
    └── train_sde_dec6-<jobid>.out
```

### 训练日志和模型
```
/scratch/u5hv/shijie.u5hv/sde_seg/logs/monodepth/
    ├── cityscapes-monodepth-101aspp-dec5-crop/
    │   └── <timestamp>/
    │       ├── events.out.tfevents.*  (TensorBoard)
    │       ├── model_best.pth         (最佳模型)
    │       └── checkpoint_*.pth       (定期保存)
    └── cityscapes-monodepth-101aspp-dec6-crop/
        └── <timestamp>/
            ├── events.out.tfevents.*
            └── model_best.pth
```

---

## 📈 预期结果

SDE 训练完成后，你应该看到：

1. **深度预测可视化** - TensorBoard 中的深度图
2. **损失下降曲线** - 光度损失、平滑损失
3. **模型权重** - `model_best.pth` 文件

这些权重将用于后续的 **Transfer Learning** 实验。

---

## 🎓 用于 Dissertation

### Background 章节可以写：

**SDE 原理：**
```latex
\subsubsection{Photometric Consistency}
Given consecutive frames $I_{t-1}, I_t, I_{t+1}$ from a video sequence,
the depth network predicts per-pixel depth $D_t$ for frame $I_t$.
Using the predicted camera pose $T_{t \to t\pm1}$, we can warp adjacent 
frames to the current frame:

\hat{I}_t = \text{Warp}(I_{t\pm1}, D_t, T_{t \to t\pm1}, K)

where $K$ is the camera intrinsic matrix. The photometric loss is:

L_{photo} = \min_{t' \in \{t-1, t+1\}} \text{pe}(I_t, \hat{I}_t)
```

**特征距离损失（Phase 2）：**
```latex
To prevent the depth pretraining from deviating too far from ImageNet
features, we introduce a feature distance loss:

L_{feat} = ||f_{depth}(I) - f_{imnet}(I)||_2

where $f_{depth}$ and $f_{imnet}$ are features from the depth-pretrained
and ImageNet-pretrained encoders respectively.
```

### Project Execution 章节可以写：

1. **SDE 训练配置和超参数**
2. **两阶段训练策略的动机**
3. **训练时间和资源使用**
4. **遇到的困难（如 24h 时间限制）和解决方案**

---

## ⏭️ 下一步

完成 SDE 训练后：

1. ✅ **验证 SDE 模型** - 查看深度预测质量
2. 📊 **运行 Transfer Learning** - 使用 SDE 权重初始化分割网络
3. 📈 **对比 Baseline** - 量化 SDE 预训练的收益
4. 🎨 **DepthMix 实验** - 基于深度的数据增强

---

## 💡 建议

**对于本科毕设，我强烈建议：**

### 选项 A: 使用预训练模型（省时）⭐
- ✅ 节省 3-5 天训练时间
- ✅ 使用论文作者的官方模型
- ✅ 结果更可靠
- ✅ 专注于分析和论文写作

### 选项 B: 自己训练（学习价值）
- ✅ 完整理解整个 pipeline
- ✅ 可以调整参数做实验
- ⚠️ 需要处理时间限制问题
- ⚠️ 需要 3-5 天等待

**推荐：选项 A**，然后在 dissertation 中说明使用了预训练模型，重点放在 Transfer Learning 和结果分析上。

---

## 🆘 故障排除

### 任务超时
```bash
# 检查最后保存的 checkpoint
ls -lt /scratch/.../monodepth/cityscapes-monodepth-101aspp-dec5-crop/*/checkpoint_*.pth

# 修改配置添加 resume，重新提交
```

### GPU 内存不足
```bash
# 减小 batch_size（编辑配置文件）
batch_size: 2  # 从 4 改为 2
```

### 找不到预训练模型
```bash
# 检查 models/utils.py 的下载逻辑
# 确保网络可以访问 Google Drive
```

---

需要更多帮助或有任何问题，请随时询问！


