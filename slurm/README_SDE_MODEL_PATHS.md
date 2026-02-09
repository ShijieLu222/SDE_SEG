# SDE 训练模型保存位置说明

## 📁 模型保存路径

### Phase 1 (dec5) - Decoder Training
```
路径：/scratch/u5hv/shijie.u5hv/sde_seg/logs/monodepth/cityscapes-monodepth-101aspp-dec5-crop/

结构：
cityscapes-monodepth-101aspp-dec5-crop/
└── <timestamp>_r101_crop_512x512_batch4_.../
    ├── encoder.pth      # ResNet-101 权重（仍是 ImageNet，未改变）
    ├── depth.pth        # ✅ 训练的 Depth Decoder 权重
    ├── pose.pth         # Pose Network 权重
    ├── optimizer.pth    # 优化器状态（用于恢复训练）
    └── events.out.tfevents.*  # TensorBoard 日志
```

### Phase 2 (dec6) - Joint Fine-tuning
```
路径：/scratch/u5hv/shijie.u5hv/sde_seg/logs/monodepth/cityscapes-monodepth-101aspp-dec6-crop/

结构：
cityscapes-monodepth-101aspp-dec6-crop/
└── <timestamp>_lr1e-5_featdist1e-2_bs4_.../
    ├── encoder.pth      # ✅ 微调后的 ResNet-101 权重（用于 Transfer！）
    ├── depth.pth        # ✅ 微调后的 Depth Decoder 权重
    ├── pose.pth         # 微调后的 Pose Network 权重
    ├── optimizer.pth    # 优化器状态
    └── events.out.tfevents.*  # TensorBoard 日志
```

---

## 🔄 如何使用训练好的模型？

### 方法 1: 直接在代码中指定路径

在 `experiments.py` 中修改（第 157 行附近）：

```python
# 原来（使用作者的预训练模型）
mono_pretrain = f'mono_cityscapes_1024x512_r101dil_aspp_dec6_lr5_fd2_crop512x512bs4'

# 改成（使用你自己训练的模型）
mono_pretrain = '/scratch/u5hv/shijie.u5hv/sde_seg/logs/monodepth/cityscapes-monodepth-101aspp-dec6-crop/<your_timestamp>'
```

### 方法 2: 将模型注册到下载路径

将你的模型复制到标准位置：

```bash
# 1. 找到你的 dec6 模型目录
DEC6_DIR="/scratch/u5hv/shijie.u5hv/sde_seg/logs/monodepth/cityscapes-monodepth-101aspp-dec6-crop/<timestamp>"

# 2. 创建目标目录
DOWNLOAD_DIR="$HOME/.cache/sdeseg/my_sde_model"
mkdir -p $DOWNLOAD_DIR

# 3. 复制权重文件
cp $DEC6_DIR/encoder.pth $DOWNLOAD_DIR/
cp $DEC6_DIR/depth.pth $DOWNLOAD_DIR/
cp $DEC6_DIR/pose.pth $DOWNLOAD_DIR/

# 4. 在 experiments.py 中使用
mono_pretrain = 'my_sde_model'
```

---

## 📊 查看训练进度

### 方法 1: 实时查看日志
```bash
# 查看 SLURM 输出
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/train_sde_dec5-<jobid>.out

# 查看训练日志
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/monodepth/cityscapes-monodepth-101aspp-dec5-crop/<timestamp>/train.log
```

### 方法 2: 使用 TensorBoard
```bash
# 在 HPC 上启动 TensorBoard
tensorboard --logdir=/scratch/u5hv/shijie.u5hv/sde_seg/logs/monodepth/ --port=6006

# 然后在本地浏览器访问（需要 SSH 端口转发）
# ssh -L 6006:localhost:6006 user@hpc
```

---

## ⏰ 训练时间预估

| 阶段 | 迭代次数 | 预计时间 | GPU 使用率 |
|------|---------|---------|-----------|
| Phase 1 (dec5) | 300,000 | ~48-72h | ~90% |
| Phase 2 (dec6) | 100,000 | ~24-36h | ~90% |
| **总计** | 400,000 | **~3-5天** | - |

**注意：** 如果超过 SLURM 的 24h/72h 时间限制，训练会中断。

---

## 🔧 恢复中断的训练

如果训练因时间限制中断，可以恢复：

### 修改配置文件添加 resume 路径

在 `configs/cityscapes_monodepth_highres_dec5_crop.yml` 末尾：

```yaml
training:
  # ... 其他配置 ...
  resume: /scratch/.../cityscapes-monodepth-101aspp-dec5-crop/<timestamp>/
```

然后重新提交相同的 SLURM 脚本，训练会从中断处继续。

---

## 📈 验证模型质量

训练完成后，可以检查深度预测质量：

```bash
# 使用 val_interval 在训练中自动验证
# 查看 TensorBoard 中的深度可视化

# 或手动评估
python evaluate_depth.py \
    --model_path /scratch/.../dec6/<timestamp>/ \
    --eval_split val
```

---

## 🎯 Transfer Learning 使用流程

```
Step 1: 训练 SDE
  sbatch slurm/train_sde_dec5.slurm  ⏰ ~3天
  ↓
  sbatch slurm/train_sde_dec6.slurm  ⏰ ~2天
  ↓
Step 2: 修改 experiments.py
  mono_pretrain = '<your_dec6_path>'
  ↓
Step 3: 运行 Transfer Learning
  sbatch slurm/train_transfer_100.slurm
  sbatch slurm/train_transfer_372.slurm
  sbatch slurm/train_transfer_744.slurm
```

---

## ❓ 常见问题

### Q1: 为什么要训练两个阶段？
A: Phase 1 保持 ImageNet 特征的同时训练 Decoder；Phase 2 微调整个网络但用特征距离损失防止过度偏离。

### Q2: 可以只训练 Phase 1 吗？
A: 可以，但性能会稍差。论文实验表明 dec6 (Phase 2) 效果最好。

### Q3: 如果我想用作者的预训练模型？
A: 保持 `experiments.py` 不变，代码会自动下载作者的模型。这样更快！

### Q4: 训练需要什么数据？
A: 只需要 Cityscapes 的视频序列数据（leftImg8bit_sequence_small），不需要任何标注！

### Q5: 我的模型比作者的好吗？
A: 理论上应该接近。可以对比 Transfer Learning 的最终 mIoU 来判断。

