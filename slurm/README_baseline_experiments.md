# Baseline 实验说明

## 概述
本目录包含用于运行不同标签数量的 baseline 语义分割实验的 SLURM 脚本。

## 实验配置

| 脚本文件 | 标签数量 | 比例 | 预计时间 | --run 参数 |
|---------|---------|------|---------|-----------|
| `train_baseline_100.slurm` | 100 | 1/30 | 24h | 0 |
| `train_baseline_372.slurm` | 372 | 1/8 | 36h | 1 |
| `train_baseline_744.slurm` | 744 | 1/4 | 48h | 2 |

## 提交任务

### 方法1：并行运行（推荐，节省时间）
```bash
# 提交所有任务（它们会并行运行）
cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth
sbatch slurm/train_baseline_100.slurm
sbatch slurm/train_baseline_372.slurm
sbatch slurm/train_baseline_744.slurm
```

## 检查任务状态
```bash
# 查看所有任务
squeue -u $USER

# 查看特定任务的输出
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/train_baseline_100-<jobid>.out
```

## 日志位置

### SLURM 日志（标准输出/错误）
```
/scratch/u5hv/shijie.u5hv/sde_seg/logs/train_baseline_<labels>-<jobid>.out
/scratch/u5hv/shijie.u5hv/sde_seg/logs/train_baseline_<labels>-<jobid>.err
```

### 训练日志（TensorBoard + checkpoints）
```
/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_210/
    ├── <timestamp>_scratch_D100random_S42_.../
    ├── <timestamp>_scratch_D372random_S42_.../
    └── <timestamp>_scratch_D744random_S42_.../
```

## 预期结果（根据论文）

| 标签数量 | Random Baseline mIoU |
|---------|---------------------|
| 100 (1/30) | ~48.75% |
| 372 (1/8) | ~59.14% |
| 744 (1/4) | ~63.46% |

## 训练参数

所有实验使用相同的配置：
- **模型**: ResNet-101 (ImageNet 预训练)
- **优化器**: SGD (lr=1e-2, momentum=0.9, weight_decay=5e-4)
- **学习率**: Backbone=1e-3, Head=1e-2, Step decay @ 30k iters
- **训练迭代**: 40,000
- **批次大小**: 2
- **图像裁剪**: 512×512
- **数据增强**: 随机水平翻转

## 修改的文件

1. **experiments.py**: 启用了 372 和 744 标签数量
   ```python
   def subsets(dataset):
       if dataset == "cityscapes":
           return [100, 372, 744]  # 启用所有三个
   ```

2. **--run 参数**: 每个脚本使用不同的索引
   - 0 → 100 labels
   - 1 → 372 labels
   - 2 → 744 labels

## 故障排除

### 任务失败
检查错误日志：
```bash
cat /scratch/u5hv/shijie.u5hv/sde_seg/logs/train_baseline_<labels>-<jobid>.err
```

## 后续步骤

完成 baseline 实验后，可以运行：
1. SDE 预训练（`train_sde_dec5.slurm`, `train_sde_dec6.slurm`）
2. Transfer learning 实验（使用 SDE 预训练权重）
3. DepthMix 数据增强实验

