# 🚀 快速启动指南：Baseline 实验

## ✅ 已完成设置

### 1. 修改了 `experiments.py`
启用了3个标签数量配置：100, 372, 744

### 2. 创建了3个 SLURM 脚本
- ✅ `train_baseline_100.slurm` - 100 labels (1/30)
- ✅ `train_baseline_372.slurm` - 372 labels (1/8)
- ✅ `train_baseline_744.slurm` - 744 labels (1/4)

### 3. ✅ 验证通过
运行 `--dry` 测试确认配置正确

---

## 🎯 现在就运行！

### 选项1：并行运行所有实验（推荐）⚡
```bash
cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth

# 一次性提交所有3个任务
sbatch slurm/train_baseline_100.slurm
sbatch slurm/train_baseline_372.slurm
sbatch slurm/train_baseline_744.slurm
```

### 选项2：一个一个运行
```bash
# 只运行 372 标签实验
sbatch slurm/train_baseline_372.slurm

# 或只运行 744 标签实验
sbatch slurm/train_baseline_744.slurm
```

---

## 📊 监控任务

### 查看任务队列
```bash
squeue -u $USER
```

### 查看实时日志
```bash
# 查看最新的日志文件
ls -lt /scratch/u5hv/shijie.u5hv/sde_seg/logs/train_baseline_*

# 实时查看输出
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/train_baseline_100-<JOBID>.out
```

### 取消任务
```bash
scancel <JOBID>
```

---

## 📈 预期时间线

| 实验 | 标签数 | 迭代次数 | 预计时间 | 预期 mIoU |
|------|--------|---------|---------|-----------|
| baseline_100 | 100 | 40k | ~12-16h | ~48.75% |
| baseline_372 | 372 | 40k | ~18-24h | ~59.14% |
| baseline_744 | 744 | 40k | ~24-36h | ~63.46% |

---

## 🗂️ 结果位置

### TensorBoard 日志
```
/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_210/
    ├── 2026-01-31_xxx_scratch_D100random_S42_.../
    ├── 2026-01-31_xxx_scratch_D372random_S42_.../
    └── 2026-01-31_xxx_scratch_D744random_S42_.../
```

### 查看 TensorBoard
```bash
# 在本地机器上通过 SSH 端口转发
ssh -L 6006:localhost:6006 u5hv@bluepebble.acrc.bris.ac.uk
tensorboard --logdir=/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_210/
# 然后在浏览器打开 http://localhost:6006
```

---

## ✨ 重要提示

1. **你之前运行的 100 标签实验已完成** ✓
   - 可以选择重新运行或跳过
   - 如果重新运行，它会生成新的时间戳目录

2. **建议并行运行**
   - 3个任务可以同时运行在不同GPU上
   - 总时间约36小时（而不是串行的 ~80小时）

3. **磁盘空间检查**
   ```bash
   df -h /scratch/u5hv/shijie.u5hv/sde_seg/
   ```

4. **如果任务失败**
   - 检查 `.err` 文件
   - 确认 GPU 可用：`nvidia-smi`
   - 确认数据路径正确

---

## 🎓 用于 Dissertation

这些实验将为你的论文提供：

### Table: Baseline Results
| Method | 1/30 (100) | 1/8 (372) | 1/4 (744) |
|--------|-----------|----------|----------|
| Random | XX.XX% | XX.XX% | XX.XX% |

### 可视化
- 训练曲线（loss vs iterations）
- mIoU 变化曲线
- 不同标签数量的分割结果对比

---

## 下一步

完成这些 baseline 实验后：
1. ✅ 分析结果
2. ⏭️ 运行 SDE 预训练
3. ⏭️ 运行 transfer learning 实验
4. ⏭️ 运行 DepthMix 实验

详见 `README_baseline_experiments.md` 获取更多信息。

