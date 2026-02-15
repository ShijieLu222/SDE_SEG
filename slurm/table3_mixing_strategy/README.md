# Table 3: Data Mixing Strategy Experiments

复刻论文 Table 3（数据混合策略对比），3 个种子 (7, 25, 42)。

## Run ID 映射

| 实验 | Run IDs | 脚本 |
|------|---------|------|
| Baseline 372 | 0, 7, 14 | train_baseline_372.slurm |
| Baseline 2975 | 1, 8, 15 | train_baseline_2975.slurm |
| Pseudo-Labels 372 | 2, 9, 16 | train_pseudo_labels_372.slurm |
| ClassMix 372 | 3, 10, 17 | train_classmix_372.slurm |
| DepthMix 372 | 4, 11, 18 | train_depthmix_372.slurm |
| DepthMix 2975 | 5, 12, 19 | train_depthmix_2975.slurm |
| ClassMix-GT 2975 | 6, 13, 20 | train_classmix_2975.slurm |

## 使用方法

```bash
# 提交全部
./submit_all.sh

# 单个
sbatch train_pseudo_labels_372.slurm
```
