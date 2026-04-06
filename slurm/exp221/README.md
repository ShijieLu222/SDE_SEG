# Exp 221 — MTL + DepthMix / Selection 消融（3 seeds）

- **固定**：PAD-MTL，N=372，40k iter，seeds 7/25/42，40h/任务
- **变量**：DepthMix（DX）和 Label Selection（S）的开关组合

## Run ID 对应表

| Run IDs | 脚本 | 配置 |
|---------|------|------|
| 0, 1, 2 | `train_mtl_dx_3seeds.slurm` | MTL + DepthMix（S=7/25/42） |
| 3, 4, 5 | `train_mtl_sel_3seeds.slurm` | MTL + Selection（S=7/25/42） |
| 6, 7, 8 | `train_mtl_dx_sel_3seeds.slurm` | MTL + DepthMix + Selection（S=7/25/42） |

共 9 runs，每个脚本跑 3 个 seed（顺序：seed 7 → 25 → 42）。

## 提交指令

```bash
sbatch slurm/exp221/train_mtl_dx_3seeds.slurm
sbatch slurm/exp221/train_mtl_sel_3seeds.slurm
sbatch slurm/exp221/train_mtl_dx_sel_3seeds.slurm
```

## 日志位置

```
/user/work/ig23200/sde_seg/logs/exp221_mtl_dx-<jobid>.out
/user/work/ig23200/sde_seg/logs/exp221_mtl_sel-<jobid>.out
/user/work/ig23200/sde_seg/logs/exp221_mtl_dx_sel-<jobid>.out
```
