# Exp 222 — Single-Rev 结果（seg -> depth, λ=1.0, w0）

> 更新时间：2026-04-22  
> 配置：MTL + Cross-Task Projection Loss（**single_rev**），N=372，Seeds 7/25/42，每 seed 40k iter。  
> 说明：single_rev 指 **segmentation 特征投影到 depth 空间** 再做一致性约束。

---

## Run 对应（single_rev）

| CT Loss | 配置 | Run IDs |
|---|---|---|
| MSE | MTL+DX | 18,19,20 |
| MSE | MTL+Sel | 21,22,23 |
| MSE | MTL+DX+Sel | 24,25,26 |
| Cosine | MTL+DX | 27,28,29 |
| Cosine | MTL+Sel | 30,31,32 |
| Cosine | MTL+DX+Sel | 33,34,35 |

---

## 最终结果（3 seeds）

| CT Loss | 配置 | S7 | S25 | S42 | 均值 | ±std | exp221 基线 | Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MSE | MTL+DX | 65.23% | 66.02% | 66.26% | **65.84%** | ±0.44% | 66.22% | ↓0.38% |
| MSE | MTL+Sel | 67.02% | 66.21% | 65.90% | **66.38%** | ±0.47% | 66.58% | ↓0.20% |
| MSE | MTL+DX+Sel | 68.27% | 66.99% | 67.81% | **67.69%** | ±0.53% | 67.74% | ↓0.05% |
| Cosine | MTL+DX | 64.47% | 65.55% | 67.79% | **65.93%** | ±1.38% | 66.22% | ↓0.29% |
| Cosine | MTL+Sel | 66.65% | 66.71% | 66.45% | **66.60%** | ±0.11% | 66.58% | ↑0.02% |
| Cosine | MTL+DX+Sel | 68.91% | 67.23% | 68.29% | **68.14%** | ±0.69% | 67.74% | **↑0.40%** |

---

## 关键结论（single_rev）

1. **single_rev 最优配置**是 **Cosine + MTL+DX+Sel**，均值 **68.14%**（较 exp221 基线 +0.40%）。
2. MSE 在 single_rev 下整体未带来稳定增益（3 组里均值均不超过对应基线）。
3. single_rev 的有效增益主要出现在 **DX+Sel 完整配置**，与纯 DX 或纯 Sel 的趋势不同。

---

## 与旧 exp222（single: depth -> seg）对比

旧 exp222（single）已记录最优为：
- **MSE + MTL+DX+Sel = 68.18%**

本次 exp222（single_rev）最优为：
- **Cosine + MTL+DX+Sel = 68.14%**

=> 两者非常接近（差 **0.04%**），在当前 3-seed 结果下可视为同一量级；
single_rev 证明了“反向单投影（seg->depth）”是可行方案，但未显著超过旧方向的最佳结果。

---

## 日志文件（single_rev）

- `/user/work/ig23200/sde_seg/logs/exp222_srev_mse_dx-16922813.out`
- `/user/work/ig23200/sde_seg/logs/exp222_srev_mse_sel-16922815.out`
- `/user/work/ig23200/sde_seg/logs/exp222_srev_mse_dx_sel-16922814.out`
- `/user/work/ig23200/sde_seg/logs/exp222_srev_cosine_dx-16922810.out`
- `/user/work/ig23200/sde_seg/logs/exp222_srev_cosine_sel-16922812.out`
- `/user/work/ig23200/sde_seg/logs/exp222_srev_cosine_dx_sel-16922811.out`
