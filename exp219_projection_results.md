# Exp 219 — No-detach Projection Ablation（初步结果）

当前状态：32 个配置中，**single projection** 的 16 个配置已完成 15 个（cosine 8 个 + mse 7 个），dual projection 尚在排队，下面先汇总已完成结果。

## 1. 单 projection + cosine（single_cosine）

| warmup | λ_ct | best mIoU |
|--------|------|-----------|
| w5k (=5000) | 0.25 | 0.6264 |
| w5k | 0.50 | 0.6169 |
| w5k | 0.75 | 0.6186 |
| w5k | 1.00 | 0.6140 |
| w0 | 0.25 | 0.6289 |
| w0 | 0.50 | 0.6142 |
| w0 | 0.75 | 0.6300 |
| w0 | 1.00 | **0.6377** |

简单结论：cosine + single proj 下，**无 warmup + λ=1.0** 目前最好（0.6377），略高于之前 exp218 的 best（约 0.634–0.635）。

## 2. 单 projection + MSE（single_mse）

| warmup | λ_ct | best mIoU |
|--------|------|-----------|
| w5k (=5000) | 0.25 | 0.6214 |
| w5k | 0.50 | 0.6241 |
| w5k | 0.75 | 0.6252 |
| w5k | 1.00 | **0.6331** |
| w0 | 0.25 | 0.6197 |
| w0 | 0.50 | 0.6168 |
| w0 | 0.75 | 0.6108 |
| w0 | 1.00 | （日志尚未生成，等待中） |

简单结论：MSE + single proj 下，**有 warmup + λ≈1.0** 表现最好（0.6331），无 warmup 明显偏低。

## 3. 当前阶段的小结

- 已完成的 15 个 run 中，**单 projection + cosine, warmup=0, λ=1.0** 暂时是整体最优（0.6377），符合你之前“no-warmup + no-detach + cosine 比较强”的趋势。  
- MSE 下仍然是 **warmup=5000 + λ 较大（1.0）** 更好，且差异比 cosine 更温和。  
- dual projection 的 16 个配置都还在排队（`AssocGrpCPUMinutesLimit`），等这批 job 出 log 后，可以在本文件补全第四部分（dual projection 结果）和一个总表（single vs dual）。

