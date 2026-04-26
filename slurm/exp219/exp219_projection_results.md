# Exp 219 Projection Ablation（**已完成**）

**状态**：共 **40** 个 run **全部跑满**（原 **32**：single/dual × cosine/mse × warmup × λ；新增 **8**：single_rev × cosine/mse × λ）。每 run **33 次验证** / 40k iter；指标均为 log 内 **best val mIoU**。默认 **单 seed S7**（runs 32–39 亦为 S7）。  
**Baseline 对照**：论文用 **exp217**（旧 HPC）三 seed **mean = 0.6269**（见 §5）。

---

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

**结论**：cosine + single proj 下 **w0 + λ=1.0** 最优（**0.6377**）。

---

## 2. 单 projection + MSE（single_mse）

| warmup | λ_ct | best mIoU |
|--------|------|-----------|
| w5k (=5000) | 0.25 | 0.6258 |
| w5k | 0.50 | 0.6241 |
| w5k | 0.75 | 0.6254 |
| w5k | 1.00 | **0.6343** |
| w0 | 0.25 | 0.6197 |
| w0 | 0.50 | 0.6168 |
| w0 | 0.75 | 0.6146 |
| w0 | 1.00 | 0.6216 |

**结论**：MSE + single 下 **w5k + λ=1.0** 最好（**0.6343**）；无 warmup 明显弱于 w5k（本组 warmup 仍是必要选项）。

---

## 3. 双 projection + cosine（dual_cosine）

| warmup | λ_ct | best mIoU |
|--------|------|-----------|
| w5k (=5000) | 0.25 | 0.6280 |
| w5k | 0.50 | 0.6249 |
| w5k | 0.75 | 0.6267 |
| w5k | 1.00 | 0.6127 |
| w0 | 0.25 | 0.6176 |
| w0 | 0.50 | **0.6366** |
| w0 | 0.75 | 0.6245 |
| w0 | 1.00 | 0.6332 |

**结论**：dual cosine **峰值在 w0、λ=0.50**（**0.6366**）；**w5k + 大 λ** 明显变差（λ=1.0 → 0.6127）。

---

## 4. 双 projection + MSE（dual_mse）

| warmup | λ_ct | best mIoU |
|--------|------|-----------|
| w5k (=5000) | 0.25 | 0.6183 |
| w5k | 0.50 | 0.6267 |
| w5k | 0.75 | 0.6270 |
| w5k | 1.00 | 0.6279 |
| w0 | 0.25 | 0.6277 |
| w0 | 0.50 | 0.6256 |
| w0 | 0.75 | **0.6307** |
| w0 | 1.00 | 0.6248 |

**结论**：dual MSE **w0 峰值在 λ=0.75**（**0.6307**）；w5k 四格在 **0.618～0.628**，整体与 w0 峰值接近但略低。

---

## 5. PAD-MTL Baseline（372，3 seeds）— 论文用 **exp217（旧 HPC）**

| seed | best mIoU |
|------|-----------|
| 7 | 0.6135 |
| 25 | 0.6285 |
| 42 | 0.6385 |
| **mean** | **0.6269** |
| **std** | **0.0126** |

> BP1 复跑 baseline（如 Job `16421316` / `16430554`）仅作对照；**主文对比以 mean 0.6269 为准**。

---

## 6. 全面分析（exp219 单 seed S7）

### 6.1 全局排序（本消融内）

| 排名 | 配置 | best mIoU |
|------|------|-----------|
| 1 | single_cosine，w0，λ=1.0 | **0.6377** |
| 2 | dual_cosine，w0，λ=0.50 | **0.6366** |
| 3 | single_mse，w5k，λ=1.0 | 0.6331 |
| 4 | dual_mse，w0，λ=0.75 | 0.6307 |

- **Cosine**：single 与 dual 峰值相差仅 **0.0011**，**no-detach + 单投影** 略占优；**dual cosine** 在 **w0、λ≈0.5** 极强。  
- **MSE**：**single + w5k** 仍优于 **dual MSE** 任一格；dual MSE 更依赖 **w0 + 中等 λ（~0.75）**。

### 6.2 与 exp217 baseline（mean 0.6269）

- **single_cosine 0.6377** 高于 **mean +0.0108**，幅度大于 baseline **std（0.0126）** 的约 **86%**，单 seed 上**有正向信号**，但**不能替代**多 seed 结论。  
- 与 **baseline 单 seed best 0.6385**（seed 42）相比，**0.6377** 低 **0.0008**，属**同一水平线**。  
- **含义**：在 **S7** 上，no-detach projection + 最优 λ 可与「三 seed baseline 里偏强的那档」对齐；**论文叙述**应强调 **相对 mean 的提升**，并计划 **同设定 3 seeds** 验证稳定性。

### 6.3 结构归纳（w0 vs w5k，single vs dual）

| 现象 | 解释（简述） |
|------|----------------|
| cosine：**w0 往往优于 w5k**（single/dual 皆然） | 与 exp218「detach」阶段经验一致：长 warmup 与 cosine 组合易压制有效区间；**w0 更利大 λ**。 |
| MSE：**w5k 在 single 上很关键** | MSE 尺度与梯度更依赖前期稳定，**warmup 后开大 λ** 更稳。 |
| **dual 仅在 cosine+w0+中 λ** 冲到第二 | 双投影参数量与优化更难；**MSE+dual** 未超过 single+MSE 峰值。 |
| **dual_cosine w5k + λ=1.0** 崩盘（0.6127） | **大 λ + 长 warmup + 双投影** 三者叠加最伤；后续 λ 细扫应**避开**该角。 |

---

## 7. 后续实验怎么安排（λ 细扫、1 seed vs 3 seeds、DepthMix、selection）

### 7.1 λ 范围细扫：**先 1 seed，再 3 seeds**

| 阶段 | 目的 | 建议 |
|------|------|------|
| **A. 细扫（1 seed，如 S7）** | 在 **0.5～1.0**（cosine,w0）与 **0.6～0.9**（MSE,w0）等**窄区间**加密步长（如步长 0.05 或 0.1） | **成本低、用于定「候选 λ」**，避免一上来 3× 算力。 |
| **B. 验证（3 seeds）** | 对 **2～3 个候选 λ**（每类 loss 各 1～2 个）跑 **7/25/42** | **论文主表**用 **mean±std**；与 exp217 baseline 协议对齐。 |

**结论**：**细扫阶段用 1 seed 合理**；**写进论文的最终数字用 3 seeds**（至少对你主推的 1～2 个配置）。

### 7.2 何时上 **3 seeds**

- **必须**：学位论文/一作论文的 **主结果表**（相对 baseline 的 Δ）。  
- **可选先不做 3 seeds**：仅当你还在大幅改结构（projection / detach）时；**exp219 已定型 no-detach**，下一步 **应进入候选 λ 的 3-seed 确认**。

### 7.3 **DepthMix** 与 **automatic selection** 何时加

- **顺序建议**（与 Hoyer 原文 Table 5/7 类组件一致）：  
  1. **固定**：PAD + L_ct + 最优 projection 模式（single vs dual 已由 exp219 缩小范围）+ **λ 由细扫+3seed 敲定**。  
  2. **再叠加 DepthMix**（未标注混合/深度相关增强）：在 **同一 λ 候选** 上做 **开/关** 或 **Table7 式** 组件消融，否则难以解释增益来源。  
  3. **最后**再上 **automatic / learned selection**（若指数据选择或策略选择）：通常 **依赖前面 pipeline 已稳定**，否则变量太多。

- **算力**：DepthMix + selection 往往要 **更长训练或更多 run**，建议 **仅在 3-seed 锁定 1～2 个最终 λ 后**再开，避免组合爆炸。

### 7.4 推荐路线图（精简）

```
exp219（32 格，1 seed）✓
    → λ 细扫（1 seed，窄区间）→ 定 2～3 个 λ*
    → 对 λ* 跑 3 seeds + 与 exp217 baseline 同表对比
    → 固定最优 (loss, w0/w5k, proj, λ*)，做 DepthMix 消融（建议仍 3 seeds 或至少 2 seeds）
    → 再叠 automatic selection（若课题需要）
```

---

## 8. 小结

- **exp219 已全部完成**；**全局最优（single/dual 原 32 run）**：**single_cosine w0 λ=1.0 → 0.6377**；**dual 最优**：**dual_cosine w0 λ=0.5 → 0.6366**。  
- **single_rev（runs 32–39）**：本组最优为 **mse w0 λ=1.0 → 0.6308**，未超过上述 **single depth→seg** 峰值（见 **§10**）。  
- 相对 **baseline mean 0.6269**，最优 cosine 配置在 **S7 上约 +0.011**；与 **baseline 单 seed best 0.6385** 几乎持平。  
- **下一步**：**λ 细扫用 1 seed** 收窄区间 → **候选 λ 用 3 seeds 定稿** → 再考虑 **DepthMix** 与 **selection**，且 **后两者应在 λ 与主结构稳定之后**再做。

---

## 9. Exp 220（3-seeds 验证，已就绪）

已根据上述候选创建 **exp 220**：**13 个配置 × 3 seeds = 39 runs**，每脚本 **36h**。

| 组 | 配置 | λ | 脚本数 |
|----|------|---|--------|
| single_cosine w0 | 验证 λ=1.0 是否最优 | 0.75, 1.0, 1.25 | 3 |
| single_mse | 对照 | w5k λ=1.0、w0 λ=1.0 | 2 |
| dual_cosine w0 | λ 细扫 | 0.5, 0.75, 1.0, 1.25 | 4 |
| dual_cosine w5k | λ 细扫 | 0.5, 0.75, 1.0, 1.25 | 4 |

- **experiments.py**：`id == 220`
- **脚本**：`slurm/exp220/train_*.slurm`，共 **19** 个（含 **6** 个 **single_rev**，runs 39–56）
- **README / 汇总表**：`slurm/exp220/README.md`、`slurm/exp220/exp220_results.md`

---

## 10. Single-rev（seg→depth），runs 32–39

**含义**：**segmentation 蒸馏特征经投影头映射到 depth 侧**，再与 depth 蒸馏特征算 \( \mathcal{L}_\text{ct} \)；与原来 **single（depth→seg）** 方向相反。全部为 **warmup=0**，**seed=7**，**λ_ct ∈ {0.25, 0.5, 0.75, 1.0}**。

| Run ID | ct_type | λ_ct | best val mIoU |
|--------|---------|------|---------------|
| 32 | cosine | 0.25 | **0.6270** |
| 33 | cosine | 0.50 | 0.6255 |
| 34 | cosine | 0.75 | 0.6126 |
| 35 | cosine | 1.00 | 0.6130 |
| 36 | mse | 0.25 | 0.6208 |
| 37 | mse | 0.50 | 0.6264 |
| 38 | mse | 0.75 | 0.6183 |
| 39 | mse | 1.00 | **0.6308** |

**小结（Phase I，S7）**：single_rev 本组 **mse λ=1.0** 与 **cosine λ=0.25** 相对较高；仍未超过 §1 中 **single cosine w0 λ=1.0（0.6377）**。三 seed 验证见 **`slurm/exp220/exp220_results.md` §10**。
