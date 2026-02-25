# Table 7 – Framework Component Ablation

*(S: Data Selection, DX: DepthMix, MTL: Multi-Task Learning. mIoU in %, std. dev. over 3 seeds)*

| 实验 | S | DX | MTL | 372 Labels (1/8) | 2975 Labels (Full) |
|------|---|---|-----|-------------------|---------------------|
| Baseline | — | — | — | 60.99 ± 1.23 | 68.98 ± 0.17 |
| MTL only | — | — | ✓ | 63.06 ± 0.80 (+2.07) | 69.85 ± 0.31 (+0.86) |
| DX only | — | ✓ | — | — | — |
| S only | ✓ | — | — | — | — |
| S + MTL | ✓ | — | ✓ | — | — |
| S + DX | ✓ | ✓ | — | — | — |
| DX + MTL | — | ✓ | ✓ | 65.11 ± 0.51 (+4.12) | 71.28 ± 0.30 (+2.29) |
| S + DX + MTL | ✓ | ✓ | ✓ | **67.66 ± 0.47 (+6.67)** | — |
