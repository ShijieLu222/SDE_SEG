# Table 5 – Comparison of SDE feature transfer methods

*(F: ImageNet feature distance loss. mIoU in %, std. dev. over 3 seeds)*

| Aux. SDE | F | 372 Labels (1/8) | 2975 Labels (Full) |
|----------|---|-------------------|---------------------|
| Baseline | — | 58.20 ± 0.42 | 67.33 ± 0.16 |
| Transfer (no F) | — | 59.90 ± 0.82 (+1.70) | 68.66 ± 0.53 (+1.32) |
| Transfer (F=✓) | ✓ | 56.68 ± 0.59 (-1.52) | 61.43 ± 0.61 (-5.90) |
| Multi-Task (F=✓) | ✓ | 56.89 ± 0.64 (-1.31) | 61.39 ± 0.41 (-5.94) |
