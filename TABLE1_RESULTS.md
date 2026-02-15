# Table 1 – Comparison of data selection methods

*DS: diversity sampling based on depth features*  
*US: uncertainty sampling based on depth student error*  
*mIoU in %, std. dev. over 3 seeds*

|  | 1/30 (100) | 1/8 (372) | 1/4 (744) |
|--|------------|-----------|-----------|
| Random | 47.57 ± 0.96 | 58.63 ± 0.67 | 63.22 ± 0.30 |
| Entropy | **53.76 ± 0.44** | **63.61 ± 0.78** | 65.57 ± 0.62 |
| Ours (US) | 50.83 ± 0.91 | 63.07 ± 0.11 | **66.84 ± 0.34** |
| Ours (DS) | 52.67 ± 0.14 | 62.86 ± 0.46 | 66.74 ± 0.15 |
| Ours (DS+US) | 53.63 ± 0.19 | 63.38 ± 0.92 | 66.35 ± 0.62 |