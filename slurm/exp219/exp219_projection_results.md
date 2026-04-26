# Exp 219 — Projection Ablation Summary

Compact English summary for Exp 219 (single-seed S7 sweep + single_rev extension).

## Setup

- Fixed: no detach, seed=7, N=372, PAD, 40k iterations
- Variables:
  - projection mode: single / dual / single_rev
  - CT type: cosine / mse
  - warmup: w5k / w0 (single_rev uses w0)
  - lambda: 0.25, 0.5, 0.75, 1.0
- Total runs: 40 (32 original + 8 single_rev)

## Best Results by Group

| Group | Best Setting | Best mIoU |
|---|---|---:|
| single + cosine | w0, lambda=1.0 | 0.6377 |
| single + mse | w5k, lambda=1.0 | 0.6343 |
| dual + cosine | w0, lambda=0.5 | 0.6366 |
| dual + mse | w0, lambda=0.75 | 0.6307 |
| single_rev | mse, w0, lambda=1.0 | 0.6308 |

## Main Observations

1. Top overall setting in Exp 219: `single_cosine w0 lambda=1.0` (0.6377).
2. Dual cosine can be competitive (`w0 lambda=0.5`, 0.6366), but single remains slightly stronger.
3. MSE benefits from warmup in single mode (w5k helps stability).
4. single_rev is feasible but does not exceed the best single (depth->seg) result in this phase.

## Next-Step Recommendation

- Use narrow lambda sweeps for candidate filtering.
- Validate final candidates with 3 seeds (7/25/42) before reporting final claims.
- Keep mean ± std as the primary reporting format for paper tables.
