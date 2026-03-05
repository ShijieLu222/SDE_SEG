#!/bin/bash
# exp 216: Cross-Task Loss Sweep — MSE(norm) vs Cosine, contribution-aligned λ, seed=42
#
# λ from exp 215 calibration (contribution-alignment method):
#   contrib%  |  MSE λ   | Cosine λ
#   ----------|----------|----------
#   baseline  |  0.00    |  —
#   1%        |  0.30    |  0.05
#   2.5%      |  0.75    |  0.12
#   5%        |  1.50    |  0.24
#   10%       |  3.00    |  0.48
#
# 9 jobs submitted in parallel, one GPU each, max 24h.
# Usage: bash submit_all.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "Submitting exp 216 Cross-Task Sweep (9 jobs)"
echo "================================================"

jids=()
submit() {
    local script="$1"
    local jid
    jid=$(sbatch --parsable "${SCRIPT_DIR}/${script}")
    echo "  Submitted ${script}  ->  Job ${jid}"
    jids+=("${jid}")
}

# Baseline (λ=0, ct_type=mse — type irrelevant when λ=0)
submit train_baseline.slurm

# MSE normalised sweep
submit train_mse_1pct.slurm
submit train_mse_2p5pct.slurm
submit train_mse_5pct.slurm
submit train_mse_10pct.slurm

# Cosine sweep
submit train_cos_1pct.slurm
submit train_cos_2p5pct.slurm
submit train_cos_5pct.slurm
submit train_cos_10pct.slurm

echo "================================================"
echo "All ${#jids[@]} jobs submitted: ${jids[*]}"
echo "Monitor: squeue -u $USER"
echo "================================================"
