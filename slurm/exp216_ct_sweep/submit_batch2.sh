#!/bin/bash
# exp 216 batch 2 — 6 new runs
#
# Cosine high-λ extension (contribution-aligned):
#   λ=0.60  12.5%  (Run  9)
#   λ=0.72  15%    (Run 10)
#   λ=0.96  20%    (Run 11)
#
# MSE fine-grained around λ=0.30:
#   λ=0.25  ~0.83% (Run 12)
#   λ=0.35  ~1.16% (Run 13)
#   λ=0.40  ~1.33% (Run 14)
#
# Usage: bash submit_batch2.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "Submitting exp 216 batch 2 (6 jobs)"
echo "================================================"

jids=()
submit() {
    local script="$1"
    local jid
    jid=$(sbatch --parsable "${SCRIPT_DIR}/${script}")
    echo "  Submitted ${script}  ->  Job ${jid}"
    jids+=("${jid}")
}

# Cosine high-λ
submit train_cos_12p5pct.slurm
submit train_cos_15pct.slurm
submit train_cos_20pct.slurm

# MSE fine-grained
submit train_mse_0p25.slurm
submit train_mse_0p35.slurm
submit train_mse_0p40.slurm

echo "================================================"
echo "All ${#jids[@]} jobs submitted: ${jids[*]}"
echo "Monitor: squeue -u $USER"
echo "================================================"
