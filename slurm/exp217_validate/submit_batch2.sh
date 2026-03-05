#!/bin/bash
# exp 217 batch 2 — cosine λ 高值延伸（4 jobs × 3 seeds）
#
# Run 15-17: cosine λ=1.25, seeds 7/25/42
# Run 18-20: cosine λ=1.50, seeds 7/25/42
# Run 21-23: cosine λ=1.75, seeds 7/25/42
# Run 24-26: cosine λ=2.00, seeds 7/25/42
#
# Usage: bash submit_batch2.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "Submitting exp 217 batch 2 (4 jobs)"
echo "================================================"

jids=()
submit() {
    local script="$1"
    local jid
    jid=$(sbatch --parsable "${SCRIPT_DIR}/${script}")
    echo "  Submitted ${script}  ->  Job ${jid}"
    jids+=("${jid}")
}

submit train_cos_1p25_3seeds.slurm
submit train_cos_1p50_3seeds.slurm
submit train_cos_1p75_3seeds.slurm
submit train_cos_2p00_3seeds.slurm

echo "================================================"
echo "All ${#jids[@]} jobs submitted: ${jids[*]}"
echo "Monitor: squeue -u $USER"
echo "================================================"
