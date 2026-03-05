#!/bin/bash
# exp 217: Multi-seed validation for best λ candidates (全部 10 jobs)
#
# 10 jobs × 3 seeds (7/25/42) = 30 runs total
# Each job runs 3 seeds sequentially on 1 GPU (max 24h)
# All 10 jobs submitted in parallel
#
# Group      | ct_type | λ     | Runs
# -----------|---------|-------|------
# mse_0p30   | mse     | 0.30  | 0-2
# cos_0p48   | cosine  | 0.48  | 3-5
# cos_0p60   | cosine  | 0.60  | 6-8
# cos_0p84   | cosine  | 0.84  | 9-11
# cos_1p00   | cosine  | 1.00  | 12-14
# cos_1p25   | cosine  | 1.25  | 15-17
# cos_1p50   | cosine  | 1.50  | 18-20
# cos_1p75   | cosine  | 1.75  | 21-23
# cos_2p00   | cosine  | 2.00  | 24-26
# baseline   | cosine  | 0.00  | 27-29  ← final validation
#
# Usage: bash submit_all.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "Submitting exp 217 Multi-seed Validation (10 jobs)"
echo "================================================"

jids=()
submit() {
    local script="$1"
    local jid
    jid=$(sbatch --parsable "${SCRIPT_DIR}/${script}")
    echo "  Submitted ${script}  ->  Job ${jid}"
    jids+=("${jid}")
}

# Batch 1
submit train_mse_0p30_3seeds.slurm
submit train_cos_0p48_3seeds.slurm
submit train_cos_0p60_3seeds.slurm
submit train_cos_0p84_3seeds.slurm
submit train_cos_1p00_3seeds.slurm

# Batch 2
submit train_cos_1p25_3seeds.slurm
submit train_cos_1p50_3seeds.slurm
submit train_cos_1p75_3seeds.slurm
submit train_cos_2p00_3seeds.slurm

# Baseline (final validation)
submit train_baseline_3seeds.slurm

echo "================================================"
echo "All ${#jids[@]} jobs submitted: ${jids[*]}"
echo "Monitor: squeue -u $USER"
echo "================================================"
