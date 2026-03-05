#!/bin/bash
# 单独提交 exp 217 baseline (λ=0) 验证 job
# Usage: bash submit_baseline.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "Submitting exp 217 baseline (λ=0) — 3 seeds"
echo "================================================"

jid=$(sbatch --parsable "${SCRIPT_DIR}/train_baseline_3seeds.slurm")
echo "  Submitted train_baseline_3seeds.slurm  ->  Job ${jid}"

echo "================================================"
echo "Monitor: squeue -u $USER"
echo "================================================"
