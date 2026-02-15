#!/bin/bash
# Batch submit script for Table 1 experiments
# This script submits all 15 experiments to SLURM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Submitting Table 1 Data Selection Experiments"
echo "=========================================="
echo ""

# Array of all experiment scripts in order
# Run ID 映射（experiments.py: seed 外层循环）
# 每个脚本跑 3 个 Run ID = 3 seeds (7, 25, 42) -> mIoU mean±std
SCRIPTS=(
    "train_random_100.slurm"      # Run 0 - Random 100 (1/30)
    "train_entropy_100.slurm"     # Run 1 - Entropy 100 (1/30)
    "train_us_100.slurm"          # Run 2 - US 100 (1/30)
    "train_ds_100.slurm"          # Run 3 - DS 100 (1/30)
    "train_ds_us_100.slurm"       # Run 4 - DS+US 100 (1/30)
    "train_random_372.slurm"      # Run 5 - Random 372 (1/8)
    "train_entropy_372.slurm"    # Run 6 - Entropy 372 (1/8)
    "train_us_372.slurm"          # Run 7 - US 372 (1/8)
    "train_ds_372.slurm"          # Run 8 - DS 372 (1/8)
    "train_ds_us_372.slurm"       # Run 9 - DS+US 372 (1/8)
    "train_random_744.slurm"      # Run 10 - Random 744 (1/4)
    "train_entropy_744.slurm"    # Run 11 - Entropy 744 (1/4)
    "train_us_744.slurm"          # Run 12 - US 744 (1/4)
    "train_ds_744.slurm"          # Run 13 - DS 744 (1/4)
    "train_ds_us_744.slurm"       # Run 14 - DS+US 744 (1/4)
)

# Submit all jobs
JOB_IDS=()
for i in "${!SCRIPTS[@]}"; do
    script="${SCRIPTS[$i]}"
    run_id=$i
    
    if [ ! -f "$script" ]; then
        echo "ERROR: Script $script not found!"
        continue
    fi
    
    echo "[$((i+1))/15] Submitting $script (Run ID: $run_id)..."
    job_id=$(sbatch "$script" | awk '{print $4}')
    JOB_IDS+=("$job_id")
    echo "  -> Job ID: $job_id"
    echo ""
done

echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To cancel all jobs:"
echo "  scancel ${JOB_IDS[*]}"
echo ""
echo "To check logs:"
echo "  tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/table1_*-\${JOB_ID}.out"
echo ""

