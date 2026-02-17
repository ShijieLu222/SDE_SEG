#!/bin/bash
# Batch submit script for Table 5 (SDE Feature Transfer)
# 复刻论文 Table 5: Baseline, Transfer (no F), Transfer (F=✓), Multi-Task (F=✓)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Submitting Table 5 SDE Feature Transfer Experiments"
echo "=========================================="
echo ""
echo "Table 5: 4 settings × 2 subsets × 3 seeds = 24 configs"
echo "  exp 210: Baseline, Transfer (no F), Transfer (F=✓)"
echo "  exp 212: Multi-Task (F=✓)"
echo ""

SCRIPTS=(
    "train_baseline_372.slurm"       # exp 210: Run 0,6,12
    "train_baseline_2975.slurm"      # exp 210: Run 3,9,15
    "train_transfer_noF_372.slurm"   # exp 210: Run 1,7,13
    "train_transfer_noF_2975.slurm"  # exp 210: Run 4,10,16
    "train_transfer_F_372.slurm"     # exp 210: Run 2,8,14
    "train_transfer_F_2975.slurm"    # exp 210: Run 5,11,17
    "train_multitask_372.slurm"      # exp 212: Run 0,2,4
    "train_multitask_2975.slurm"     # exp 212: Run 1,3,5
)

JOB_IDS=()
for i in "${!SCRIPTS[@]}"; do
    script="${SCRIPTS[$i]}"
    if [ ! -f "$script" ]; then
        echo "ERROR: Script $script not found!"
        continue
    fi

    echo "[$((i+1))/8] Submitting $script (3 seeds)..."
    job_id=$(sbatch "$script" | awk '{print $4}')
    JOB_IDS+=("$job_id")
    echo "  -> Job ID: $job_id"
    echo ""
done

echo "=========================================="
echo "All Table 5 jobs submitted!"
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
echo "  tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/table5_*-\${JOB_ID}.out"
echo ""
