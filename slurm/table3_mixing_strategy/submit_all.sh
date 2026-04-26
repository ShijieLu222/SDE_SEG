#!/bin/bash
# Batch submit script for Table 3 experiments (Data Mixing Strategy)
# Reproduce paper Table 3: Baseline, Pseudo-Labels, ClassMix, DepthMix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Submitting Table 3 Mixing Strategy Experiments"
echo "=========================================="
echo ""
echo "Table 3: 3 seeds per script (7,25,42), 21 configs total"
echo "  Baseline 372:      Run 0,7,14"
echo "  Baseline 2975:     Run 1,8,15"
echo "  Pseudo-Labels 372: Run 2,9,16"
echo "  ClassMix 372:      Run 3,10,17"
echo "  DepthMix 372:      Run 4,11,18"
echo "  DepthMix 2975:     Run 5,12,19"
echo "  ClassMix-GT 2975:  Run 6,13,20"
echo ""

SCRIPTS=(
    "train_baseline_372.slurm"      # Run 0,7,14
    "train_baseline_2975.slurm"     # Run 1,8,15
    "train_pseudo_labels_372.slurm" # Run 2,9,16
    "train_classmix_372.slurm"      # Run 3,10,17
    "train_depthmix_372.slurm"      # Run 4,11,18
    "train_depthmix_2975.slurm"     # Run 5,12,19
    "train_classmix_2975.slurm"     # Run 6,13,20 (ClassMix-GT)
)

JOB_IDS=()
for i in "${!SCRIPTS[@]}"; do
    script="${SCRIPTS[$i]}"
    if [ ! -f "$script" ]; then
        echo "ERROR: Script $script not found!"
        continue
    fi

    echo "[$((i+1))/7] Submitting $script (3 seeds)..."
    job_id=$(sbatch "$script" | awk '{print $4}')
    JOB_IDS+=("$job_id")
    echo "  -> Job ID: $job_id"
    echo ""
done

echo "=========================================="
echo "All Table 3 jobs submitted!"
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
echo "  tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/table3_*-\${JOB_ID}.out"
echo ""
