#!/bin/bash
# Batch submit script for Table 7 (Framework Component Ablation)
# Reproduce paper Table 7: combinations of S (Data Selection), DX (DepthMix), and MTL (SDE Multi-Task Learning).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Submitting Table 7 Component Ablation Experiments"
echo "=========================================="
echo ""
echo "Table 7: 8 component combinations × 372/2975 labels × 3 seeds = 36 configs"
echo "  exp 213: Baseline, MTL, DX, S, S+MTL, S+DX, DX+MTL, S+DX+MTL"
echo "  Time limit: 24 hours per job"
echo ""

SCRIPTS=(
    "train_baseline_372.slurm"       # Run 0,1,2
    "train_baseline_2975.slurm"      # Run 3,4,5
    "train_mtl_372.slurm"            # Run 6,7,8
    "train_mtl_2975.slurm"           # Run 9,10,11
    "train_dx_372.slurm"             # Run 12,13,14
    "train_dx_2975.slurm"            # Run 15,16,17
    "train_s_372.slurm"              # Run 18,19,20
    "train_s_mtl_372.slurm"          # Run 21,22,23
    "train_s_dx_372.slurm"           # Run 24,25,26
    "train_dx_mtl_372.slurm"         # Run 27,28,29
    "train_dx_mtl_2975.slurm"        # Run 30,31,32
    "train_s_dx_mtl_372.slurm"       # Run 33,34,35
)

JOB_IDS=()
for i in "${!SCRIPTS[@]}"; do
    script="${SCRIPTS[$i]}"
    if [ ! -f "$script" ]; then
        echo "ERROR: Script $script not found!"
        continue
    fi

    echo "[$((i+1))/12] Submitting $script (3 seeds)..."
    job_id=$(sbatch "$script" | awk '{print $4}')
    JOB_IDS+=("$job_id")
    echo "  -> Job ID: $job_id"
    echo ""
done

echo "=========================================="
echo "All Table 7 jobs submitted!"
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
echo "  tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/table7_*-\${JOB_ID}.out"
echo ""
