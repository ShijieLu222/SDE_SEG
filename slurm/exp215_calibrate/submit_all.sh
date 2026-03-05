#!/bin/bash
# Submit both calibration jobs in parallel (~1h each)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Submitting exp 215: Cross-Task Calibration"
echo "=========================================="
echo "  Run 0: MSE    (normalised)  ~1h"
echo "  Run 1: cosine               ~1h"
echo "  Both jobs run in parallel."
echo "=========================================="
echo ""

mkdir -p /scratch/u5hv/shijie.u5hv/sde_seg/logs

JOB_MSE=$(sbatch calibrate_mse.slurm | awk '{print $4}')
echo "[1/2] MSE    submitted -> Job ID: ${JOB_MSE}"

JOB_COS=$(sbatch calibrate_cosine.slurm | awk '{print $4}')
echo "[2/2] cosine submitted -> Job ID: ${JOB_COS}"

echo ""
echo "Check status : squeue -u \$USER"
echo "Cancel all   : scancel ${JOB_MSE} ${JOB_COS}"
echo ""
echo "Logs:"
echo "  tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/exp215_calibrate_mse-${JOB_MSE}.out"
echo "  tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/exp215_calibrate_cosine-${JOB_COS}.out"
