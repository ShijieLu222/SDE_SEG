#!/bin/bash
# Submit all 32 Exp 219 jobs in one command (Run ID 0-31).
# Usage: run with `bash submit_all.sh`. Using `sbatch` on this file submits only one job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Order matches Exp 219 cfgs in experiments.py:
# proj(single,dual) x ct(cosine,mse) x warmup(w5k,w0) x lam(0p25,0p50,0p75,1p00)
JOBS=(
  train_single_cosine_w5k_lam0p25.slurm
  train_single_cosine_w5k_lam0p50.slurm
  train_single_cosine_w5k_lam0p75.slurm
  train_single_cosine_w5k_lam1p00.slurm
  train_single_cosine_w0_lam0p25.slurm
  train_single_cosine_w0_lam0p50.slurm
  train_single_cosine_w0_lam0p75.slurm
  train_single_cosine_w0_lam1p00.slurm
  train_single_mse_w5k_lam0p25.slurm
  train_single_mse_w5k_lam0p50.slurm
  train_single_mse_w5k_lam0p75.slurm
  train_single_mse_w5k_lam1p00.slurm
  train_single_mse_w0_lam0p25.slurm
  train_single_mse_w0_lam0p50.slurm
  train_single_mse_w0_lam0p75.slurm
  train_single_mse_w0_lam1p00.slurm
  train_dual_cosine_w5k_lam0p25.slurm
  train_dual_cosine_w5k_lam0p50.slurm
  train_dual_cosine_w5k_lam0p75.slurm
  train_dual_cosine_w5k_lam1p00.slurm
  train_dual_cosine_w0_lam0p25.slurm
  train_dual_cosine_w0_lam0p50.slurm
  train_dual_cosine_w0_lam0p75.slurm
  train_dual_cosine_w0_lam1p00.slurm
  train_dual_mse_w5k_lam0p25.slurm
  train_dual_mse_w5k_lam0p50.slurm
  train_dual_mse_w5k_lam0p75.slurm
  train_dual_mse_w5k_lam1p00.slurm
  train_dual_mse_w0_lam0p25.slurm
  train_dual_mse_w0_lam0p50.slurm
  train_dual_mse_w0_lam0p75.slurm
  train_dual_mse_w0_lam1p00.slurm
)

echo "=============================================="
echo "Exp 219: submitting ${#JOBS[@]} jobs"
echo "=============================================="

for i in "${!JOBS[@]}"; do
  f="${JOBS[$i]}"
  if [[ ! -f "$f" ]]; then
    echo "Skipping (not found): $f"
    continue
  fi
  id=$(sbatch "$f" | awk '{print $4}')
  echo "Run $i  $f  -> job $id"
done

echo "=============================================="
echo "Submission complete. Check queue: squeue -u \$USER"
echo "=============================================="
