#!/bin/bash
# 一键提交 exp219 全部 32 个任务（Run ID 0–31）
# 用法: bash submit_all.sh  或  sbatch 只提交此脚本会只提交一个 job，请直接 bash 运行

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 顺序与 experiments.py 中 exp 219 的 cfgs 顺序一致：proj(single,dual) × ct(cosine,mse) × warmup(w5k,w0) × lam(0p25,0p50,0p75,1p00)
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
echo "Exp 219: 提交 ${#JOBS[@]} 个任务"
echo "=============================================="

for i in "${!JOBS[@]}"; do
  f="${JOBS[$i]}"
  if [[ ! -f "$f" ]]; then
    echo "跳过 (不存在): $f"
    continue
  fi
  id=$(sbatch "$f" | awk '{print $4}')
  echo "Run $i  $f  -> job $id"
done

echo "=============================================="
echo "提交完成。查看队列: squeue -u \$USER"
echo "=============================================="
