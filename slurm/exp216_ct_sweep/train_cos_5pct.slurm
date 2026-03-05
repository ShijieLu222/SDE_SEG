#!/bin/bash
#SBATCH --job-name=exp216_train_cos_5pct
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp216_train_cos_5pct-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp216_train_cos_5pct-%j.err

set -euo pipefail

# ===== exp 216: cosine λ=0.24 (contribution 5%) =====
# Run 7: cosine λ=0.24, 1 seed (seed 42)
RUN_IDS="7"
CONFIG="configs/cityscapes_joint.yml"
# ====================================================

echo "=========================================="
echo "exp 216: cosine λ=0.24  (5% contribution)"
echo "=========================================="
echo "Run ID  : ${RUN_IDS}"
echo "Config  : ${CONFIG}"
echo "Time    : 24h"
echo "=========================================="
echo "Host    : $(hostname)"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Start   : $(date)"
echo "GPU     : $CUDA_VISIBLE_DEVICES"
echo ""

source ~/fyp/venvs/sde_seg/bin/activate
cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth

python run_experiments.py --config "${CONFIG}" --machine ws --exp 216 --run "${RUN_IDS}"

echo ""
echo "=========================================="
echo "Done!  End: $(date)"
echo "=========================================="
