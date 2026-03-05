#!/bin/bash
#SBATCH --job-name=exp216_train_baseline
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp216_train_baseline-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp216_train_baseline-%j.err

set -euo pipefail

# ===== exp 216: baseline (contribution 0%) =====
# Run 0: baseline, 1 seed (seed 42)
RUN_IDS="0"
CONFIG="configs/cityscapes_joint.yml"
# ====================================================

echo "=========================================="
echo "exp 216: baseline  (0% contribution)"
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
