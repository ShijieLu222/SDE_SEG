#!/bin/bash
#SBATCH --job-name=exp217_train_cos_1p25_3seeds
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp217_train_cos_1p25_3seeds-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp217_train_cos_1p25_3seeds-%j.err

set -euo pipefail

# ===== exp 217 batch 2: cosine λ=1.25 × 3 seeds =====
# seeds: 7 / 25 / 42 (顺序执行)
RUN_IDS="15,16,17"
CONFIG="configs/cityscapes_joint.yml"
# =============================================

echo "============================================="
echo "exp 217 batch 2: cosine λ=1.25 × 3 seeds"
echo "============================================="
echo "Run IDs : ${RUN_IDS}  (seeds 7 / 25 / 42)"
echo "Config  : ${CONFIG}"
echo "Time    : 24h"
echo "============================================="
echo "Host    : $(hostname)"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Start   : $(date)"
echo "GPU     : $CUDA_VISIBLE_DEVICES"
echo ""

source ~/fyp/venvs/sde_seg/bin/activate
cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth

echo "Starting 3 seeds (7, 25, 42) sequentially..."
echo ""

python run_experiments.py --config "${CONFIG}" --machine ws --exp 217 --run "${RUN_IDS}"

echo ""
echo "============================================="
echo "Done!  End: $(date)"
echo "============================================="
