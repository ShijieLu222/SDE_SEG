#!/bin/bash
#SBATCH --job-name=exp215_calibrate_cosine
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp215_calibrate_cosine-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp215_calibrate_cosine-%j.err

set -euo pipefail

# ===== exp 215: Calibration — cosine =====
# Run 1: cosine, λ_ct=1.0, seed 42, 600 iter only
# Purpose: measure raw L_seg / L_depth / L_ct(cosine,λ=1) magnitudes
RUN_IDS="1"
CONFIG="configs/cityscapes_joint.yml"
# =========================================

echo "============================================"
echo "exp 215: Calibration — cross_task_type=cosine"
echo "============================================"
echo "Run IDs : ${RUN_IDS}  (cosine, λ=1.0, seed 42, 600 iter)"
echo "Config  : ${CONFIG}"
echo "Time    : 1h (calibration only)"
echo "============================================"
echo "Host    : $(hostname)"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Start   : $(date)"
echo "GPU     : $CUDA_VISIBLE_DEVICES"
echo ""

source ~/fyp/venvs/sde_seg/bin/activate
cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth

python run_experiments.py --config "${CONFIG}" --machine ws --exp 215 --run "${RUN_IDS}"

echo ""
echo "============================================"
echo "Done!  End: $(date)"
echo ""
echo "Read these 3 values from tensorboard (iter 200-600 mean):"
echo "  training/segmentation_loss"
echo "  training/mono_loss"
echo "  training/cross_task_loss"
echo "============================================"
