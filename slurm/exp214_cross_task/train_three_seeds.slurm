#!/bin/bash
#SBATCH --job-name=exp214_ct_3seeds
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp214_cross_task_three_seeds-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp214_cross_task_three_seeds-%j.err

set -euo pipefail

# ===== exp 214: Cross-Task Consistency Ablation — 3 seeds (full ablation) =====
# Same data/config as Table 7 MTL (exp 213): 2975 frames, 372 labeled, only_sequences_with_segmentation=True
# Run IDs: 0-12  =>  4 lambdas × 3 seeds = 12 runs sequentially
# If 12 runs exceed 24h, consider splitting into 4 jobs (e.g. --run 0,1,2 then 3,4,5 etc.)
RUN_IDS="0-12"
CONFIG="configs/cityscapes_joint.yml"
# ================================================================================

echo "=========================================="
echo "exp 214: Cross-Task Consistency (3 seeds)"
echo "=========================================="
echo "Run IDs: ${RUN_IDS}  =>  4 λ × 3 seeds = 12 runs"
echo "Config: ${CONFIG}"
echo "Time limit: 24h"
echo "=========================================="
echo ""
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Workdir: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

source ~/fyp/venvs/sde_seg/bin/activate

cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth

echo "Starting training (12 runs sequentially)..."
echo "Command: python run_experiments.py --config ${CONFIG} --machine ws --exp 214 --run ${RUN_IDS}"
echo ""

python run_experiments.py --config "${CONFIG}" --machine ws --exp 214 --run ${RUN_IDS}

echo ""
echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="
