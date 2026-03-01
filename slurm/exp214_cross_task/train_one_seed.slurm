#!/bin/bash
#SBATCH --job-name=exp214_ct_1seed
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp214_cross_task_one_seed-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp214_cross_task_one_seed-%j.err

set -euo pipefail

# ===== exp 214: Cross-Task Consistency Ablation — 1 seed (seed 7 only) =====
# Same data/config as Table 7 MTL (exp 213): 2975 frames, 372 labeled, only_sequences_with_segmentation=True
# Run IDs: 0 (λ=0), 3 (λ=0.001), 6 (λ=0.0015), 9 (λ=0.002) — 4 runs, seed 7 only
RUN_IDS="0,3,6,9"
CONFIG="configs/cityscapes_joint.yml"
# ===========================================================================

echo "=========================================="
echo "exp 214: Cross-Task Consistency (1 seed)"
echo "=========================================="
echo "Run IDs: ${RUN_IDS}  =>  λ=0, 0.001, 0.0015, 0.002  (seed 7 only)"
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

echo "Starting training (4 runs sequentially)..."
echo "Command: python run_experiments.py --config ${CONFIG} --machine ws --exp 214 --run ${RUN_IDS}"
echo ""

python run_experiments.py --config "${CONFIG}" --machine ws --exp 214 --run ${RUN_IDS}

echo ""
echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="
