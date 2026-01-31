#!/bin/bash
#SBATCH --job-name=baseline_372
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/train_baseline_372-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/train_baseline_372-%j.err

set -euo pipefail

echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Workdir: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Training with 372 labels (1/8)"

source ~/fyp/venvs/sde_seg/bin/activate

cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth

python run_experiments.py --machine ws --exp 210 --run 1

echo "Training completed!"
echo "End time: $(date)"

