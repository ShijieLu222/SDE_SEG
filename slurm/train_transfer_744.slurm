#!/bin/bash
#SBATCH --job-name=transfer_744
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/train_transfer_744-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/train_transfer_744-%j.err

set -euo pipefail

echo "================================================"
echo "  Transfer Learning: 744 labels (1/4)"
echo "================================================"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Configuration:"
echo "  - Method: Transfer Learning from SDE"
echo "  - Labels: 744 (1/4 of Cityscapes)"
echo "  - Encoder init: SDE pretrained ✨"
echo "  - Task: Semantic Segmentation"
echo "================================================"
echo ""

source ~/fyp/venvs/sde_seg/bin/activate

cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth

echo "Starting Transfer Learning training..."
python run_experiments.py --machine ws --exp 210 --run 5

echo ""
echo "================================================"
echo "✅ Transfer Learning (744 labels) completed!"
echo "End time: $(date)"
echo "================================================"

