#!/bin/bash
#SBATCH --job-name=sde_dec5
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/train_sde_dec5-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/train_sde_dec5-%j.err

set -euo pipefail

echo "================================================"
echo "  SDE Training Phase 1 (dec5)"
echo "================================================"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Configuration:"
echo "  - Phase: 1 (Decoder Training)"
echo "  - Iterations: 300,000"
echo "  - Encoder: FROZEN ❄️ (ImageNet pretrained)"
echo "  - Decoder: TRAINING ✅"
echo "  - Batch Size: 4"
echo "  - Data: Cityscapes video sequences (unlabeled)"
echo "  - Task: Self-Supervised Depth Estimation"
echo "================================================"
echo ""

source ~/fyp/venvs/sde_seg/bin/activate

cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth

echo "Starting SDE Phase 1 training..."
echo ""

python train.py --machine ws \
    --config configs/cityscapes_monodepth_highres_dec5_crop.yml

echo ""
echo "================================================"
echo "✅ SDE Phase 1 (dec5) completed!"
echo "End time: $(date)"
echo ""
echo "📁 Model saved to:"
echo "/scratch/u5hv/shijie.u5hv/sde_seg/logs/monodepth/cityscapes-monodepth-101aspp-dec5-crop/"
echo ""
echo "📋 Next step: Run Phase 2 (dec6)"
echo "   sbatch slurm/train_sde_dec6.slurm"
echo "================================================"

