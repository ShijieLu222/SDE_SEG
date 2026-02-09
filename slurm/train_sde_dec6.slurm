#!/bin/bash
#SBATCH --job-name=sde_dec6
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/train_sde_dec6-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/train_sde_dec6-%j.err

set -euo pipefail

echo "================================================"
echo "  SDE Training Phase 2 (dec6)"
echo "================================================"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Configuration:"
echo "  - Phase: 2 (Joint Fine-tuning)"
echo "  - Iterations: 100,000"
echo "  - Encoder: UNFROZEN (Fine-tuning)"
echo "  - Decoder: TRAINING ✅"
echo "  - Feature Distance Loss: 1e-2 (prevent drift)"
echo "  - Batch Size: 4"
echo "  - Learning Rate: 1e-5"
echo "  - Loads: Phase 1 checkpoint (dec5)"
echo "================================================"
echo ""

source ~/fyp/venvs/sde_seg/bin/activate

cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth

echo "⚠️  IMPORTANT: This requires Phase 1 (dec5) checkpoint!"
echo "Checking for Phase 1 checkpoint..."
echo ""

# Check if dec5 checkpoint exists
DEC5_DIR="/scratch/u5hv/shijie.u5hv/sde_seg/logs/monodepth/cityscapes-monodepth-101aspp-dec5-crop"
if [ ! -d "$DEC5_DIR" ] || [ -z "$(ls -A $DEC5_DIR 2>/dev/null)" ]; then
    echo "❌ ERROR: Phase 1 checkpoint not found!"
    echo "   Expected location: $DEC5_DIR"
    echo ""
    echo "Please run Phase 1 first:"
    echo "   sbatch slurm/train_sde_dec5.slurm"
    echo ""
    exit 1
fi

echo "✅ Phase 1 checkpoint found!"
echo ""

echo "Starting SDE Phase 2 training..."
echo ""

python train.py --machine ws \
    --config configs/cityscapes_monodepth_highres_dec6_crop.yml

echo ""
echo "================================================"
echo "✅ SDE Phase 2 (dec6) completed!"
echo "End time: $(date)"
echo ""
echo "📁 Model saved to:"
echo "/scratch/u5hv/shijie.u5hv/sde_seg/logs/monodepth/cityscapes-monodepth-101aspp-dec6-crop/"
echo ""
echo "📋 Next step: Use this model for Transfer Learning"
echo "   Update experiments.py to use your trained model"
echo "   Then run: sbatch slurm/train_transfer_100.slurm"
echo "================================================"

