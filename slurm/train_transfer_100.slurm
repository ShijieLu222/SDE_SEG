#!/bin/bash
#SBATCH --job-name=transfer_100
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/train_transfer_100-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/train_transfer_100-%j.err

set -euo pipefail

echo "================================================"
echo "  Transfer Learning: 100 labels (1/30)"
echo "================================================"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""
echo "Configuration:"
echo "  - Method: Transfer Learning from SDE"
echo "  - Labels: 100 (1/30 of Cityscapes)"
echo "  - Encoder init: SDE pretrained ✨"
echo "  - Task: Semantic Segmentation"
echo ""
echo "📥 First run will download SDE pretrained model (~500MB)"
echo "   from Google Drive (automatic)"
echo "================================================"
echo ""

source ~/fyp/venvs/sde_seg/bin/activate

cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth

echo "Starting Transfer Learning training..."
python run_experiments.py --machine ws --exp 210 --run 1

echo ""
echo "================================================"
echo "✅ Transfer Learning (100 labels) completed!"
echo "End time: $(date)"
echo "================================================"
echo ""
echo "📊 Compare with baseline:"
echo "  Baseline: Check previous results"
echo "  Transfer: Check TensorBoard logs"
echo ""

