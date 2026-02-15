#!/bin/bash
#SBATCH --job-name=table1_us_100
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/table1_us_100-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/table1_us_100-%j.err

set -euo pipefail

# ===== 实验配置信息 =====
EXP_TYPE="US"
N_LABELS=100
RUN_IDS="6,21,36"   # 3 seeds (7, 25, 42) -> mIoU mean±std
SELECTION_METHOD="Uncertainty Sampling (depth estimation error)"
FRACTION="1/30"
# =========================

echo "=========================================="
echo "Table 1 Experiment: Data Selection"
echo "=========================================="
echo "Experiment Type: ${EXP_TYPE}"
echo "Number of Labels: ${N_LABELS}"
echo "Selection Method: ${SELECTION_METHOD}"
echo "Data Fraction: ${FRACTION}"
echo "Run IDs: ${RUN_IDS} (seeds 7, 25, 42)"
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

echo "Starting training..."
echo "Command: python run_experiments.py --machine ws --exp 210 --run ${RUN_IDS}"
echo ""

python run_experiments.py --machine ws --exp 210 --run ${RUN_IDS}

echo ""
echo "=========================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=========================================="
