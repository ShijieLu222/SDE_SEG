#!/bin/bash
#SBATCH --job-name=cs_preprocess
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/preprocess-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/preprocess-%j.err

set -euo pipefail

echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Workdir: $(pwd)"

source ~/fyp/venvs/sde_seg/bin/activate

cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth

python data_preprocessing/prepare_cityscapes.py

echo "Data preprocessing completed!"
echo "End time: $(date)"
