#!/bin/bash
# 纯 CPU 预处理 sequence；用 compute 分区，避免与训练抢 gpu 队列。
# sinfo: compute* 最长 14 天，无 GRES。若 sbatch 被拒再改 --time / --mem。
#SBATCH --job-name=prep_cs_seq
#SBATCH --partition=compute
#SBATCH --account=coms037985
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/user/work/ig23200/sde_seg/logs/prepare_cityscapes_sequence-%j.out
#SBATCH --error=/user/work/ig23200/sde_seg/logs/prepare_cityscapes_sequence-%j.err

set -euo pipefail
export CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/user/work/ig23200/sde_seg/raw/}"
source /user/work/ig23200/miniconda3/etc/profile.d/conda.sh
conda activate sde_seg
cd /user/home/ig23200/projects/SDE_SEG
echo "CITYSCAPES_ROOT=${CITYSCAPES_ROOT}"
python data_preprocessing/prepare_cityscapes.py --only sequence
echo "Done."
