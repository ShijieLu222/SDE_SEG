#!/bin/bash
#SBATCH --job-name=exp219_d_cos_w5k_lam0p75
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp219_dual_cosine_w5k_lam0p75-%j.out
#SBATCH --error=/scratch/u5hv/shijie.u5hv/sde_seg/logs/exp219_dual_cosine_w5k_lam0p75-%j.err

set -euo pipefail
RUN_IDS="18"
CONFIG="configs/cityscapes_joint.yml"
echo "exp 219 run ${RUN_IDS}: dual_cosine_w5k_lam0p75 (1 seed, 24h)"
source ~/fyp/venvs/sde_seg/bin/activate
cd /home/u5hv/shijie.u5hv/fyp/improving_segmentation_with_selfsupervised_depth
python run_experiments.py --config "${CONFIG}" --machine ws --exp 219 --run "${RUN_IDS}"
