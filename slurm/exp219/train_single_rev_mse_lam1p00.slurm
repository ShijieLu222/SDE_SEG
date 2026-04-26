#!/bin/bash
#SBATCH --job-name=exp219_srev_mse_lam1p00
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/user/work/ig23200/sde_seg/logs/exp219_single_rev_mse_lam1p00-%j.out
#SBATCH --error=/user/work/ig23200/sde_seg/logs/exp219_single_rev_mse_lam1p00-%j.err

set -euo pipefail
RUN_IDS="39"
CONFIG="configs/cityscapes_joint.yml"
echo "exp 219 run ${RUN_IDS}: single_rev_mse_w0_lam1p00 (1 seed, 24h)"
source /user/work/ig23200/miniconda3/etc/profile.d/conda.sh && conda activate sde_seg
cd /user/home/ig23200/projects/SDE_SEG
python run_experiments.py --config "${CONFIG}" --machine bp1 --exp 219 --run "${RUN_IDS}"
