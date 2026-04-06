#!/bin/bash
# exp220: single_mse w0 λ=1.0, 3 seeds (7,25,42)
#SBATCH --job-name=exp220_s_mse_w0_lam1p00
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/user/work/ig23200/sde_seg/logs/exp220_single_mse_w0_lam1p00-%j.out
#SBATCH --error=/user/work/ig23200/sde_seg/logs/exp220_single_mse_w0_lam1p00-%j.err

set -euo pipefail
RUN_IDS="12,13,14"
CONFIG="configs/cityscapes_joint.yml"
echo "exp 220 run ${RUN_IDS}: single_mse_w0_lam1p00 (3 seeds, 40h)"
source /user/work/ig23200/miniconda3/etc/profile.d/conda.sh && conda activate sde_seg
cd /user/home/ig23200/projects/SDE_SEG
python run_experiments.py --config "${CONFIG}" --machine bp1 --exp 220 --run "${RUN_IDS}"
