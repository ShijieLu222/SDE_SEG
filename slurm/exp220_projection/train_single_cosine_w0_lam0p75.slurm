#!/bin/bash
# exp220: single_cosine w0 λ=0.75, 3 seeds (7,25,42)
#SBATCH --job-name=exp220_s_cos_w0_lam0p75
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/user/work/ig23200/sde_seg/logs/exp220_single_cosine_w0_lam0p75-%j.out
#SBATCH --error=/user/work/ig23200/sde_seg/logs/exp220_single_cosine_w0_lam0p75-%j.err

set -euo pipefail
RUN_IDS="0,1,2"
CONFIG="configs/cityscapes_joint.yml"
echo "exp 220 run ${RUN_IDS}: single_cosine_w0_lam0p75 (3 seeds, 40h)"
source /user/work/ig23200/miniconda3/etc/profile.d/conda.sh && conda activate sde_seg
cd /user/home/ig23200/projects/SDE_SEG
python run_experiments.py --config "${CONFIG}" --machine bp1 --exp 220 --run "${RUN_IDS}"
