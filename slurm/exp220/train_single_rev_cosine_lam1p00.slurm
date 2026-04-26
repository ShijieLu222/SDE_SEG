#!/bin/bash
# exp220: single_rev_cosine w0 λ=1.00, 3 seeds (7,25,42)
#SBATCH --job-name=exp220_srev_cos_lam1p00
#SBATCH --partition=mlcnu
#SBATCH --account=coms037985
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/user/work/ig23200/sde_seg/logs/exp220_single_rev_cosine_lam1p00-%j.out
#SBATCH --error=/user/work/ig23200/sde_seg/logs/exp220_single_rev_cosine_lam1p00-%j.err

set -euo pipefail
RUN_IDS="45,46,47"
CONFIG="configs/cityscapes_joint.yml"
echo "exp 220 run ${RUN_IDS}: single_rev_cosine_w0_lam1p00 (3 seeds, 40h)"
source /user/work/ig23200/miniconda3/etc/profile.d/conda.sh && conda activate sde_seg
cd /user/home/ig23200/projects/SDE_SEG
python run_experiments.py --config "${CONFIG}" --machine bp1 --exp 220 --run "${RUN_IDS}"
