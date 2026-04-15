#!/bin/bash
#SBATCH --job-name=exp222_mse_dx_sel
#SBATCH --partition=mlcnu
#SBATCH --account=coms037985
#SBATCH --time=42:00:00
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/user/work/ig23200/sde_seg/logs/exp222_mse_mtl_dx_sel-%j.out
#SBATCH --error=/user/work/ig23200/sde_seg/logs/exp222_mse_mtl_dx_sel-%j.err

set -euo pipefail
RUN_IDS="6,7,8"
CONFIG="configs/cityscapes_joint.yml"
echo "exp 222 runs ${RUN_IDS}: MTL + DX + Sel + CT(MSE,lam1.0,w0), N=372, seeds 7/25/42"
source /user/work/ig23200/miniconda3/etc/profile.d/conda.sh && conda activate sde_seg
cd /user/home/ig23200/projects/SDE_SEG
python run_experiments.py --config "${CONFIG}" --machine bp1 --exp 222 --run "${RUN_IDS}"
