#!/bin/bash
#SBATCH --job-name=exp221_mtl_dx_sel
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --time=42:00:00
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/user/work/ig23200/sde_seg/logs/exp221_mtl_dx_sel-%j.out
#SBATCH --error=/user/work/ig23200/sde_seg/logs/exp221_mtl_dx_sel-%j.err

set -euo pipefail
RUN_IDS="6,7,8"
CONFIG="configs/cityscapes_joint.yml"
echo "exp 221 runs ${RUN_IDS}: MTL + DepthMix + Selection, N=372, seeds 7/25/42 (40h)"
source /user/work/ig23200/miniconda3/etc/profile.d/conda.sh && conda activate sde_seg
cd /user/home/ig23200/projects/SDE_SEG
python run_experiments.py --config "${CONFIG}" --machine bp1 --exp 221 --run "${RUN_IDS}"
