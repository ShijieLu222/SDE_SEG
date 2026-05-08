#!/bin/bash
# Full stack (S + DepthMix + PAD-MTL) + CT: single d→s projection, cosine, λ_ct=1, warmup 0.
# exp222 run indices 15–17 → seeds 7, 25, 42. Writes best_model.pkl when val mIoU improves.
#SBATCH --job-name=ct_cos_full
#SBATCH --partition=mlcnu
#SBATCH --account=coms037985
#SBATCH --time=44:00:00
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/user/work/ig23200/sde_seg/logs/exp222_cosine_ct_full_stack_saveckpt-%j.out
#SBATCH --error=/user/work/ig23200/sde_seg/logs/exp222_cosine_ct_full_stack_saveckpt-%j.err

set -euo pipefail
RUN_IDS="15,16,17"
CONFIG="configs/cityscapes_joint.yml"
echo "exp 222 runs ${RUN_IDS}: S+DX+MTL + CT(cosine, λ=1, w0), N=372 fixed, 3 seeds — save_model ON"
source /user/work/ig23200/miniconda3/etc/profile.d/conda.sh && conda activate sde_seg
cd /user/home/ig23200/projects/SDE_SEG
python run_experiments.py --config "${CONFIG}" --machine bp1 --exp 222 --run "${RUN_IDS}" --save-model
