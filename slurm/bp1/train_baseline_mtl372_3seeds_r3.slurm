#!/bin/bash
# BluePebble: PAD-MTL baseline (Exp 213 runs 6,7,8), same as train_baseline_mtl372_3seeds.slurm.
# Log prefix r3 to distinguish from r2 / older jobs.
#
# Submit:  sbatch slurm/bp1/train_baseline_mtl372_3seeds_r3.slurm

#SBATCH --job-name=bp1_base_r3
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/user/work/ig23200/sde_seg/logs/bp1_baseline_mtl372_r3-%j.out
#SBATCH --error=/user/work/ig23200/sde_seg/logs/bp1_baseline_mtl372_r3-%j.err

set -euo pipefail

CONFIG="configs/cityscapes_joint.yml"
EXP_ID=213
RUN_IDS="6,7,8"

echo "=========================================="
echo "BP1 baseline R3: MTL only (PAD), 372 labels, 3 seeds (7,25,42)"
echo "=========================================="
echo "Config: ${CONFIG}"
echo "exp: ${EXP_ID}   run: ${RUN_IDS}"
echo "Host: $(hostname)   Job: ${SLURM_JOB_ID}   Start: $(date)"
echo "Workdir: $(pwd)   GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

source /user/work/ig23200/miniconda3/etc/profile.d/conda.sh
conda activate sde_seg

cd /user/home/ig23200/projects/SDE_SEG

python run_experiments.py --config "${CONFIG}" --machine bp1 --exp "${EXP_ID}" --run "${RUN_IDS}"

echo ""
echo "Finished: $(date)"
echo "=========================================="
