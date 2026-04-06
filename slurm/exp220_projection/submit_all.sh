#!/bin/bash
# 提交 exp220 全部 13 个脚本
set -euo pipefail
cd "$(dirname "$0")/../.."
for f in slurm/exp220_projection/train_*.slurm; do
    sbatch "$f"
done
