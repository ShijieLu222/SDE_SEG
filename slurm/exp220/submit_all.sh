#!/bin/bash
# Submit all 13 Exp 220 scripts
set -euo pipefail
cd "$(dirname "$0")/../.."
for f in slurm/exp220_projection/train_*.slurm; do
    sbatch "$f"
done
