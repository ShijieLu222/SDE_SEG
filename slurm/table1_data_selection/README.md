# Table 1: Automatic Data Selection Experiments

This directory contains SLURM scripts for reproducing Table 1 from the paper: "Improving Semi-Supervised and Domain-Adaptive Semantic Segmentation with Self-Supervised Depth Estimation"

## Experiments Overview

Table 1 compares different data selection methods:
- **Random**: Random sampling (baseline)
- **Entropy**: Entropy-based selection
- **US**: Uncertainty Sampling (depth-based)
- **DS**: Diversity Sampling (depth-based)
- **DS+US**: Combined Diversity + Uncertainty Sampling

Each method is tested with 3 different label amounts:
- 100 labels (1/30 of dataset)
- 372 labels (1/8 of dataset)
- 744 labels (1/4 of dataset)

Total: **15 experiments** (5 methods × 3 label amounts)

## Scripts

### Individual Scripts
每个脚本跑 **3 个 Run ID**（对应 seeds 7, 25, 42），用于计算 mIoU mean±std：
- `train_random_*.slurm` - Random: RUN_IDS="0,15,30" / "1,16,31" / "2,17,32"
- `train_entropy_*.slurm` - Entropy: RUN_IDS="3,18,33" / "4,19,34" / "5,20,35"
- `train_us_*.slurm` - US: RUN_IDS="6,21,36" / "7,22,37" / "8,23,38"
- `train_ds_*.slurm` - DS: RUN_IDS="9,24,39" / "10,25,40" / "11,26,41"
- `train_ds_us_*.slurm` - DS+US: RUN_IDS="12,27,42" / "13,28,43" / "14,29,44"

### Batch Submission
- `submit_all.sh` - Submit all 15 experiments at once

## Usage

### Submit All Experiments
```bash
cd slurm/table1_data_selection
./submit_all.sh
```

### Submit Individual Experiment
```bash
sbatch train_random_100.slurm
```

### Check Job Status
```bash
squeue -u $USER
```

### Check Logs
```bash
# Output logs
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/table1_*-<JOB_ID>.out

# Error logs
tail -f /scratch/u5hv/shijie.u5hv/sde_seg/logs/table1_*-<JOB_ID>.err
```

### Cancel All Jobs
```bash
scancel -u $USER
```

## Expected Results

According to the paper, expected mIoU results:

| Method | 100 labels | 372 labels | 744 labels |
|--------|------------|------------|------------|
| Random | 48.75 ± 1.61 | 59.14 ± 1.02 | 63.46 ± 0.38 |
| Entropy | 53.63 ± 0.77 | 63.51 ± 0.68 | 66.18 ± 0.50 |
| US | 51.75 ± 1.12 | 62.77 ± 0.46 | 66.76 ± 0.45 |
| DS | 53.00 ± 0.51 | 63.23 ± 0.69 | 66.37 ± 0.20 |
| DS+US | **54.37 ± 0.36** | **64.25 ± 0.18** | **66.94 ± 0.59** |

## Notes

- All experiments use experiment ID 210
- Each script runs **3 Run IDs** (3 seeds: 7, 25, 42) to compute mIoU mean±std
- Experiments run independently and can be submitted in parallel
- Each job requests: 1 GPU, 8 CPUs, 64GB RAM, 24 hours

