"""
生成 exp217 各配置的 train loss 和 val mIoU 曲线图
用法：python3 scripts/plot_curves.py [--exp 217] [--outdir figures/]
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorboard.backend.event_processing import event_accumulator

# ──────────────────────────────────────────────────────────────
# 配置：要绘制的 run，每组取 3 seeds 的均值 ± std
# ──────────────────────────────────────────────────────────────
BASELINE_PATTERN = "cityscapes_pad_ct_lam0"   # exp214 baseline runs

EXP217_GROUPS = {
    "Baseline (λ=0)": {
        "logbase": "/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_214",
        "tag_patterns": [
            "2026-03-03_19-40-070_tag=cityscapes_pad_transfer_ct0_0_D372_S7",
            "2026-03-04_00-01-141_tag=cityscapes_pad_transfer_ct0_0_D372_S25",
            "2026-03-04_04-22-202_tag=cityscapes_pad_transfer_ct0_0_D372_S42",
        ],
    },
    "Cosine λ=0.48": {
        "logbase": "/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_217",
        "tag_patterns": ["cos_0p48"],
    },
    "Cosine λ=0.60": {
        "logbase": "/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_217",
        "tag_patterns": ["cos_0p60"],
    },
    "Cosine λ=0.84": {
        "logbase": "/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_217",
        "tag_patterns": ["cos_0p84"],
    },
    "Cosine λ=1.00": {
        "logbase": "/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_217",
        "tag_patterns": ["cos_1p00"],
    },
    "MSE λ=0.30": {
        "logbase": "/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_217",
        "tag_patterns": ["mse_0p30"],
    },
    "Cosine λ=1.25": {
        "logbase": "/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_217",
        "tag_patterns": ["cos_1p25"],
    },
    "Cosine λ=1.50": {
        "logbase": "/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_217",
        "tag_patterns": ["cos_1p50"],
    },
    "Cosine λ=1.75": {
        "logbase": "/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_217",
        "tag_patterns": ["cos_1p75"],
    },
    "Cosine λ=2.00": {
        "logbase": "/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_217",
        "tag_patterns": ["cos_2p00"],
    },
}


def load_scalar(run_dir, tag):
    """从 *.metrics event 文件中读取 scalar tag，返回 (steps, values)。"""
    evt_files = [f for f in os.listdir(run_dir) if f.endswith('.metrics')]
    if not evt_files:
        return None, None
    ea = event_accumulator.EventAccumulator(
        os.path.join(run_dir, evt_files[0]),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events])
    return steps, vals


def find_runs(logbase, patterns):
    """找到 logbase 下所有包含任意 pattern 的 run 目录。"""
    if not os.path.isdir(logbase):
        return []
    runs = []
    for d in sorted(os.listdir(logbase)):
        full = os.path.join(logbase, d)
        if os.path.isdir(full) and any(p in d for p in patterns):
            runs.append(full)
    return runs


def interpolate_to_common(steps_list, vals_list):
    """将多条曲线插值到公共 x 轴（最短的那条），返回 (x, mean, std)。"""
    min_len = min(len(s) for s in steps_list)
    common_steps = steps_list[0][:min_len]
    aligned = []
    for steps, vals in zip(steps_list, vals_list):
        aligned.append(np.interp(common_steps, steps, vals))
    arr = np.stack(aligned, axis=0)
    return common_steps, arr.mean(axis=0), arr.std(axis=0)


def plot_metric(groups, scalar_tag, ylabel, title, outpath, smooth=0.0, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    for (name, cfg), color in zip(groups.items(), colors):
        runs = find_runs(cfg["logbase"], cfg["tag_patterns"])
        if not runs:
            print(f"  [WARN] No runs found for '{name}'")
            continue

        seeds_steps, seeds_vals = [], []
        for r in runs:
            s, v = load_scalar(r, scalar_tag)
            if s is None:
                continue
            if smooth > 0:
                # exponential moving average
                alpha = 1 - smooth
                ema = np.zeros_like(v)
                ema[0] = v[0]
                for i in range(1, len(v)):
                    ema[i] = alpha * v[i] + (1 - alpha) * ema[i - 1]
                v = ema
            seeds_steps.append(s)
            seeds_vals.append(v)

        if not seeds_steps:
            continue

        if len(seeds_steps) == 1:
            ax.plot(seeds_steps[0], seeds_vals[0], label=name, color=color)
        else:
            x, mean, std = interpolate_to_common(seeds_steps, seeds_vals)
            ax.plot(x, mean, label=name, color=color)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="figures", help="Output directory for figures")
    parser.add_argument("--smooth", type=float, default=0.7,
                        help="EMA smoothing factor for train loss (0=none, 0.9=heavy)")
    args = parser.parse_args()

    print("Plotting val mIoU curves...")
    plot_metric(
        EXP217_GROUPS,
        scalar_tag="val_metrics/Mean IoU : \t",
        ylabel="mIoU",
        title="Validation mIoU — exp217 (3-seed mean ± std)",
        outpath=os.path.join(args.outdir, "exp217_val_miou.png"),
        smooth=0.0,
    )

    print("Plotting train total loss curves...")
    plot_metric(
        EXP217_GROUPS,
        scalar_tag="training/total_loss",
        ylabel="Total Loss",
        title="Training Total Loss — exp217 (3-seed mean ± std)",
        outpath=os.path.join(args.outdir, "exp217_train_loss.png"),
        smooth=args.smooth,
    )

    print("Plotting train segmentation loss curves...")
    plot_metric(
        EXP217_GROUPS,
        scalar_tag="training/segmentation_loss",
        ylabel="Segmentation Loss",
        title="Training Segmentation Loss — exp217 (3-seed mean ± std)",
        outpath=os.path.join(args.outdir, "exp217_train_seg_loss.png"),
        smooth=args.smooth,
    )

    print("Plotting train cross-task loss curves...")
    plot_metric(
        EXP217_GROUPS,
        scalar_tag="training/cross_task_loss",
        ylabel="Cross-Task Consistency Loss (L_ct)",
        title="Training L_ct — exp217 (3-seed mean ± std)",
        outpath=os.path.join(args.outdir, "exp217_train_lct.png"),
        smooth=args.smooth,
    )

    print("Plotting val mIoU (zoomed, iter 25000+)...")
    plot_metric(
        EXP217_GROUPS,
        scalar_tag="val_metrics/Mean IoU : \t",
        ylabel="mIoU",
        title="Validation mIoU (last ~15k iters) — exp217",
        outpath=os.path.join(args.outdir, "exp217_val_miou_zoom.png"),
        smooth=0.0,
        xlim=(25000, None),
        ylim=(0.60, 0.67),
    )

    print("Done.")


if __name__ == "__main__":
    main()
