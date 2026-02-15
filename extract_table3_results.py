#!/usr/bin/env python3
"""Extract Table 3 mIoU results from training log files."""

import re
import numpy as np
from pathlib import Path

LOG_DIR = Path("/scratch/u5hv/shijie.u5hv/sde_seg/logs")

# Map: (display name, log filename) -> (372 col name, 2975 col name)
# None means this experiment doesn't have that column
FILES = [
    ("Baseline", "table3_baseline_372-2318209.out", "table3_baseline_2975-2318210.out"),
    ("Pseudo-Labels", "table3_pseudo_372-2318211.out", None),
    ("ClassMix", "table3_classmix_372-2318212.out", None),
    ("ClassMix-GT", None, "table3_classmixgt_2975-2318215.out"),
    ("DepthMix", "table3_depthmix_372-2318213.out", "table3_depthmix_2975-2318214.out"),
]


def extract_miou_from_log(path: Path) -> list[float]:
    """Extract final mIoU (0-1) for each run. Each run separated by 'Dispatch job'."""
    text = path.read_text()
    chunks = re.split(r"^Dispatch job \d+_tag=", text, flags=re.MULTILINE)
    ious = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        matches = re.findall(r"Mean IoU\s*:\s*([\d.]+)", chunk)
        if matches:
            ious.append(float(matches[-1]))
    return ious


def format_mean_std(vals: list[float]) -> str:
    """Format as 'mean ± std' (percent)."""
    if not vals:
        return "—"
    arr = np.array(vals) * 100
    return f"{arr.mean():.2f} ± {arr.std():.2f}"


def main():
    # Collect results: row -> (col_372, col_2975)
    rows = []
    for name, f372, f2975 in FILES:
        vals_372 = extract_miou_from_log(LOG_DIR / f372) if f372 else []
        vals_2975 = extract_miou_from_log(LOG_DIR / f2975) if f2975 else []

        col_372 = format_mean_std(vals_372)
        col_2975 = format_mean_std(vals_2975)
        rows.append((name, col_372, col_2975, vals_372, vals_2975))

    # Print raw per-seed for debugging
    print("=" * 60)
    print("Per-seed mIoU (×100)")
    print("=" * 60)
    for name, _, _, v372, v2975 in rows:
        s372 = ", ".join(f"{x*100:.2f}" for x in v372) if v372 else "—"
        s2975 = ", ".join(f"{x*100:.2f}" for x in v2975) if v2975 else "—"
        print(f"{name}: 372={s372} | 2975={s2975}")

    # Print Table 3 layout
    print()
    print("=" * 70)
    print("Table 3 – 混合策略 (Mixing Strategy)")
    print("=" * 70)
    print(f"{'混合策略':<18} {'372 Labels (1/8)':<22} {'2975 Labels (Full)':<22}")
    print("-" * 70)
    for name, col_372, col_2975, _, _ in rows:
        print(f"{name:<18} {col_372:<22} {col_2975:<22}")
    print("=" * 70)


if __name__ == "__main__":
    main()
