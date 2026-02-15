#!/usr/bin/env python3
"""Extract Table 1 mIoU results from training log files.

Table 1: Comparison of data selection methods.
DS: diversity sampling based on depth features
US: uncertainty sampling based on depth student error
mIoU in %, std. dev. over 3 seeds.
"""

import argparse
import re
import numpy as np
from pathlib import Path

DEFAULT_LOG_DIR = Path("/scratch/u5hv/shijie.u5hv/sde_seg/logs")

# Table 1: (method, n_labels) -> log prefix
# Each slurm outputs table1_<method>_<n>-<jobid>.out
METHODS = [
    ("Random", "random"),
    ("Entropy", "entropy"),
    ("Ours (US)", "us"),
    ("Ours (DS)", "ds"),
    ("Ours (DS+US)", "ds_us"),
]
N_LABELS = [(100, "1/30"), (372, "1/8"), (744, "1/4")]


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


def find_latest_log(log_dir: Path, prefix: str) -> Path | None:
    """Find the most recently modified log file matching table1_<prefix>-*.out"""
    pattern = f"table1_{prefix}-*.out"
    candidates = list(log_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Extract Table 1 results from log files")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory containing table1_*.out log files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output markdown file (default: print to stdout)",
    )
    args = parser.parse_args()
    log_dir = args.log_dir

    # Build result matrix: rows = methods, cols = n_labels
    results = {}  # (method_key, n) -> (mean_std_str, raw_vals)
    for display_name, method_key in METHODS:
        for n, frac in N_LABELS:
            prefix = f"{method_key}_{n}"
            log_path = find_latest_log(log_dir, prefix)
            if log_path:
                ious = extract_miou_from_log(log_path)
                results[(method_key, n)] = (format_mean_std(ious), ious, str(log_path.name))
            else:
                results[(method_key, n)] = ("—", [], None)

    # Print raw per-seed for debugging
    print("=" * 70)
    print("Per-seed mIoU (×100)")
    print("=" * 70)
    for display_name, method_key in METHODS:
        parts = []
        for n, _ in N_LABELS:
            _, vals, _ = results[(method_key, n)]
            s = ", ".join(f"{x*100:.2f}" for x in vals) if vals else "—"
            parts.append(f"{n}={s}")
        print(f"{display_name}: {' | '.join(parts)}")
    print()

    # Build Table 1 markdown (Paper format)
    # Find best (max mean) in each column for bold
    best_per_col = []
    for col_idx in range(len(N_LABELS)):
        best_mean = -1
        for _, method_key in METHODS:
            _, vals, _ = results[(method_key, N_LABELS[col_idx][0])]
            if vals:
                m = np.mean(vals) * 100
                if m > best_mean:
                    best_mean = m
        best_per_col.append(best_mean)

    md_lines = [
        "# Table 1 – Comparison of data selection methods",
        "",
        "*DS: diversity sampling based on depth features*  ",
        "*US: uncertainty sampling based on depth student error*  ",
        "*mIoU in %, std. dev. over 3 seeds*",
        "",
        "|  | 1/30 (100) | 1/8 (372) | 1/4 (744) |",
        "|--|------------|-----------|-----------|",
    ]

    for display_name, method_key in METHODS:
        row_vals = []
        for col_idx, (n, _) in enumerate(N_LABELS):
            mean_std, vals, _ = results[(method_key, n)]
            if vals and np.mean(vals) * 100 >= best_per_col[col_idx] - 0.01:
                row_vals.append(f"**{mean_std}**")
            else:
                row_vals.append(mean_std)
        md_lines.append(f"| {display_name} | {row_vals[0]} | {row_vals[1]} | {row_vals[2]} |")

    md_content = "\n".join(md_lines)

    # Print to stdout
    print("=" * 70)
    print("Table 1 – Comparison of data selection methods")
    print("=" * 70)
    print()
    for line in md_lines:
        print(line)
    print()

    if args.output:
        args.output.write_text(md_content, encoding="utf-8")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
