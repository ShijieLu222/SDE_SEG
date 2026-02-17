#!/usr/bin/env python3
"""Extract Table 5 mIoU results from training log files.

Table 5: Comparison of SDE feature transfer methods.
F: ImageNet feature distance loss.
mIoU in %, std. dev. over 3 seeds.
"""

import re
import numpy as np
from pathlib import Path

LOG_DIR = Path("/scratch/u5hv/shijie.u5hv/sde_seg/logs")

# (display_name, F column, 372 log prefix, 2975 log prefix)
ROWS = [
    ("Baseline", "", "table5_baseline_372", "table5_baseline_2975"),
    ("Transfer (no F)", "", "table5_transfer_noF_372", "table5_transfer_noF_2975"),
    ("Transfer (F=✓)", "✓", "table5_transfer_F_372", "table5_transfer_F_2975"),
    ("Multi-Task (F=✓)", "✓", "table5_multitask_372", "table5_multitask_2975"),
]


def extract_miou_from_log(path: Path) -> list[float]:
    """Extract final mIoU (0-1) for each run. Each run separated by 'Dispatch job'."""
    if not path.exists():
        return []
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
    """Find the most recently modified log file matching <prefix>-*.out"""
    pattern = f"{prefix}-*.out"
    candidates = list(log_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract Table 5 results from log files")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Save markdown to file")
    args = parser.parse_args()

    log_dir = LOG_DIR

    # Build results: (row_key, n_labels) -> (mean_std_str, raw_vals, mean)
    results = {}
    for display_name, f_col, prefix_372, prefix_2975 in ROWS:
        for prefix, n in [(prefix_372, 372), (prefix_2975, 2975)]:
            log_path = find_latest_log(log_dir, prefix)
            if log_path:
                ious = extract_miou_from_log(log_path)
                mean_val = np.mean(ious) * 100 if ious else None
                results[(display_name, n)] = (format_mean_std(ious), ious, mean_val, str(log_path.name))
            else:
                results[(display_name, n)] = ("—", [], None, None)

    # Baseline means for improvement calculation
    base_372 = results.get(("Baseline", 372), (None, [], None, None))[2]
    base_2975 = results.get(("Baseline", 2975), (None, [], None, None))[2]

    # Per-seed
    print("=" * 70)
    print("Per-seed mIoU (×100)")
    print("=" * 70)
    for display_name, _, prefix_372, prefix_2975 in ROWS:
        _, vals_372, _, _ = results.get((display_name, 372), (None, [], None, None))
        _, vals_2975, _, _ = results.get((display_name, 2975), (None, [], None, None))
        s372 = ", ".join(f"{x*100:.2f}" for x in vals_372) if vals_372 else "—"
        s2975 = ", ".join(f"{x*100:.2f}" for x in vals_2975) if vals_2975 else "—"
        print(f"{display_name}:  372=[{s372}]  2975=[{s2975}]")
    print()

    # Paper Table 5 format
    print("=" * 70)
    print("Table 5 – Comparison of SDE feature transfer methods")
    print("(F: ImageNet feature distance loss. mIoU in %, std. dev. over 3 seeds)")
    print("=" * 70)
    print()

    # Header
    print("| Aux. SDE | F | 372 Labels (1/8) | 2975 Labels (Full) |")
    print("|----------|---|-------------------|---------------------|")

    for display_name, f_col, _, _ in ROWS:
        mean_std_372, vals_372, mean_372, _ = results.get((display_name, 372), ("—", [], None, None))
        mean_std_2975, vals_2975, mean_2975, _ = results.get((display_name, 2975), ("—", [], None, None))

        # Add improvement (+X.XX or -X.XX) for non-baseline
        cell_372 = mean_std_372
        if display_name != "Baseline" and mean_372 is not None and base_372 is not None:
            delta = mean_372 - base_372
            sign = "+" if delta >= 0 else ""
            cell_372 = f"{mean_std_372} ({sign}{delta:.2f})"

        cell_2975 = mean_std_2975
        if display_name != "Baseline" and mean_2975 is not None and base_2975 is not None:
            delta = mean_2975 - base_2975
            sign = "+" if delta >= 0 else ""
            cell_2975 = f"{mean_std_2975} ({sign}{delta:.2f})"

        f_display = f_col if f_col else "—"
        print(f"| {display_name} | {f_display} | {cell_372} | {cell_2975} |")

    print()
    print("(Baseline: scratch, no SDE. Transfer: SDE init. Multi-Task: seg+depth with fd2 init.)")

    if args.output:
        md = "\n".join([
            "# Table 5 – Comparison of SDE feature transfer methods",
            "",
            "*(F: ImageNet feature distance loss. mIoU in %, std. dev. over 3 seeds)*",
            "",
            "| Aux. SDE | F | 372 Labels (1/8) | 2975 Labels (Full) |",
            "|----------|---|-------------------|---------------------|",
        ])
        for display_name, f_col, _, _ in ROWS:
            mean_std_372, _, mean_372, _ = results.get((display_name, 372), ("—", [], None, None))
            mean_std_2975, _, mean_2975, _ = results.get((display_name, 2975), ("—", [], None, None))
            cell_372 = mean_std_372
            if display_name != "Baseline" and mean_372 is not None and base_372 is not None:
                delta = mean_372 - base_372
                cell_372 = f"{mean_std_372} ({'+' if delta >= 0 else ''}{delta:.2f})"
            cell_2975 = mean_std_2975
            if display_name != "Baseline" and mean_2975 is not None and base_2975 is not None:
                delta = mean_2975 - base_2975
                cell_2975 = f"{mean_std_2975} ({'+' if delta >= 0 else ''}{delta:.2f})"
            f_display = f_col if f_col else "—"
            md += f"\n| {display_name} | {f_display} | {cell_372} | {cell_2975} |"
        md += "\n"
        args.output.write_text(md, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
