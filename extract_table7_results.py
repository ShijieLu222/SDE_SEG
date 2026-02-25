#!/usr/bin/env python3
"""Extract Table 7 mIoU results from training log files.

Table 7: Framework Component Ablation
S: Data Selection (preselected labels)
DX: DepthMix
MTL: Multi-Task Learning (SDE + Segmentation)
mIoU in %, std. dev. over 3 seeds.
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Optional

LOG_DIR = Path("/scratch/u5hv/shijie.u5hv/sde_seg/logs")
TRAINING_DIR = Path("/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_213")

# Table 7 rows: (display_name, S, DX, MTL, tag_pattern_372, tag_pattern_2975, exclude_patterns)
# Tag patterns to match training directories
ROWS = [
    ("Baseline", "—", "—", "—", 
     r"^cityscapes_transfer_D372random", r"^cityscapes_transfer_D2975random", []),
    ("MTL only", "—", "—", "✓", 
     r"pad_transfer_D372random.*Unlab1_0None", r"pad_transfer_D2975random.*Unlab1_0None", [r"dcompgt", r"sel_ds_us"]),
    ("DX only", "—", "✓", "—", 
     r"transfer_dcompgt0030_D372random", r"transfer_dcompgt0030_D2975random", [r"pad_transfer", r"sel_ds_us"]),
    ("S only", "✓", "—", "—", 
     r"sel_ds_us_transfer_D372fixed", None, [r"dcompgt", r"pad_transfer"]),
    ("S + MTL", "✓", "—", "✓", 
     r"sel_ds_us_pad_transfer_D372fixed.*Unlab1_0None", None, [r"dcompgt"]),
    ("S + DX", "✓", "✓", "—", 
     r"sel_ds_us_transfer_dcompgt0030_D372fixed", None, [r"pad_transfer"]),
    ("DX + MTL", "—", "✓", "✓", 
     r"pad_transfer_dcompgt0030_D372random", r"pad_transfer_dcompgt0030_D2975random", [r"sel_ds_us"]),
    ("S + DX + MTL", "✓", "✓", "✓", 
     r"sel_ds_us_pad_transfer_dcompgt0030_D372fixed", None, []),
]


def extract_miou_from_log(path: Path) -> List[float]:
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


def extract_miou_from_training_dir(training_dir: Path) -> List[float]:
    """Extract mIoU from training output directory (run_*.log files)."""
    if not training_dir.exists():
        return []
    log_files = sorted(training_dir.glob("run_*.log"))
    ious = []
    for log_file in log_files:
        try:
            text = log_file.read_text()
            matches = re.findall(r"Mean IoU\s*:\s*([\d.]+)", text)
            if matches:
                # Get the last (final) mIoU value
                ious.append(float(matches[-1]))
        except Exception as e:
            # Skip if file can't be read
            continue
    return ious


def format_mean_std(vals: List[float]) -> str:
    """Format as 'mean ± std' (percent)."""
    if not vals:
        return "—"
    arr = np.array(vals) * 100
    return f"{arr.mean():.2f} ± {arr.std():.2f}"


def find_latest_log(log_dir: Path, prefix: Optional[str]) -> Optional[Path]:
    """Find the most recently modified log file matching <prefix>-*.out"""
    if prefix is None:
        return None
    pattern = f"{prefix}-*.out"
    candidates = list(log_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_training_dirs_by_pattern(training_base: Path, pattern: str, n_labels: int, exclude_patterns: Optional[List[str]] = None) -> List[Path]:
    """Find training directories matching pattern and label count."""
    if not training_base.exists():
        return []
    dirs = [d for d in training_base.iterdir() if d.is_dir()]
    matching = []
    exclude_patterns = exclude_patterns or []
    for d in dirs:
        tag = d.name.split("tag=", 1)[-1] if "tag=" in d.name else d.name
        # Check label count first
        label_match = re.search(rf"D{n_labels}(?:random|fixed)", tag)
        if not label_match:
            continue
        # Check exclusion patterns
        excluded = False
        for excl_pattern in exclude_patterns:
            if re.search(excl_pattern, tag):
                excluded = True
                break
        if excluded:
            continue
        # Check pattern match
        if re.search(pattern, tag):
            matching.append(d)
    return matching


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract Table 7 results from log files")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Save markdown to file")
    args = parser.parse_args()

    log_dir = LOG_DIR

    # Build results: (row_key, n_labels) -> (mean_std_str, raw_vals, mean)
    results = {}
    for row_data in ROWS:
        display_name = row_data[0]
        s_col = row_data[1]
        dx_col = row_data[2]
        mtl_col = row_data[3]
        pattern_372 = row_data[4]
        pattern_2975 = row_data[5]
        exclude_patterns = row_data[6] if len(row_data) > 6 else []
        
        for pattern, n in [(pattern_372, 372), (pattern_2975, 2975)]:
            if pattern is None:
                results[(display_name, n)] = ("—", [], None, None)
                continue
            
            # Try SLURM log files first (most reliable)
            # Map display names to actual log file prefixes
            prefix_map = {
                "Baseline": f"table7_baseline_{n}",
                "MTL only": f"table7_mtl_{n}",
                "DX only": f"table7_dx_{n}",
                "S only": f"table7_s_{n}",
                "S + MTL": f"table7_s_mtl_{n}",
                "S + DX": f"table7_s_dx_{n}",
                "DX + MTL": f"table7_dx_mtl_{n}",
                "S + DX + MTL": f"table7_s_dx_mtl_{n}",
            }
            prefix = prefix_map.get(display_name, f"table7_{display_name.lower().replace(' ', '_').replace('+', '_')}_{n}")
            log_path = find_latest_log(log_dir, prefix)
            ious = []
            if log_path:
                ious = extract_miou_from_log(log_path)
            
            # Fallback to training directories if SLURM log not found
            if not ious:
                training_dirs = find_training_dirs_by_pattern(TRAINING_DIR, pattern, n, exclude_patterns)
                if training_dirs:
                    # Extract from all matching directories (should be 3 for 3 seeds)
                    # Sort by name to ensure consistent order
                    for td in sorted(training_dirs):
                        dir_ious = extract_miou_from_training_dir(td)
                        if dir_ious:
                            # Take the last (final) mIoU from each directory
                            ious.append(dir_ious[-1] if isinstance(dir_ious, list) else dir_ious)
            
            if ious:
                mean_val = np.mean(ious) * 100
                results[(display_name, n)] = (format_mean_std(ious), ious, mean_val, f"{len(ious)} seeds")
            else:
                results[(display_name, n)] = ("—", [], None, None)

    # Baseline means for improvement calculation
    base_372 = results.get(("Baseline", 372), (None, [], None, None))[2]
    base_2975 = results.get(("Baseline", 2975), (None, [], None, None))[2]

    # Per-seed
    print("=" * 70)
    print("Per-seed mIoU (×100)")
    print("=" * 70)
    for row_data in ROWS:
        display_name = row_data[0]
        _, vals_372, _, _ = results.get((display_name, 372), (None, [], None, None))
        _, vals_2975, _, _ = results.get((display_name, 2975), (None, [], None, None))
        s372 = ", ".join(f"{x*100:.2f}" for x in vals_372) if vals_372 else "—"
        s2975 = ", ".join(f"{x*100:.2f}" for x in vals_2975) if vals_2975 else "—"
        print(f"{display_name}:  372=[{s372}]  2975=[{s2975}]")
    print()

    # Table 7 format
    print("=" * 80)
    print("Table 7 – Framework Component Ablation")
    print("(S: Data Selection, DX: DepthMix, MTL: Multi-Task Learning. mIoU in %, std. dev. over 3 seeds)")
    print("=" * 80)
    print()

    # Header
    print("| 实验 | S | DX | MTL | 372 Labels (1/8) | 2975 Labels (Full) |")
    print("|------|---|---|-----|-------------------|---------------------|")

    md_lines = [
        "# Table 7 – Framework Component Ablation",
        "",
        "*(S: Data Selection, DX: DepthMix, MTL: Multi-Task Learning. mIoU in %, std. dev. over 3 seeds)*",
        "",
        "| 实验 | S | DX | MTL | 372 Labels (1/8) | 2975 Labels (Full) |",
        "|------|---|---|-----|-------------------|---------------------|",
    ]

    for row_data in ROWS:
        display_name = row_data[0]
        s_col = row_data[1]
        dx_col = row_data[2]
        mtl_col = row_data[3]
        mean_std_372, vals_372, mean_372, _ = results.get((display_name, 372), ("—", [], None, None))
        mean_std_2975, vals_2975, mean_2975, _ = results.get((display_name, 2975), ("—", [], None, None))

        # Add improvement (+X.XX) for non-baseline
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

        # Bold the best result (S + DX + MTL)
        if display_name == "S + DX + MTL":
            cell_372 = f"**{cell_372}**"
            if cell_2975 != "—":
                cell_2975 = f"**{cell_2975}**"

        row_str = f"| {display_name} | {s_col} | {dx_col} | {mtl_col} | {cell_372} | {cell_2975} |"
        print(row_str)
        md_lines.append(row_str)

    print()
    print("=" * 80)

    if args.output:
        md_content = "\n".join(md_lines) + "\n"
        args.output.write_text(md_content, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
