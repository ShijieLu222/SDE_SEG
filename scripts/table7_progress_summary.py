#!/usr/bin/env python3
"""Parse check_table7_progress.sh output and group by experiment + seed."""
import re
import sys
from collections import defaultdict

def tag_to_name(tag):
    if "cityscapes_" not in tag:
        return None, None
    # D372 or D2975
    m = re.search(r"D(\d+)", tag)
    n = m.group(1) if m else ""
    # S7, S25, S42
    s = re.search(r"S(\d+)[_\s]", tag)
    seed = int(s.group(1)) if s else 0
    # experiment type (tag uses underscores)
    if "sel_ds_us_pad_transfer_dcompgt" in tag:
        name = f"table7_s_dx_mtl_{n}"
    elif "pad_transfer_dcompgt" in tag and "sel_ds_us" not in tag:
        name = f"table7_dx_mtl_{n}"
    elif "sel_ds_us_transfer_dcompgt" in tag or ("sel_ds_us" in tag and "transfer_dcompgt" in tag and "pad" not in tag):
        name = f"table7_s_dx_{n}"
    elif "sel_ds_us_pad_transfer" in tag and "dcompgt" not in tag:
        name = f"table7_s_mtl_{n}"
    elif "sel_ds_us_transfer" in tag and "pad" not in tag and "dcompgt" not in tag:
        name = f"table7_s_{n}"
    elif "pad_transfer" in tag and "None" in tag:  # Unlab1_0None
        name = f"table7_mtl_{n}"
    elif "transfer_dcompgt" in tag and "pad" not in tag:
        name = f"table7_dx_{n}"
    elif re.search(r"cityscapes_transfer_D", tag) and "pad" not in tag and "dcompgt" not in tag and "sel" not in tag:
        name = f"table7_baseline_{n}"
    else:
        name = f"table7_?_{n}"
    return name, seed

def main():
    data = defaultdict(dict)  # name -> { seed -> (step, pct) }
    for line in sys.stdin:
        line = line.strip()
        if not line or "Iter" not in line:
            continue
        run, rest = line.split(":", 1)
        m = re.search(r"Iter \[(\d+)/40000\] \(([0-9.]+)%\)", rest)
        if not m:
            continue
        step, pct = m.group(1), m.group(2)
        # get tag from run: ...tag=...
        tag = run.split("tag=", 1)[-1] if "tag=" in run else run
        name, seed = tag_to_name(tag)
        if name and seed:
            data[name][seed] = (int(step), float(pct))

    order = [
        "table7_baseline_372", "table7_baseline_2975",
        "table7_mtl_372", "table7_mtl_2975",
        "table7_dx_372", "table7_dx_2975",
        "table7_s_372", "table7_s_mtl_372", "table7_s_dx_372",
        "table7_dx_mtl_372", "table7_dx_mtl_2975", "table7_s_dx_mtl_372",
    ]
    for name in order:
        if name not in data:
            continue
        seeds = sorted(data[name].keys())
        parts = [f"  seed {s}: {data[name][s][0]}/40000 ({data[name][s][1]:.1f}%)" for s in seeds]
        print(f"{name}:")
        print("\n".join(parts))
        print()

if __name__ == "__main__":
    main()
