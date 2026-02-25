#!/bin/bash
BASE="${1:-/scratch/u5hv/shijie.u5hv/sde_seg/logs/cityscapes_joint_213}"
for d in "$BASE"/202*; do
  [ -d "$d" ] || continue
  run=$(basename "$d")
  log=$(ls -t "$d"/run_*.log 2>/dev/null | head -1)
  [ -z "$log" ] && continue
  last=$(grep -o 'Iter \[[0-9]*/40000\]' "$log" 2>/dev/null | tail -1)
  if [ -n "$last" ]; then
    cur=$(echo "$last" | sed 's/.*\[\([0-9]*\)\/40000\].*/\1/')
    pct=$(awk "BEGIN {printf \"%.1f\", ($cur/40000)*100}")
    echo "$run: $last ($pct%)"
  fi
done
