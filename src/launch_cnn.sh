#!/usr/bin/env bash
# CNN queue (GPU), one run at a time: all six 3-class configs, then 10, then 50 classes.
# Collapse (exit 3, dev acc at chance after 5 epochs): retry with the next seed (up to 5 seeds), noted in DECISIONS.md.
# Any other crash only skips that run (see run_one.sh). Rerun to resume: finished runs are skipped.
cd "$(dirname "$0")/.." || exit 1
for k in 3 10 50; do
  for cfg in "V1 raw" "V2 raw" "V1 binarized" "V2 binarized" "V3 binarized" "V4 binarized"; do
    set -- $cfg
    for seed in 7 8 9 10 11; do
      src/run_one.sh "CNN_$1_${k}_$2_s$seed" python src/cnn.py "$1" "$2" "$k" "$seed"
      [ $? -eq 3 ] || break
      echo "- Collapse (Q3): CNN $1 $2 K=$k seed $seed stayed at chance after 5 epochs; reran with seed $((seed + 1))." >> DECISIONS.md
    done
  done
done
echo "$(date +%F_%T) | CNN queue | DONE | 0s" >> logs/status.txt
