#!/usr/bin/env bash
# Transfer-learning queue (GPU, Option A): for each base, extract features once, then train the head.
# A crash only skips that step (see run_one.sh). Rerun to resume: finished runs are skipped and
# tl_features.py skips bases whose features exist.
cd "$(dirname "$0")/.." || exit 1
for base in vgg19 mobilenet resnet50 inception; do
  src/run_one.sh "TLFEAT_$base" python src/tl_features.py "$base"
  src/run_one.sh "TL_${base}_3_resized_s7" python src/tl.py "$base"
done
echo "$(date +%F_%T) | TL queue | DONE | 0s" >> logs/status.txt
