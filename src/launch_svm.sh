#!/usr/bin/env bash
# SVM queue (CPU): 4 kernels x {3,10} classes, 4 at a time, 3-class jobs first.
# A crash only skips that job (see run_one.sh). Rerun this script to resume: finished runs are skipped.
cd "$(dirname "$0")/.." || exit 1
printf '%s\n' "linear 3" "rbf 3" "poly 3" "sigmoid 3" "linear 10" "rbf 10" "poly 10" "sigmoid 10" |
  xargs -P 4 -L1 bash -c 'src/run_one.sh "SVM_$0_$1_scaled_s0" python src/svm.py $0 $1' || true
echo "$(date +%F_%T) | SVM queue | DONE | 0s" >> logs/status.txt
