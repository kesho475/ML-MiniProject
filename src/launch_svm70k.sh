#!/usr/bin/env bash
# Optional svm70k queue (CPU, data-size check): same 8 SVM runs on the first 70,000 drawings per class.
# Extra rows with family "svm70k"; the main 20k SVM results are untouched.
# RAM: measured peak RSS ~2.3 GB per 3-class job and ~7.5 GB per 10-class job, so 4 at a time
# for 3 classes and 2 at a time for 10 classes (<= ~15 GB, leaving room for XGBoost on 50 classes).
# A crash only skips that job (see run_one.sh). Rerun this script to resume: finished runs are skipped.
cd "$(dirname "$0")/.." || exit 1
run() { xargs -P "$1" -L1 bash -c 'src/run_one.sh "svm70k_$0_$1_scaled_s0" python src/svm.py $0 $1 --70k' || true; }
printf '%s\n' "linear 3" "rbf 3" "poly 3" "sigmoid 3" | run 4
printf '%s\n' "rbf 10" "poly 10" "linear 10" "sigmoid 10" | run 2
echo "$(date +%F_%T) | svm70k queue | DONE | 0s" >> logs/status.txt
