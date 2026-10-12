#!/usr/bin/env bash
# Phase 8 queue (GPU): XGBoost + Optuna, stage 1 (10 classes) -> stage 2 (50) -> stage 3 (3).
# A crashed trial is recorded as FAIL inside the study and the search goes on; a crashed stage
# only skips that stage (see run_one.sh). One logs/status.txt line per stage.
# To rerun a failed stage, delete its study from results/optuna/xgb.db first (xgb.py starts clean).
cd "$(dirname "$0")/.." || exit 1
for s in "1 10" "2 50" "3 3"; do
  set -- $s
  src/run_one.sh "XGB_xgb_$2_raw_s0" python src/xgb.py "stage$1"
done
echo "$(date +%F_%T) | XGB queue | DONE | 0s" >> logs/status.txt
