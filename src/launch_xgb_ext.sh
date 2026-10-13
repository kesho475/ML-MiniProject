#!/usr/bin/env bash
# Phase 8b queue (GPU, one run at a time): A1 -> B (xgb10_wide) -> C -> A2-1..3 -> final.
# Budget 11.5 h from this launch (results/xgb_extended/queue_start): enforced inside src/xgb_ext.py.
# One logs/status.txt line per step. A crash is logged FAILED(rc=N) and the step is retried once,
# resuming from its last checkpoint; then the queue goes on. Output of step X: logs/XGBX_X.log.
# Usage: src/launch_xgb_ext.sh [--smoke]   (smoke: separate folder, status in that folder)
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
smoke=$1
out=results/xgb_extended; status_file=logs/status.txt; logdir=logs
if [ "$smoke" = --smoke ]; then out=results/xgb_extended-smoke; status_file=$out/status.txt; logdir=$out/logs; fi
mkdir -p "$out" "$logdir"
date +%s > "$out/queue_start"
line() { echo "$(date +%F_%T) | XGBX_$1 | $2 | ${3:-0}s" >> "$status_file"; }

step() {  # step <name> <xgb_ext.py args...>
  name=$1; shift
  for try in 1 2; do
    start=$SECONDS
    python -u src/xgb_ext.py "$@" $smoke 2>&1 | tee -a "$logdir/XGBX_$name.log"
    rc=${PIPESTATUS[0]}
    t=$((SECONDS - start))
    if [ "$rc" = 0 ]; then line "$name" "$(cat "$out/status/$name" 2>/dev/null || echo OK)" $t; return; fi
    line "$name" "FAILED(rc=$rc, try $try)" $t
    { echo "== XGBX_$name rc=$rc $(date +%F_%T)"; tail -n 8 "$logdir/XGBX_$name.log"; } >> "$logdir/errors.log"
  done
}

step A1 run A1
step B wide
step C run C
for i in 1 2 3; do step "A2-$i" run "A2-$i"; done
step final final
echo "$(date +%F_%T) | XGB-extended queue | DONE | 0s" >> "$status_file"
