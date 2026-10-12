#!/usr/bin/env bash
# Run one job so a failure never stops a queue.  Usage: run_one.sh <run_id> <cmd...>
# Skips finished runs (results/runs/<id>.json). Appends one line to logs/status.txt:
#   <time> | <run_id> | OK / FAILED(rc=N) / COLLAPSED / SKIPPED | <seconds>
# Exit code = the job's (3 = CNN collapse); stderr goes to logs/<id>.err, last lines to logs/errors.log.
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
id=$1; shift
status() { echo "$(date +%F_%T) | $id | $1 | ${2:-0}s" >> logs/status.txt; }
if [ -f "results/runs/$id.json" ]; then status SKIPPED; exit 0; fi
start=$SECONDS
"$@" 2>> "logs/$id.err"
rc=$?
t=$((SECONDS - start))
case $rc in
  0) status OK $t ;;
  3) status COLLAPSED $t ;;
  *) status "FAILED(rc=$rc)" $t
     { echo "== $id rc=$rc $(date +%F_%T)"; tail -n 8 "logs/$id.err"; } >> logs/errors.log ;;
esac
exit $rc
