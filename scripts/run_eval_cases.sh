#!/bin/bash
# Live runs of the eval cases, one at a time, each only when Studio load1 < LOAD_MAX (ORCH gate).
# Pulse line every wait tick; per-case timeout; stops after 2 failures in a row. Log: runs/eval-live.log
cd "$(dirname "$0")/.."
P="/Users/oskolamicheal/Projects/Retailbox - Meta AI Developer Hackathon"
MEDIA="$P/code/discovery-clock/data/media"
VE="$P/docs/mikhail-cg1/tools/vault_env.py"
LOAD_MAX=${LOAD_MAX:-14}
LOG=runs/eval-live.log
ts(){ date -u +%H:%M:%SZ; }
fails=0
for c in "$@"; do
  id="${c##*-}"; [ "$c" = "motorcycle-r-ub7DZN8BA" ] && id="r-ub7DZN8BA"
  [ -f "runs/$c/result.json" ] && { echo "$(ts) SKIP $c has result" >> $LOG; continue; }
  w=0
  while :; do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    awk -v x="$l" -v m="$LOAD_MAX" 'BEGIN{exit !(x<m)}' && break
    echo "$(ts) PULSE $c waiting load1=$l" >> $LOG; sleep 30; w=$((w+1))
    [ $w -ge 60 ] && { echo "$(ts) ERR load gate timeout $c" >> $LOG; exit 7; }
  done
  echo "$(ts) START $c load1=$l" >> $LOG
  VAULT_EGRESS_OK=api.meta.ai timeout 1200 python3 "$VE" meta-hackathon-mikhail META_MODEL_API_KEY:MODEL_API_KEY META_MODEL_API_BASE_URL:META_MODEL_API_BASE_URL -- \
    env VAULT_EGRESS_OK=inference.baseten.co python3 "$VE" meta-ai-hackathon-baseten BASETEN_API_KEY_MIKHAIL:BASETEN_API_KEY -- \
    env DC_CAP_USD=${DC_CAP_USD:-8} .venv/bin/python -m discovery_clock run --case "$c" --video "$MEDIA/$id.mp4" --report "eval/cases/$c/claims.txt" --claims lines >> runs/eval-live-$c.out 2>&1
  rc=$?
  if [ -f "runs/$c/result.json" ]; then echo "$(ts) OK $c rc=$rc" >> $LOG; fails=0
  else echo "$(ts) FAIL $c rc=$rc (see runs/eval-live-$c.out)" >> $LOG; fails=$((fails+1)); [ $fails -ge 2 ] && { echo "$(ts) ERR 2 fails in a row" >> $LOG; exit 8; }; fi
done
echo "$(ts) DONE $*" >> $LOG
