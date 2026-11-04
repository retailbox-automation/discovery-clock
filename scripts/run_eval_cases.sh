#!/bin/bash
# Live runs of the evaluation cases, one at a time.
# Needs MODEL_API_KEY (Meta Model API) in the environment; BASETEN_API_KEY is optional.
# Videos are read from data/media/<video id>.mp4 (scripts/fetch_media.py downloads them).
# Usage: scripts/run_eval_cases.sh fisherman-fy6dRGapw_4 housefire-phHtxd9Ln3Q ...
# Log: runs/eval-live.log; per-case output: runs/eval-live-<case>.out
cd "$(dirname "$0")/.."
: "${MODEL_API_KEY:?set MODEL_API_KEY (Meta Model API key)}"
MEDIA=${MEDIA:-data/media}
LOG=runs/eval-live.log
mkdir -p runs
ts(){ date -u +%H:%M:%SZ; }
fails=0
for c in "$@"; do
  id=$(python3 -c "import json,sys; print(json.load(open('eval/cases/$c/key.json'))['video_id'])")
  [ -f "runs/$c/result.json" ] && { echo "$(ts) SKIP $c has result" >> $LOG; continue; }
  echo "$(ts) START $c" >> $LOG
  timeout 1200 env DC_CAP_USD=${DC_CAP_USD:-8} .venv/bin/python -m discovery_clock run --case "$c" \
    --video "$MEDIA/$id.mp4" --report "eval/cases/$c/claims.txt" --claims lines >> "runs/eval-live-$c.out" 2>&1
  rc=$?
  if [ -f "runs/$c/result.json" ]; then echo "$(ts) OK $c rc=$rc" >> $LOG; fails=0
  else echo "$(ts) FAIL $c rc=$rc (see runs/eval-live-$c.out)" >> $LOG; fails=$((fails+1)); [ $fails -ge 2 ] && { echo "$(ts) ERR 2 fails in a row" >> $LOG; exit 8; }; fi
done
echo "$(ts) DONE $*" >> $LOG
