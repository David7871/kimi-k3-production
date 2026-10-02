#!/bin/bash
# 16-rank nsys timelines + interactive HTML for the Moet follow-up off/on arms,
# same method as nsys/timeline-c16-4s.html / timeline-c256-3.5s.html
# (long-c16, long-c256): k3-nsys-cap.sh, MNS256, PROFILE_SEC=4, then
# ~/nsys3/gen_html.py. All arms on image moet-mvp2 (both hosts).
# Captures: fuoff-c16, fuon-c16, fuoff-c256, fuon-c256
# (results/nsys-json/*.json in this export).
# Run on rtx-01:  tmux new -d -s moetnsys 'bash ~/k3-moet/moet-nsys.sh'
set -uo pipefail
R=~/k3-results; LOG=$R/moet-nsys.log; mkdir -p $R
log() { echo "[moet-nsys $(date -Is)] $*" | tee -a "$LOG"; }
exec 9>/tmp/k3-lane.lock
flock 9 || { log "LOCK FAIL"; exit 1; }
log "lock acquired"
cleanup() {
  docker rm -f kimi-k3-qsrt-pp4 >/dev/null 2>&1
  ssh -o BatchMode=yes -o ConnectTimeout=20 10.10.10.2 'docker rm -f k3-qsrt-ray-worker' >/dev/null 2>&1
}
cleanup
INT6=$(cat ~/int6.envstr)
MOETENV=$(sed -n 's/^MOETENV="\(.*\)"$/\1/p' ~/k3-moet/moet-followup.sh)
[ -n "$MOETENV" ] || { log "MOETENV not found"; exit 1; }
export UP_IMAGE=kimi-k3-up-ray:moet-mvp2 PROFILE_SEC=4
cap() {  # TAG CONC ENV ; up to 3 attempts: rtx-01 workers intermittently miss
         # stop_profile, get force-stopped, and their reports have no kernel table
  [ -s $R/timeline-$1.html ] && { log "$1 already done, skip"; return 0; }
  local a
  for a in 1 2 3; do
    log "== $1 (C$2) capture, attempt $a"
    if bash ~/k3-moet/k3-nsys-cap-moet.sh "$1" "nsys-$1" "$2" 256 "$3" > /dev/null 2>&1; then
      grep -E '^C[0-9]+ decode' ~/$1-bench.txt 2>/dev/null | tail -1 | sed "s/^/[$1] load during capture: /" | tee -a "$LOG"
      if (cd ~/nsys3 && python3 gen_html.py ~/nsys-$1 $R/timeline-$1.html \
            "K3 PP4xTP4 $1: C$2, 4 s, 16 ranks, moet-mvp2") >> "$LOG" 2>&1; then
        log "$1 OK -> $R/timeline-$1.html"
      else log "$1 HTML FAILED"; fi
      return 0
    fi
    log "$1 attempt $a FAILED: $(grep -E '^FAIL|ranks_with_kernels' ~/$1.cap.log | tail -2 | tr '\n' ' ')"
  done
  log "$1 capture FAILED after 3 attempts"
}
cap fuoff-c16   16  "$INT6"
cap fuon-c16    16  "$INT6 $MOETENV"
cap fuoff-c256  256 "$INT6"
cap fuon-c256   256 "$INT6 $MOETENV"
cleanup
log "=== MOET-NSYS DONE"
