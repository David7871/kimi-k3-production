#!/bin/bash
# Moet follow-up batch, one lane-lock hold, all arms on image moet-mvp2:
#   fu-off    flag unset (same-image control)
#   fu-on     Moet 2-bit (MVP env)
#   fu-rb     Moet + VLLM_PP_LAYER_PARTITION=24,24,24,21 (KV capacity)
#   (the live script also ran fu-topp95/90/80 expert-top-p arms; those arms were
#    dropped — incomplete probes — and are removed from this export)
# Per arm: C16 x3, then C256 x3 bracketed by /metrics snapshots of
# vllm:iteration_tokens_total (tokens per scheduled step = per PP micro-batch),
# probe391, GSM8K-200 with completion-token counts (skipped for fu-rb).
# Run on rtx-01:  tmux new -d -s moetfu 'bash ~/k3-moet/moet-followup.sh'
set -uo pipefail
exec 9>/tmp/k3-lane.lock
flock 9 || { echo "LOCK FAIL"; exit 1; }
R=~/k3-results; LOG=$R/moet-followup.log; mkdir -p $R
log() { echo "[moet-fu $(date -Is)] $*" | tee -a "$LOG"; }
cleanup() {
  docker rm -f kimi-k3-qsrt-pp4 >/dev/null 2>&1
  ssh -o BatchMode=yes -o ConnectTimeout=20 10.10.10.2 \
    'docker rm -f k3-qsrt-ray-worker' >/dev/null 2>&1
}
trap 'cleanup; log "cleanup (exit)"' EXIT
INT6=$(cat ~/int6.envstr)
MOETENV="VLLM_K3_MOET=1 VLLM_K3_MOET_PACKS=/data/models/k3-moet-packs VLLM_MOE_W2=1 VLLM_MOE_W2_CUBIT_DIR=/opt/moet/cubins VLLM_MOE_W2_KS=3584,768 VLLM_MOE_W2_CKPT_ID=k3-mxfp4-moet-v1 VLLM_MOE_W2_NUM_LAYERS=93 VLLM_MOE_W2_DELTA=0"
IMG=kimi-k3-up-ray:moet-mvp2
BENCH=/data/models/k3-run/overlays-mb/decode_bench.py
itok() { curl -s -m 10 http://127.0.0.1:8001/metrics | grep -E '^vllm:iteration_tokens_total_(bucket|sum|count)' > "$1"; }
probe391() {
  curl -s -m 300 http://127.0.0.1:8001/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"Kimi-K3-QSRT-K2","messages":[{"role":"user","content":"What is 17*23? Think step by step, then give the final number."}],"max_tokens":2048,"temperature":0}' \
    > "$R/$1/probe391.json"
  python3 - "$1" <<'EOF' | tee -a "$LOG"
import json, os, sys
t = sys.argv[1]
d = json.load(open(os.path.expanduser(f"~/k3-results/{t}/probe391.json")))
m = d["choices"][0]["message"]
s = (m.get("content") or "") + (m.get("reasoning_content") or "")
print(f"[{t}] probe391: {'PASS' if '391' in s else 'FAIL'} tokens={d.get('usage',{}).get('completion_tokens')}")
EOF
}
arm() {  # TAG ENV GSM(0/1)
  local tag=$1 env=$2 gsm=$3
  cleanup; sleep 5
  log "== $tag boot"
  if ! EXEC=up2l3 UP_IMAGE=$IMG bash ~/k3-moet/k3-arm-moet.sh "$tag" "$env" "16,16,16" >> "$LOG" 2>&1; then
    log "$tag FAILED (boot/bench)"; docker logs kimi-k3-qsrt-pp4 > "$R/$tag/head-fail.log" 2>&1; return 1
  fi
  itok "$R/$tag/itok-pre256.txt"
  for i in 1 2 3; do python3 $BENCH 8001 256 30 10 | tee -a "$R/$tag/decode.txt" "$LOG"; done
  itok "$R/$tag/itok-post256.txt"
  docker logs kimi-k3-qsrt-pp4 2>&1 | grep -E "KV cache size|Maximum concurrency|Model loading took|CUDA graph.*took|moe_w2_cubit: (PP|cubins)" | tail -10 | tee -a "$LOG"
  probe391 "$tag"
  if [ "$gsm" = 1 ]; then
    python3 ~/k3-moet/k3_gsm8k.py 8001 200 "$R/$tag/gsm8k.jsonl" 8 2>&1 | tail -1 | sed "s/^/[$tag] /" | tee -a "$LOG"
  fi
  docker logs kimi-k3-qsrt-pp4 > "$R/$tag/head.log" 2>&1
  log "== $tag done"
}
log "lock acquired; image $IMG $(docker image inspect $IMG --format '{{.Id}}')"
arm fu-off    "$INT6"                                        1
arm fu-on     "$INT6 $MOETENV"                               1
arm fu-rb     "$INT6 $MOETENV VLLM_PP_LAYER_PARTITION=24,24,24,21" 0
cleanup
python3 ~/k3-moet/moet_fu_summary.py | tee -a "$LOG"
log "=== MOET-FOLLOWUP DONE ==="
