#!/bin/bash
# K3 PP4xTP4 stage-rebalance A/B. Run on rtx-01 inside tmux, under flock.
# A = default split [23,23,24,23] (no VLLM_PP_LAYER_PARTITION)
# B = [24,24,24,21]  (offload S3 by 2 layers; +16.7% KV on S3 per prior CPU work)
# Optional C = [23,24,24,22] (byte-balanced) if ARGC passed as 3rd variant env.
# Warm median-of-3 at C16 and C256: CONC list repeats each conc 3x after k3-arm's
# built-in warm pass (C4) which serves as the discard bench.
set -uo pipefail
exec 9>/tmp/k3-lane.lock
flock 9 || { echo "LOCK FAIL"; exit 1; }
echo "[lock] acquired $(date -Is)"

cleanup() {
  docker rm -f kimi-k3-qsrt-pp4 >/dev/null 2>&1
  ssh -o BatchMode=yes -o ConnectTimeout=20 10.10.10.2 'docker rm -f k3-qsrt-ray-worker' >/dev/null 2>&1
  echo "[cleanup] containers removed $(date -Is)"
}
trap cleanup EXIT

INT6=$(cat ~/int6.envstr)
CONCS="16,16,16,256,256,256"

runv() {  # TAG PARTITION(or empty)
  local TAG=$1 PART=$2
  local ENV="$INT6"
  if [ -n "$PART" ]; then ENV="VLLM_PP_LAYER_PARTITION=$PART $ENV"; fi
  echo "=== variant $TAG part='${PART:-default}' $(date -Is) ==="
  if ! EXEC=up2l3 bash /data/models/k3-run/scripts/k3-arm.sh "$TAG" "$ENV" "$CONCS"; then
    echo "[$TAG] ARM FAILED"
    return 1
  fi
  # correctness probe: 17*23=391 (chat endpoint; reasoning may hold the answer)
  curl -s -m 180 http://127.0.0.1:8001/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"Kimi-K3-QSRT-K2","messages":[{"role":"user","content":"What is 17*23? Think step by step, then give the final number."}],"max_tokens":2000,"temperature":0}' \
    > ~/k3-results/"$TAG"/probe391.json
  python3 - "$TAG" <<'EOF'
import json,os,sys
tag=sys.argv[1]
d=json.load(open(os.path.expanduser(f"~/k3-results/{tag}/probe391.json")))
m=d["choices"][0]["message"]
txt=(m.get("content") or "")+(m.get("reasoning_content") or "")
print(f"[{tag}] probe391: {'PASS' if '391' in txt else 'FAIL'} :: {txt[-120:]!r}")
EOF
  # per-variant KV capacity + partition log lines
  docker logs kimi-k3-qsrt-pp4 2>&1 | grep -E "KV cache size|Maximum concurrency|unevenly partitioned|layer_partition|PP_LAYER" | tail -8 > ~/k3-results/"$TAG"/kvlines.txt
  cleanup
  sleep 5
}



runv rb-c "23,24,24,22"
echo "=== ALL DONE $(date -Is) ==="
