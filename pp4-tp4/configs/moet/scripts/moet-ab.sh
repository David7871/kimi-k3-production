#!/bin/bash
# Moet MVP A/B: flock-queued lane window.
# (a) single-GPU op test -> (b) moet-on arm + bench + probe + GSM8K ->
# (d) moet-off control arm (same moet-mvp image, flag off) + same bench/GSM8K.
# Control uses the moet-mvp image with VLLM_K3_MOET unset so the ONLY delta
# between arms is the flag (flag-off path is byte-identical to :local).
set -uo pipefail
exec 9>/tmp/k3-lane.lock
flock 9 || { echo "LOCK FAIL"; exit 1; }
mkdir -p ~/k3-results
LOG=~/k3-results/moet-ab.log
log() { echo "[moet-ab $(date -Is)] $*" | tee -a "$LOG"; }
log "lock acquired"
cleanup() {
  docker rm -f kimi-k3-qsrt-pp4 >/dev/null 2>&1
  ssh -o BatchMode=yes -o ConnectTimeout=20 10.10.10.2 \
    'docker rm -f k3-qsrt-ray-worker' >/dev/null 2>&1
  log "cleanup done"
}
trap cleanup EXIT
INT6=$(cat ~/int6.envstr)
MOETENV="VLLM_K3_MOET=1 VLLM_K3_MOET_PACKS=/data/models/k3-moet-packs VLLM_MOE_W2=1 VLLM_MOE_W2_CUBIT_DIR=/opt/moet/cubins VLLM_MOE_W2_KS=3584,768 VLLM_MOE_W2_CKPT_ID=k3-mxfp4-moet-v1 VLLM_MOE_W2_NUM_LAYERS=93 VLLM_MOE_W2_DELTA=0"
IMG=kimi-k3-up-ray:moet-mvp

# ---------- (a) single-GPU op test ----------
log "(a) op test on GPU 0"
docker run --rm --gpus '"device=0"' --network none \
  --entrypoint /opt/venv/bin/python \
  -v /data/models:/data/models -v /data/models/k3-moet-work:/work \
  $(for kv in $MOETENV; do echo -n "-e $kv "; done) \
  $IMG /work/moet_op_test.py --layer 1 --rank 0 \
  > /tmp/moet-op.log 2>&1
RC=$?
tail -10 /tmp/moet-op.log | tee -a "$LOG"
if [ $RC -ne 0 ]; then log "OP TEST CRASH rc=$RC - aborting"; exit 1; fi
WORST=$(grep -oP 'kernel_rel_rms=\K[0-9.e-]+' /tmp/moet-op.log | sort -g | tail -1)
log "op test worst rel_rms=$WORST"
python3 -c "import sys; sys.exit(0 if float('$WORST') < 0.05 else 1)" \
  || { log "OP TEST kernel_rel_rms $WORST >= 0.05 - aborting"; exit 1; }

probe391() {
  local tag=$1
  curl -s -m 300 http://127.0.0.1:8001/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"Kimi-K3-QSRT-K2","messages":[{"role":"user","content":"What is 17*23? Think step by step, then give the final number."}],"temperature":0,"max_tokens":2048}' \
    > ~/k3-results/"$tag"/probe391.json
  python3 - "$tag" <<'EOF' | tee -a "$LOG"
import json, os, sys
tag = sys.argv[1]
d = json.load(open(os.path.expanduser(f"~/k3-results/{tag}/probe391.json")))
m = d["choices"][0]["message"]
txt = (m.get("content") or "") + (m.get("reasoning_content") or "")
print(f"[{tag}] probe391: {'PASS' if '391' in txt else 'FAIL'} :: {txt[-120:]!r}")
EOF
}

kv_info() {
  local tag=$1
  docker logs kimi-k3-qsrt-pp4 2>&1 \
    | grep -E "KV cache size|Maximum concurrency|moet|moe_w2|pack" | tail -12 \
    | tee -a "$LOG"
}

# ---------- (b) flag-ON arm ----------
log "(b) moet-on arm boot $(date -Is)"
ON_OK=0
if EXEC=up2l3 UP_IMAGE=$IMG \
   bash ~/k3-moet/k3-arm-moet.sh moet-on "$INT6 $MOETENV" "16,16" \
   >> "$LOG" 2>&1; then
  ON_OK=1
  log "moet-on arm done; probe391 + KV/pack evidence:"
  probe391 moet-on
  kv_info moet-on
  log "(c) GSM8K-200 flag-on"
  python3 ~/k3-moet/k3_gsm8k.py 8001 200 ~/k3-results/moet-on/gsm8k.jsonl 8 \
    2>&1 | tail -2 | tee -a "$LOG"
else
  log "MOET-ON ARM FAILED (boot OOM or crash) - recorded, continuing to (d)"
  docker logs kimi-k3-qsrt-pp4 > ~/k3-results/moet-on-head-crash.log 2>&1
  tail -20 ~/k3-results/moet-on-head-crash.log | tee -a "$LOG"
fi
cleanup

# ---------- (d) flag-OFF control (same image, flag unset) ----------
log "(d) moet-off control arm boot $(date -Is)"
if EXEC=up2l3 UP_IMAGE=$IMG \
   bash ~/k3-moet/k3-arm-moet.sh moet-off "$INT6" "16,16" \
   >> "$LOG" 2>&1; then
  probe391 moet-off
  kv_info moet-off
  log "GSM8K-200 flag-off"
  python3 ~/k3-moet/k3_gsm8k.py 8001 200 ~/k3-results/moet-off/gsm8k.jsonl 8 \
    2>&1 | tail -2 | tee -a "$LOG"
else
  log "MOET-OFF CONTROL FAILED"
fi
cleanup

# ---------- McNemar discordant counts ----------
if [ $ON_OK -eq 1 ] && [ -f ~/k3-results/moet-on/gsm8k.jsonl ] \
   && [ -f ~/k3-results/moet-off/gsm8k.jsonl ]; then
  python3 - <<'EOF' | tee -a "$LOG"
import json, os
on = [json.loads(l)["ok"] for l in open(
    os.path.expanduser("~/k3-results/moet-on/gsm8k.jsonl"))]
off = [json.loads(l)["ok"] for l in open(
    os.path.expanduser("~/k3-results/moet-off/gsm8k.jsonl"))]
n = min(len(on), len(off))
b = sum(1 for i in range(n) if on[i] and not off[i])   # on-only correct
c = sum(1 for i in range(n) if off[i] and not on[i])   # off-only correct
print(f"McNemar: n={n} on_only={b} off_only={c} "
      f"both={sum(1 for i in range(n) if on[i] and off[i])} "
      f"neither={sum(1 for i in range(n) if not on[i] and not off[i])}")
EOF
fi
log "=== MOET-AB DONE $(date -Is) ==="
