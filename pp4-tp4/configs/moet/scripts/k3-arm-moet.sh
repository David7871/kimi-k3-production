#!/usr/bin/env bash
# K3 QSRT PP4xTP4 arm on rtx-01+rtx-02: launch worker+head -> ready -> constant-concurrency decode bench.
# Usage: k3-arm.sh TAG "ENV_KV ..." [CONC=1,4,8,16]
# ENV_KV like VLLM_PP_CPU_BROADCAST=0 K3_MNS=16 K3_EXTRA_ARGS=--pp-max-micro-batch-size=0 (no spaces in values)
set -uo pipefail
TAG=$1; KV=${2:-}; CONC=${3:-1,4,8,16}
S=/data/models/k3-run/scripts; OUT=$HOME/k3-results/$TAG; mkdir -p "$OUT"
log() { echo "[$TAG $(date -Is)] $*" | tee -a "$OUT/arm.log"; }
EA=""; for kv in $KV; do EA+="-e $kv "; done
log "env: $KV"
if [ "${EXEC:-mb}" = up ]; then ssh -o BatchMode=yes 10.10.10.2 "EXTRA_ENV_ARGS='$EA' bash $S/k3-up-launch.sh worker" >>"$OUT/arm.log" 2>&1; sleep 5; EXTRA_ENV_ARGS="$EA" bash $S/k3-up-launch.sh head >>"$OUT/arm.log" 2>&1; else
case ${EXEC:-mb} in up2l2) W=k3-qsrt-launch-worker-up2-l2.sh;; up2dsm) W=k3-qsrt-launch-worker-up2-dsmem.sh;; up2ima) W=k3-qsrt-launch-worker-up2-ima.sh;; up2imadsm) W=k3-qsrt-launch-worker-up2-imadsm.sh;; up2l3) W=k3-qsrt-launch-worker-up2-l3.sh;; *) W=k3-qsrt-launch-worker-${EXEC:-mb}.sh;; esac
ssh -o BatchMode=yes 10.10.10.2 "UP_IMAGE='${UP_IMAGE:-}' K3_L3V='${K3_L3V:-}' K3_L3B12X='${K3_L3B12X:-}' K3_L3_TWOSHOT_ROUTE='${K3_L3_TWOSHOT_ROUTE:-}' K3_L3_TWOSHOT_ROUTE_SRC='${K3_L3_TWOSHOT_ROUTE_SRC:-}' EXTRA_ENV_ARGS='$EA' bash $S/$W" >>"$OUT/arm.log" 2>&1 || { log "WORKER LAUNCH FAILED"; exit 1; }
sleep 10
case ${EXEC:-mb} in mp) H=k3-qsrt-launch-head-mp.sh;; up2) H=k3-qsrt-launch-head-nospec-up2.sh;; up2l2) H=k3-qsrt-launch-head-nospec-up2-l2.sh;; up2dsm) H=k3-qsrt-launch-head-nospec-up2-dsmem.sh;; up2ima) H=k3-qsrt-launch-head-nospec-up2-ima.sh;; up2imadsm) H=k3-qsrt-launch-head-nospec-up2-imadsm.sh;; up2l3) H=k3-qsrt-launch-head-nospec-up2-l3.sh;; nsys) H=k3-qsrt-launch-head-nospec-nsys.sh;; *) H=k3-qsrt-launch-head-nospec-mb.sh;; esac
EXTRA_ENV_ARGS="$EA" bash $S/$H >>"$OUT/arm.log" 2>&1 || { log "HEAD LAUNCH FAILED"; exit 1; }
fi
for i in $(seq 1 360); do
  docker ps --format '{{.Names}}' | grep -qx kimi-k3-qsrt-pp4 || { log "HEAD DIED"; docker logs kimi-k3-qsrt-pp4 >"$OUT/boot.log" 2>&1; tail -30 "$OUT/boot.log"; exit 1; }
  curl -sf -m 5 http://127.0.0.1:8001/v1/models | grep -q Kimi-K3 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8001/v1/models | grep -q Kimi-K3 || { log "READY TIMEOUT"; docker logs kimi-k3-qsrt-pp4 >"$OUT/boot.log" 2>&1; exit 1; }
log "ready"
docker logs kimi-k3-qsrt-pp4 2>&1 | grep -E "micro-batching|KV cache size|Maximum concurrency" | head -6 | tee -a "$OUT/arm.log"
# coherence probe
curl -s -m 120 http://127.0.0.1:8001/v1/completions -H 'content-type: application/json' \
  -d '{"model":"Kimi-K3-QSRT-K2","prompt":"The capital of France is","max_tokens":24,"temperature":0}' >"$OUT/probe.json"
log "probe: $(python3 -c 'import json;print(repr(json.load(open("'"$OUT"'/probe.json"))["choices"][0]["text"]))' 2>&1)"
# warm pass (graph/JIT), then measured
drain(){ for i in $(seq 120); do r=$(curl -s 127.0.0.1:8001/metrics | grep -oP "^vllm:num_requests_running\{[^}]*\} \K[0-9.]+"); [ "$r" = "0.0" ] && return; sleep 5; done; log "DRAIN TIMEOUT running=$r"; }
python3 /data/models/k3-run/overlays-mb/decode_bench.py 8001 4 10 5 >/dev/null 2>&1; drain
: >"$OUT/decode.txt"
for c in ${CONC//,/ }; do python3 /data/models/k3-run/overlays-mb/decode_bench.py 8001 "$c" 30 10 | tee -a "$OUT/decode.txt" | tee -a "$OUT/arm.log"; drain; done
docker logs kimi-k3-qsrt-pp4 >"$OUT/head.log" 2>&1
log "ARM_DONE"
