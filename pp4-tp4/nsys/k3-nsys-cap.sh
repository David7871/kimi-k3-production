#!/usr/bin/env bash
# Fail-fast 16-rank nsys capture for the K3 PP4xTP4 lane.
# Usage: k3-nsys-cap.sh TAG OUTDIR CONC MNS "ENV_KV..."
# Every phase has a hard deadline and a crash watchdog; prints PHASE/FAIL/OK lines to grep.
set -u
TAG=$1 OUT=$HOME/$2 CONC=$3 MNS=$4 ENVKV=$5
H=kimi-k3-qsrt-pp4 W=k3-qsrt-ray-worker R2=10.10.10.2
LOG=$HOME/$TAG.cap.log; exec > >(tee -a "$LOG") 2>&1
ph() { echo "PHASE $(date -u +%T) $*"; }
fail() { echo "FAIL $(date -u +%T) $*"; echo "containers kept for inspection; clean: docker rm -f $H $W (both hosts)"; exit 1; }
crashed() {  # engine dead / CUDA fault in the last N seconds of head logs, or GPU Xid on either host
  docker logs --since "${1:-30}s" $H 2>&1 | grep -qE "illegal memory|EngineDeadError|Engine core.*died|SIGABRT|NCCL error" && return 0
  for h in localhost $R2; do ssh -o BatchMode=yes $h "sudo -n dmesg --since '-${1:-30}s' 2>/dev/null | grep -q 'NVRM: Xid'" && return 0; done
  return 1
}
reps() { docker exec $H bash -c "ls /tmp/nsys-root/*.nsys-rep 2>/dev/null | wc -l"; }
reps2() { ssh $R2 "docker exec $W bash -c 'ls /tmp/nsys-root/*.nsys-rep 2>/dev/null | wc -l'"; }

ph clean
docker rm -f $H $W >/dev/null 2>&1; ssh $R2 "docker rm -f $H $W" >/dev/null 2>&1; sleep 5

ph boot   # k3-arm.sh boots + single-request probe; hard 15 min cap
EXEC=up2l3 timeout 900 bash /data/models/k3-run/scripts/k3-arm.sh $TAG \
  "$ENVKV K3_MNS=$MNS K3_EXTRA_ARGS_CSV=--ray-workers-use-nsight,--profiler-config.profiler=cuda" 1 \
  || fail "boot/probe failed or >15 min"
docker exec $H rm -rf /tmp/nsys-root; ssh $R2 docker exec $W rm -rf /tmp/nsys-root

ph load   # warm load; abort within 10 s of any crash
(cd /data/models/k3-run/overlays-mb && timeout $((300 + ${PROFILE_SEC%.*})) python3 decode_bench.py 8001 $CONC $((240 + ${PROFILE_SEC%.*})) 10 > $HOME/$TAG-bench.txt 2>&1 &)
T0=$SECONDS; while [ $((SECONDS - T0)) -lt 30 ]; do sleep 1; crashed 15 && fail "engine crashed under C$CONC load (see docker logs $H)"; done
curl -m 5 -sf localhost:8001/health >/dev/null || fail "server unhealthy before profiling"

ph profile
curl -m 60 -sf -XPOST localhost:8001/start_profile >/dev/null || fail "start_profile timed out/failed"
sleep ${PROFILE_SEC:-1.5}
curl -m 60 -sf -XPOST localhost:8001/stop_profile >/dev/null || fail "stop_profile timed out/failed"

ph flush  # wait for all 16 reports; sessions stuck in RangeCollection never got stop -> force-stop them
for i in $(seq 1 60); do
  n1=$(reps); n2=$(reps2); echo "reps rtx-01=$n1 rtx-02=$n2"
  [ "$n1" -ge 8 ] && [ "$n2" -ge 8 ] && break
  crashed 30 && fail "crash during flush"
  if [ $i -eq 9 ]; then   # 90 s: anything still collecting missed the stop -> nsys stop it
    for c in "docker exec $H" "ssh $R2 docker exec $W"; do
      $c bash -c 'for s in $(nsys sessions list | awk "/RangeCollection/{print \$1}"); do echo force-stop $s; nsys stop --session=$s >/dev/null 2>&1 & done; wait'
    done
  fi
  sleep 10
done
[ "$(reps)" -ge 8 ] && [ "$(reps2)" -ge 8 ] || fail "reports incomplete after 10 min: rtx-01=$(reps) rtx-02=$(reps2)"
mkdir -p $OUT.logs; docker logs $H > $OUT.logs/head.log 2>&1
docker exec $H bash -c "nsys sessions list" > $OUT.logs/sessions-rtx01.txt 2>&1
ssh $R2 "docker exec $W bash -c 'nsys sessions list'" > $OUT.logs/sessions-rtx02.txt 2>&1

ph collect
rm -rf $OUT; mkdir -p $OUT/rtx-01 $OUT/rtx-02
docker cp $H:/tmp/nsys-root/. $OUT/rtx-01/ >/dev/null
ssh $R2 "rm -rf /tmp/nsx; docker cp $W:/tmp/nsys-root /tmp/nsx" >/dev/null && rsync -a $R2:/tmp/nsx/ $OUT/rtx-02/
docker run --rm --entrypoint bash -v $OUT:/n kimi-k3-up-ray:local -c \
  'for f in /n/*/*.nsys-rep; do nsys export -t sqlite --force-overwrite true -o ${f%.nsys-rep}.sqlite $f >/dev/null & done; wait'

ph verify  # 16 sqlite with kernels, all overlapping in time; only then remove containers
python3 - "$OUT" "${PROFILE_SEC:-1.5}" <<'EOF' || fail "verification failed (containers kept)"
import sqlite3, glob, sys
sp = []
for f in sorted(glob.glob(sys.argv[1] + "/*/*.sqlite")):
    c = sqlite3.connect(f)
    try:
        t0 = c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
        a, b, n = c.execute("select min(start), max(end), count(*) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
        sp.append((t0 + a, t0 + b)); print(f.split("/")[-2], "kernels", n)
    except Exception as e: print("NO KERNELS", f, e)
lo, hi = max(s for s, _ in sp), min(e for _, e in sp)
print("ranks_with_kernels", len(sp), "common_overlap_ms %.1f" % ((hi - lo) / 1e6))
need = 0.8 * float(sys.argv[2]) * 1e9  # all 16 ranks must overlap for >=80% of the requested profile length
print("required_overlap_ms %.0f" % (need / 1e6))
sys.exit(0 if len(sp) == 16 and hi - lo >= need else 1)
EOF
docker rm -f $H $W >/dev/null; ssh $R2 "docker rm -f $H $W" >/dev/null
echo "OK $(date -u +%T) $TAG -> $OUT"
