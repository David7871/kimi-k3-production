#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/g0san/kimi-k3-production
COMPOSE=compose.tp8-dflash-fine3072.yml
TS=$(TZ=Asia/Seoul date +%Y%m%d-%H%M%S-KST)
OUT="$ROOT/logs/shmfix-cutover-$TS"
ASKPASS=/tmp/kimi-shmfix-askpass.sh
API_KEY='sk-v14-CHANGE_ME'
mkdir -p "$OUT"
exec > >(tee -a "$OUT/cutover.log") 2>&1

gateway_started=0
cleanup() {
  rc=$?
  if [[ $gateway_started -eq 0 ]]; then
    echo "Restoring llmconduit-kimi after cutover exit rc=$rc"
    SUDO_ASKPASS="$ASKPASS" sudo -A systemctl start llmconduit-kimi.service || true
  fi
  rm -f "$ASKPASS"
  exit "$rc"
}
trap cleanup EXIT

cd "$ROOT"
echo "CUTOVER_START $(TZ=Asia/Seoul date --iso-8601=seconds)"
chmod 700 "$ASKPASS"
python3 -m pytest tests/test_lmcache_shm_preflight.py -q
python3 -m py_compile bin/lmcache_shm_preflight.py
bash -n bin/entrypoint-issue75-remote-dflash-k7-dcp8.sh
docker compose -f "$COMPOSE" config --quiet

# Gate new traffic before sampling idleness.
SUDO_ASKPASS="$ASKPASS" sudo -A systemctl stop llmconduit-kimi.service
for sample in 1 2 3 4 5; do
  curl -fsS http://127.0.0.1:8090/metrics > "$OUT/idle-$sample.metrics"
  python3 - "$OUT/idle-$sample.metrics" <<'PY'
import sys
text=open(sys.argv[1]).read().splitlines()
def metric(prefix):
    return sum(float(line.rsplit(' ',1)[1]) for line in text if line.startswith(prefix) and '{' in line)
r=metric('vllm:num_requests_running')
w=metric('vllm:num_requests_waiting')
if r != 0 or w != 0:
    raise SystemExit(f'not idle: running={r} waiting={w}')
print(f'idle running={r} waiting={w}')
PY
  sleep 2
done

docker logs --timestamps kimi-k3-production > "$OUT/pre-recreate-target.log" 2>&1
docker inspect kimi-k3-production > "$OUT/pre-recreate-target.inspect.json"
docker inspect kimi-k3-draft > "$OUT/pre-recreate-draft.inspect.json"

echo "Recreating target only; draft remains live"
docker compose -f "$COMPOSE" up -d --force-recreate kimi-k3-production
new_id=$(docker inspect kimi-k3-production --format '{{.Id}}')
echo "new_target_id=$new_id"

ready=0
for attempt in $(seq 1 120); do
  state=$(docker inspect kimi-k3-production --format '{{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}}' 2>/dev/null || echo missing)
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8090/health || true)
  echo "readiness attempt=$attempt state=$state health=$code"
  if [[ "$code" == 200 ]]; then ready=1; break; fi
  if [[ "$state" == missing || "$state" == exited* || "$state" == dead* ]]; then
    docker logs --timestamps kimi-k3-production > "$OUT/failed-target.log" 2>&1 || true
    raise_message="target exited before readiness"
    echo "$raise_message"
    exit 20
  fi
  sleep 10
done
if [[ $ready -ne 1 ]]; then
  docker logs --timestamps kimi-k3-production > "$OUT/readiness-timeout.log" 2>&1 || true
  echo "target readiness timeout"
  exit 21
fi

# Require complete LMCache registration and direct-SHM transport.
for attempt in $(seq 1 30); do
  if curl -fsS --max-time 5 http://127.0.0.1:8088/status > "$OUT/lmcache-status.json"; then
    if python3 - "$OUT/lmcache-status.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert s['is_healthy'] is True
assert s['storage_manager']['is_healthy'] is True
assert s['storage_manager']['l1_manager']['memory_total_bytes'] == 48 * 2**30
assert s['worker_liveness']['tracked_instances'] == 8
assert len(s['registered_non_cuda_instance_ids']) == 8
print('LMCACHE_STATUS_OK 8/8 direct contexts, 48 GiB L1')
PY
    then break; fi
  fi
  [[ $attempt -eq 30 ]] && { echo "LMCache registration timeout"; exit 22; }
  sleep 5
done

docker logs --timestamps kimi-k3-production > "$OUT/post-recreate-target.log" 2>&1
python3 - "$OUT/post-recreate-target.log" <<'PY'
import sys
text=open(sys.argv[1],errors='replace').read()
created=text.count('Creating EngineDrivenContextShm')
pinned=text.count('SHM pinned=True')
registered=text.count('mode=SHM, groups=4')
pickle=text.count('Using pickle non-GPU transfer strategy')
insufficient=text.count('Insufficient /dev/shm capacity')
capacity=text.count('CAPACITY_OK')
print(
    f'log markers: created={created} pinned={pinned} registered={registered} '
    f'pickle={pickle} insufficient={insufficient} capacity={capacity}'
)
assert created == 8, f'expected 8 SHM contexts, got {created}'
assert pinned == 8, f'expected 8 pinned SHM contexts, got {pinned}'
assert registered == 8, f'expected 8 SHM registrations, got {registered}'
assert pickle == 0, f'pickle fallback active: {pickle}'
assert insufficient == 0, 'insufficient SHM fallback occurred'
assert capacity >= 1, 'entrypoint capacity guard missing'
PY

pool=/dev/shm/lmcache_l1_pool_kimik3_remote9g_dflash3_fine3072_l1_48g_v1
python3 - "$pool" <<'PY'
import os,sys
p=sys.argv[1]
s=os.stat(p)
assert s.st_size == 48 * 2**30, (p,s.st_size)
print(f'SHM_POOL_OK path={p} size={s.st_size}')
PY

docker inspect kimi-k3-production --format 'status={{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}} started={{.State.StartedAt}}' | tee "$OUT/post-state.txt"

cat > "$OUT/direct-smoke.json" <<'JSON'
{"model":"Kimi-K3","messages":[{"role":"user","content":"Return exactly POST_SHM_FIX_OK and nothing else."}],"temperature":0,"seed":0,"max_tokens":512,"stream":false,"cache_salt":"post-shm-fix-cold-20260829"}
JSON
curl -fsS --max-time 180 -H 'Content-Type: application/json' --data-binary @"$OUT/direct-smoke.json" http://127.0.0.1:8090/v1/chat/completions > "$OUT/direct-smoke-response.json"
python3 - "$OUT/direct-smoke-response.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); c=d['choices'][0]; m=c['message']
assert c['finish_reason']=='stop', c
assert m.get('content')=='POST_SHM_FIX_OK', m
print('DIRECT_SMOKE_OK')
PY

# Restore the gateway and verify the authenticated model catalog.
SUDO_ASKPASS="$ASKPASS" sudo -A systemctl start llmconduit-kimi.service
gateway_started=1
for attempt in $(seq 1 30); do
  code=$(curl -sS -o "$OUT/gateway-models.json" -w '%{http_code}' --max-time 5 -H "Authorization: Bearer $API_KEY" http://127.0.0.1:5318/v1/models || true)
  [[ "$code" == 200 ]] && break
  [[ $attempt -eq 30 ]] && { echo "gateway readiness timeout code=$code"; exit 23; }
  sleep 2
done

free -h | tee "$OUT/free-after.txt"
swapon --show | tee "$OUT/swap-after.txt"
df -h /dev/shm | tee "$OUT/shm-after.txt"
docker stats --no-stream kimi-k3-production kimi-k3-draft | tee "$OUT/docker-stats-after.txt"
journalctl -k --since "$(TZ=Asia/Seoul date -d '20 minutes ago' '+%Y-%m-%d %H:%M:%S')" --no-pager > "$OUT/kernel-after.log"
if grep -qiE 'out of memory|oom-kill|killed process.*lmcache|nvrm.*xid' "$OUT/kernel-after.log"; then
  echo "fatal kernel marker after cutover"
  exit 24
fi

echo "CUTOVER_SUCCESS $(TZ=Asia/Seoul date --iso-8601=seconds) artifacts=$OUT"
