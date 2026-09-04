#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT=/home/g0san/kimi-k3-production
RUNTIME="$ROOT/candidates/all-compatible-20260829/runtime"
BASE="$RUNTIME/compose.tp8-dflash-flashkda-320k-w4a8.yml"
MXFP8="$ROOT/research/mxfp8-entry-20260830/compose.mxfp8-entry.override.yml"
PROFILE="$RUNTIME/compose.mythos5-dflash-stock-fine1536.override.yml"
compose=(docker compose --project-name kimi-k3-production --project-directory "$RUNTIME" -f "$BASE" -f "$MXFP8" -f "$PROFILE")

[[ "${1:-}" == '--execute' ]] || {
  echo 'Refusing to switch without --execute' >&2
  exit 2
}

set -a
# Reuse the established local gateway credential without embedding it here.
# shellcheck disable=SC1091
source /home/g0san/glm53-r28/.env
set +a

stamp="$(TZ=Asia/Seoul date +%Y%m%d-%H%M%S-KST)"
logdir="$ROOT/logs/mythos5-fine512-cutover-$stamp"
mkdir -p "$logdir"

rollback_on_error() {
  rc=$?
  trap - ERR INT TERM
  echo "MYTHOS5_CUTOVER_FAILED rc=$rc; leaving GLM stopped as requested" >&2
  docker logs --timestamps kimi-k3-production >"$logdir/target-failure.log" 2>&1 || true
  docker logs --timestamps kimi-k3-draft >"$logdir/draft-failure.log" 2>&1 || true
  sudo -n systemctl stop llmconduit-kimi.service >/dev/null 2>&1 || true
  sudo -n systemctl stop llmconduit.service >/dev/null 2>&1 || true
  "${compose[@]}" stop -t 120 kimi-k3-production kimi-k3-draft >/dev/null 2>&1 || true
  exit "$rc"
}
trap rollback_on_error ERR INT TERM

"$ROOT/bin/preflight-mythos5-fine512.sh"
docker inspect glm53-r28 >"$logdir/glm-before.json"
docker inspect kimi-k3-production >"$logdir/kimi-target-before.json"
docker inspect kimi-k3-draft >"$logdir/kimi-draft-before.json"

echo 'Stopping ingress while the Kimi backend is recreated'
sudo -n systemctl stop llmconduit-kimi.service
sudo -n systemctl stop llmconduit.service
if [[ "$(docker inspect glm53-r28 --format '{{.State.Status}}' 2>/dev/null || true)" == running ]]; then
  echo 'Draining and stopping GLM model'
  idle=0
  for attempt in $(seq 1 120); do
    metrics="$(curl -fsS --max-time 5 http://127.0.0.1:5317/metrics)"
    busy="$(awk '$1 ~ /^vllm:num_requests_(running|waiting)(\{|$)/ {sum += $NF} END {print sum + 0}' <<<"$metrics")"
    if awk -v value="$busy" 'BEGIN {exit !(value == 0)}'; then idle=1; break; fi
    (( attempt % 12 == 0 )) && echo "GLM_DRAIN busy=$busy elapsed=$((attempt * 5))s"
    sleep 5
  done
  [[ "$idle" == 1 ]] || { echo 'GLM did not drain within 10 minutes' >&2; false; }
  docker stop -t 120 glm53-r28 >/dev/null
fi

echo 'Starting original DFlash draft and Kimi target'
"${compose[@]}" up -d --force-recreate --no-build --pull never

draft_ready=0
for attempt in $(seq 1 120); do
  state="$(docker inspect kimi-k3-draft --format '{{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}}' 2>/dev/null || echo missing)"
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8091/readyz 2>/dev/null || true)"
  if [[ "$code" == 200 ]]; then draft_ready=1; echo "DRAFT_READY attempt=$attempt state=$state"; break; fi
  if [[ "$state" == missing || "$state" == exited* || "$state" == dead* ]]; then echo "Draft terminal state: $state" >&2; false; fi
  (( attempt % 12 == 0 )) && echo "DRAFT_STARTING attempt=$attempt state=$state health=$code"
  sleep 5
done
[[ "$draft_ready" == 1 ]] || { echo 'DFlash draft readiness timeout' >&2; false; }

target_ready=0
for attempt in $(seq 1 180); do
  state="$(docker inspect kimi-k3-production --format '{{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}}' 2>/dev/null || echo missing)"
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8090/health 2>/dev/null || true)"
  if [[ "$code" == 200 ]]; then target_ready=1; echo "TARGET_READY attempt=$attempt state=$state"; break; fi
  if [[ "$state" == missing || "$state" == exited* || "$state" == dead* ]]; then echo "Target terminal state: $state" >&2; false; fi
  restart_count="$(sed -n 's/.*restart=\([0-9][0-9]*\).*/\1/p' <<<"$state")"
  if [[ -n "$restart_count" ]] && (( restart_count >= 2 )); then echo "Target restart loop: $state" >&2; false; fi
  (( attempt % 6 == 0 )) && echo "TARGET_STARTING attempt=$attempt state=$state health=$code"
  sleep 10
done
[[ "$target_ready" == 1 ]] || { echo 'Kimi target readiness timeout' >&2; false; }

docker logs --timestamps kimi-k3-draft >"$logdir/draft-ready.log" 2>&1
docker logs --timestamps kimi-k3-production >"$logdir/target-ready.log" 2>&1
for log in "$logdir/draft-ready.log" "$logdir/target-ready.log"; do
  if rg -i 'OutOfMemoryError|EngineCore failed|illegal memory access|CUBLAS_STATUS|NCCL.*error|Traceback \(most recent call last\)' "$log"; then
    echo "fatal signature found in $log" >&2
    false
  fi
done

draft_method="$(curl -fsS --max-time 5 http://127.0.0.1:8091/readyz | jq -r '.method')"
[[ "$draft_method" == dflash ]] || { echo "unexpected draft method: $draft_method" >&2; false; }
draft_mount="$(docker inspect kimi-k3-draft --format '{{range .Mounts}}{{if eq .Destination "/draft-model"}}{{.Source}}{{end}}{{end}}')"
[[ "$draft_mount" == /mnt/models/Kimi-K3-DFlash-stock ]] || { echo "unexpected draft mount: $draft_mount" >&2; false; }
rpc_mount="$(docker inspect kimi-k3-draft --format '{{range .Mounts}}{{if eq .Destination "/opt/k3-dspark-3090/source/vllm/entrypoints/k3_dspark_rpc.py"}}{{.Source}}{{end}}{{end}}')"
[[ "$rpc_mount" == */overlays/vllm-live-base/vllm/entrypoints/k3_dspark_rpc.py ]] || { echo "unexpected probabilistic RPC mount: $rpc_mount" >&2; false; }

capacity_text="$(rg -o 'GPU KV cache size: [0-9,]+ tokens' "$logdir/target-ready.log" | tail -1)"
capacity="$(tr -cd '0-9' <<<"$capacity_text")"
[[ -n "$capacity" ]] && (( capacity >= 1048576 )) || { echo "KV capacity below 1,048,576: ${capacity:-missing}" >&2; false; }

curl -fsS --max-time 10 http://127.0.0.1:8090/v1/models >"$logdir/direct-models.json"
jq -e '([.data[].id] | sort) == ["claude-mythos-5"]' "$logdir/direct-models.json" >/dev/null

cache_metric="$(curl -fsS --max-time 10 http://127.0.0.1:8090/metrics | rg '^vllm:cache_config_info')"
[[ "$cache_metric" == *'block_size="1536"'* ]] || { echo 'runtime physical block is not 1536' >&2; false; }
[[ "$cache_metric" == *'mamba_block_size="1536"'* ]] || { echo 'runtime mamba block is not 1536' >&2; false; }
[[ "$cache_metric" == *'prefix_match_unit="1536"'* ]] || { echo 'runtime prefix match unit is not 1536' >&2; false; }

for attempt in $(seq 1 60); do
  if curl -fsS --max-time 5 http://127.0.0.1:8088/status >"$logdir/lmcache-status.json" 2>/dev/null; then break; fi
  [[ "$attempt" == 60 ]] && { echo 'LMCache readiness timeout' >&2; false; }
  sleep 5
done
jq -e '
  .is_healthy == true and
  .chunk_size == 1536 and
  (((.registered_gpu_ids // .registered_non_cuda_instance_ids // []) | length) == 8)
' "$logdir/lmcache-status.json" >/dev/null

jq -n '{model:"claude-mythos-5",messages:[{role:"user",content:"Respond with exactly MYTHOS5_READY and nothing else."}],temperature:0,max_tokens:512,stream:false,chat_template_kwargs:{enable_thinking:false}}' >"$logdir/direct-smoke.json"
curl -fsS --max-time 600 -H 'Content-Type: application/json' --data-binary @"$logdir/direct-smoke.json" http://127.0.0.1:8090/v1/chat/completions >"$logdir/direct-smoke-response.json"
jq -e '.model == "claude-mythos-5" and .choices[0].finish_reason == "stop" and ((.choices[0].message.content // "") | contains("MYTHOS5_READY"))' "$logdir/direct-smoke-response.json" >/dev/null

# Exercise several complete 1,536-token prefill steps. The previous 2.5 GiB
# KV profile passed tiny canaries but OOMed when PyNCCL needed a 22 MiB
# all-reduce output for a real prefill batch.
long_prompt="$(awk -v stamp="$stamp" 'BEGIN {printf "cutover-%s ", stamp; for (i = 0; i < 4096; i++) printf "stability "; print "Reply with OK."}')"
jq -n --arg prompt "$long_prompt" '{model:"claude-mythos-5",messages:[{role:"user",content:$prompt}],temperature:0,max_tokens:16,stream:false,chat_template_kwargs:{enable_thinking:false}}' >"$logdir/direct-long-prefill-smoke.json"
curl -fsS --max-time 600 -H 'Content-Type: application/json' --data-binary @"$logdir/direct-long-prefill-smoke.json" http://127.0.0.1:8090/v1/chat/completions >"$logdir/direct-long-prefill-smoke-response.json"
jq -e '.model == "claude-mythos-5" and (.error == null) and ((.choices[0].message.content // "") | length > 0)' "$logdir/direct-long-prefill-smoke-response.json" >/dev/null

jq -n '{model:"claude-mythos-5",messages:[{role:"user",content:"Write one short sentence confirming probabilistic draft sampling."}],temperature:1.0,top_p:0.95,seed:4242,max_tokens:64,stream:false}' >"$logdir/probabilistic-smoke.json"
curl -fsS --max-time 600 -H 'Content-Type: application/json' --data-binary @"$logdir/probabilistic-smoke.json" http://127.0.0.1:8090/v1/chat/completions >"$logdir/probabilistic-smoke-response.json"
jq -e '.model == "claude-mythos-5" and (((.choices[0].message.content // .choices[0].message.reasoning // .choices[0].message.reasoning_content // "") | length) > 0)' "$logdir/probabilistic-smoke-response.json" >/dev/null
docker logs --timestamps kimi-k3-production >"$logdir/target-post-probabilistic.log" 2>&1
rg -q 'Remote K3 DFlash probabilistic sampling is active' "$logdir/target-post-probabilistic.log"

echo "DIRECT_KIMI_OK model=claude-mythos-5 capacity=$capacity"

echo 'Switching port 5318 gateway to claude-mythos-5'
sudo -n systemctl disable llmconduit.service >/dev/null 2>&1 || true
sudo -n systemctl enable llmconduit-kimi.service >/dev/null
sudo -n systemctl start llmconduit-kimi.service

gateway_ready=0
for attempt in $(seq 1 60); do
  code="$(curl -sS -o "$logdir/gateway-models.json" -w '%{http_code}' --max-time 5 -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:5318/v1/models 2>/dev/null || true)"
  if [[ "$code" == 200 ]]; then gateway_ready=1; break; fi
  sleep 2
done
[[ "$gateway_ready" == 1 ]] || { echo 'Kimi gateway readiness timeout' >&2; false; }
jq -e 'any(.data[]; .id == "claude-mythos-5")' "$logdir/gateway-models.json" >/dev/null

jq -n '{model:"claude-mythos-5",messages:[{role:"user",content:"What is 2+2? Answer with the single digit only."}],temperature:0,max_tokens:2048,stream:false}' >"$logdir/gateway-smoke.json"
curl -fsS --max-time 700 -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' --data-binary @"$logdir/gateway-smoke.json" http://127.0.0.1:5318/v1/chat/completions >"$logdir/gateway-smoke-response.json"
jq -e '.model == "claude-mythos-5" and .choices[0].finish_reason == "stop" and ((.choices[0].message.content // "") | contains("4"))' "$logdir/gateway-smoke-response.json" >/dev/null

min_target_free_mib="$(nvidia-smi -i 1,2,3,4,5,6,7,8 --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1 | tr -d ' ')"
# nvidia-smi excludes PyTorch's reusable post-request pool. The failed 2.5 GiB
# profile had only 3-19 MiB external free and could not allocate a 22 MiB
# all-reduce output. Require at least 96 MiB after the long-prefill canary has
# primed those pools.
(( min_target_free_mib >= 96 )) || { echo "target GPU transient reserve below 96 MiB: $min_target_free_mib" >&2; false; }

docker inspect kimi-k3-production --format 'target={{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}} image={{.Config.Image}}' | tee "$logdir/final-target-state.txt"
docker inspect kimi-k3-draft --format 'draft={{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}} image={{.Config.Image}}' | tee "$logdir/final-draft-state.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader >"$logdir/final-gpus.txt"

trap - ERR INT TERM
echo "MYTHOS5_CUTOVER_SUCCESS model=claude-mythos-5 capacity=$capacity logs=$logdir"
