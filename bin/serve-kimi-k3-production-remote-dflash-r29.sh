#!/usr/bin/env bash
# TP8/DCP8 Kimi-K3 target on the qualified r29 image with a standalone BF16
# DFlash process on GPU 0 (RTX 3090). Outer wrapper: LMCache policy
# envs plus the remote-draft speculative config; the inner launcher owns the
# engine flags.
set -euo pipefail

export LMCACHE_MODE="${LMCACHE_MODE:-ram}"
export LMCACHE_L1_GB="${LMCACHE_L1_GB:-16}"
export LMCACHE_L1_INIT_GB="${LMCACHE_L1_INIT_GB:-0}"
export LMCACHE_TRANSFER_MODE="${LMCACHE_TRANSFER_MODE:-engine_driven}"
export LMCACHE_EXPECTED_SOURCE_ROOT="${LMCACHE_EXPECTED_SOURCE_ROOT:-/opt/infernal-invocation/lmcache}"
export LMCACHE_SHM_NAME="${LMCACHE_SHM_NAME-}"
export LMCACHE_SEPARATE_OBJECT_GROUPS="${LMCACHE_SEPARATE_OBJECT_GROUPS:-1}"
export LMCACHE_SERVER_ENV="${LMCACHE_SERVER_ENV-CUDA_VISIBLE_DEVICES= CUDA_MODULE_LOADING=LAZY}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-12288}"
# 5s heartbeat cadence with the adapter-side 30s PING timeout decouple (see
# entrypoint-aliases): bounded server busy spells must not flip worker health,
# while a dead server still degrades promptly.
export LMCACHE_HEARTBEAT_INTERVAL="${LMCACHE_HEARTBEAT_INTERVAL:-5}"
export LMCACHE_WORKER_REAP_TIMEOUT="${LMCACHE_WORKER_REAP_TIMEOUT:-120}"
export LMCACHE_WORKER_REGISTRATION_GRACE="${LMCACHE_WORKER_REGISTRATION_GRACE:-86400}"
export LMCACHE_KV_LOAD_FAILURE_POLICY="${LMCACHE_KV_LOAD_FAILURE_POLICY:-recompute}"

# Source trees win over site-packages: the LMCache adapter operational patches
# and the remote-draft vLLM overlay live in the source checkouts.
export PYTHONPATH="/opt/infernal-invocation/lmcache:/opt/infernal-invocation/b12x:/opt/infernal-invocation/vllm${PYTHONPATH:+:${PYTHONPATH}}"

export MODEL="${MODEL:-/model}"
export DRAFT_MODEL="${DRAFT_MODEL:-/draft-model}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8090}"
export TP_SIZE="8"
export DCP_SIZE="8"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-512000}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1548}"
export MAX_NUM_SCHEDULED_TOKENS="${MAX_NUM_SCHEDULED_TOKENS:-1536}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-2348810240}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.985}"
export ENABLE_PREFIX_CACHING="1"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [[ ! -f "${DRAFT_MODEL}/config.json" || ! -f "${DRAFT_MODEL}/model.safetensors" ]]; then
  echo "Kimi-K3 DFlash BF16 checkpoint is incomplete: ${DRAFT_MODEL}" >&2
  exit 1
fi

# Only verifier-side metadata is constructed here. The six-layer BF16 DFlash
# weights, private BF16 sliding-window KV, and all 14 B1/B2 K1..K7 CUDA graphs
# live in the standalone RTX 3090 process behind the ZMQ proposal endpoint.
printf -v DFLASH_SPECULATIVE_CONFIG \
  '{"method":"dflash","model":"%s","num_speculative_tokens":7,"draft_tensor_parallel_size":1,"max_model_len":%s,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"bfloat16","draft_sample_method":"greedy","rejection_sample_method":"block","draft_load_config":{"load_format":"safetensors"}}' \
  "${DRAFT_MODEL}" \
  "${MAX_MODEL_LEN}"

exec /usr/local/bin/lmcache-mp-wrapper.sh \
  /usr/local/bin/serve-kimi-k3-qsrt-dspark-r29 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --max-num-scheduled-tokens "${MAX_NUM_SCHEDULED_TOKENS}" \
  --speculative-config "${DFLASH_SPECULATIVE_CONFIG}" \
  --host "${HOST}" \
  "$@"