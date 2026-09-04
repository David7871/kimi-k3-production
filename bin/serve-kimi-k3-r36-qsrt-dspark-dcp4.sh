#!/usr/bin/env bash
# Serve local Kimi-K3 QSRT-K2 with the immutable r36 installed packages.
# This profile is TP8/DCP4 with colocated Inferact DSpark K7 and native host KV.
set -euo pipefail

MODEL="${MODEL:-/model}"
DRAFT_MODEL="${DRAFT_MODEL:-/draft-model}"
PORT="${PORT:-8090}"
TP_SIZE="${TP_SIZE:-8}"
DCP_SIZE="${DCP_SIZE:-4}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1879048192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-330000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1542}"
MAX_NUM_SCHEDULED_TOKENS="${MAX_NUM_SCHEDULED_TOKENS:-1536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.985}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3}"
HOST_KV_BACKEND="${HOST_KV_BACKEND:-native}"
KV_OFFLOADING_SIZE="${KV_OFFLOADING_SIZE:-32}"
PYTHON_BIN="${K3_PYTHON_BIN:-/opt/venv/bin/python}"

if [[ "${TP_SIZE}" != 8 || "${DCP_SIZE}" != 4 ]]; then
  printf 'This qualified host wrapper requires TP_SIZE=8 and DCP_SIZE=4; got TP=%s DCP=%s\n' \
    "${TP_SIZE}" "${DCP_SIZE}" >&2
  exit 2
fi
if (( MAX_NUM_BATCHED_TOKENS != MAX_NUM_SCHEDULED_TOKENS + MAX_NUM_SEQS * 6 )); then
  printf 'DSpark K7 geometry requires batched=scheduled+seqs*6; got %s, %s, %s\n' \
    "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SCHEDULED_TOKENS}" "${MAX_NUM_SEQS}" >&2
  exit 2
fi

required=(
  "${MODEL}/model.safetensors.index.json"
  "${DRAFT_MODEL}/model.safetensors"
  "${PYTHON_BIN}"
  /opt/venv/lib/python3.12/site-packages/vllm/__init__.py
  /opt/venv/lib/python3.12/site-packages/b12x/attention/dense_mla/__init__.py
)
for path in "${required[@]}"; do
  [[ -e "${path}" ]] || {
    printf 'Required Kimi-K3 artifact is missing: %s\n' "${path}" >&2
    exit 1
  }
done

printf -v speculative_config \
  '{"method":"dspark","model":"%s","num_speculative_tokens":7,"attention_backend":"B12X_MLA","kv_cache_dtype":"fp8","draft_sample_method":"greedy","rejection_sample_method":"block","quantization":"mxfp8","quantization_config":{"linear":"mxfp8","ignore":["re:.*fused_qkv_a_proj$"]}}' \
  "${DRAFT_MODEL}"

export VLLM_SOURCE_OVERLAY_ACTIVE=0
export PYTHONPATH="/opt/exllamav3:/opt/venv/lib/python3.12/site-packages"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/cache/hf-modules}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/cache/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/cache/cuda}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-/cache/cupy}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/cache/numba}"
mkdir -p "${HF_MODULES_CACHE}" "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${CUPY_CACHE_DIR}" "${NUMBA_CACHE_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export CUDA_MODULE_DATA_LOADING="${CUDA_MODULE_DATA_LOADING:-LAZY}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
export B12X_MOE_WORKSPACE_TOKEN_LIMIT="${B12X_MOE_WORKSPACE_TOKEN_LIMIT:-4096}"
export B12X_W4A16_PREFILL_FUSED_SUM="${B12X_W4A16_PREFILL_FUSED_SUM:-1}"
export B12X_W4A16_STABLE_ROUTE_PACK="${B12X_W4A16_STABLE_ROUTE_PACK:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_SINGLE_CHANNEL="${VLLM_PCIE_ONESHOT_SINGLE_CHANNEL:-1}"
export B12X_PCIE_ALLREDUCE_ALGORITHM="${B12X_PCIE_ALLREDUCE_ALGORITHM:-island_rs}"
export VLLM_USE_B12X_DCP_A2A="${VLLM_USE_B12X_DCP_A2A:-1}"
export VLLM_DCP_A2A_MAX_TOKENS="${VLLM_DCP_A2A_MAX_TOKENS:-8}"
export VLLM_DCP_A2A_LARGE_BACKEND="${VLLM_DCP_A2A_LARGE_BACKEND:-ag_rs}"
export VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT:-0}"
export VLLM_K3_KV_GROUP_SIZE="${VLLM_K3_KV_GROUP_SIZE:-6}"
export VLLM_KIMI_SHARD_QKV_A="${VLLM_KIMI_SHARD_QKV_A:-1}"
export VLLM_KIMI_USE_B12X_PROJECTION_GATHER="${VLLM_KIMI_USE_B12X_PROJECTION_GATHER:-1}"
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER="${VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER:-1}"
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK="${VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK:-1}"
export VLLM_KIMI_FUSED_TOPK16="${VLLM_KIMI_FUSED_TOPK16:-1}"
export VLLM_KIMI_K3_B12X_DSPARK_ARGMAX="${VLLM_KIMI_K3_B12X_DSPARK_ARGMAX:-1}"
export VLLM_DSPARK_DRAFT_KV_WINDOW="${VLLM_DSPARK_DRAFT_KV_WINDOW:-32768}"
export VLLM_DSPARK_COMPACT_ROPE="${VLLM_DSPARK_COMPACT_ROPE:-1}"
export VLLM_DSPARK_SHARD_MARKOV_HEAD="${VLLM_DSPARK_SHARD_MARKOV_HEAD:-1}"
export VLLM_DSPARK_CAPTURE_SHARDED_MARKOV="${VLLM_DSPARK_CAPTURE_SHARDED_MARKOV:-1}"
export VLLM_DSPARK_REPLICATE_MARKOV_W1="${VLLM_DSPARK_REPLICATE_MARKOV_W1:-1}"
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE="${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-4096}"
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-0}"
export B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION="${B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION:-1}"
export B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER="${B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER:-0}"
export B12X_PCIE_HIERARCHICAL_THREADS="${B12X_PCIE_HIERARCHICAL_THREADS:-256}"
export B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES="${B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES:-24}"
export B12X_PCIE_HIERARCHICAL_BF16X2="${B12X_PCIE_HIERARCHICAL_BF16X2:-1}"
export B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS="${B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS:-7168}"
export B12X_PCIE_DCP_THREADS="${B12X_PCIE_DCP_THREADS:-512}"
export B12X_PCIE_DCP_BLOCK_LIMIT="${B12X_PCIE_DCP_BLOCK_LIMIT:-8}"
export B12X_PCIE_KIMI_TOPK_THREADS="${B12X_PCIE_KIMI_TOPK_THREADS:-384}"
export INSTANTTENSOR_COPY="${INSTANTTENSOR_COPY:-0}"
export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-AIO}"
export INSTANTTENSOR_BUFFER_SIZE="${INSTANTTENSOR_BUFFER_SIZE:-536870912}"
export INSTANTTENSOR_MAX_FREE_MEM_USAGE="${INSTANTTENSOR_MAX_FREE_MEM_USAGE:-0.6}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
unset NCCL_GRAPH_FILE

host_kv_args=()
case "${HOST_KV_BACKEND}" in
  native)
    host_kv_args=(--kv-offloading-size "${KV_OFFLOADING_SIZE}" --kv-offloading-backend native)
    ;;
  off)
    ;;
  *)
    printf 'HOST_KV_BACKEND must be native or off; got %s\n' "${HOST_KV_BACKEND}" >&2
    exit 2
    ;;
esac

printf 'Kimi-K3 r36 QSRT launch: TP=%s DCP=%s spec=7 max_len=%s batch=%s scheduled=%s seqs=%s host_kv=%sGiB\n' \
  "${TP_SIZE}" "${DCP_SIZE}" "${MAX_MODEL_LEN}" "${MAX_NUM_BATCHED_TOKENS}" \
  "${MAX_NUM_SCHEDULED_TOKENS}" "${MAX_NUM_SEQS}" \
  "$([[ "${HOST_KV_BACKEND}" == native ]] && printf '%s' "${KV_OFFLOADING_SIZE}" || printf '0')"

exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --language-model-only \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --decode-context-parallel-size "${DCP_SIZE}" \
  --dcp-comm-backend a2a \
  --dcp-kv-cache-interleave-size 1 \
  --load-format instanttensor \
  --moe-backend b12x \
  --linear-backend b12x \
  --attention-backend B12X_MLA \
  --kda-prefill-backend triton \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-scheduled-tokens "${MAX_NUM_SCHEDULED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --block-size 128 \
  --mamba-block-size 1536 \
  --prefix-match-unit 1536 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config "${speculative_config}" \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"],"cudagraph_capture_sizes":[8],"pass_config":{"fuse_allreduce_rms":true}}' \
  --generation-config vllm \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  --attention-config.flash_attn_version=2 \
  "${host_kv_args[@]}" \
  "$@"
