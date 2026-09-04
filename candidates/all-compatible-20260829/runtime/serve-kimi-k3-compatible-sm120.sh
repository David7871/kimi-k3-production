#!/usr/bin/env bash
# Serve the issue-75 QSRT target with remote BF16 DFlash K3 on eight RTX PRO
# 6000 GPUs (TP8/DCP8). The target-token and remote-draft input budgets are
# independent (vLLM #53080), so the full 3,080 target-token budget is retained.
# FlashKDA intermediate checkpoints preserve a fine prefix inside the 3,072
# Mamba page without replaying the complete page.
# The draft forward runs in the standalone process addressed by
# VLLM_K3_DRAFT_REMOTE_ADDRESS; this launcher only builds verifier metadata.
set -euo pipefail

MODEL="${MODEL:-/model}"
DRAFT_MODEL="${DRAFT_MODEL:-/draft-model}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
DCP_SIZE="${DCP_SIZE:-8}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-2348810240}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-372000}"
# DFlash consumes one bonus anchor plus K parallel draft rows per active
# request; that budget is now accounted independently from target tokens.
DFLASH_K="${DFLASH_K:-3}"
SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-dflash}"
DRAFT_ATTENTION_BACKEND="${DRAFT_ATTENTION_BACKEND:-TRITON_ATTN}"
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-greedy}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-3080}"
MAX_NUM_SCHEDULED_TOKENS="${MAX_NUM_SCHEDULED_TOKENS:-3080}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# Optional physical KV block override. Hybrid KDA+MLA sizes the attention
# block to the minimum that fits one recurrent state (1536); LMCache align
# mode requires block_size <= max_num_batched_tokens < 2*block_size, so a
# 4096-token chunk budget needs the physical block raised to 4096 too.
# Empty (default) keeps the engine-computed block for other profiles.
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.985}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3}"
ENABLE_VISION="${ENABLE_VISION:-0}"
MAMBA_BLOCK_SIZE="${MAMBA_BLOCK_SIZE:-1536}"
PREFIX_MATCH_UNIT="${PREFIX_MATCH_UNIT:-1536}"
PYTHON_BIN="${K3_PYTHON_BIN:-/opt/venv/bin/python}"

if (( TP_SIZE < 1 || DCP_SIZE < 1 || TP_SIZE % DCP_SIZE != 0 )); then
  printf 'TP_SIZE must be positive and divisible by DCP_SIZE: TP_SIZE=%s DCP_SIZE=%s\n' \
    "${TP_SIZE}" "${DCP_SIZE}" >&2
  exit 2
fi
if [[ "${TP_SIZE}" != 8 || "${DCP_SIZE}" != 8 ]]; then
  printf 'This host profile requires TP_SIZE=8 and DCP_SIZE=8; got TP=%s DCP=%s\n' \
    "${TP_SIZE}" "${DCP_SIZE}" >&2
  exit 2
fi
case "${SPECULATIVE_METHOD}" in
  dflash|dspark) ;;
  *)
    printf 'SPECULATIVE_METHOD must be dflash or dspark; got %s\n' \
      "${SPECULATIVE_METHOD}" >&2
    exit 2
    ;;
esac
if (( MAX_NUM_SCHEDULED_TOKENS < 1 || MAX_NUM_SCHEDULED_TOKENS > MAX_NUM_BATCHED_TOKENS )); then
  printf 'Target scheduled budget must be in [1, batched]: batched=%s scheduled=%s\n' \
    "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SCHEDULED_TOKENS}" >&2
  exit 2
fi
if (( MAX_NUM_BATCHED_TOKENS < MAX_NUM_SEQS * (DFLASH_K + 1) )); then
  printf 'DFlash K%s needs at least seqs*(K+1) draft rows: batched=%s seqs=%s\n' \
    "${DFLASH_K}" "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SEQS}" >&2
  exit 2
fi

required=(
  "${MODEL}/model.safetensors.index.json"
  "${DRAFT_MODEL}/config.json"
  "${DRAFT_MODEL}/model.safetensors"
  "${PYTHON_BIN}"
  /opt/infernal-invocation/vllm/vllm/__init__.py
  /opt/infernal-invocation/b12x/b12x/attention/dense_mla/__init__.py
)
for path in "${required[@]}"; do
  [[ -e "${path}" ]] || {
    printf 'Required Kimi-K3 artifact is missing: %s\n' "${path}" >&2
    exit 1
  }
done

# Verifier-side metadata only. Draft weights, private BF16 KV, and standalone
# CUDA graphs live in the remote process.  DFlash remains the default; a
# full-stack override can select the already-supported remote DSpark path.
printf -v speculative_config \
  '{"method":"%s","model":"%s","num_speculative_tokens":%s,"draft_tensor_parallel_size":1,"max_model_len":%s,"attention_backend":"%s","kv_cache_dtype":"bfloat16","draft_sample_method":"%s","rejection_sample_method":"block","draft_load_config":{"load_format":"safetensors"}}' \
  "${SPECULATIVE_METHOD}" \
  "${DRAFT_MODEL}" \
  "${DFLASH_K}" \
  "${MAX_MODEL_LEN}" \
  "${DRAFT_ATTENTION_BACKEND}" \
  "${DRAFT_SAMPLE_METHOD}"

export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/huggingface-modules}"
export VLLM_SOURCE_OVERLAY_ACTIVE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${KIMI_PCIE_ALLREDUCE_BACKEND:-b12x}"
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
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE="${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-4096}"
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

case "${ENABLE_VISION}" in
  0) multimodal_args=(--language-model-only) ;;
  1)
    multimodal_args=(
      --limit-mm-per-prompt "{\"image\":${MAX_IMAGES_PER_PROMPT:-5}}"
      --mm-processor-kwargs "{\"in_patch_limit\":${VISION_IN_PATCH_LIMIT:-40960},\"patch_limit_on_one_side\":${VISION_PATCH_LIMIT_ON_ONE_SIDE:-512}}"
      --mm-processor-cache-gb "${MM_PROCESSOR_CACHE_GB:-1}"
      --mm-encoder-tp-mode "${MM_ENCODER_TP_MODE:-weights}"
    )
    ;;
  *)
    echo "ENABLE_VISION must be 0 or 1" >&2
    exit 2
    ;;
esac

exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  "${multimodal_args[@]}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --decode-context-parallel-size "${DCP_SIZE}" \
  --dcp-comm-backend a2a \
  --dcp-kv-cache-interleave-size 1 \
  ${KV_BLOCK_SIZE:+--block-size "${KV_BLOCK_SIZE}"} \
  ${VLLM_PROFILER_DIR:+--profiler-config.profiler=torch --profiler-config.torch_profiler_dir="${VLLM_PROFILER_DIR}"} \
  ${VLLM_PROFILER_EXTRA_ARGS:-} \
  --mamba-block-size "${MAMBA_BLOCK_SIZE}" \
  --load-format instanttensor \
  --moe-backend b12x \
  --linear-backend b12x \
  --attention-backend B12X_MLA \
  --kda-prefill-backend flashkda \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-scheduled-tokens "${MAX_NUM_SCHEDULED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit "${PREFIX_MATCH_UNIT}" \
  --speculative-config "${speculative_config}" \
  --compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"],\"cudagraph_capture_sizes\":[${K3_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,16}],\"pass_config\":{\"fuse_allreduce_rms\":true}}" \
  --generation-config vllm \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  --attention-config.flash_attn_version=2 \
  "$@"
