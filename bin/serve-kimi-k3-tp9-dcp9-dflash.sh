#!/usr/bin/env bash
# Serve the issue-75 QSRT target with COLOCATED BF16 DFlash K3 on
# nine GPUs (draft sharded across the TP9 ranks; no remote proposal RPC).
# Derived from serve-kimi-k3-tp9-dcp9.sh; only the speculative config and
# the DFlash draft-KV window export differ.
# NINE RTX PRO 6000 GPUs. This host profile is TP9/DCP9 and relies on B12X
# virtual-TP padding (heads 96->99, MoE intermediate 3072->3456) plus the
# odd-shard pure-K2 atom partition patch applied by the TP9 entrypoint.
#
# Geometry: physical attention block 1536 (unchanged); LMCache/recurrent
# cadence 4608 = 3 blocks, giving 4608 / 9 = 512 rank-local rows per chunk.
set -euo pipefail

MODEL="${MODEL:-/model}"
DRAFT_MODEL="${DRAFT_MODEL:-/draft-model}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-9}"
DCP_SIZE="${DCP_SIZE:-9}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-2348810240}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-372000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1544}"
MAX_NUM_SCHEDULED_TOKENS="${MAX_NUM_SCHEDULED_TOKENS:-1536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.985}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Kimi-K3}"
ENABLE_VISION="${ENABLE_VISION:-0}"
MAMBA_BLOCK_SIZE="${MAMBA_BLOCK_SIZE:-4608}"
# Optional physical KV block override (p4608 profile). LMCache align mode
# requires block_size <= max_num_batched_tokens < 2*block_size, so a
# 4608-token scheduler budget needs the physical block raised to 4608.
# Empty (default) keeps the engine-computed 1536 block.
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-}"
PREFIX_MATCH_UNIT="${PREFIX_MATCH_UNIT:-1536}"
PYTHON_BIN="${K3_PYTHON_BIN:-/opt/venv/bin/python}"

if (( TP_SIZE < 1 || DCP_SIZE < 1 || TP_SIZE % DCP_SIZE != 0 )); then
  printf 'TP_SIZE must be positive and divisible by DCP_SIZE: TP_SIZE=%s DCP_SIZE=%s\n' \
    "${TP_SIZE}" "${DCP_SIZE}" >&2
  exit 2
fi
if [[ "${TP_SIZE}" != 9 || "${DCP_SIZE}" != 9 ]]; then
  printf 'This host profile requires TP_SIZE=9 and DCP_SIZE=9; got TP=%s DCP=%s\n' \
    "${TP_SIZE}" "${DCP_SIZE}" >&2
  exit 2
fi
if (( MAX_NUM_BATCHED_TOKENS != MAX_NUM_SCHEDULED_TOKENS + MAX_NUM_SEQS * 2 )); then
  printf 'DSpark K3 requires batched=scheduled+seqs*2; got %s, %s, %s\n' \
    "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SCHEDULED_TOKENS}" "${MAX_NUM_SEQS}" >&2
  exit 2
fi
# TP9 cadence: the physical attention block stays 1536 (mamba page math with
# KDA local 12). The LMCache chunk / recurrent cadence is 4608 = 3 blocks =
# lcm(1536, 9-rank rows): 4608 / 9 = 512 rank-local rows per chunk. Scheduler
# steps stay 1536 (one block per step); a mamba snapshot lands exactly at
# every third step boundary.

required=(
  "${MODEL}/model.safetensors.index.json"
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

# Colocated DFlash: the 2026-08-23-qualified remote-DFlash draft config
# minus the remote proposal fields. The six-layer SWA draft shards across
# the nine TP ranks in BF16 (no MXFP8 32-block constraint), and aux hidden
# states stay in-process instead of crossing the ZMQ hop.
printf -v speculative_config \
  '{"method":"dflash","model":"%s","num_speculative_tokens":3,"max_model_len":%s,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"bfloat16","draft_sample_method":"greedy","rejection_sample_method":"block","draft_load_config":{"load_format":"safetensors"}}' \
  "${DRAFT_MODEL}" \
  "${MAX_MODEL_LEN}"

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
# Kimi-K3 DCP profiles use the topology-aware B12X all-reduce and must not
# inherit the generic C++ backend configured by the base image.
export VLLM_PCIE_ALLREDUCE_BACKEND="${KIMI_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_SINGLE_CHANNEL="${VLLM_PCIE_ONESHOT_SINGLE_CHANNEL:-1}"
export B12X_PCIE_ALLREDUCE_ALGORITHM="${B12X_PCIE_ALLREDUCE_ALGORITHM:-island_rs}"
export VLLM_USE_B12X_DCP_A2A="${VLLM_USE_B12X_DCP_A2A:-1}"
export VLLM_DCP_A2A_MAX_TOKENS="${VLLM_DCP_A2A_MAX_TOKENS:-8}"
export VLLM_DCP_A2A_LARGE_BACKEND="${VLLM_DCP_A2A_LARGE_BACKEND:-ag_rs}"
export VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT:-0}"
export VLLM_K3_KV_GROUP_SIZE="${VLLM_K3_KV_GROUP_SIZE:-6}"
# TP9: q_lora_rank (1536) and kv_lora_rank+rope (576) must both divide by
# TP for the sharded fused qkv_a path; 1536 % 9 != 0, so use the replicated
# qkv_a projection (the well-tested disable_tp branch).
export VLLM_KIMI_SHARD_QKV_A="${VLLM_KIMI_SHARD_QKV_A:-0}"
export VLLM_KIMI_USE_B12X_PROJECTION_GATHER="${VLLM_KIMI_USE_B12X_PROJECTION_GATHER:-1}"
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER="${VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER:-1}"
export VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK="${VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK:-1}"
export VLLM_KIMI_FUSED_TOPK16="${VLLM_KIMI_FUSED_TOPK16:-1}"
export VLLM_KIMI_K3_B12X_DSPARK_ARGMAX="${VLLM_KIMI_K3_B12X_DSPARK_ARGMAX:-1}"
export VLLM_DSPARK_DRAFT_KV_WINDOW="${VLLM_DSPARK_DRAFT_KV_WINDOW:-32768}"
# DFlash attention window is 4096; retaining more projected prefix rows
# consumes VRAM without improving draft attention (2026-08-23 lesson).
export VLLM_K3_DRAFT_PREFIX_CACHE_TOKENS="${VLLM_K3_DRAFT_PREFIX_CACHE_TOKENS:-4096}"
export VLLM_DSPARK_COMPACT_ROPE="${VLLM_DSPARK_COMPACT_ROPE:-1}"
export VLLM_DSPARK_SHARD_MARKOV_HEAD="${VLLM_DSPARK_SHARD_MARKOV_HEAD:-1}"
export VLLM_DSPARK_CAPTURE_SHARDED_MARKOV="${VLLM_DSPARK_CAPTURE_SHARDED_MARKOV:-1}"
export VLLM_DSPARK_REPLICATE_MARKOV_W1="${VLLM_DSPARK_REPLICATE_MARKOV_W1:-1}"
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

# Vision profile (upstream-qualified PRs #459/#464): request-sized encoder
# allocation and exact-row TP gather keep per-GPU vision memory bounded.
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
  --mamba-block-size "${MAMBA_BLOCK_SIZE}" \
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
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-match-unit "${PREFIX_MATCH_UNIT}" \
  --speculative-config "${speculative_config}" \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,16],"pass_config":{"fuse_allreduce_rms":true}}' \
  --generation-config vllm \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  --attention-config.flash_attn_version=2 \
  "$@"
