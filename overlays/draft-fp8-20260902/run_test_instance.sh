#!/bin/bash
# Standalone fp8 DFlash draft test instance on GPU 0 (ports 8095/8096).
# Pass extra args (e.g. --exit-after-load).
set -euo pipefail
O=/home/g0san/kimi-k3-production/overlays/draft-fp8-20260902
K=/home/g0san/kimi-k3-production
docker run --rm --gpus '"device=0"' --network host --ipc=host -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=0 -e VLLM_K3_DRAFT_PREFIX_CACHE_TOKENS=32768 \
  -v /mnt/models/Kimi-K3-DFlash-stock:/draft-model:ro -v /mnt/models/Kimi-K3-QSRT-K2:/target-weights:ro -v /mnt/models/kimi-k3-draft3090-cache:/cache \
  -v $K/overlays/vllm-live-base/vllm/entrypoints/k3_dspark_rpc.py:/opt/k3-dspark-3090/source/vllm/entrypoints/k3_dspark_rpc.py \
  -v $K/bin/patch-dspark-standalone-grouped-op-gate.py:/usr/local/bin/patch-dspark-standalone-grouped-op-gate.py:ro \
  -v $K/bin/patch-dspark-reconnect-partial-window.py:/usr/local/bin/patch-dspark-reconnect-partial-window.py:ro \
  -v $K/bin/entrypoint-draft-9thgpu.sh:/usr/local/bin/entrypoint-draft-9thgpu.sh:ro \
  -v $O/k3_dspark_standalone.py:/opt/k3-dspark-3090/source/vllm/entrypoints/k3_dspark_standalone.py:ro \
  -v $O/qwen3_dflash.py:/opt/k3-dspark-3090/source/vllm/model_executor/models/qwen3_dflash.py:ro \
  -v $O/utils.py:/opt/k3-dspark-3090/source/vllm/v1/worker/gpu/spec_decode/dflash/utils.py:ro \
  --entrypoint /usr/local/bin/entrypoint-draft-9thgpu.sh kimi-k3-dspark-3090:bf16-cudagraph \
  --draft-model /draft-model --target-weights /target-weights --target-config /opt/k3-dspark-3090/target-config --method dflash --device 0 --host 127.0.0.1 --port 8095 --num-speculative-tokens 3 --max-model-len 1048576 --max-num-batched-tokens 1536 --max-num-seqs 4 --max-retained-requests 8 --draft-kv-cache-gib 8.0 --draft-kv-window 32768 --enable-cuda-graph --cuda-graph-warmups 2 --proposal-address tcp://127.0.0.1:8096 "$@"
