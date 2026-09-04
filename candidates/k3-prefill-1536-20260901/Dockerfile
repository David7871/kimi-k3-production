# syntax=docker/dockerfile:1.7
FROM kimi-k3-production-issue75-dspark@sha256:087276f1ec791909e71055d655eeadb3b35c246e46f2be21af9d210aa970a60b

ARG VLLM_COMMIT=unknown
ARG STABLE_EXTENSION_SHA256=unknown

COPY --from=artifacts /cmake/_C_stable_libtorch.abi3.so \
    /opt/infernal-invocation/vllm/vllm/_C_stable_libtorch.abi3.so
COPY --from=artifacts /cmake/_C_stable_libtorch.abi3.so \
    /opt/venv/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so

COPY --from=candidate /vllm/envs.py \
    /opt/infernal-invocation/vllm/vllm/envs.py
COPY --from=candidate /vllm/model_executor/layers/attention/mla_attention.py \
    /opt/infernal-invocation/vllm/vllm/model_executor/layers/attention/mla_attention.py
COPY --from=candidate /vllm/model_executor/layers/attention/sparse_mla_attention.py \
    /opt/infernal-invocation/vllm/vllm/model_executor/layers/attention/sparse_mla_attention.py
COPY --from=candidate /vllm/model_executor/layers/quantization/kquant_hybrid.py \
    /opt/infernal-invocation/vllm/vllm/model_executor/layers/quantization/kquant_hybrid.py
COPY --from=candidate /vllm/models/kimi_k3/nvidia/mla.py \
    /opt/infernal-invocation/vllm/vllm/models/kimi_k3/nvidia/mla.py
COPY --from=candidate /vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py \
    /opt/infernal-invocation/vllm/vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py
COPY --from=candidate /vllm/v1/attention/backends/mla/b12x_mla.py \
    /opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/b12x_mla.py
COPY --from=candidate /vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py \
    /opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py
COPY --from=candidate /vllm/v1/attention/backends/mla/prefill/base.py \
    /opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/prefill/base.py
COPY --from=candidate /vllm/v1/attention/backends/mla/prefill/flash_attn.py \
    /opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/prefill/flash_attn.py
COPY --from=candidate /vllm/v1/attention/backends/mla/prefill/flashinfer.py \
    /opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/prefill/flashinfer.py
COPY --from=candidate /vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py \
    /opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py
COPY --from=candidate /vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py \
    /opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py
COPY --from=candidate /vllm/v1/attention/ops/dcp_utils.py \
    /opt/infernal-invocation/vllm/vllm/v1/attention/ops/dcp_utils.py

RUN set -eu; \
    source_root=/opt/infernal-invocation/vllm/vllm; \
    package_root=/opt/venv/lib/python3.12/site-packages/vllm; \
    for relative_path in \
      envs.py \
      model_executor/layers/attention/mla_attention.py \
      model_executor/layers/attention/sparse_mla_attention.py \
      model_executor/layers/quantization/kquant_hybrid.py \
      models/kimi_k3/nvidia/mla.py \
      models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py \
      v1/attention/backends/mla/b12x_mla.py \
      v1/attention/backends/mla/prefill/aiter_flash_attn.py \
      v1/attention/backends/mla/prefill/base.py \
      v1/attention/backends/mla/prefill/flash_attn.py \
      v1/attention/backends/mla/prefill/flashinfer.py \
      v1/attention/backends/mla/prefill/tokenspeed_mla.py \
      v1/attention/backends/mla/prefill/trtllm_ragged.py \
      v1/attention/ops/dcp_utils.py; \
    do \
      install -m 0644 "$source_root/$relative_path" "$package_root/$relative_path"; \
    done; \
    chmod 0755 \
      "$source_root/_C_stable_libtorch.abi3.so" \
      "$package_root/_C_stable_libtorch.abi3.so"; \
    /opt/venv/bin/python -m py_compile \
      "$source_root/model_executor/layers/attention/mla_attention.py" \
      "$source_root/models/kimi_k3/nvidia/mla.py" \
      "$source_root/v1/attention/ops/dcp_utils.py"

LABEL local.kimi-k3.prefill-policy="fixed-1536-exact" \
      local.kimi-k3.prefill-optimizations="final-layout-dcp-peer-publish,fused-context-kv-pack,caller-owned-context-output" \
      local.kimi-k3.prefill-quality="dense-exact" \
      local.kimi-k3.vllm.commit="${VLLM_COMMIT}" \
      local.kimi-k3.stable-extension.sha256="${STABLE_EXTENSION_SHA256}" \
      org.opencontainers.image.version="k3-prefill-1536-exact-20260901-r1"
