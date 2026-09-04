# syntax=docker/dockerfile:1.7
FROM kimi-k3-production-issue75-dspark:k3-prefill-1536-exact-20260901-r1

ARG VLLM_COMMIT=unknown

COPY --from=candidate /vllm/envs.py \
    /opt/infernal-invocation/vllm/vllm/envs.py
COPY --from=candidate /vllm/model_executor/layers/attention/mla_attention.py \
    /opt/infernal-invocation/vllm/vllm/model_executor/layers/attention/mla_attention.py
COPY --from=candidate /vllm/v1/attention/backends/mla/prefill/flash_attn.py \
    /opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/prefill/flash_attn.py
COPY --from=candidate /vllm/models/kimi_k3/nvidia/mla.py \
    /opt/infernal-invocation/vllm/vllm/models/kimi_k3/nvidia/mla.py
COPY --from=candidate /vllm/models/kimi_k3/nvidia/model.py \
    /opt/infernal-invocation/vllm/vllm/models/kimi_k3/nvidia/model.py

RUN set -eu; \
    source_root=/opt/infernal-invocation/vllm/vllm; \
    package_root=/opt/venv/lib/python3.12/site-packages/vllm; \
    test -d "$package_root/vllm_flash_attn/cute"; \
    mkdir -p "$source_root/vllm_flash_attn"; \
    cp -a "$package_root/vllm_flash_attn/cute" \
      "$source_root/vllm_flash_attn/cute"; \
    for relative_path in \
      envs.py \
      model_executor/layers/attention/mla_attention.py \
      models/kimi_k3/nvidia/mla.py \
      models/kimi_k3/nvidia/model.py \
      v1/attention/backends/mla/prefill/flash_attn.py; \
    do \
      install -m 0644 "$source_root/$relative_path" "$package_root/$relative_path"; \
    done; \
    /opt/venv/bin/python -m py_compile \
      "$source_root/envs.py" \
      "$source_root/model_executor/layers/attention/mla_attention.py" \
      "$source_root/models/kimi_k3/nvidia/mla.py" \
      "$source_root/models/kimi_k3/nvidia/model.py" \
      "$source_root/v1/attention/backends/mla/prefill/flash_attn.py"

LABEL local.kimi-k3.prefill-policy="scheduler1536-internal24576" \
      local.kimi-k3.prefill-attention="sm120-fa4-kimi192x128-lse" \
      local.kimi-k3.prefill-projection="retained-shared-workspace" \
      local.kimi-k3.prefill-quality="dense-bf16-exact" \
      local.kimi-k3.vllm.commit="${VLLM_COMMIT}" \
      org.opencontainers.image.version="k3-prefill-sm120-fa4-20260901-r3"
