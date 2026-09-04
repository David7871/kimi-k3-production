FROM kimi-k3-production-issue75-dspark@sha256:f18184728c27959de26cf11952d7a8b96e57b513a6badf8bf3e09fd4cda16a76

LABEL org.opencontainers.image.title="Kimi-K3 QSRT/EXL3 SM120 compatible acceleration candidate" \
      local.kimi-k3.base-digest="sha256:f18184728c27959de26cf11952d7a8b96e57b513a6badf8bf3e09fd4cda16a76" \
      local.kimi-k3.vllm-prs="52388,52789,53614,53080,54348" \
      local.kimi-k3.lmcache-prs="44101" \
      local.kimi-k3.flashkda.commit="ee0be888cd0e972f9409bf53756f8c38c6652173" \
      local.kimi-k3.flashkda.arch="sm_120a" \
      local.kimi-k3.flashkda.runtime="sm120-tensormap-no-bulk-syscall" \
      local.kimi-k3.flashkda.binary-sha256="c0b440ab87e6cfec16b23138f95a67d8fe9142836c040a56c41a574dc7d27ed1" \
      local.kimi-k3.b12x.policy="retain-production-qsrt-k2" \
      local.kimi-k3.moon-ep.policy="excluded-incompatible-sqg-k2-single-node"

COPY PATCHSET.md /opt/kimi-k3-candidate/PATCHSET.md
COPY artifacts/flashkda/ee0be888-prod-sm120a-cu133-torch213-tensormap/_flashkda_C.abi3.so \
    /opt/infernal-invocation/vllm/vllm/_flashkda_C.abi3.so

COPY vllm/cmake/external_projects/flashkda.cmake \
    /opt/infernal-invocation/vllm/cmake/external_projects/flashkda.cmake
COPY vllm/csrc/flashkda_registration.cpp \
    /opt/infernal-invocation/vllm/csrc/flashkda_registration.cpp
COPY vllm/vllm/config/speculative.py \
    /opt/infernal-invocation/vllm/vllm/config/speculative.py
COPY vllm/vllm/config/vllm.py \
    /opt/infernal-invocation/vllm/vllm/config/vllm.py
COPY vllm/vllm/models/kimi_k3/nvidia/kda.py \
    /opt/infernal-invocation/vllm/vllm/models/kimi_k3/nvidia/kda.py
COPY vllm/vllm/models/kimi_k3/nvidia/kda_metadata.py \
    /opt/infernal-invocation/vllm/vllm/models/kimi_k3/nvidia/kda_metadata.py
COPY vllm/vllm/v1/core/block_pool.py \
    /opt/infernal-invocation/vllm/vllm/v1/core/block_pool.py
COPY vllm/vllm/v1/core/sched/scheduler.py \
    /opt/infernal-invocation/vllm/vllm/v1/core/sched/scheduler.py
COPY vllm/vllm/v1/core/single_type_kv_cache_manager.py \
    /opt/infernal-invocation/vllm/vllm/v1/core/single_type_kv_cache_manager.py
COPY vllm/vllm/v1/kv_cache_interface.py \
    /opt/infernal-invocation/vllm/vllm/v1/kv_cache_interface.py
COPY vllm/vllm/v1/request.py \
    /opt/infernal-invocation/vllm/vllm/v1/request.py
COPY vllm/vllm/v1/worker/gpu/model_states/mamba_hybrid.py \
    /opt/infernal-invocation/vllm/vllm/v1/worker/gpu/model_states/mamba_hybrid.py
COPY vllm/vllm/v1/worker/mamba_utils.py \
    /opt/infernal-invocation/vllm/vllm/v1/worker/mamba_utils.py
COPY lmcache/lmcache/integration/vllm/lmcache_mp_connector.py \
    /opt/infernal-invocation/lmcache/lmcache/integration/vllm/lmcache_mp_connector.py

RUN set -eu; \
    source_root=/opt/infernal-invocation/vllm/vllm; \
    package_root=/opt/venv/lib/python3.12/site-packages/vllm; \
    for relative_path in \
      _flashkda_C.abi3.so \
      config/speculative.py \
      config/vllm.py \
      models/kimi_k3/nvidia/kda.py \
      models/kimi_k3/nvidia/kda_metadata.py \
      v1/core/block_pool.py \
      v1/core/sched/scheduler.py \
      v1/core/single_type_kv_cache_manager.py \
      v1/kv_cache_interface.py \
      v1/request.py \
      v1/worker/gpu/model_states/mamba_hybrid.py \
      v1/worker/mamba_utils.py; \
    do \
      install -m 0644 "$source_root/$relative_path" "$package_root/$relative_path"; \
    done; \
    install -m 0644 \
      /opt/infernal-invocation/lmcache/lmcache/integration/vllm/lmcache_mp_connector.py \
      /opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/lmcache_mp_connector.py; \
    chmod 0755 "$source_root/_flashkda_C.abi3.so" \
      "$package_root/_flashkda_C.abi3.so"
