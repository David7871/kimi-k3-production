# Kimi-K3 production runtime plus a self-contained deployment bundle.
# Model weights and persistent cache data remain host-managed and are excluded.
FROM kimi-k3-production-issue75-dspark:all-compatible-20260830-r5

COPY runtime-patches/vllm/vllm/v1/core/sched/scheduler.py \
  /opt/infernal-invocation/vllm/vllm/v1/core/sched/scheduler.py
COPY runtime-patches/lmcache/lmcache/integration/vllm/lmcache_mp_connector.py \
  /opt/infernal-invocation/lmcache/lmcache/integration/vllm/lmcache_mp_connector.py
COPY deploy/bin/patch-lmcache-mixed-recurrent-guard.py \
  /usr/local/bin/patch-lmcache-mixed-recurrent-guard.py
COPY deploy/bin/entrypoint-issue75-remote-dflash-k7-dcp8.sh \
  /usr/local/bin/entrypoint-issue75-remote-dflash-k7-dcp8.sh
COPY deploy/bin/lmcache_shm_preflight.py \
  /usr/local/bin/lmcache_shm_preflight.py

RUN set -eu; \
    chmod 0755 \
      /usr/local/bin/patch-lmcache-mixed-recurrent-guard.py \
      /usr/local/bin/entrypoint-issue75-remote-dflash-k7-dcp8.sh \
      /usr/local/bin/lmcache_shm_preflight.py; \
    K3_LMCACHE_SOURCE_ROOT=/opt/infernal-invocation/lmcache/lmcache \
    K3_VLLM_SOURCE_ROOT=/opt/infernal-invocation/vllm/vllm \
      python3 /usr/local/bin/patch-lmcache-mixed-recurrent-guard.py; \
    install -m 0644 \
      /opt/infernal-invocation/vllm/vllm/v1/core/sched/scheduler.py \
      /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py; \
    install -m 0644 \
      /opt/infernal-invocation/lmcache/lmcache/integration/vllm/lmcache_mp_connector.py \
      /opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/lmcache_mp_connector.py; \
    install -m 0644 \
      /opt/infernal-invocation/lmcache/lmcache/__init__.py \
      /opt/venv/lib/python3.12/site-packages/lmcache/__init__.py; \
    python3 -m py_compile \
      /opt/infernal-invocation/vllm/vllm/v1/core/sched/scheduler.py \
      /opt/infernal-invocation/lmcache/lmcache/integration/vllm/lmcache_mp_connector.py \
      /opt/infernal-invocation/lmcache/lmcache/__init__.py; \
    /opt/venv/bin/python -c "from pathlib import Path; site_pkg=Path('/opt/venv/lib/python3.12/site-packages/lmcache'); source_pkg=Path('/opt/infernal-invocation/lmcache/lmcache'); assert list(site_pkg.glob('c_ops*.so')); assert '_LMCACHE_BINARY_PACKAGE_PATHS' in (source_pkg/'__init__.py').read_text(); assert 'resuming_completed_remote_kv' in Path('/opt/infernal-invocation/vllm/vllm/v1/core/sched/scheduler.py').read_text(); assert 'num_lmcache_hit_tokens' in Path('/opt/infernal-invocation/lmcache/lmcache/integration/vllm/lmcache_mp_connector.py').read_text()"

COPY deploy /opt/kimi-k3-deploy
LABEL com.mydatacenter.deploy-bundle="all-compatible-20260901-r6" \
      com.mydatacenter.runtime-fixes="async-kv-resume-admission,lmcache-abort-cleanup,native-c-ops-package-path,fresh-l1-arena" \
      org.opencontainers.image.version="all-compatible-20260901-r6-deploy" \
      org.opencontainers.image.description="Kimi-K3 TP8/DCP8 runtime with LMCache and DFlash deployment assets; model weights excluded"
