#!/usr/bin/env bash
# Issue-75 QSRT target entrypoint for the REMOTE DFlash K7 profile (9-GPU).
# Identical patch/LMCache chain to entrypoint-issue75-colocated-dspark-k3-dcp8.sh;
# only the final launcher differs (DFlash K7 speculative config + geometry).
set -euo pipefail

# Apply the already-qualified fine-cache, heartbeat, scheduler, AttnRes, and
# grammar composition guards to the exact issue-75 source tree.
K3PATCH_ONLY=1 \
K3PATCH_INNER_LAUNCHER=/usr/local/bin/serve-kimi-k3-qsrt-dspark-r29 \
  /usr/local/bin/entrypoint-aliases-remote-dspark-r29-fine1536.sh

# Hybrid MLA + recurrent attention needs the source-compatible LMCache
# classifier backport (pure-MLA TP rank sharing would restore the wrong
# recurrent shard on nonzero ranks). Apply before any worker imports LMCache.
python3 /usr/local/bin/patch-lmcache-mla-only.py

# Hybrid recurrent state is not yet qualified for a request that splices an
# LMCache tail onto a nonzero local APC prefix. Recompute only that tail while
# preserving full external resumes and full local APC hits.
python3 /usr/local/bin/patch-lmcache-mixed-recurrent-guard.py

# fine4608 profile: validate the LMCache mamba step alignment against the
# snapshot cadence (mamba_block_size) instead of the physical block, so
# 4608-token scheduler steps pass. Skipped when the profile does not mount
# the patch; cadence==block profiles are byte-identical anyway.
if [[ -f /usr/local/bin/patch-lmcache-mamba-cadence-validator.py ]]; then
  python3 /usr/local/bin/patch-lmcache-mamba-cadence-validator.py
fi

# decode-80 diagnostic: split the remote-speculator metadata_d2h wait into
# entry_drain / prep_drain / copy_sync. Skipped when not mounted.
if [[ -f /usr/local/bin/patch-remote-speculator-timing-detail.py ]]; then
  python3 /usr/local/bin/patch-remote-speculator-timing-detail.py
fi

# DFlash draft fine-tuning: persist the served auxiliary-feature stream.
# Inert without VLLM_K3_DRAFT_CAPTURE_DIR; skipped when not mounted.
if [[ -f /usr/local/bin/patch-remote-speculator-capture.py ]]; then
  python3 /usr/local/bin/patch-remote-speculator-capture.py
fi

if [[ -f /usr/local/bin/patch-runner-graph-sync-diag.py ]]; then
  python3 /usr/local/bin/patch-runner-graph-sync-diag.py
fi

if [[ -f /usr/local/bin/patch-worker-profile-rank0-only.py ]]; then
  python3 /usr/local/bin/patch-worker-profile-rank0-only.py
fi

# The remote target parses the DFlash2 config for auxiliary taps and K, but
# never loads draft weights. Register its architecture as DFlash-compatible
# metadata before ModelConfig validation. Skipped on the DFlash1 profile.
if [[ -f /usr/local/bin/patch-target-dflash2-config.py ]]; then
  python3 /usr/local/bin/patch-target-dflash2-config.py
fi

# Colocated DFlash: keep draft SWA layers out of KV-connector registration.
if [[ -f /usr/local/bin/patch-kv-connector-skip-dflash-draft-tp9.py ]]; then
  python3 /usr/local/bin/patch-kv-connector-skip-dflash-draft-tp9.py
fi

# decode-80: stripe split-K only for decode-sized MoE launches.
if [[ -f /usr/local/bin/patch-w4a16-splitk-decode-only.py ]]; then
  python3 /usr/local/bin/patch-w4a16-splitk-decode-only.py
fi

# decode-80: open island_rs (cascade-aware AR) to TP8 two-island worlds.
if [[ -f /usr/local/bin/patch-island-rs-ws8.py ]]; then
  python3 /usr/local/bin/patch-island-rs-ws8.py
fi

export PYTHONPATH="/opt/infernal-invocation/lmcache:/opt/infernal-invocation/b12x:/opt/infernal-invocation/vllm:/opt/venv/lib/python3.12/site-packages"

export LMCACHE_MODE="${LMCACHE_MODE:-disk}"
export LMCACHE_TRANSFER_MODE="${LMCACHE_TRANSFER_MODE:-engine_driven}"
export LMCACHE_EXPECTED_SOURCE_ROOT="${LMCACHE_EXPECTED_SOURCE_ROOT:-/opt/infernal-invocation/lmcache}"
export LMCACHE_SEPARATE_OBJECT_GROUPS="${LMCACHE_SEPARATE_OBJECT_GROUPS:-1}"
export LMCACHE_SERVER_ENV="${LMCACHE_SERVER_ENV-CUDA_VISIBLE_DEVICES= CUDA_MODULE_LOADING=LAZY PYTHONPATH=/opt/infernal-invocation/b12x:/opt/infernal-invocation/vllm}"
export LMCACHE_SHM_NAME="${LMCACHE_SHM_NAME:-kimik3_remote9g_dflash7_fine1536_l1_48g_v1}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-1536}"
export LMCACHE_L1_INIT_GB="${LMCACHE_L1_INIT_GB:-48}"
export LMCACHE_L1_GB="${LMCACHE_L1_GB:-48}"
export LMCACHE_SHM_RESERVE_GB="${LMCACHE_SHM_RESERVE_GB:-4}"
export LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT="${LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT:-1}"
export LMCACHE_L2_GB="${LMCACHE_L2_GB:-512}"
export LMCACHE_L2_WORKERS="${LMCACHE_L2_WORKERS:-4}"
export LMCACHE_MAX_GPU_WORKERS="${LMCACHE_MAX_GPU_WORKERS:-8}"
export LMCACHE_MAX_CPU_WORKERS="${LMCACHE_MAX_CPU_WORKERS:-4}"
export LMCACHE_PORT="${LMCACHE_PORT:-5555}"
export LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8088}"
export LMCACHE_START_TIMEOUT="${LMCACHE_START_TIMEOUT:-120}"
export LMCACHE_LOG="${LMCACHE_LOG:-/tmp/lmcache-mp-8090.log}"
export LMCACHE_HEARTBEAT_INTERVAL="${LMCACHE_HEARTBEAT_INTERVAL:-5}"
export LMCACHE_WORKER_REAP_TIMEOUT="${LMCACHE_WORKER_REAP_TIMEOUT:-120}"
export LMCACHE_WORKER_REGISTRATION_GRACE="${LMCACHE_WORKER_REGISTRATION_GRACE:-86400}"
export LMCACHE_KV_LOAD_FAILURE_POLICY="${LMCACHE_KV_LOAD_FAILURE_POLICY:-recompute}"

# Docker automatic restarts do not rerun Compose dependencies. Remove every
# unmapped LMCache pool before starting: the raw SHM arena does not persist
# the L1 key/allocation index, so keeping a same-name file preserves no usable
# cache and makes the independent capacity check select pickle transport.
# Require a fresh direct-SHM pool plus reserve for this TP8 profile.
python3 /usr/local/bin/lmcache_shm_preflight.py \
  --required-gib "${LMCACHE_L1_GB}" \
  --reserve-gib "${LMCACHE_SHM_RESERVE_GB}"

# Full-external restores across process generations were qualified
# byte-identical; the unsafe mixed APC+external tail case recompute-guards
# at runtime via patch-lmcache-mixed-recurrent-guard.py, so the namespace no
# longer needs a per-boot wipe. LMCACHE_L2_RESET_ON_START=1 forces one.
if [[ "${LMCACHE_L2_RESET_ON_START:-0}" == "1" ]]; then
  python3 /usr/local/bin/lmcache_l2_process_generation.py \
    --path /lmcache/l2 \
    --expected-leaf "l2"
fi

exec /usr/local/bin/lmcache-mp-wrapper.sh \
  /usr/local/bin/serve-kimi-k3-issue75-dflash-k7-dcp8.sh
