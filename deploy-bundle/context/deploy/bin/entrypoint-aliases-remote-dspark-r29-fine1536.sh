#!/usr/bin/env bash
set -euo pipefail

# Kimi-K3 r29 (upstream-aligned 20260821) remote-DSpark production image
# LMCache operational patches. Runs before the packaged launcher on every
# container start. All edits are idempotent: each block verifies its target
# state and fails loudly if the packaged files drift from the expected anchors.

# The packaged lmcache-mp-wrapper.sh hard-codes the retain prefetch policy and
# does not bound concurrent prefetch jobs. Retain keeps every L2-prefetched
# chunk resident in L1; overlapping long-prefix restores also multiply
# temporary transfer buffers. Both conditions can grow the LMCache server far
# beyond its 16 GiB L1 budget and trigger the kernel OOM killer. Use temporary
# staging plus one in-flight prefetch by default.
python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ.get("K3PATCH_WRAPPER", "/usr/local/bin/lmcache-mp-wrapper.sh"))
old = "--l2-prefetch-policy retain"
new = "--l2-prefetch-policy default"
text = path.read_text()
if new in text:
    print(f"LMCache prefetch policy already patched: {new}")
elif text.count(old) == 1:
    path.write_text(text.replace(old, new, 1))
    print(f"Applied LMCache prefetch policy patch: {old} -> {new}")
else:
    raise SystemExit(
        f"ERROR: expected exactly one {old!r} in {path}, found {text.count(old)}"
    )

text = path.read_text()
anchor = "--l2-prefetch-policy default"
max_in_flight = (
    '--l2-prefetch-max-in-flight '
    '"${LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT:-1}"'
)
if max_in_flight in text:
    print(f"LMCache prefetch concurrency already patched: {max_in_flight}")
elif text.count(anchor) == 1:
    path.write_text(text.replace(anchor, anchor + "\n  " + max_in_flight, 1))
    print(f"Applied LMCache prefetch concurrency patch: {max_in_flight}")
else:
    raise SystemExit(
        f"ERROR: expected exactly one {anchor!r} in {path}, "
        f"found {text.count(anchor)}"
    )
PY

# L1 TTLs. The packaged wrapper expires L1 entries 600s after write / 300s
# after read, so any coding session paused for >10 minutes loses its L1
# residency and falls back to coarse L2 disk chunks. L1 capacity is already
# governed by the LRU eviction watermarks, so lengthen the TTLs to cover
# realistic pauses. Worker registration grace is env-configurable in this
# wrapper revision (LMCACHE_WORKER_REGISTRATION_GRACE, set to 86400 by the
# outer launcher), so no file edit is needed for it.
python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ.get("K3PATCH_WRAPPER", "/usr/local/bin/lmcache-mp-wrapper.sh"))
text = path.read_text()

def sub_once(text, old, new):
    if new in text:
        print(f"already patched: {new}")
        return text
    if text.count(old) != 1:
        raise SystemExit(
            f"ERROR: expected exactly one {old!r} in {path}, found {text.count(old)}"
        )
    print(f"applied: {old} -> {new}")
    return text.replace(old, new, 1)

text = sub_once(text, "--l1-write-ttl-seconds 600", "--l1-write-ttl-seconds 7200")
text = sub_once(text, "--l1-read-ttl-seconds 300", "--l1-read-ttl-seconds 3600")
path.write_text(text)
PY

# Inject an independent heartbeat PING timeout into the connector transfer
# config the wrapper generates. The wrapper only emits heartbeat_interval;
# the adapter-side decouple patch below reads heartbeat_timeout from
# extra_config. Ambient LMCACHE_HEARTBEAT_TIMEOUT overrides the 30s default.
python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ.get("K3PATCH_WRAPPER", "/usr/local/bin/lmcache-mp-wrapper.sh"))
anchor = 'heartbeat_interval = float(os.environ["LMCACHE_JSON_HEARTBEAT_INTERVAL"])'
addition = (
    anchor
    + '\nheartbeat_timeout = float(os.environ.get("LMCACHE_HEARTBEAT_TIMEOUT", "30"))'
)
json_anchor = '"lmcache.mp.heartbeat_interval": heartbeat_interval,'
json_addition = (
    json_anchor
    + '\n                "lmcache.mp.heartbeat_timeout": heartbeat_timeout,'
)

text = path.read_text()
if '"lmcache.mp.heartbeat_timeout"' in text:
    print("wrapper already patched: heartbeat_timeout present")
else:
    if text.count(anchor) != 1:
        raise SystemExit(
            f"ERROR: expected exactly one heartbeat_interval assignment anchor, "
            f"found {text.count(anchor)}"
        )
    if text.count(json_anchor) != 1:
        raise SystemExit(
            f"ERROR: expected exactly one heartbeat_interval JSON anchor, "
            f"found {text.count(json_anchor)}"
        )
    text = text.replace(anchor, addition, 1)
    text = text.replace(json_anchor, json_addition, 1)
    path.write_text(text)
    print("wrapper patched: heartbeat_timeout injected into transfer config")
PY

# Start each worker's per-instance LMCache heartbeat immediately after KV-cache
# registration instead of lazily on the first store/retrieve. With the lazy
# start, a fresh engine that idles past worker_registration_grace_seconds
# without a chunk-sized request has its registrations reaped while still
# healthy, and the next store crashes all workers (observed 2026-08-20 15:02
# KST, full engine restart). Pinging from registration onward latches
# has_liveness_signal and keeps last_seen fresh, so the reaper never targets
# a live worker.
python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ.get(
    "K3PATCH_ADAPTER",
    "/opt/infernal-invocation/lmcache/lmcache/integration/vllm/vllm_multi_process_adapter.py",
))
marker = "heartbeat-at-registration"
anchor = '''        except TimeoutError:
            raise ConnectionError(
                "LMCache server did not respond to "
                "register_kv_caches within "
                f"{self._mq_timeout}s. Is the server running?"
            ) from None
'''
insert = '''
        # %s: a lazy (first store/retrieve) heartbeat lets the server reap a
        # live but never-pinged registration after the grace window; the next
        # PREPARE_STORE then raises and kills every worker. Ping from
        # registration onward so the reaper never targets a live worker.
        self._ensure_heartbeat_started()
''' % marker

text = path.read_text()
if marker in text:
    print(f"adapter already patched: {marker}")
elif text.count(anchor) == 1:
    path.write_text(text.replace(anchor, anchor + insert, 1))
    print(f"applied adapter patch: {marker}")
else:
    raise SystemExit(
        f"ERROR: expected exactly one register anchor in {path}, found {text.count(anchor)}"
    )
PY

# Heartbeat PING timeout decouple (PR #4593 backport for this LMCache revision).
#
# The stock adapter couples the heartbeat PING response timeout to the
# heartbeat interval, so any server busy spell longer than the interval — e.g.
# a large external-cache restore with dynamic pinning, observed 2026-08-21
# 13:31/17:29 KST — flips workers into degraded mode and back. Give the PING
# an independent 30s timeout: bounded NORMAL-pool queue latency no longer
# causes health flaps, while a genuinely dead server still degrades after 30s.
# Source-matched adapter suite: 28 passed in a disposable container.
python3 - <<'PY'
import os
from pathlib import Path

ADAPTER = Path(os.environ.get(
    "K3PATCH_ADAPTER",
    "/opt/infernal-invocation/lmcache/lmcache/integration/vllm/vllm_multi_process_adapter.py",
))
MARKER = "heartbeat-timeout-decouple"


def sub(text, old, new, count=1):
    found = text.count(old)
    if found != count:
        raise SystemExit(
            f"ERROR: expected {count} occurrence(s) of anchor {old[:70]!r}..., found {found}"
        )
    return text.replace(old, new, count)


text = ADAPTER.read_text()
if MARKER in text:
    print(f"adapter already patched: {MARKER}")
else:
    # ExtraConfigDefault gains heartbeat_timeout (default 0.0 = legacy).
    text = sub(
        text,
        "    # Interval (seconds) between periodic heartbeat pings\n"
        "    # to the server.\n"
        "    heartbeat_interval = 10.0\n",
        "    # Interval (seconds) between periodic heartbeat pings\n"
        "    # to the server.\n"
        "    heartbeat_interval = 10.0\n"
        "    # Timeout (seconds) for each heartbeat PING. Zero preserves the\n"
        "    # historical behavior of using heartbeat_interval as the timeout.\n"
        f"    # {MARKER}\n"
        "    heartbeat_timeout = 0.0\n",
    )
    # Both adapters resolve heartbeat_timeout from extra_config.
    text = sub(
        text,
        "            heartbeat_interval = cfg[ExtraConfigDefault.heartbeat_interval.name]\n",
        "            heartbeat_interval = cfg[ExtraConfigDefault.heartbeat_interval.name]\n"
        "            heartbeat_timeout = cfg[ExtraConfigDefault.heartbeat_timeout.name]\n",
        count=2,
    )
    # Defaults when extra_config is None (scheduler, then worker).
    text = sub(
        text,
        "        self.mq_clients: dict[str, MessageQueueClient] = {\n"
        "            url: MessageQueueClient(url, context) for url in self._server_urls\n"
        "        }\n",
        "        self.mq_clients: dict[str, MessageQueueClient] = {\n"
        "            url: MessageQueueClient(url, context) for url in self._server_urls\n"
        "        }\n"
        "        heartbeat_timeout = 0.0\n",
    )
    text = sub(
        text,
        "            legacy_block_size,\n"
        "            mq_timeout,\n"
        "        )\n"
        "        if extra_config is not None:\n",
        "            legacy_block_size,\n"
        "            mq_timeout,\n"
        "        )\n"
        "        heartbeat_timeout = 0.0\n"
        "        if extra_config is not None:\n",
    )
    # Both adapters store the resolved value.
    text = sub(
        text,
        "        self._heartbeat_interval = heartbeat_interval\n",
        "        self._heartbeat_interval = heartbeat_interval\n"
        "        self._heartbeat_timeout = heartbeat_timeout\n",
        count=2,
    )
    # HeartbeatThread accepts an independent timeout.
    text = sub(
        text,
        "        interval: float = DEFAULT_HEARTBEAT_INTERVAL,\n"
        "        instance_id: int | None = None,\n"
        "    ):\n",
        "        interval: float = DEFAULT_HEARTBEAT_INTERVAL,\n"
        "        instance_id: int | None = None,\n"
        "        timeout: float = 0.0,\n"
        "    ):\n",
    )
    text = sub(
        text,
        "        self._interval = interval\n",
        "        self._interval = interval\n"
        "        # heartbeat_timeout > 0 decouples the PING response timeout from\n"
        "        # the cadence; bounded queue latency must not flip health.\n"
        "        self._effective_timeout = timeout if timeout > 0.0 else interval\n",
    )
    text = sub(
        text,
        "self._mq_client, timeout=self._interval, instance_id=self._instance_id",
        "self._mq_client, timeout=self._effective_timeout, instance_id=self._instance_id",
    )
    # Both heartbeat creation sites pass the configured timeout.
    text = sub(
        text,
        "                    interval=self._heartbeat_interval,\n"
        "                )\n",
        "                    interval=self._heartbeat_interval,\n"
        "                    timeout=self._heartbeat_timeout,\n"
        "                )\n",
    )
    text = sub(
        text,
        "                interval=self._heartbeat_interval,\n"
        "                instance_id=self.instance_id,\n",
        "                interval=self._heartbeat_interval,\n"
        "                timeout=self._heartbeat_timeout,\n"
        "                instance_id=self.instance_id,\n",
    )
    # Worker-side reap-floor warning accounts for the effective timeout.
    old_warn = (
        "        if 3 * heartbeat_interval > _SERVER_REAP_TIMEOUT_FLOOR_SECONDS:\n"
        "            logger.warning(\n"
        "                \"lmcache.mp.heartbeat_interval is %.1fs, so 3 x \"\n"
        "                \"heartbeat_interval (%.1fs) exceeds the MP server's \"\n"
        "                \"default worker reap timeout floor (%.1fs). Raise the \"\n"
        "                \"server's worker reap timeout to at least 3 x the \"\n"
        "                \"heartbeat interval, or the server may reap this \"\n"
        "                \"worker between heartbeats.\",\n"
        "                heartbeat_interval,\n"
        "                3 * heartbeat_interval,\n"
        "                _SERVER_REAP_TIMEOUT_FLOOR_SECONDS,\n"
        "            )\n"
    )
    new_warn = (
        "        effective_heartbeat_timeout = (\n"
        "            heartbeat_timeout if heartbeat_timeout > 0.0 else heartbeat_interval\n"
        "        )\n"
        "        heartbeat_refresh_gap = max(\n"
        "            3 * heartbeat_interval,\n"
        "            heartbeat_interval + effective_heartbeat_timeout,\n"
        "        )\n"
        "        if heartbeat_refresh_gap > _SERVER_REAP_TIMEOUT_FLOOR_SECONDS:\n"
        "            logger.warning(\n"
        "                \"LMCache heartbeat refresh gap can reach %.1fs with \"\n"
        "                \"heartbeat_interval=%.1fs and heartbeat_timeout=%.1fs, \"\n"
        "                \"which exceeds the MP server's default worker reap \"\n"
        "                \"timeout floor (%.1fs). Raise the server's worker reap \"\n"
        "                \"timeout above the refresh gap, or the server may reap \"\n"
        "                \"this worker between heartbeats.\",\n"
        "                heartbeat_refresh_gap,\n"
        "                heartbeat_interval,\n"
        "                effective_heartbeat_timeout,\n"
        "                _SERVER_REAP_TIMEOUT_FLOOR_SECONDS,\n"
        "            )\n"
    )
    text = sub(text, old_warn, new_warn)
    compile(text, str(ADAPTER), "exec")
    ADAPTER.write_text(text)
    print(f"adapter patched: {MARKER}")
PY

# Decode-step CUDA-graph padding elimination. The packaged launcher captures
# graph sizes [8..64], but a single-stream DSpark3 decode step presents exactly
# 1+3=4 tokens and is padded to the size-8 graph. Padded rows still route
# through the MoE, so they activate extra experts and burn HBM bandwidth on
# every decode step. Capture size 4 lets single-stream steps run unpadded;
# two-sequence steps (8 tokens) still hit the size-8 graph.
python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ.get(
    "K3PATCH_INNER_LAUNCHER",
    "/usr/local/bin/serve-kimi-k3-qsrt-dspark-r29",
))
old = '"cudagraph_capture_sizes":[8,16,24,32,40,48,56,64]'
new = '"cudagraph_capture_sizes":[4,8,16,24,32,40,48,56,64]'
text = path.read_text()
if new in text:
    print(f"capture-size-4 already patched: {new}")
elif text.count(old) == 1:
    path.write_text(text.replace(old, new, 1))
    print(f"applied capture-size-4 patch: {old} -> {new}")
else:
    raise SystemExit(
        f"ERROR: expected exactly one {old!r} in {path}, found {text.count(old)}"
    )
PY

# vLLM PR #467 backport: preserve the final committed AttnRes residual block.
# Kimi-K3 has 93 layers with AttnRes block size 12, so layer 84 commits the
# eighth residual-bank slot. Reusing that storage for the post-attention
# output overwrites a residual source consumed by layers 85-92 and the final
# AttnRes mixture. Stop output reuse at the final block-write boundary.
python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ.get(
    "K3PATCH_KIMI_MODEL",
    "/opt/infernal-invocation/vllm/vllm/models/kimi_k3/nvidia/model.py",
))
marker = "attnres-final-block-preserve"
init_old = (
    "            self.is_block_write_layer = layer_idx % self.attn_res_block_size == 0\n"
    "            self.block_write_idx = layer_idx // self.attn_res_block_size\n"
)
init_new = init_old + (
    "            # %s: the final block-write layer commits the last residual-bank\n"
    "            # slot; its storage must stay immutable for layers 85-92.\n"
    "            self.is_final_block_write_layer = (\n"
    "                self.is_block_write_layer\n"
    "                and self.block_write_idx\n"
    "                == cdiv(config.num_hidden_layers, self.attn_res_block_size) - 1\n"
    "            )\n"
) % marker
post_old = (
    "        if self.is_block_write_layer:\n"
    "            output = prefix_sum if self.reuse_attn_res_output else None\n"
    "            prefix_sum = hidden_states\n"
    "            prefix_delta = None\n"
)
post_new = (
    "        if self.is_block_write_layer:\n"
    "            # The old prefix becomes the last committed residual block at the\n"
    "            # final block boundary. It must remain immutable for every later\n"
    "            # AttnRes mixture and therefore cannot also hold the new delta.\n"
    "            output = (\n"
    "                prefix_sum\n"
    "                if self.reuse_attn_res_output and not self.is_final_block_write_layer\n"
    "                else None\n"
    "            )\n"
    "            prefix_sum = hidden_states\n"
    "            prefix_delta = None\n"
)
text = path.read_text()
semantic_applied = (
    "self.is_final_block_write_layer = (" in text
    and "and not self.is_final_block_write_layer" in text
)
if marker in text or semantic_applied:
    print(f"model.py semantic already present: {marker}")
else:
    if text.count(init_old) != 1:
        raise SystemExit(
            f"ERROR: expected exactly one AttnRes init anchor, found {text.count(init_old)}"
        )
    if text.count(post_old) != 1:
        raise SystemExit(
            f"ERROR: expected exactly one _post_attn_norm anchor, found {text.count(post_old)}"
        )
    text = text.replace(init_old, init_new, 1)
    text = text.replace(post_old, post_new, 1)
    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"applied model.py patch: {marker}")

# The PR also updates the mock-layer helper so existing AttnRes tests keep
# working with the new attribute. Tests are not runtime-loaded, but keeping
# the suite green protects future regression runs.
test_path = Path(os.environ.get(
    "K3PATCH_KIMI_TEST_EAGLE3",
    "/opt/infernal-invocation/vllm/tests/models/kimi_k3/test_eagle3.py",
))
helper_old = '    object.__setattr__(layer, "is_block_write_layer", block_write)\n'
helper_new = (
    helper_old
    + '    object.__setattr__(layer, "is_final_block_write_layer", False)\n'
)
if test_path.is_file():
    test_text = test_path.read_text()
    if '"is_final_block_write_layer", False' in test_text:
        print("test_eagle3 helper already patched")
    elif test_text.count(helper_old) == 1:
        test_path.write_text(test_text.replace(helper_old, helper_new, 1))
        print("applied test_eagle3 helper patch")
    else:
        raise SystemExit(
            f"ERROR: expected exactly one test helper anchor, found {test_text.count(helper_old)}"
        )
PY

# vLLM PR #439 backport: skip speculative decoding when every scheduled
# request needs at most one output token. Drafting cannot pay off for a
# 1-token output, so the draft pass plus verification is pure added latency.
# Only fires when no running request is scheduled, so an in-flight multi-token
# generation can never lose its draft tokens.
python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ.get(
    "K3PATCH_VLLM_SCHEDULER",
    "/opt/infernal-invocation/vllm/vllm/v1/core/sched/scheduler.py",
))
marker = "skip-spec-max1-token"
anchor = (
    "self.dynamic_sd_lookup[len(num_scheduled_tokens)],\n"
    "            )\n"
    "\n"
    "        scheduled_encoder_input_stats = None"
)
addition = (
    "self.dynamic_sd_lookup[len(num_scheduled_tokens)],\n"
    "            )\n"
    "\n"
    "        # %s: drafting cannot pay off when every scheduled request needs\n"
    "        # at most one output token; the draft pass plus verification is\n"
    "        # pure added latency. Only fires when no running request is\n"
    "        # scheduled, so an in-flight multi-token generation keeps drafts.\n"
    "        if (\n"
    "            num_spec_tokens_to_schedule > 0\n"
    "            and scheduled_new_reqs\n"
    "            and not scheduled_running_reqs\n"
    "            and all(req.max_tokens <= 1 for req in scheduled_new_reqs)\n"
    "        ):\n"
    "            num_spec_tokens_to_schedule = 0\n"
    "\n"
    "        scheduled_encoder_input_stats = None"
) % marker
text = path.read_text()
semantic_applied = "and all(req.max_tokens <= 1 for req in scheduled_new_reqs)" in text
if marker in text or semantic_applied:
    print(f"scheduler.py semantic already present: {marker}")
else:
    if text.count(anchor) != 1:
        raise SystemExit(
            f"ERROR: expected exactly one scheduler insertion anchor, found {text.count(anchor)}"
        )
    text = text.replace(anchor, addition, 1)
    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"applied scheduler.py patch: {marker}")
PY

# Remote-DSpark compatibility for PR #439: the scheduler can intentionally
# request zero speculative tokens for a one-token response. Local model-runner
# code accepts an empty [batch, 0] draft tensor, but the remote proxy previously
# rejected active_k=0 before reaching that contract.
python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ.get(
    "K3PATCH_REMOTE_SPECULATOR",
    "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/spec_decode/dspark/remote_speculator.py",
))
marker = "skip-zero-depth-remote-dspark"
anchor = (
    "        if not 1 <= active_k <= self.num_speculative_steps:\n"
    "            raise ValueError(\n"
)
addition = (
    "        # %s: scheduler-side speculative skip uses a zero-width\n"
    "        # draft tensor; model_runner handles this as a non-spec step.\n"
    "        if active_k == 0:\n"
    "            return self.draft_tokens[: input_batch.num_reqs, :0].contiguous()\n"
    "        if not 1 <= active_k <= self.num_speculative_steps:\n"
    "            raise ValueError(\n"
) % marker
text = path.read_text()
semantic_applied = (
    "if active_k == 0:" in text
    and "return self.draft_tokens[: input_batch.num_reqs, :0].contiguous()" in text
)
if marker in text or semantic_applied:
    print(f"remote_speculator.py semantic already present: {marker}")
else:
    if text.count(anchor) != 1:
        raise SystemExit(
            "ERROR: expected exactly one remote speculator depth anchor, "
            f"found {text.count(anchor)}"
        )
    text = text.replace(anchor, addition, 1)
    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"applied remote_speculator.py patch: {marker}")
PY

# V2 model-runner compatibility for PR #439: after a speculator returns the
# scheduler-requested empty draft, limit_draft_tokens must accept depth zero.
# Keep negative and over-limit counts invalid, and preserve the existing
# contract that a speculator may decline a valid draft with an empty tensor.
python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ.get(
    "K3PATCH_SPEC_DECODE_UTILS",
    "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/spec_decode/utils.py",
))
marker = "allow-zero-width-draft-limit"
anchor = (
    "    if not 1 <= num_speculative_tokens <= max_num_speculative_tokens:\n"
    "        raise RuntimeError(\n"
)
addition = (
    "    # %s: depth zero means the scheduler intentionally disabled\n"
    "    # speculation for this step; ignore any wider persistent buffer.\n"
    "    if num_speculative_tokens == 0:\n"
    "        return draft_tokens[:, :0]\n"
    "    if not 1 <= num_speculative_tokens <= max_num_speculative_tokens:\n"
    "        raise RuntimeError(\n"
) % marker
text = path.read_text()
semantic_applied = (
    "if num_speculative_tokens == 0:" in text
    and "return draft_tokens[:, :0]" in text
)
if marker in text or semantic_applied:
    print(f"spec_decode/utils.py semantic already present: {marker}")
else:
    if text.count(anchor) != 1:
        raise SystemExit(
            "ERROR: expected exactly one draft-limit range anchor, "
            f"found {text.count(anchor)}"
        )
    text = text.replace(anchor, addition, 1)
    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"applied spec_decode/utils.py patch: {marker}")
PY

# The immutable child image carries both explicit token geometry and fine
# 1,536-token external-object projection. Assert the complete semantic contract
# after the heartbeat/runtime patches above; do not mutate cache geometry here.
python3 - <<'PY'
import os
import py_compile
from pathlib import Path

paths = {
    "adapter": Path(os.environ.get(
        "K3PATCH_ADAPTER",
        "/opt/infernal-invocation/lmcache/lmcache/integration/vllm/vllm_multi_process_adapter.py",
    )),
    "connector": Path(os.environ.get(
        "K3PATCH_CONNECTOR",
        "/opt/infernal-invocation/lmcache/lmcache/integration/vllm/lmcache_mp_connector.py",
    )),
    "group_view": Path(os.environ.get(
        "K3PATCH_GROUP_VIEW",
        "/opt/infernal-invocation/lmcache/lmcache/v1/multiprocess/group_view.py",
    )),
    "worker": Path(os.environ.get(
        "K3PATCH_WORKER_TRANSFER",
        "/opt/infernal-invocation/lmcache/lmcache/v1/multiprocess/transfer_context/worker_transfer.py",
    )),
    "vllm_mamba_manager": Path(os.environ.get(
        "K3PATCH_VLLM_MAMBA_MANAGER",
        "/opt/infernal-invocation/vllm/vllm/v1/core/single_type_kv_cache_manager.py",
    )),
    "vllm_scheduler": Path(os.environ.get(
        "K3PATCH_VLLM_SCHEDULER",
        "/opt/infernal-invocation/vllm/vllm/v1/core/sched/scheduler.py",
    )),
    "remote_speculator": Path(os.environ.get(
        "K3PATCH_REMOTE_SPECULATOR",
        "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/spec_decode/dspark/remote_speculator.py",
    )),
    "spec_decode_utils": Path(os.environ.get(
        "K3PATCH_SPEC_DECODE_UTILS",
        "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/spec_decode/utils.py",
    )),
    "kimi_model": Path(os.environ.get(
        "K3PATCH_KIMI_MODEL",
        "/opt/infernal-invocation/vllm/vllm/models/kimi_k3/nvidia/model.py",
    )),
    "inner_launcher": Path(os.environ.get(
        "K3PATCH_INNER_LAUNCHER",
        "/usr/local/bin/serve-kimi-k3-qsrt-dspark-r29",
    )),
}
texts = {name: path.read_text() for name, path in paths.items()}
checks = {
    "adapter_explicit_chunk_forward": texts["adapter"].count(
        "tokens_per_chunk=self.lmcache_tokens_per_chunk"
    ),
    "adapter_multigroup_groupwise_guard": texts["adapter"].count(
        "and len(self.engine_group_infos) <= 1"
    ),
    "connector_fine_slices": texts["connector"].count(
        "external_chunk_size=lmcache_tokens_per_chunk"
    ),
    "connector_geometry_validation": texts["connector"].count(
        "validate_external_chunk_geometry("
    ),
    "group_view_geometry_definition": texts["group_view"].count(
        "def validate_external_chunk_geometry("
    ),
    "group_view_fine_slice_argument": texts["group_view"].count(
        "external_chunk_size: int | None = None"
    ),
    "worker_chunk_annotations": texts["worker"].count(
        "tokens_per_chunk: int | None = None"
    ),
    "worker_explicit_chunk_calculation": texts["worker"].count(
        "chunk_tokens = (\n"
        "            tokens_per_chunk\n"
        "            if tokens_per_chunk is not None\n"
        "            else blocks_in_chunk * block_size\n"
        "        )"
    ),
    "worker_external_chunk_state": texts["worker"].count(
        "self._external_chunk_size = chunk_tokens"
    ),
    "worker_resolve_geometry": texts["worker"].count(
        "def resolve_external_chunk_group_geometry("
    ),
    "worker_subblock_view": texts["worker"].count(
        "def make_external_subblock_view("
    ),
    "worker_block_projection": texts["worker"].count(
        "def project_external_chunk_block_ids("
    ),
    "worker_skip_projection": texts["worker"].count(
        "def project_external_chunk_skip_tokens("
    ),
    "vllm_running_state_markers": texts["vllm_mamba_manager"].count(
        "self._allocated_block_reqs.add(request_id)"
    ),
    "vllm_dspark_cache_policy": texts["vllm_scheduler"].count(
        "def use_eagle_for_target_cache("
    ),
    "vllm_dspark_cache_wiring": texts["vllm_scheduler"].count(
        "use_eagle=self.use_eagle_for_target_cache"
    ),
    "vllm_grammar_spec_width_collection": texts["vllm_scheduler"].count(
        "num_spec_tokens = [\n"
        "            len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))\n"
        "            for req_id in structured_output_request_ids\n"
        "        ]"
    ),
    "vllm_grammar_spec_width_serialization": texts["vllm_scheduler"].count(
        "return GrammarOutput(\n"
        "            structured_output_request_ids,\n"
        "            bitmask,\n"
        "            num_spec_tokens,\n"
        "        )"
    ),
    "vllm_grammar_commit_validation": texts["vllm_scheduler"].count(
        "self.structured_output_manager.filter_speculative_grammar_tokens("
    ),
    "vllm_groupwise_load_recovery": texts["vllm_scheduler"].count(
        "self.kv_cache_manager.get_block_ids_for_computed_tokens("
    ),
    "vllm_skip_spec_max1": texts["vllm_scheduler"].count(
        "and all(req.max_tokens <= 1 for req in scheduled_new_reqs)"
    ),
    "remote_skip_zero_depth": texts["remote_speculator"].count(
        "if active_k == 0:"
    ),
    "remote_zero_width_return": texts["remote_speculator"].count(
        "return self.draft_tokens[: input_batch.num_reqs, :0].contiguous()"
    ),
    "draft_limit_zero_depth": texts["spec_decode_utils"].count(
        "if num_speculative_tokens == 0:"
    ),
    "draft_limit_zero_slice": texts["spec_decode_utils"].count(
        "return draft_tokens[:, :0]"
    ),
    "kimi_attnres_final_preserve": texts["kimi_model"].count(
        "self.is_final_block_write_layer = ("
    ),
    "kimi_attnres_final_condition": texts["kimi_model"].count(
        "and not self.is_final_block_write_layer"
    ),
    "launcher_capture_size_4": texts["inner_launcher"].count(
        '"cudagraph_capture_sizes":[4,8,16,24,32,40,48,56,64]'
    ),
}
expected = {
    "adapter_explicit_chunk_forward": 1,
    "adapter_multigroup_groupwise_guard": 1,
    "connector_fine_slices": 2,
    "connector_geometry_validation": 1,
    "group_view_geometry_definition": 1,
    "group_view_fine_slice_argument": 1,
    "worker_chunk_annotations": 3,
    "worker_explicit_chunk_calculation": 1,
    "worker_external_chunk_state": 1,
    "worker_resolve_geometry": 1,
    "worker_subblock_view": 1,
    "worker_block_projection": 1,
    "worker_skip_projection": 1,
    "vllm_running_state_markers": 2,
    "vllm_dspark_cache_policy": 1,
    "vllm_dspark_cache_wiring": 1,
    "vllm_grammar_spec_width_collection": 1,
    "vllm_grammar_spec_width_serialization": 1,
    "vllm_grammar_commit_validation": 1,
    "vllm_groupwise_load_recovery": 2,
    "vllm_skip_spec_max1": 1,
    "remote_skip_zero_depth": 1,
    "remote_zero_width_return": 1,
    "draft_limit_zero_depth": 1,
    "draft_limit_zero_slice": 1,
    "kimi_attnres_final_preserve": 1,
    "kimi_attnres_final_condition": 1,
    "launcher_capture_size_4": 1,
}
if checks != expected:
    raise SystemExit(
        "ERROR: Kimi fine1536 semantic contract mismatch: "
        f"observed={checks} expected={expected}"
    )
for path in paths.values():
    if path.suffix == ".py":
        py_compile.compile(str(path), doraise=True)
print(f"Kimi fine1536 semantic contract verified: {checks}")
PY

if [[ "${K3PATCH_ONLY:-0}" == "1" ]]; then
  echo "K3PATCH_ONLY=1: runtime patch preflight complete"
  exit 0
fi

# The packaged remote-DSpark launcher already uses speculative depth 3.
# Keep that production default unchanged while explicitly forwarding the
# prefix-hash/match unit into the inner vLLM CLI.
launcher="${K3PATCH_LAUNCHER:-/usr/local/bin/serve-kimi-k3-production-remote-dspark-r29}"
launch_args=()
if [[ -n "${PREFIX_MATCH_UNIT:-}" ]]; then
  if [[ ! "${PREFIX_MATCH_UNIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: PREFIX_MATCH_UNIT must be a positive integer; got ${PREFIX_MATCH_UNIT}" >&2
    exit 2
  fi
  launch_args+=(--prefix-match-unit "${PREFIX_MATCH_UNIT}")
fi
exec "${launcher}" "${launch_args[@]}" "$@"
