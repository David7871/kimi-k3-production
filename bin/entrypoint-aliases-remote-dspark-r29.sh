#!/usr/bin/env bash
set -euo pipefail

# Kimi-K3 r29 (upstream-aligned 20260821) remote-DSpark production image
# LMCache operational patches. Runs before the packaged launcher on every
# container start. All edits are idempotent: each block verifies its target
# state and fails loudly if the packaged files drift from the expected anchors.

# The packaged lmcache-mp-wrapper.sh hard-codes the retain prefetch policy.
# Retain keeps every L2-prefetched chunk resident in L1, and on 2026-08-20 the
# LMCache server process grew to 54.7 GiB RSS (3.4x the 16 GiB L1 budget) and
# was reaped by the kernel OOM killer, which triggered the wrapper's
# fail-together shutdown of the vLLM API server. Use the default policy:
# prefetched chunks are temporary staging buffers and are released after use.
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

# Pass LMCache's configured token span explicitly to the engine-driven transfer
# registration. r29's recurrent cache page is 12,288 tokens while the legacy
# vLLM block size remains 1,536. The stock path multiplies the legacy
# blocks-per-chunk (8) by the first recurrent page (12,288), inflating one
# LMCache chunk to 98,304 tokens. Every group then requires eight manager block
# IDs although the scheduler supplies one, so all workers commit zero chunks.
# The explicit span preserves the configured 12,288-token object geometry:
# recurrent groups gather one 12,288-token page and the DCP MLA group gathers
# one rank-local 1,536-slot page.
python3 - <<'PY'
import os
from pathlib import Path

ADAPTER = Path(os.environ.get(
    "K3PATCH_ADAPTER",
    "/opt/infernal-invocation/lmcache/lmcache/integration/vllm/vllm_multi_process_adapter.py",
))
WORKER_TRANSFER = Path(os.environ.get(
    "K3PATCH_WORKER_TRANSFER",
    "/opt/infernal-invocation/lmcache/lmcache/v1/multiprocess/transfer_context/worker_transfer.py",
))
MARKER = "explicit-chunk-token-geometry"
annotation = "        tokens_per_chunk: int | None = None,\n"
signature_anchor = "        engine_group_infos: Sequence[EngineGroupInfo] = (),\n    ) -> None:\n"
adapter_anchor = (
    "                layout_hints=layout_hints,\n"
    "                engine_group_infos=self.engine_group_infos,\n"
    "            )\n"
)
adapter_replacement = (
    "                layout_hints=layout_hints,\n"
    "                engine_group_infos=self.engine_group_infos,\n"
    "                tokens_per_chunk=self.lmcache_tokens_per_chunk,\n"
    "            )\n"
)
calculation_anchor = "        chunk_tokens = blocks_in_chunk * block_size\n"
calculation_replacement = (
    f"        # {MARKER}\n"
    "        chunk_tokens = (\n"
    "            tokens_per_chunk\n"
    "            if tokens_per_chunk is not None\n"
    "            else blocks_in_chunk * block_size\n"
    "        )\n"
)

adapter_text = ADAPTER.read_text()
worker_text = WORKER_TRANSFER.read_text()
adapter_done = adapter_text.count("tokens_per_chunk=self.lmcache_tokens_per_chunk")
worker_annotations = worker_text.count(annotation)
worker_marker = worker_text.count(MARKER)

if (adapter_done, worker_annotations, worker_marker) == (1, 3, 1):
    print(f"LMCache geometry already patched: {MARKER}")
elif (adapter_done, worker_annotations, worker_marker) == (0, 0, 0):
    if adapter_text.count(adapter_anchor) != 1:
        raise SystemExit(
            "ERROR: expected one LMCache adapter registration anchor, found "
            f"{adapter_text.count(adapter_anchor)} in {ADAPTER}"
        )
    if worker_text.count(signature_anchor) != 3:
        raise SystemExit(
            "ERROR: expected three transfer-context register signatures, found "
            f"{worker_text.count(signature_anchor)} in {WORKER_TRANSFER}"
        )
    if worker_text.count(calculation_anchor) != 1:
        raise SystemExit(
            "ERROR: expected one legacy chunk-token calculation, found "
            f"{worker_text.count(calculation_anchor)} in {WORKER_TRANSFER}"
        )
    adapter_text = adapter_text.replace(adapter_anchor, adapter_replacement, 1)
    worker_text = worker_text.replace(
        signature_anchor,
        "        engine_group_infos: Sequence[EngineGroupInfo] = (),\n"
        + annotation
        + "    ) -> None:\n",
        3,
    )
    worker_text = worker_text.replace(
        calculation_anchor, calculation_replacement, 1
    )
    compile(adapter_text, str(ADAPTER), "exec")
    compile(worker_text, str(WORKER_TRANSFER), "exec")
    ADAPTER.write_text(adapter_text)
    WORKER_TRANSFER.write_text(worker_text)
    print(f"applied LMCache geometry patch: {MARKER}")
else:
    raise SystemExit(
        "ERROR: partial LMCache geometry patch detected: "
        f"adapter={adapter_done}, annotations={worker_annotations}, "
        f"marker={worker_marker}"
    )
PY

# The packaged remote-DSpark launcher already uses speculative depth 3.
# Keep that production default unchanged.
exec /usr/local/bin/serve-kimi-k3-production-remote-dspark-r29 "$@"
