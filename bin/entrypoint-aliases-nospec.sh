#!/usr/bin/env bash
set -euo pipefail

# Kimi-K3 remote-DSpark production image LMCache operational patches.
# Runs before the packaged launcher on every container start. All edits are
# idempotent: each block verifies its target state and fails loudly if the
# packaged files drift from the expected anchors.

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

# L1 TTLs and worker registration grace.
#
# The packaged wrapper expires L1 entries 600s after write / 300s after read,
# so any coding session paused for >10 minutes loses its L1 residency and
# falls back to coarse L2 disk chunks. L1 capacity is already governed by the
# LRU eviction watermarks, so lengthen the TTLs to cover realistic pauses.
#
# worker_registration_grace_seconds defaults to 3600. The stock vLLM adapter
# only starts its per-instance heartbeat lazily on the first store/retrieve,
# so a fresh engine that serves no chunk-sized request within 3600s gets all
# 8 worker registrations reaped; the next PREPARE_STORE then raises
# "non-GPU context not registered (or reaped)" and kills every worker
# (observed 2026-08-20 15:02 KST, full engine restart). Raise the grace as a
# second line of defense behind the heartbeat-at-registration patch below.
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

grace_line = "  --worker-registration-grace-seconds 86400"
anchor = "  --eviction-ratio 0.10"
if "--worker-registration-grace-seconds" in text:
    print("already patched: worker-registration-grace-seconds present")
elif text.count(anchor) == 1:
    text = text.replace(anchor, anchor + "\n" + grace_line, 1)
    print(f"applied: inserted {grace_line.strip()} after {anchor.strip()}")
else:
    raise SystemExit(
        f"ERROR: expected exactly one {anchor!r} in {path}, found {text.count(anchor)}"
    )

path.write_text(text)
PY

# Start each worker's per-instance LMCache heartbeat immediately after KV-cache
# registration instead of lazily on the first store/retrieve. With the lazy
# start, a fresh engine that idles past worker_registration_grace_seconds
# without a chunk-sized request has its registrations reaped while still
# healthy, and the next store crashes all workers (see note above). Pinging
# from registration onward latches has_liveness_signal and keeps last_seen
# fresh, so the reaper never targets a live worker.
python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ.get(
    "K3PATCH_ADAPTER",
    "/opt/kimi-k3-qsrt/lmcache/lmcache/integration/vllm/vllm_multi_process_adapter.py",
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
# heartbeat interval (5s), so any server busy spell longer than 5s — e.g. a
# large external-cache restore with dynamic pinning, observed 2026-08-21
# 13:31/17:29 KST — flips workers into degraded mode and back. Give the PING
# an independent 30s timeout: bounded NORMAL-pool queue latency no longer
# causes health flaps, while a genuinely dead server still degrades after 30s.
# Source-matched adapter suite: 28 passed in a disposable container.
python3 - <<'PY'
from pathlib import Path

ADAPTER = Path("/opt/kimi-k3-qsrt/lmcache/lmcache/integration/vllm/vllm_multi_process_adapter.py")
WRAPPER = Path("/usr/local/bin/lmcache-mp-wrapper.sh")
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

wrapper = WRAPPER.read_text()
if '"lmcache.mp.heartbeat_timeout"' in wrapper:
    print("wrapper already patched: heartbeat_timeout present")
else:
    wrapper = sub(
        wrapper,
        '                "lmcache.mp.heartbeat_interval": 5,\n',
        '                "lmcache.mp.heartbeat_interval": 5,\n'
        '                "lmcache.mp.heartbeat_timeout": 30,\n',
    )
    WRAPPER.write_text(wrapper)
    print("wrapper patched: heartbeat_timeout=30")
PY

# The packaged generic QSRT launcher has an explicit non-speculative mode.
# Keep the current TP8/DCP8 target, LMCache, and operational fixes while
# removing the draft model and every speculative-decoding argument.
export K3_PYTHON_BIN="${K3_PYTHON_BIN:-/opt/venv/bin/python}"
export K3_MODEL_DIR="${K3_MODEL_DIR:-${MODEL:-/model}}"
export K3_SERVED_MODEL_NAME="${K3_SERVED_MODEL_NAME:-${SERVED_MODEL_NAME:-Kimi-K3}}"
export K3_HOST="${K3_HOST:-${HOST:-127.0.0.1}}"
export K3_PORT="${K3_PORT:-${PORT:-8090}}"
export K3_TP_SIZE="${K3_TP_SIZE:-${TP_SIZE:-8}}"
export K3_MAX_MODEL_LEN="${K3_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-512000}}"
export K3_KV_CACHE_MEMORY_BYTES="${K3_KV_CACHE_MEMORY_BYTES:-${KV_CACHE_MEMORY_BYTES:-2348810240}}"
export K3_GPU_MEMORY_UTILIZATION="${K3_GPU_MEMORY_UTILIZATION:-${GPU_MEMORY_UTILIZATION:-0.985}}"
export K3_MAX_NUM_BATCHED_TOKENS="${K3_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS:-1536}}"
export K3_MAX_NUM_SEQS="${K3_MAX_NUM_SEQS:-${MAX_NUM_SEQS:-2}}"
export K3_ENABLE_DSPARK=0
export K3_NUM_SPECULATIVE_TOKENS=0
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-0}"

exec /usr/local/bin/lmcache-mp-wrapper.sh \
  /usr/local/bin/serve-kimi-k3-qsrt.sh \
  --language-model-only \
  --decode-context-parallel-size "${DCP_SIZE:-8}" \
  --dcp-comm-backend a2a \
  --dcp-kv-cache-interleave-size 1 \
  --no-enable-flashinfer-autotune \
  --mamba-cache-mode align \
  --max-num-scheduled-tokens "${MAX_NUM_SCHEDULED_TOKENS:-1536}" \
  --host "${HOST:-127.0.0.1}"
