#!/usr/bin/env bash
set -euo pipefail

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

# The packaged launcher owns every model/runtime policy.
exec /usr/local/bin/serve-kimi-k3-production-tp8-dcp8 "$@"
