#!/usr/bin/env python3
"""Apply LMCache safety fixes required by the Kimi-K3 source overlay.

A deterministic Kimi-K3 qualification reproduced non-equivalent output only
when one request combined a nonzero vLLM APC prefix with a deeper LMCache tail.
Cold, full-APC, full-LMCache-L1, and APC-plus-recompute arms were byte-identical.
Until per-group recurrent-state composition is proven, keep full external resume
and full local APC, but recompute any tail beyond a nonzero local recurrent hit.

The source checkout shadows the installed ``lmcache`` package but intentionally
does not contain compiled extensions. Extend the package search path to the
matching installed package so workers load ``lmcache.c_ops`` instead of the
Python transfer fallback.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

ROOT = Path(
    os.environ.get(
        "K3_LMCACHE_SOURCE_ROOT",
        "/opt/infernal-invocation/lmcache/lmcache",
    )
)
CONNECTOR = ROOT / "integration/vllm/lmcache_mp_connector.py"
PACKAGE_INIT = ROOT / "__init__.py"
VLLM_ROOT = Path(
    os.environ.get(
        "K3_VLLM_SOURCE_ROOT",
        "/opt/infernal-invocation/vllm/vllm",
    )
)
SCHEDULER = VLLM_ROOT / "v1/core/sched/scheduler.py"
MARKER = "def _should_skip_mixed_recurrent_retrieve("
BINARY_PATH_MARKER = "_LMCACHE_BINARY_PACKAGE_PATHS = tuple("
REMOTE_RESUME_MARKER = "resuming_completed_remote_kv = ("
ABORT_CLEANUP_OLD = '''            if tracker.num_lmcache_hit_blocks > 0:
                self.scheduler_adapter.free_lookup_locks(
                    token_ids=list(tracker.all_token_ids),
                    start=0,
                    end=tracker.num_lmcache_hit_blocks * self.vllm_block_size,
                    request_id=request_id,
                )
'''
ABORT_CLEANUP_NEW = '''            if tracker.num_lmcache_hit_tokens > 0:
                self.scheduler_adapter.free_lookup_locks(
                    token_ids=list(tracker.all_token_ids),
                    start=0,
                    end=tracker.num_lmcache_hit_tokens,
                    request_id=request_id,
                    cache_salt=tracker.cache_salt,
                )
'''


def replace_once(text: str, old: str, new: str, path: Path = CONNECTOR) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected one occurrence, found {count}: {old!r}"
        )
    return text.replace(old, new, 1)


def patch(text: str) -> str:
    helper_anchor = "@dataclass\nclass LMCacheMPRequestTracker:"
    helper = '''def _should_skip_mixed_recurrent_retrieve(
    has_recurrent_cache: bool,
    num_computed_tokens: int,
    num_lmcache_hit_tokens: int,
) -> bool:
    """Reject an unqualified external tail spliced onto local recurrent state."""
    return (
        has_recurrent_cache
        and num_computed_tokens > 0
        and num_lmcache_hit_tokens > num_computed_tokens
    )


'''
    text = replace_once(text, helper_anchor, helper + helper_anchor)

    fields = '''    num_vllm_hit_tokens: int = 0
    num_lmcache_hit_tokens: int = 0
'''
    text = replace_once(
        text,
        fields,
        fields + "    skip_mixed_recurrent_retrieve: bool = False\n",
    )

    init = '''        self.num_vllm_hit_tokens = 0
        self.num_lmcache_hit_tokens = 0
'''
    text = replace_once(
        text,
        init,
        init + "        self.skip_mixed_recurrent_retrieve = False\n",
    )

    needs = '''        return (
            self.num_lmcache_hit_tokens > self.num_vllm_hit_tokens
            and self.state != LMCacheMPRequestState.READY
        )
'''
    text = replace_once(
        text,
        needs,
        '''        return (
            not self.skip_mixed_recurrent_retrieve
            and self.num_lmcache_hit_tokens > self.num_vllm_hit_tokens
            and self.state != LMCacheMPRequestState.READY
        )
''',
    )

    decision = '''        tracker.num_lmcache_hit_tokens = ret

        need_to_load = max(0, ret - num_computed_tokens)
'''
    text = replace_once(
        text,
        decision,
        '''        tracker.num_lmcache_hit_tokens = ret

        # Hybrid recurrent state is not composable across local APC and a
        # deeper external tail in this runtime. Keep the external objects
        # locked only until update_state_after_alloc releases them, then
        # recompute the tail from the trusted local state.
        if _should_skip_mixed_recurrent_retrieve(
            self._has_recurrent_cache,
            num_computed_tokens,
            ret,
        ):
            tracker.skip_mixed_recurrent_retrieve = True
            return 0, False

        need_to_load = max(0, ret - num_computed_tokens)
''',
    )
    return text


def patch_abort_cleanup(text: str) -> str:
    """Keep pre-allocation abort cleanup in the tracker's token units."""
    if ABORT_CLEANUP_NEW in text:
        return text
    return replace_once(text, ABORT_CLEANUP_OLD, ABORT_CLEANUP_NEW)


def patch_binary_package_path(text: str) -> str:
    """Expose matching wheel-provided native extensions to the source overlay."""
    if BINARY_PATH_MARKER in text:
        return text
    anchor = "# Standard\nimport sys\n"
    replacement = '''# Standard
from pathlib import Path
import site
import sys

# This source overlay carries patched Python modules but not wheel-built native
# extensions. Keep source modules first and use the matching installed package
# only as a fallback for submodules such as lmcache.c_ops.
_LMCACHE_BINARY_PACKAGE_PATHS = tuple(
    str(Path(site_dir) / "lmcache")
    for site_dir in site.getsitepackages()
    if (Path(site_dir) / "lmcache").is_dir()
)
for binary_package_path in _LMCACHE_BINARY_PACKAGE_PATHS:
    if binary_package_path not in __path__:
        __path__.append(binary_package_path)
'''
    return replace_once(text, anchor, replacement, path=PACKAGE_INIT)


def patch_vllm_scheduler(text: str) -> str:
    """Do not repeat full-input admission after an async KV load completes."""
    if REMOTE_RESUME_MARKER in text:
        return text

    state_anchor = '''                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
'''
    state_replacement = '''                num_external_computed_tokens = 0
                load_kv_async = False
                resuming_completed_remote_kv = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
'''
    text = replace_once(
        text, state_anchor, state_replacement, path=SCHEDULER
    )

    resume_anchor = '''                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
'''
    resume_replacement = '''                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    resuming_completed_remote_kv = (
                        request.status == RequestStatus.WAITING
                    )
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
'''
    text = replace_once(
        text, resume_anchor, resume_replacement, path=SCHEDULER
    )

    admission_anchor = (
        "                    full_sequence_must_fit="
        "self.scheduler_reserve_full_isl,\n"
    )
    admission_replacement = '''                    # An async KV load passes full-sequence admission before it
                    # allocates transfer blocks. After completion, the request
                    # already holds that reservation. Reapplying admission can
                    # double-count held HMA group blocks; per-step allocation
                    # still enforces the available-block limit.
                    full_sequence_must_fit=(
                        self.scheduler_reserve_full_isl
                        and not resuming_completed_remote_kv
                    ),
'''
    return replace_once(
        text, admission_anchor, admission_replacement, path=SCHEDULER
    )


def verify(text: str) -> None:
    ast.parse(text, filename=str(CONNECTOR))
    required = {
        MARKER: 1,
        "skip_mixed_recurrent_retrieve: bool = False": 1,
        "self.skip_mixed_recurrent_retrieve = False": 1,
        "not self.skip_mixed_recurrent_retrieve": 1,
        "tracker.skip_mixed_recurrent_retrieve = True": 1,
        ABORT_CLEANUP_NEW: 1,
    }
    for token, expected in required.items():
        count = text.count(token)
        if count != expected:
            raise SystemExit(
                f"{CONNECTOR}: expected {expected} occurrence(s) of {token!r}, "
                f"found {count}"
            )


def verify_package_init(text: str) -> None:
    ast.parse(text, filename=str(PACKAGE_INIT))
    if text.count(BINARY_PATH_MARKER) != 1:
        raise SystemExit(
            f"{PACKAGE_INIT}: expected one occurrence of "
            f"{BINARY_PATH_MARKER!r}"
        )


def verify_vllm_scheduler(text: str) -> None:
    ast.parse(text, filename=str(SCHEDULER))
    required = {
        REMOTE_RESUME_MARKER: 1,
        "and not resuming_completed_remote_kv": 1,
    }
    for token, expected in required.items():
        count = text.count(token)
        if count != expected:
            raise SystemExit(
                f"{SCHEDULER}: expected {expected} occurrence(s) of "
                f"{token!r}, found {count}"
            )


original = CONNECTOR.read_text()
if MARKER in original:
    updated = original
else:
    updated = patch(original)
updated = patch_abort_cleanup(updated)
verify(updated)
if updated == original:
    print("LMCache mixed recurrent retrieve guard and abort cleanup: already present")
else:
    CONNECTOR.write_text(updated)
    print(f"LMCache mixed recurrent retrieve guard and abort cleanup: patched {CONNECTOR}")

package_original = PACKAGE_INIT.read_text()
package_updated = patch_binary_package_path(package_original)
verify_package_init(package_updated)
if package_updated == package_original:
    print("LMCache native-extension package path: already present")
else:
    PACKAGE_INIT.write_text(package_updated)
    print(f"LMCache native-extension package path: patched {PACKAGE_INIT}")

scheduler_original = SCHEDULER.read_text()
scheduler_updated = patch_vllm_scheduler(scheduler_original)
verify_vllm_scheduler(scheduler_updated)
if scheduler_updated == scheduler_original:
    print("vLLM async-KV resume admission: already present")
else:
    SCHEDULER.write_text(scheduler_updated)
    print(f"vLLM async-KV resume admission: patched {SCHEDULER}")
