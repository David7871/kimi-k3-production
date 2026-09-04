#!/usr/bin/env python3
"""Background (three-phase) store for multi-group LMCache transfer contexts.

Marker: kimi-k3-async-multigroup-store.

`AsyncEngineDrivenTransferContext.submit_store` pipelines prepare, gather and
commit in a background executor for single-group models, but sends hybrid
(multi-group: KDA state groups plus the MLA group) stores through the
synchronous `_submit_store_multigroup`, which on the forward thread does a
full device synchronize, a PREPARE_STORE round trip to the LMCache server,
the GPU->CPU gathers, a second synchronize and the COMMIT round trip (about
90 ms per 1,536-token prefill chunk on Kimi-K3). This patch runs the same
three phases for multi-group stores in the executor, gathering on the copy
stream after the forward's CUDA event, with the same in-flight bookkeeping
(`_pending_stores`, `_inflight_gather_events`) that `flush_inflight_stores`
relies on before paged blocks may be reused. `LMCACHE_ASYNC_MULTIGROUP_STORE=0`
restores the synchronous path.
"""
import sys, py_compile
from pathlib import Path
path = Path(sys.argv[1]); src = path.read_text()
MARKER = "kimi-k3-async-multigroup-store"
if MARKER in src:
    print("already patched"); sys.exit(0)
def replace_once(old, new):
    global src
    n = src.count(old); assert n == 1, f"anchor count {n} for {old[:60]!r}"
    src = src.replace(old, new)
replace_once(
"""from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    IPCEvent,
    _single_group_block_ids,
)
""",
f"""from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    IPCEvent,
    _single_group_block_ids,
)
import os  # {MARKER}
from lmcache.v1.multiprocess.transfer_context.pickle import (  # {MARKER}
    EngineDrivenContextPickle,
)

_ASYNC_MULTIGROUP_STORE = os.environ.get("LMCACHE_ASYNC_MULTIGROUP_STORE", "1") == "1"  # {MARKER}
""")
replace_once(
"""        if self._group_states:
            # Multi-group stores take the synchronous per-group path for now;
            # the wait-for-forward-event + copy-stream pipelining below is
            # single-group. TODO: pipeline the grouped gather as well.
            _event.wait()
            return self._submit_store_multigroup(key, instance_id, kv_caches, block_ids)
""",
f"""        if self._group_states:
            # {MARKER}: multi-group stores run the same three-phase background
            # pipeline as single-group stores unless disabled by env.
            if _ASYNC_MULTIGROUP_STORE and not isinstance(
                self._engine_driven_context, EngineDrivenContextPickle
            ):
                return self._submit_store_multigroup_async(
                    _request_id, key, instance_id, kv_caches, block_ids, _event
                )
            _event.wait()
            return self._submit_store_multigroup(key, instance_id, kv_caches, block_ids)
""")
replace_once(
"""    def flush_inflight_stores(self) -> None:
        \"\"\"Synchronize all in-flight gather (GPU->CPU) events.
""",
f'''    # {MARKER}: begin
    def _submit_store_multigroup_async(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
    ) -> MessagingFuture:
        """Three-phase background store for multi-group (hybrid) KV layouts.

        Mirrors `_submit_store_multigroup` (same prepare/gather/commit calls
        and per-group slot mapping) but off the forward thread: the copy
        stream waits on the forward's CUDA event instead of a device
        synchronize, and the gather's completion event is tracked in
        `_inflight_gather_events` so `flush_inflight_stores` blocks before
        vLLM may overwrite the paged blocks. PREPARE_STORE and COMMIT hold
        `_commit_lock` so stores of one worker stay ordered on the socket.
        """
        ctx = self._engine_driven_context
        assert ctx is not None
        completion: MessagingFuture[bool] = MessagingFuture()
        if len(block_ids) != len(self._group_states):
            raise RuntimeError(
                f"got {{len(block_ids)}} block-id lists for "
                f"{{len(self._group_states)}} registered groups"
            )
        # Validate and project every group's inputs on the forward thread so a
        # malformed store fails before any server-side slot is reserved.
        transfer_inputs = [
            self._group_transfer_inputs(state, key, kv_caches, block_ids[gid])
            for gid, state in enumerate(self._group_states)
        ]
        group_states = list(self._group_states)
        gather_launched = threading.Event()
        try:
            with self._inflight_lock:
                if self._is_closing:
                    completion.set_result(False)
                    return completion
                self._pending_stores.add(gather_launched)

            def _prepare_gather_and_commit_grouped() -> None:
                gather_done: Any | None = None
                ok = False
                try:
                    with self._commit_lock:
                        result = ctx.prepare_store_grouped(key, instance_id)
                    if result is None:
                        return
                    tensors, chunk_indices, group_ids = result
                    if not tensors:
                        ok = True
                        return
                    with torch.inference_mode(), torch_dev.stream(self._copy_stream):
                        _event.wait(stream=self._copy_stream)
                        for gid, state in enumerate(group_states):
                            out_g, chunks_g = self._group_slots(
                                tensors, group_ids, gid, chunk_indices
                            )
                            if not out_g:
                                continue
                            transfer_kv_caches, transfer_block_ids = transfer_inputs[gid]
                            gather_paged_kv_to_cpu(
                                transfer_kv_caches,
                                transfer_block_ids,
                                state.blocks_in_chunk,
                                layout_hints=self._layout_hints,
                                engine_kv_format=state.engine_kv_format,
                                out=out_g,
                                chunk_indices=chunks_g,
                                blocks_per_window=state.blocks_per_window,
                            )
                        gather_done = torch_dev.Event()
                        gather_done.record(self._copy_stream)
                    with self._inflight_lock:
                        self._inflight_gather_events.add(gather_done)
                        self._pending_stores.discard(gather_launched)
                    gather_launched.set()
                    gather_done.synchronize()
                    with self._commit_lock:
                        ok = ctx.commit_store(key, instance_id, [])
                    if not ok:
                        logger.error(
                            "Async grouped commit_store failed for request_id=%s",
                            _request_id,
                        )
                except Exception:
                    logger.exception(
                        "Async grouped engine-driven store failed for request_id=%s",
                        _request_id,
                    )
                    ok = False
                finally:
                    with self._inflight_lock:
                        if gather_done is not None:
                            self._inflight_gather_events.discard(gather_done)
                        self._pending_stores.discard(gather_launched)
                    gather_launched.set()
                    completion.set_result(ok)

            self._commit_executor.submit(_prepare_gather_and_commit_grouped)
        except Exception:
            logger.exception("Failed to submit async grouped engine-driven store")
            with self._inflight_lock:
                self._pending_stores.discard(gather_launched)
            gather_launched.set()
            completion.set_result(False)
            return completion
        return completion
    # {MARKER}: end

    def flush_inflight_stores(self) -> None:
        """Synchronize all in-flight gather (GPU->CPU) events.
''')
path.write_text(src); py_compile.compile(str(path), doraise=True); print("patched + compiled:", path)
