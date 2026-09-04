#!/usr/bin/env python3
"""Defer the remote-draft context ingest for pure mid-prefill steps.

Marker: kimi-k3-draft-async-ingest.

For a scheduler step in which no active request sampled a token (every
request is still inside its prompt), the draft server's proposal is never
consumed, but rank 0 still serialized 1536 x 43008 bf16 aux rows and waited
for a blocking ZMQ round trip while the other ranks idled. This patch hands
those steps to a worker thread that owns the socket in FIFO order; the main
thread returns after the pinned-memory copy. Steps that need drafts (a token
was sampled) keep the synchronous path, and every synchronous RPC first
drains the queue so request ordering on the draft server is unchanged.

`VLLM_K3_DRAFT_ASYNC_PREFILL_INGEST` = ring depth (default 2; 0 disables).
"""
import sys, py_compile
from pathlib import Path

path = Path(sys.argv[1]); src = path.read_text()
MARKER = "kimi-k3-draft-async-ingest"
if MARKER in src:
    print("already patched"); sys.exit(0)

def replace_once(old, new):
    global src
    n = src.count(old); assert n == 1, f"anchor count {n} for {old[:70]!r}"
    src = src.replace(old, new)

# A. imports
replace_once("import os\nimport time\n", f"import os\nimport queue  # {MARKER}\nimport threading  # {MARKER}\nimport time\n")

# B. job record + init of ring/thread after the anchor staging allocation
replace_once(
'''            self._anchor_staging = torch.empty(
                self.max_num_reqs,
                dtype=torch.int64,
                pin_memory=True,
            )
''',
f'''            self._anchor_staging = torch.empty(
                self.max_num_reqs,
                dtype=torch.int64,
                pin_memory=True,
            )
            # {MARKER}: begin
            self._sampled_staging = torch.empty(
                self.max_num_reqs,
                dtype=torch.int32,
                pin_memory=True,
            )
            self._async_depth = int(
                os.environ.get("VLLM_K3_DRAFT_ASYNC_PREFILL_INGEST", "2") or 0
            )
            self._async_ring: list[tuple[torch.Tensor, torch.Tensor]] = []
            self._async_slot_free: list[threading.Event] = []
            self._async_slot = 0
            self._async_queue: queue.Queue = queue.Queue()
            self._async_error: tuple[BaseException, list[str]] | None = None
            self._async_lock = threading.Lock()
            self._async_stats = {{"deferred": 0, "deferred_ms": 0.0}}
            for _ in range(self._async_depth):
                self._async_ring.append(
                    (
                        torch.empty(
                            self.max_num_tokens,
                            dtype=torch.int64,
                            pin_memory=True,
                        ),
                        torch.empty(
                            (self.max_num_tokens, self.raw_context_width),
                            dtype=vllm_config.model_config.dtype,
                            pin_memory=True,
                        ),
                    )
                )
                free = threading.Event()
                free.set()
                self._async_slot_free.append(free)
            if self._async_depth > 0:
                threading.Thread(
                    target=self._async_ingest_worker,
                    name="k3-draft-async-ingest",
                    daemon=True,
                ).start()
                logger.info(
                    "Remote K3 %s deferred prefill ingest enabled: ring depth %d",
                    self.method,
                    self._async_depth,
                )
            # {MARKER}: end
''')

# C. _rpc: drain queued deferred jobs first, then use the socket under the lock
replace_once(
'''    def _rpc(self, frames: list[bytes]) -> dict[str, Any]:
        assert self._socket is not None
        try:
            self._socket.send_multipart(frames)
            response_frames = self._socket.recv_multipart()
        except Exception:
            self._connect()
            raise
''',
f'''    # {MARKER}: begin
    def _drain_async_ingest(self) -> None:
        pending = getattr(self, "_async_queue", None)
        if pending is not None:
            pending.join()

    def _apply_async_error(self) -> None:
        error = getattr(self, "_async_error", None)
        if error is None:
            return
        self._async_error = None
        exc, request_ids = error
        self._disabled_requests.update(request_ids)
        logger.warning(
            "Remote K3 %s deferred prefill ingest failed (%s); drafting is "
            "disabled for %d request(s) until they leave the batch",
            self.method,
            exc,
            len(request_ids),
        )

    def _async_ingest_worker(self) -> None:
        while True:
            job = self._async_queue.get()
            slot, frames, request_ids, enqueued = job
            try:
                with self._async_lock:
                    try:
                        self._socket.send_multipart(frames, copy=False)
                        response_frames = self._socket.recv_multipart()
                    except Exception:
                        self._connect()
                        raise
                response = json.loads(response_frames[0])
                if not isinstance(response, dict) or not response.get("ok", False):
                    raise RuntimeError(f"K3 draft deferred ingest failed: {{response}}")
                self._async_stats["deferred_ms"] += (
                    time.perf_counter() - enqueued
                ) * 1000
            except Exception as exc:  # noqa: BLE001
                logger.exception("Remote K3 deferred prefill ingest raised")
                self._async_error = (exc, request_ids)
            finally:
                self._async_slot_free[slot].set()
                self._async_queue.task_done()
    # {MARKER}: end

    def _rpc(self, frames: list[bytes]) -> dict[str, Any]:
        assert self._socket is not None
        self._drain_async_ingest()
        with self._async_lock:
            try:
                self._socket.send_multipart(frames)
                response_frames = self._socket.recv_multipart()
            except Exception:
                self._connect()
                raise
''')

# D. stage sampled counts alongside rejected/anchor and apply async errors first
replace_once(
'''        rejected_staging = self._rejected_staging[:num_reqs]
        anchor_staging = self._anchor_staging[:num_reqs]
        rejected_staging.copy_(num_rejected[:num_reqs], non_blocking=True)
        anchor_staging.copy_(anchor_tokens, non_blocking=True)
''',
f'''        self._apply_async_error()  # {MARKER}
        rejected_staging = self._rejected_staging[:num_reqs]
        anchor_staging = self._anchor_staging[:num_reqs]
        sampled_staging = self._sampled_staging[:num_reqs]  # {MARKER}
        rejected_staging.copy_(num_rejected[:num_reqs], non_blocking=True)
        anchor_staging.copy_(anchor_tokens, non_blocking=True)
        sampled_staging.copy_(sampled_counts, non_blocking=True)  # {MARKER}
''')
replace_once(
'''        rejected_counts = rejected_staging.tolist()
        anchor_tokens_cpu = anchor_staging.tolist()
''',
f'''        rejected_counts = rejected_staging.tolist()
        anchor_tokens_cpu = anchor_staging.tolist()
        sampled_counts_cpu = sampled_staging.tolist()  # {MARKER}
''')

# E. staging slot selection
replace_once(
'''        context_ready = time.perf_counter()
        positions_staging = self._positions_staging[:num_context_rows]
        context_staging = self._context_staging[:num_context_rows]
        positions_staging.copy_(positions, non_blocking=True)
''',
f'''        context_ready = time.perf_counter()
        # {MARKER}: a step in which no active request sampled a token is a
        # pure mid-prefill step; its proposal is never consumed, so only the
        # context ingest matters and it can complete off the critical path.
        deferred = self._async_depth > 0 and not _K3_CAPTURE_DIR and all(
            sampled_counts_cpu[request_idx] == 0 for request_idx in active_indices
        )
        if deferred:
            slot = self._async_slot
            self._async_slot = (slot + 1) % self._async_depth
            self._async_slot_free[slot].wait()
            self._async_slot_free[slot].clear()
            positions_full, context_full = self._async_ring[slot]
            positions_staging = positions_full[:num_context_rows]
            context_staging = context_full[:num_context_rows]
        else:
            slot = -1
            positions_staging = self._positions_staging[:num_context_rows]
            context_staging = self._context_staging[:num_context_rows]
        positions_staging.copy_(positions, non_blocking=True)
''')

# F. frames: zero-copy views for deferred jobs; deferred enqueue instead of RPC
replace_once(
'''        positions_frame = positions_staging.numpy().tobytes()
        # NumPy has inconsistent bfloat16 support; preserve its exact bits as u16.
        context_frame = context_staging.view(torch.uint16).numpy().tobytes()
''',
f'''        if deferred:  # {MARKER}: the ring slot stays reserved until the reply
            positions_frame = memoryview(positions_staging.numpy())
            context_frame = memoryview(context_staging.view(torch.uint16).numpy())
        else:
            positions_frame = positions_staging.numpy().tobytes()
            # NumPy has inconsistent bfloat16 support; preserve its exact bits as u16.
            context_frame = context_staging.view(torch.uint16).numpy().tobytes()
''')
replace_once(
'''        response = self._rpc(
            [json.dumps(header).encode(), positions_frame, context_frame]
        )
        rpc_done = time.perf_counter()
        self._copy_tokens_from_response(
            response,
            active_indices,
            num_speculative_tokens,
        )
        if self._probabilistic:
            self._copy_logits_from_response(
                response,
                active_indices,
                num_speculative_tokens,
            )
        output_copied = time.perf_counter()
''',
f'''        frames = [json.dumps(header).encode(), positions_frame, context_frame]
        if deferred:  # {MARKER}
            self._async_queue.put(
                (
                    slot,
                    frames,
                    [str(request["request_id"]) for request in requests],
                    time.perf_counter(),
                )
            )
            self._async_stats["deferred"] += 1
            response = {{}}
            rpc_done = output_copied = time.perf_counter()
        else:
            response = self._rpc(frames)
            rpc_done = time.perf_counter()
            self._copy_tokens_from_response(
                response,
                active_indices,
                num_speculative_tokens,
            )
            if self._probabilistic:
                self._copy_logits_from_response(
                    response,
                    active_indices,
                    num_speculative_tokens,
                )
            output_copied = time.perf_counter()
''')
replace_once(
'''            "client_total": (output_copied - started) * 1000,
        }
''',
f'''            "client_total": (output_copied - started) * 1000,
            "deferred": 1.0 if deferred else 0.0,  # {MARKER}
        }}
''')
path.write_text(src)
py_compile.compile(str(path), doraise=True)
print("patched + compiled:", path)
