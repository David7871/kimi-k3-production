# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

import os
import queue
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

# PP sampled-token broadcast transport, selected once per process at
# PPHandler construction via VLLM_K3_PP_TOKEN_XFER:
#
#   "" | "async_bcast" (default): historical behavior, bit-for-bit. NCCL
#       broadcast on a sibling device group, posted async_op at step T on a
#       side stream and consumed at T+pp_size. The recv kernels are enqueued
#       pp_size steps before the last stage sends, so an NCCL kernel spins
#       resident on the non-last ranks' SMs for 1..pp_size-1 steps -- which
#       blocks device-wide cooperative kernels (b12x grid barriers) that need
#       all their CTAs co-resident.
#
#   "gloo": CPU (gloo) transport, ported from the shipped GLM-5.3-Flash PP4
#       fix (libref/overlay_pp_utils.py, measured C32 247 -> 846 tok/s).
#       No GPU kernel is ever resident waiting for the last stage:
#         - last rank D2H-stages the (few-hundred-byte) payload into pinned
#           memory on a side stream; a daemon thread waits the copy event and
#           posts the blocking gloo broadcasts, so the worker main thread
#           never blocks on the GPU or on peers;
#         - non-last ranks post the gloo receives host-side (async_op) at
#           step T -- they progress on gloo's own CPU threads -- and copy the
#           payload H2D just-in-time at consume (T+pp_size) on an idle side
#           stream, gating the main stream with an event (no host sync).
#       Sub-knobs (gloo mode only):
#         VLLM_K3_PP_SYNC_STAGE=1   synchronous D2H staging on the main
#                                   stream at the last rank (~20us/step;
#                                   libref's fix for side-stream staging
#                                   bimodality). Default 0 = async staging.
#         VLLM_K3_PP_TOKEN_DELAY=N  consume delay in steps (queue depth).
#                                   Default = pp world size (same consume
#                                   cadence as the NCCL path). Must be
#                                   1 <= N <= pp world size: N larger than
#                                   pp_size would apply tokens after the
#                                   microbatch's next decode already built
#                                   its inputs.
#
# The env value MUST be identical on every rank: group creation is a
# world-collective and the two modes create different communicators.
XFER_ASYNC_BCAST = "async_bcast"
XFER_GLOO = "gloo"


def _pp_token_xfer_mode() -> str:
    mode = os.environ.get("VLLM_K3_PP_TOKEN_XFER", "") or XFER_ASYNC_BCAST
    if mode not in (XFER_ASYNC_BCAST, XFER_GLOO):
        raise ValueError(
            f"Unknown VLLM_K3_PP_TOKEN_XFER={mode!r}; "
            f"expected one of: '', '{XFER_ASYNC_BCAST}', '{XFER_GLOO}'"
        )
    return mode


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    # Async broadcast Work handles, retained until the slot is consumed so
    # the collectives are neither GC'd nor watchdog-orphaned while pending.
    works: tuple = ()


@dataclass
class GlooPendingRecv:
    """Per-step slot for the gloo/CPU transport: pinned host payloads plus
    the async gloo broadcast Works, consumed pp_delay steps later."""

    # Async gloo broadcast Work handles (CPU-side), waited at consume time
    # so the pinned buffers are never read while gloo may still write them.
    works: list

    sampled_tokens_cpu: torch.Tensor  # [num_reqs, max_sample_len] int64, pinned
    combined_cpu: torch.Tensor  # [2, num_reqs] int32, pinned
    idx_mapping: torch.Tensor  # [num_reqs] (GPU)
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    with outputs that might be needed in a subsequent (decode) step.
    Returns None if no sampled outputs are needed in the requests' next step."""
    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    max_seq_len = input_batch.max_seq_len_np
    assert max_seq_len is not None  # always populated under PP
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    # Exclude requests that we know are finished.
    not_finishing = np.maximum(old_computed, prefill_len) + 1 < max_seq_len
    need_sampled_mask = produces_sample & not_finishing
    return need_sampled_mask if need_sampled_mask.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv so the default stream isn't
    gated by the matching peer call. Step T's recv is consumed at
    T+token_delay (default pp_size) via `get_prev_sampled_outputs`.

    Transport is selected by VLLM_K3_PP_TOKEN_XFER (see module docstring):
    "async_bcast" (default) keeps the historical NCCL sibling-group path;
    "gloo" moves the payload over CPU so no NCCL recv kernel is resident on
    the non-last ranks while compute runs.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        self.xfer_mode = _pp_token_xfer_mode()
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)

        pp_world = get_pp_group().world_size
        if self.xfer_mode == XFER_GLOO:
            token_delay = int(os.environ.get("VLLM_K3_PP_TOKEN_DELAY", pp_world))
            assert 1 <= token_delay <= pp_world, (
                "VLLM_K3_PP_TOKEN_DELAY must be in [1, pp_world_size]; a delay "
                "larger than pp_size would apply sampled tokens after the "
                "microbatch's next decode step already built its inputs."
            )
            self._sync_stage = os.environ.get("VLLM_K3_PP_SYNC_STAGE", "0") == "1"
        else:
            token_delay = pp_world
            self._sync_stage = False
        self.token_delay = token_delay

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed token_delay steps later.
        # Pre-seeded with token_delay None placeholders so the first
        # token_delay consumes are no-ops. None means no postprocess is
        # pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | GlooPendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * token_delay)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        if self.xfer_mode == XFER_GLOO:
            # Last rank: side stream for async D2H staging. Non-last ranks:
            # idle side stream for the just-in-time H2D at consume time.
            self.broadcast_stream = (
                torch.cuda.Stream(device) if self.is_last_rank else None
            )
            self.recv_stage_stream = (
                None if self.is_last_rank else torch.cuda.Stream(device)
            )
            # Keeps pinned H2D sources alive until the non-blocking copies on
            # recv_stage_stream (idle stream: execute ~immediately after
            # enqueue) are long done. Bounded; oldest entries drop out.
            self._h2d_retention: deque = deque(maxlen=2 * token_delay + 2)

            # Dedicated CPU (gloo) group for the sampled-token broadcast.
            # Mirrors GroupCoordinator.make_sibling_device_group: every world
            # rank must create every lane's group in the same order (PyTorch
            # auto-names groups via a shared store counter; per-lane single
            # creation collides and hangs one lane's handshake). Each rank
            # keeps only its own group.
            pp_group = get_pp_group()
            self.broadcast_group = None
            for ranks in pp_group.group_ranks:
                pg = dist.new_group(ranks, backend="gloo")
                if pp_group.rank in ranks:
                    self.broadcast_group = pg
            assert self.broadcast_group is not None

            # Last rank only: FIFO of (event, staged cpu tensors) drained by
            # the sender thread in enqueue order, preserving the collective
            # order the receivers posted in.
            self._send_queue: queue.Queue | None = None
            if self.is_last_rank:
                self._send_queue = queue.Queue()
                self._send_thread = threading.Thread(
                    target=self._send_loop, name="pp-token-broadcast", daemon=True
                )
                self._send_thread.start()
            return

        self.broadcast_stream = torch.cuda.Stream(device)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )

        # Last-rank only: bounded retention of fire-and-forget send Works
        # (async_op broadcasts) so they stay alive and watchdog-tracked
        # until long completed; oldest entries are dropped as new sends
        # replace them.
        self.send_works: deque = deque(maxlen=2 * (pp_world + 1))

    # ------------------------------------------------------------------
    # gloo transport (last-rank send side)
    # ------------------------------------------------------------------

    def _send_loop(self) -> None:
        """Waits for staged D2H copies and posts the gloo broadcasts.

        Runs on a daemon thread so the worker main thread never blocks on the
        GPU-side D2H event or on the peers' receives. The receiver side posted
        its (async_op) gloo broadcasts pp_delay steps ago and consumes at
        T+token_delay, so by the time this thread broadcasts step T's payload
        the matching recvs are already pending on every non-last rank.
        """
        while True:
            item = self._send_queue.get()
            if item is None:
                return
            event, tensors = item
            try:
                if event is not None:
                    event.synchronize()
                for tensor in tensors:
                    dist.broadcast(
                        tensor, src=self.last_rank, group=self.broadcast_group
                    )
            except Exception:
                logger.exception("PP sampled-token gloo broadcast failed")
                raise

    def _stage_and_send(self, *gpu_tensors: torch.Tensor) -> None:
        """Stage token payloads to pinned CPU and hand to the sender thread.

        VLLM_K3_PP_SYNC_STAGE=1 (libref b12x-PP bimodality fix): synchronous
        copy on the main stream. The payloads are a few hundred bytes per
        step, so the sync DtoH costs ~20us but removes the side-stream
        wait_stream ordering (whose semaphore can land behind already-enqueued
        next-step work). Default (unset): async side-stream path.
        """
        assert self.is_last_rank and self._send_queue is not None
        if self._sync_stage:
            if torch.cuda.is_current_stream_capturing():
                # Capture must not see a blocking sync; fall back to async.
                self._stage_and_send_async(*gpu_tensors)
                return
            staged = []
            for tensor in gpu_tensors:
                cpu = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
                cpu.copy_(tensor, non_blocking=False)
                staged.append(cpu)
            self._send_queue.put((None, staged))
            return
        self._stage_and_send_async(*gpu_tensors)

    def _stage_and_send_async(self, *gpu_tensors: torch.Tensor) -> None:
        """Async D2H-copy token payloads into pinned staging, hand to sender."""
        assert self.broadcast_stream is not None
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            staged = []
            for tensor in gpu_tensors:
                cpu = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
                cpu.copy_(tensor, non_blocking=True)
                staged.append(cpu)
            event = torch.cuda.Event()
            event.record(self.broadcast_stream)
        # Sources were allocated on the main stream and are read on the
        # broadcast stream.
        for tensor in gpu_tensors:
            tensor.record_stream(self.broadcast_stream)
        self._send_queue.put((event, staged))

    # ------------------------------------------------------------------
    # common
    # ------------------------------------------------------------------

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor] | None:
        """Consume the entry from token_delay steps ago and make its payload
        visible to the main stream, then filter out entries whose request was
        freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None
        if isinstance(slot, GlooPendingRecv):
            return self._consume_gloo(slot)

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )

    def _consume_gloo(self, slot: GlooPendingRecv) -> dict[str, torch.Tensor] | None:
        """gloo-mode consume: wait for the (normally long-complete) host-side
        broadcasts, then copy the pinned payloads to the GPU just-in-time on
        an idle side stream. The host never blocks on the GPU: the H2D copies
        execute immediately on recv_stage_stream and the main stream is gated
        with an event so the deferred postprocess sees the payload.
        """
        # Wait first so the pinned staging tensors are never read (or
        # dropped) while gloo may still be writing them. Posted token_delay
        # steps ago and sent by the last rank ~1 step ago, so this is
        # normally an already-satisfied check.
        for work in slot.works:
            work.wait()

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        with torch.cuda.stream(self.recv_stage_stream):
            sampled_tokens = torch.empty(
                slot.sampled_tokens_cpu.shape, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(
                slot.combined_cpu.shape, dtype=torch.int32, device=self.device
            )
            sampled_tokens.copy_(slot.sampled_tokens_cpu, non_blocking=True)
            combined.copy_(slot.combined_cpu, non_blocking=True)
            event = self.recv_stage_stream.record_event()
        # Must record_stream since these were allocated on recv_stage_stream
        # but are used on the main stream.
        sampled_tokens.record_stream(self.main_stream)
        combined.record_stream(self.main_stream)
        self.main_stream.wait_event(event)
        # Retain the pinned sources until the async H2D is long complete
        # (recv_stage_stream is otherwise idle: copies run ~immediately).
        self._h2d_retention.append((slot.sampled_tokens_cpu, slot.combined_cpu))

        num_sampled, num_rejected = combined.unbind(dim=0)
        return dict(
            sampled_tokens=sampled_tokens,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            idx_mapping=idx_mapping,
        )

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        if self.xfer_mode == XFER_GLOO:
            # Post the gloo receives immediately (host-side, async_op): no
            # GPU kernel is left resident waiting for the last stage. gloo
            # progresses the broadcast on its own CPU threads; the payload
            # lands in pinned host memory and is H2D-copied at consume time.
            sampled_tokens_cpu = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, pin_memory=True
            )
            combined_cpu = torch.empty(
                2, num_reqs, dtype=torch.int32, pin_memory=True
            )
            works = [
                dist.broadcast(
                    sampled_tokens_cpu,
                    src=self.last_rank,
                    group=self.broadcast_group,
                    async_op=True,
                ),
                dist.broadcast(
                    combined_cpu,
                    src=self.last_rank,
                    group=self.broadcast_group,
                    async_op=True,
                ),
            ]
            self.queue[-1] = GlooPendingRecv(
                works,
                sampled_tokens_cpu,
                combined_cpu,
                input_batch.idx_mapping,
                input_batch.idx_mapping_np,
                need_sampled_mask,
                gen_at_receive_np,
            )
            return bool(need_sampled_mask.all())

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            # async_op + block_current_stream keeps this HOST-non-blocking:
            # the recv kernels are enqueued on the comm's internal stream,
            # broadcast_stream is made to wait on their end events at the
            # stream level only, and the event recorded below (consumed
            # pp_size steps later via main_stream.wait_event) fires once
            # both recvs actually completed. A synchronous broadcast() here
            # calls Work.wait(), which host-blocks until the last rank
            # reaches the matching send for THIS batch -- i.e. for a full
            # pipeline pass. Since the worker MQ loop is serial, that
            # stalls dequeue of the next execute_model and serializes the
            # whole PP schedule at one batch in flight.
            w_tok = torch.distributed.broadcast(
                sampled_tokens,
                src=self.last_rank,
                group=self.broadcast_group,
                async_op=True,
            )
            w_tok.block_current_stream()
            w_meta = torch.distributed.broadcast(
                combined,
                src=self.last_rank,
                group=self.broadcast_group,
                async_op=True,
            )
            w_meta.block_current_stream()
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            (w_tok, w_meta),
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        if self.xfer_mode == XFER_GLOO:
            # Pad to the fixed receive width so the payload shape matches the
            # pre-posted pinned recvs regardless of how many columns the
            # sampler materialized this step (spec-decode agnostic; a no-op
            # when sampled_token_ids already has max_sample_len columns).
            send_tokens = torch.nn.functional.pad(
                sampled_token_ids,
                (0, max(0, self.max_sample_len - sampled_token_ids.shape[-1])),
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            self._stage_and_send(send_tokens.contiguous(), combined.contiguous())
            return

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            # Fire-and-forget sends: async_op avoids host-blocking the last
            # rank until every receiver has posted its matching recv (under
            # a cold/slow earlier stage that would stall this rank's
            # response path by the same margin). block_current_stream keeps
            # broadcast_stream ordered after the sends so the record_stream
            # lifetime guards below remain sound.
            w_tok = torch.distributed.broadcast(
                sampled_token_ids.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
                async_op=True,
            )
            w_tok.block_current_stream()
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            w_meta = torch.distributed.broadcast(
                combined,
                src=self.last_rank,
                group=self.broadcast_group,
                async_op=True,
            )
            w_meta.block_current_stream()
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
        self.send_works.append((w_tok, w_meta))
