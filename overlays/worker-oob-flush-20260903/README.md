# Out-of-band RPC flush for the deferred draft resolve (target side)

Served-lineage files of vLLM PR #610 (`agent/kimi-k3-worker-oob-rpc-flush-20260903-pr`,
commit cbd21fc97d on #608's head 03928fdc48):

| file | mount target | content |
|---|---|---|
| `model_runner.py` | `vllm/v1/worker/gpu/model_runner.py` | the served deferred-resolve runner (`overlays/draft-deferred-resolve-20260903/model_runner.py`: ubatch prefill, graph-sync diagnostic, `_PendingDraft`) plus `GPUModelRunner.flush_deferred_draft()` |
| `gpu_worker.py` | `vllm/v1/worker/gpu_worker.py` | the branch file (`Worker.flush_deferred_output()`; `profile()` flushes and drains the device first) with `bin/patch-worker-profile-rank0-only.py` pre-applied, so the boot patch reports "already patched" against the read-only mount |
| `multiproc_executor.py` | `vllm/v1/executor/multiproc_executor.py` | the branch file: `WorkerProc.worker_busy_loop` calls `worker.flush_deferred_output()` before any RPC other than `execute_model`/`sample_tokens` |

The image's `gpu_worker.py` and `multiproc_executor.py` equal the branch base,
so the mounts add exactly the #610 diff.

Defect closed: with `VLLM_K3_DRAFT_DEFERRED_RESOLVE=1` a step's asynchronous
output is completed by the next step (`_resolve_pending_draft` records the
draft copy). The executor answers RPCs in order and the engine drains its
futures in order, so an RPC issued between two steps made the engine wait
for the previous output before issuing the next step while that output
waited for the next step (2026-09-03 06:33 UTC: `/stop_profile` during a
request, EngineCore reader logging "No available shared memory broadcast
block found in 60 seconds", 13-minute outage). The flush consumes the
proposal on every rank in the same RPC position (TP broadcasts stay
aligned); the next step finds nothing pending and reads the stored tokens.

Status: implemented and unit-tested (9 passed in the production image:
`tests/v1/executor/test_multiproc_worker_oob_flush.py`,
`tests/v1/worker/test_gpu_model_runner_v2_deferred_draft.py`); offline
probe of the torch profiler around gate-parked streams on GPU 0 passes.
Production deployment via `candidates/k3-worker-oob-flush-20260903/deploy.sh`
requires approval; its qualification replays the profiler round trip.
