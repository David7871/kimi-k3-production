# Deferred consumption of the remote draft reply (target side)

Served-lineage files of vLLM PR #604 (`agent/kimi-k3-draft-deferred-resolve-20260903-pr`,
commit 655c3240aa on #585's head 5444576fff):

| file | mount target | content |
|---|---|---|
| `remote_speculator.py` | `vllm/v1/worker/gpu/spec_decode/dspark/remote_speculator.py` | the served top-k transport file plus the deferred resolve: reply thread, pinned staging of every reply field, `_StreamGate` (`cudaLaunchHostFunc(pthread_barrier_wait)`), `resolve_pending()` (gate, copies, broadcast, masked Gumbel sampling) |
| `model_runner.py` | `vllm/v1/worker/gpu/model_runner.py` | the served ubatch-prefill runner plus: `_PendingDraft`, `_resolve_pending_draft()` before the input-id combine, the combine moved after the attention metadata (`_late_input_ids`), the deferred draft copy of the step's `AsyncOutput` |
| `async_utils.py` | `vllm/v1/worker/gpu/async_utils.py` | `AsyncOutput.get_output()` waits for a draft copy recorded by a later step |

Mechanism: rank 0 issues the draft RPC on a reply thread; `propose()` returns
at once; the next `execute_model` prepares the step (update_batch,
prepare_inputs, block tables, attention metadata) while the round trip is in
flight, then parks the stream on the gate and enqueues the reply copies, the
TP broadcast, the draft sampling, the token store and the input-id combine
behind it. No host thread waits for the reply. Draft tokens, cached draft
logits and sample positions equal the synchronous path's (unit test on GPU 0).

Options: `VLLM_K3_DRAFT_DEFERRED_RESOLVE` (1 = deferred, 0 = synchronous on
the same files); `VLLM_K3_DRAFT_TIMING_LOG_INTERVAL` (N = log the per-phase
means every N proposals, including the reply thread's `reply_rpc`).

Status: implemented and unit-tested (47 passed with these files mounted in
the production image); production deployment via
`candidates/k3-draft-deferred-resolve-20260903/deploy.sh` requires approval.
