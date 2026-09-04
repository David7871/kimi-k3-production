# Peer-memory draft transport (target rank 0 <-> draft server)

Served-lineage files of vLLM PR #608 (`agent/kimi-k3-draft-p2p-20260903-pr`,
commit 03928fdc48; on #604 with #584 merged so the draft server file equals
the served union of the ingest-graph and top-k lineages):

| file | mount target | content |
|---|---|---|
| `remote_speculator.py` | target `vllm/v1/worker/gpu/spec_decode/dspark/remote_speculator.py` | the deferred-resolve speculator plus the peer transport: `P2P_OPEN` handshake, context push into the draft's ring slot, verifier-named reply slot, pull behind the gate, stale-mapping detection by `server_id` |
| `p2p_transport.py` | target `vllm/v1/worker/gpu/spec_decode/dspark/p2p_transport.py` | ctypes bindings (`cudaIpcOpenMemHandle`, `cudaMemcpyAsync`), share-blob parsing (2-byte header + 64-byte handle; cudaMalloc allocations only), `DraftPeerBuffers` |
| `k3_dspark_rpc.py` | draft `/opt/k3-dspark-3090/source/vllm/entrypoints/k3_dspark_rpc.py` | `P2P_OPEN` (ring + reply slots, exported handles), `PROPOSE` with `p2p_context_slot` / `p2p_reply_slot`, `server_id` in every reply |

Mechanism: rank 0 opens the draft's context ring (3 x 1,536 x 43,008 bf16 =
66 MB on GPU 0) and reply slots (2 x 4 x 3 x 16,384 bf16 + int32) once; per
proposal it copies the context rows into a ring slot on its stream
(cudaMemcpyAsync over the PCIe fabric, then the existing synchronization)
and sends a header with the slot and the positions frame only; the draft
reads the slot into its ingest graph input, writes the top-k reply into
the slot the header named, synchronizes, and replies with a JSON header;
rank 0 pulls the reply slot into device staging behind the deferred-resolve
gate and scatters it as the frame path did. The draft never maps verifier
memory, so only the target enumerates GPU 0 (`CUDA_VISIBLE_DEVICES=1..8,0`).

Served measurements (single stream, `itl_series.py` 600 tokens): the draft
round trip `reply_rpc` 4.14 -> 2.15 ms; ctx0 ITL p50 34.5 -> 34.3 ms
(79.0 -> 79.5-82.7 tok/s); ctx64k p50 37.3 -> 36.9 ms and TTFT 51.9 -> 50.6 s
(the 22 MB context of every chunked-prefill step no longer crosses the host
twice). Acceptance unchanged (~2.1 of 3).

Options: `VLLM_K3_DRAFT_P2P` (1 = peer transport, 0 = frame transport on the
same files; any handle-open or push failure falls back to frames and logs
"peer transport unavailable/disabled"). A draft-server restart is detected
by the `server_id` in its replies; the mappings are reopened before the next
proposal.

Status: **deployed 2026-09-03 06:20 UTC** (third recreate of the day: the
first shipped `propose()` without its inference-mode decorator, the second
pulled the reply slot named by the reply — which does not exist when the
pull is enqueued — so the resolve read unfilled staging and nothing was
accepted; both fixed in the PR).
