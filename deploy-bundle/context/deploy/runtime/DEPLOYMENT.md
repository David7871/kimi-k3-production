# Kimi-K3 TP8/DCP8 runtime specification

Status: qualified

The deployment image is
`ghcr.io/myshytf/kimi-k3:all-compatible-20260901-r6-deploy`. It serves the
Kimi-K3 target model under the API identifier `claude-mythos-5` and uses a
dedicated DFlash draft process.

## Scheduling and cache invariants

- Target parallelism: tensor parallel 8 and decode-context parallel 8.
- Maximum model context: 1,048,576 tokens.
- Target scheduling chunk: 1,536 tokens.
- Maximum concurrent sequences: 4.
- Speculative depth: 3.
- GPU KV reservation: 3 GiB per target rank.
- Mamba, prefix-match, and LMCache chunk boundaries: 1,536 tokens.
- LMCache L1: 80 GiB direct shared memory with a 4 GiB host reserve.
- LMCache L2: host-mounted filesystem storage; configured capacity 2,457.6
  GiB.

The L1 arena is process-local and is recreated at startup. Reusing an unmapped
arena is unsupported because LMCache does not persist its L1 key/allocation
index. L2 objects are indexed at startup and remain reusable when model,
tokenizer, cache geometry, and token prefix are identical.

Async external-KV loads pass full-input admission before allocating transfer
blocks. A request returning from a completed load skips that duplicate
admission gate and remains subject to the per-step free-block check. This
prevents heterogeneous cache groups from double-counting held transfer blocks.

LMCache Python source is loaded from `/opt/infernal-invocation/lmcache`. Its
package path includes the matching installed wheel directory so
`lmcache.c_ops` resolves to the compiled extension and the transfer interface
uses native pointers.

## Qualification evidence

- Container health endpoint returned HTTP 200.
- Eight target workers registered `mode=SHM` and pinned the shared-memory arena.
- Native block transfer reported `multi_layer_block_kv_transfer mode: ptr`.
- A 163,651-token Claude Code request moved from external-KV loading to model
  execution, completed prefill, entered decode, and executed shell tools.
- The request queue remained at zero waiting requests during prefill and
  decode.
- LMCache L2 accepted and completed every observed store task; no store task
  remained pending or in flight.
- EngineCore reported no fatal error, allocator OOM, or abort-cleanup exception
  during qualification.

## Limitations

- Model weights are not included in the image.
- Prefix reuse requires an exact token-prefix match. Claude Code project-memory
  changes can invalidate the prefix near the beginning of a request.
- The profile requires nine GPUs and a host shared-memory mount large enough
  for the configured L1 arena.
