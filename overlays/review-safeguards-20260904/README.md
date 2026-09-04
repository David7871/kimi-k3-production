# Kimi-K3 decode and draft-server safeguards

Status: **implemented** and operationally qualified in the serving target and
draft services. Production-cache compatibility is **unsupported** because the
fail-closed PCIe capability check changes the large-prefill all-reduce route.

The files in `overlays/review-safeguards-20260904/` preserve the Kimi-K3 model
and cache layout while enforcing bounds, ordering, and ownership invariants at
the target/draft boundary. The PCIe capability guard can select a different
all-reduce algorithm and therefore a different floating-point reduction order.

The target service mounts implement these invariants:

- a pending remote-draft reply is resolved before request slots can be reused;
- CUDA IPC exports cover complete `cudaMalloc` allocations, and peer copies are
  rejected when their byte ranges exceed the exported allocation;
- the remote draft server must export the context ring and reply slots through
  the shared IPC contract;
- out-of-place collective results are consumed by the Kimi-K3 MLA path;
- the PCIe DMA route is disabled unless the runtime explicitly advertises
  auxiliary allocations that are peer-accessible on every participating GPU;
- the 64 KiB W4A16 direct lookup table is materialized outside CUDA graph
  capture and retained by the compiled kernel object.

The draft service mounts implement these invariants:

- deferred requests do not emit an early reply;
- concurrent ingest failures are returned to the caller;
- capture-index records are appended without rewriting prior records;
- cache smoke tests use the configured draft width and clear every cache block
  touched by the test sequence;
- DFlash model loading receives the configured draft load policy;
- the server and target use the same CUDA IPC export implementation.

Source qualification evidence:

- the served-lineage W4A16 CUDA suite passed 9 tests;
- the served-lineage model-runner suite passed 6 tests;
- the integrated Python files compile without syntax errors;
- the contributing vLLM branches passed 8 peer-transport GPU tests, 6 final
  runner tests, and 8 focused CPU tests;
- the contributing b12x branch passed 10 master-lineage tests and 9 benchmark
  unit tests.

`candidates/k3-review-safeguards-20260904/restart-draft.sh` recreates only the
draft service. `candidates/k3-review-safeguards-20260904/restart.sh` recreates
only the target service through the configuration-fingerprint namespace gate.
The target configuration fingerprint is
`9edabb5620e30d56b900db61a8a9bea998c889a691917111a16abaa94daaac73`.
It resolves to LMCache namespace `iso-9edabb5620e30d56`; do not accept that
fingerprint into the `legacy` namespace. The capability guard reports
`DMA min=off`, so large prefill all-reduces use PyNCCL instead of the DMA ring.
All three cold decode probes differ from the `legacy` reference at token 0,
which is consistent with the changed reduction order.

Operational qualification on 2026-09-04 UTC verified exact hashes for all ten
mounted files, active CUDA peer transport, a successful profiler RPC sequence
during decoding, no fatal target or draft log entries, and a clean 600-token
stream at 81.1 tokens/s with 35.1 ms p99 inter-token latency. Evidence is in
`candidates/k3-review-safeguards-20260904/qualification/`.
