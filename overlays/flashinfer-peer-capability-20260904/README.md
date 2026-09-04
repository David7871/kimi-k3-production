# FlashInfer all-peer auxiliary capability

Status: **qualified** by unit tests. TP8 combined-workspace qualification is
required before deployment. The serving target does not mount this overlay.

`FlashInferPcieIpcAllReducePool` creates an all-peer CUDA IPC workspace for
each semantic stream channel. Each prepared workspace allocates one slab per
rank and opens every nonlocal rank's IPC handle. The pool now exposes
`supports_all_peer_auxiliary=True` only while at least one such workspace
exists. An unprepared, rolled-back, or closed pool reports false.

The consumer in `custom_all_reduce.py` remains fail-closed: an adapter without
the capability, or a FlashInfer pool whose workspace did not initialize, keeps
large all-reduces on PyNCCL. A successfully prepared TP2/TP4/TP8 pool permits
the configured B12X DMA crossover because the auxiliary runtime maps additional
allocations across the same peer set.

Source identity:

- vLLM PR-stack heads: #564 `3545324f2d`, #585 `b8aff7eb87`, #604
  `82d2e436d4`, #608 `ec40bfe85d`, and #610 `7c911c4396`;
- overlay SHA-256:
  `f5d2b592e40978b6198dcf4866892394f8e67152a72f240d16ca9dd39956d060`.

Unit evidence:

- `tests/distributed/test_flashinfer_pcie_all_reduce.py`: 7 passed;
- Ruff check and format check: passed;
- the lifecycle test verifies false before preparation, true after successful
  workspace preparation, and false after close.

The deployment gate is `preflight_tp8.py`: construct the FlashInfer and B12X
DMA workspaces together on the production TP8 device order, exercise an exact
BF16 payload and a random BF16 payload, verify rank-identical DMA output, and
report error against an FP32 reduction. After deployment, the target log must
report `DMA min=6291456`; all three cold decode fingerprints must be compared
with the saved `legacy` reference before any namespace acceptance.

`candidates/k3-flashinfer-peer-capability-20260904/run-preflight.sh` launches
that probe but never stops or recreates a service. It refuses to run while the
target container is active. The candidate restart fingerprint is
`c24d9801a28726b4eaeeb190d18af939446b38a8c74311f40cd5c2832979510e`,
which resolves to `iso-c24d9801a28726b4` until qualification establishes cache
compatibility.
