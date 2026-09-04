# Balanced dense-MLA split ranges (b12x PR #291, vLLM PR #587)

Files mounted by `candidates/k3-longctx-20260902/compose.dense-mla-balanced-splits.override.yml`:

- `_forward.py` — `b12x/attention/dense_mla/_forward.py` from the served
  kernel lineage (`~/k3-longctx-b12x-mla-splits`, branch
  `agent/kimi-k3-dense-mla-balanced-splits-20260902`, 00a14d3). Every
  launched split scans `ceil(selected chunks / launched splits)` 64-token
  chunks; the plan's `chunks_per_split` only sizes the split scratch.
- `b12x_mla.py` — `vllm/v1/attention/backends/mla/b12x_mla.py` from the
  served vLLM lineage (`~/k3-longctx-vllm-mla-splits`, same branch name,
  2ca9d200e). Eager launches use `min(num_splits, live chunks)` splits;
  CUDA-graph launches keep the full plan.

Effect (RTX PRO 6000, bf16 KV, 131,072-token plan, 47 splits, forward +
merge): 126.6 -> 10.5 us at 4k live tokens, 134.2 -> 47.9 us at 40k;
max |out - reference| unchanged. Production expectation: -5 ms per decode
step at >= 64k context (DenseMlaForwardKernel 5.7 ms -> < 1 ms across the
24 MLA layers).

Numerics: the same partial products are combined through the same
log-sum-exp merge in a different split order; outputs are within the
kernel's reference tolerance but not bit-identical to the served split
order. Prefill numerics (and therefore the KV cache contents) are
unaffected.

Status: offline-qualified (`research/longctx-20260902/run_dense_mla_tests.sh`
on the served, #271 and master lineages). Not deployed.
