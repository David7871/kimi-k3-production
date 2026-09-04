# DFlash draft server: captured context ingest and fused projection kernels

Overlay for the standalone Kimi-K3 DFlash draft server (container
`kimi-k3-draft`, image `kimi-k3-dspark-3090:bf16-cudagraph`, GPU 0).
Mounted by `candidates/k3-draft-ingest-graph-20260902/compose.draft-ingest-graph.override.yml`
over the served copies (`overlays/vllm-live-base/vllm/entrypoints/k3_dspark_rpc.py`,
`overlays/draft-fp8-20260902/k3_dspark_standalone.py`).

## What it changes

`k3_dspark_standalone.py`
- The draft engine config selects the CUDA per-token fp8 quantization op
  (`custom_ops: ["none", "+quant_fp8"]`), the CUTLASS fp8 GEMM
  (`linear_backend: cutlass`, per-token x per-channel scaling in the GEMM
  epilogue) and the vLLM C RMSNorm kernels (`ir_op_priority` vllm_c first).
  The torch-native equivalents issued 8-13 tiny kernels per projection
  inside the captured proposal graph (about 600 per proposal).
- The runtime installs the configured IR op priorities
  (`kernel_config.ir_op_priority.set_default()`); vLLM workers do this at
  init, the standalone runtime has no worker, so every IR op ran
  torch-native regardless of the configuration.

`k3_dspark_rpc.py`
- The per-step context append (the accepted rows of one decode step:
  1..max_num_seqs x (1 + K) rows, at most one KV block) replays a CUDA graph
  per row count: raw target auxiliary rows -> combined draft states -> every
  layer's K/V -> RoPE -> KV cache write at the staged slots. Larger appends
  (prefill-sized, reconnects) and projected rows keep the eager path.
  `K3_DRAFT_INGEST_GRAPH=0` disables the capture (diagnostics).
- The probabilistic logits frame is copied to the host through a pinned
  buffer instead of a pageable copy.
- Top-k draft-distribution transport (vLLM PR #585, see
  `overlays/draft-topk-logits-20260902/README.md`): when the target's
  request carries `logits_topk: k`, the server answers with the k largest
  logits per position (BF16 values + int32 indices) instead of the full
  vocabulary. Dormant while the target keeps `VLLM_K3_DRAFT_LOGITS_TOPK=0`.
- `K3_DRAFT_LOGITS_MASS_DIAG=k1,k2,...` logs the draft probability mass
  outside each top-k over served proposals (per 1024 positions); the
  candidate enables it for 256/1024/4096/16384.

Draft-side only. The target's verification is untouched; draft logits
change at fp8-rounding level (GEMM kernel), so acceptance is expected to be
unchanged within noise. The captured ingest is exact: draft tokens are
identical with the graphs on and off.

## Evidence (GPU 0, second draft instance, `research/decode-step-20260902/run_draft_profile.sh`)

| configuration | batch | proposal total (ms) | gpu_context | gpu_query | host_submit |
| --- | --- | --- | --- | --- | --- |
| served (torch kernels, eager ingest) | 2 x 3 rows | 5.89 | 1.15 | 4.13 | 1.85 |
| fused kernels only | 2 x 3 rows | 4.61 | 1.02 | 3.04 | 1.40 |
| fused kernels + captured ingest | 2 x 3 rows | 3.72 | 0.51 | 2.81 | 0.57 |
| fused kernels + captured ingest | 1 x 3 rows | 3.42 | 0.47 | 2.67 | 0.48 |
| fused kernels + captured ingest | 4 x 4 rows | 3.95 | 0.58 | 2.76 | 0.72 |

Logs: `out/draft_profile_{served,varA,varA2,varB,varB_noingest,varB_b1r3,varB_b4r4,varB_b2r1}_*.log`.
Remaining GPU time is the fp8 GEMMs (25 projections 1.71 ms, lm_head
0.89 ms) at ~75 % of HBM bandwidth.

PRs: local-inference-lab/vllm #584 (`agent/kimi-k3-draft-ingest-graph-20260902-pr`,
stacked on the fp8 draft PR #570 lineage, which predates the probabilistic
logits frame) and #585 (`agent/kimi-k3-draft-topk-logits-20260902-pr`,
stacked on #564: served-copy sync + pinned logits + top-k transport +
mass diagnostic). The overlay file is the served server with both PRs'
hunks applied.

Status: deployed 2026-09-02 20:37 KST (`candidates/k3-draft-ingest-graph-20260902/qualification/`):
target-side server_total ~5.0 -> ~3.4 ms, rpc_roundtrip 5.6-6.2 -> 4.5-4.9 ms,
acceptance 2.07-2.51, no draft errors.
