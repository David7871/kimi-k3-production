# Small-M MXFP8 dense linears: activation quantization inside the GEMM

`_kernel.py` = the served `b12x/gemm/mxfp8_linear/_kernel.py` (b12x snapshot
46c84bab) + b12x branch `agent/kimi-k3-mxfp8-linear-fused-quant-a-20260902-served`
(the same change on origin/master: PR #288, branch `...-20260902-pr`). Mounted
at `/opt/infernal-invocation/b12x/b12x/gemm/mxfp8_linear/_kernel.py` by
`candidates/k3-mxfp8-linear-fused-quant-20260902/compose.mxfp8-linear-fused-quant.override.yml`.

The Kimi-K3 serialized-MXFP8 dense projections (`B12xMxfp8LinearKernel`,
four per layer) ran, before every dense GEMM, the standalone activation
quantizer: two E8M0 scale-buffer fills plus the row quantization kernel.
For up to eight tokens with BF16 activations, an unpadded K and an N that
is a whole number of 64-wide tiles and at most
`B12X_MXFP8_LINEAR_FUSED_QUANT_A_MAX_N` (default 4096), the linear now
runs `dense_gemm_fused_quant_a` (A quantized in each CTA; the block-FP8
linear already uses it for the same token range). Bit-identical output;
three launches fewer per eligible projection. Wider N keeps the separate
path (the fused GEMM has no split-K and is slower at N = 7168).

Evidence: `research/decode-step-20260902/out/mxfp8_fused_quant_bench_gpu.log`
(GPU time per projection at 8 tokens: N=1536 13.0 -> 10.0 us, N=576
9.3 -> 6.4, N=896 12.6 -> 6.6, N=3584 11.8 -> 10.8; all shapes and
1/2/4/8 tokens bit-identical), `out/mxfp8_fused_quant_tests2.log`
(new tests pass; the MXFP8/block-FP8 suites keep the served failure set).

Expected production effect: the eligible projections lose ~5 us of kernel
time and three launches each; about 1-1.5 ms per decode step at batch 8.
Takes effect on the next target boot. `B12X_MXFP8_LINEAR_FUSED_QUANT_A_MAX_N=0`
restores the separate quantizer without changing the mount.

Status: deployed 2026-09-02 20:40 KST (`candidates/k3-decode-20260902`).
