# Lever: small-kernel fusion (#4) and GEMM merging (#9)

Date: 2026-10-01. Lane: Kimi-K3-QSRT-K2 PP4xTP4/DCP4, rtx-01 (S0,S1) + rtx-02 (S2,S3), EXEC=up2l3, production env `~/int6.envstr` (K3_MNS=256, prefix caching ON, b12x PCIe allreduce, DCP b12x opts). Overlay copy: `/data/models/k3-run/overlay-vllm-l3-fusion` (both hosts), = live `overlay-vllm-l3` + (a) env-gated layer tracer in `model.py` (inert unless `VLLM_K3_LAYER_TRACE_TOKENS` set) and (b) `latent_moe_runner.py` (new overlay file, MOUNTS.txt line added) with the tail-norm fusion behind default-off `VLLM_K3_FUSE_TAIL_NORM=1`.

## 1. Kernel inventory (what the small time actually is)

Sources: `nsys/fuse.py` on `roof-c16-s3.txt`/`roof-c256-s3.txt`, the `~/nsys6-c16` sqlite (full kernel names+grids), and an eager layer trace with python stacks (new: `VLLM_K3_LAYER_TRACE_TOKENS=16` hook in the overlay copy; output `rtx-01:~/k3-results/fuse-trace/trace-dump.txt`).

fuse.py totals per S3 cycle (12.540 ms C16 / 54.219 ms C256):

| class | C16 ms | C256 ms |
|---|---|---|
| small (norms/elementwise/quant/split-K reduce/router glue) | 0.65 | 0.85 |
| attn_res (2 fused kernels/layer) | 0.35 | 0.47 |
| inter-kernel gaps | 0.39 | 0.72 |
| L1 estimate (merge each chain to one kernel) | 0.50 | 0.54 |
| L2 estimate (fold into AR epilogue / GEMM prologue) | 1.05 | 1.70 |
| GEMM fixed-cost bound (#9; 4.4 µs/GEMM) | 0.87 | 0.90 |

Per decode layer at 16 tokens (eager trace, exact kernel order; comm durations omitted — inflated by rank skew in the eager pass):

1. `_attn_res_kernel` 7.0 (pre-attn norm, already fused Triton)
2. MXFP8RowsQuant 1.3 + dense_gemm 52.4 (KDA in_proj, MXFP8) + CuteSkinnyGemm 44.0 + nvjet 2.5
3. `kda_decode_fusion_many_heads_kernel` 16.0
4. MXFP8RowsQuant 1.2 + dense_gemm 18.8 (o_proj, MXFP8) → TP all-reduce
5. `_attn_res_kernel` 6.9 (post-attn norm, already fused)
6. router gate: nvjet splitK 4.5 + cublasLt `splitKreduce_kernel` 1.8 (bf16→fp32 `torch.mm`) → all-gather
7. elementwise 1.5 + routed down_proj: CuteSkinnyGemm 14.6 (bf16 skinny, `VLLM_K3_SKINNY_GEMM`) → all-gather
8. elementwise 2.2 + `single_group_topk_block_kernel` 4.1 (`VLLM_K3_FAST_TOPK`, already single launch) + elementwise 1.4 + Fill 0.9 + `_pack_topk_routes_small_prefix` 1.6 + `_pack_topk_routes_sort` 1.1
9. routed experts: W4A16FusedMoe + W4A16TopKSum (QSRT, b12x; off-limits to external kernels)
10. shared experts on aux stream (already overlapped, CANDIDATES #25): copy 1.0 + quant 1.2 + dense_gemm 24.0 + `situ_and_mul` 2.1 + quant 1.2 + dense_gemm 10.3
11. latent-MoE tail tier-1 (`LatentMoERunner._overlap_allreduce_tail`): latent all-reduce → **eager 9-kernel RMSNorm chain 18.7** → up_proj CuteSkinnyGemm 14.6 (+ shared AR overlapped) → add 1.1

What existing flags already cover: `VLLM_K3_PREFILL_FUSE` = scheduler micro-batch admission (not kernels); `VLLM_K3_SKINNY_GEMM` = CuTe DSL skinny GEMM for BF16 linears M≤16 (the `vllmmode_*` kernels); `VLLM_K3_FAST_TOPK` = single-launch grouped top-k. attn_res and q/kv RMSNorm (`_fused_q_kv_rmsnorm_kernel`, MLA only) are already fused custom kernels. torch.compile is a NO-GO for decode (CANDIDATES #40: eager model + FULL cudagraph, no FX passes; #18 fused-AR-RMSNorm compile pass CLOSED for the same reason). Tier-0 latent tail fusion (`KimiK3LatentMoETailOp`) is SM100-only — dead on SM120.

## 2. The eager chain: IR native rms_norm fallback (root cause)

The 9-kernel chain (cast→pow→mean 6.5→add→rsqrt→mul→cast→mul→cast, 18.7 µs at T=16, in BOTH KDA and MLA layers, right after the latent all-reduce) is `routed_expert_norm(reduced)` in `LatentMoERunner.allreduce_norm_latent_out` (fallback path; the flashinfer fused AR+norm is unavailable with the b12x PCIe all-reduce). The eager trace proves `RMSNorm.forward` runs the **native PyTorch reference** (`vllm/ir/ops/layernorm.py`) here, and the worker logs the matching warning (`Priority not set for op rms_norm, using native implementation.`). The stated mechanism — an empty IR dispatch priority list — was inspected in a CPU-only container, not dumped from a GPU worker; treat it as the likely, not proven, cause. Once per layer per step (23 layers/stage on S3): ~17 µs/layer critical path at C16 ≈ 0.39 ms/cycle (3.1%); similar at C256 (~0.8%).

## 3. Fusion implemented: fused tail RMSNorm (`VLLM_K3_FUSE_TAIL_NORM=1`)

`overlay-vllm-l3-fusion/latent_moe_runner.py` (site-packages `vllm/models/kimi_k3/nvidia/latent_moe_runner.py` + additive patch, 39 added lines, zero deletions): in the tier-1 tail fallback, serve the norm with the single fused vLLM C kernel (`torch.ops._C.rms_norm(out, x, weight, eps)`) when dtypes/shape allow, else keep the old path. Same math, fp32 accumulation both ways, different reduction order → not bit-exact. Default off. (First A/B attempt died at arch-inspection: the patch had initially been based on the newer `k3-up-tree` source, which imports `invariant_gemm` absent from site-packages; rebased onto the real site-packages file, import-verified at the exact mount path on both hosts.)

### Unit test (1 GPU, under lock; `test_tail_norm.py` in this folder)

| shape | scale | max-abs-diff | rel-max | bit-exact |
|---|---|---|---|---|
| T∈{16,64,256} | w-scale ∈ {0.05, 1, 20} | 3.1e-2 – 6.3e-2 | 2.9e-3 – 6.9e-3 | no (1–2 bf16 ulp) |

Eager microbench (best of 5×300, launch-dominated): native 56.6 µs vs fused 5.7 µs at T=16 and T=64. Inside the cudagraph the chain's GPU time is 18.7 µs vs ~2 µs for the single kernel.

Mean abs diff 8.7e-4 across all nine cells; max-abs equals 1 bf16 ulp at the max-magnitude element (values ≈4–8). The run also prints the smoking gun live: `Priority not set for op rms_norm, using native implementation.`

## 4. #9 GEMM merging — assessment

The only back-to-back weight GEMMs sharing one input are router gate (bf16→fp32 `torch.mm`, nvjet splitK 4.5+splitKreduce 1.8) and routed down_proj (CuteSkinnyGemm 14.6), both reading the post-attn-norm hidden. At C16/C256 they already run concurrently on separate streams (`_maybe_overlap_router_and_down_proj`), so a load-time weight concat would only save one fixed start-up (~4.4 µs) and the split-K reduce, at the cost of bf16-rounding the fp32 router logits (numerics change to expert selection). The assignment's other named pairs (o_proj with router/latent-down, shared gate/up with latent up) do NOT share inputs and cannot be concatenated. Estimated ceiling ≤0.3 ms/cycle (<2.5%) → **rejected on analysis** (the measured tail-norm result below, +0.36%, confirms the small-kernel classes here are all sub-1% each).

## 5. Lane A/B (C16 + C256; script `rtx-01:~/k3-fusion-ab.sh`, tmux `fuseab`, lock held)

decode_bench: C4 warm + C16 warm discard, then 3×30 s measured runs per conc; median (band) tok/s. A = fusion overlay, flags OFF; B = same overlay, `VLLM_K3_FUSE_TAIL_NORM=1`.

| arm | flags | C16 median (band) | C256 median (band) | KV tokens | probe 17×23 |
|---|---|---|---|---|---|
| A | — | 335.5 (335.5–335.8) | 1259.6 (1257.1–1279.0) | 13,080,081 | PASS |
| B | VLLM_K3_FUSE_TAIL_NORM=1 | 336.7 (336.6–336.8) | 1262.2 (1256.7–1279.0) | 13,080,081 | PASS |

Deltas: **C16 +0.36%** (bands disjoint, real), **C256 +0.2%** (bands overlap, noise). KV capacity identical; DCP untouched. The A/B establishes only the throughput deltas; the explanation for why the gain is far below the ~3% eager-trace estimate (launch-gap vs on-graph kernel time, critical-path position) is a hypothesis, not traced here.

Quality: 391-probe PASS both arms with coherent reasoning. Greedy 20-prompt diff (sequential, temperature 0, so batching is not an evidenced cause): **15/20 completions differ**, including three first-token divergences; 5/20 identical; divergent entries mostly share long prefixes (median lcp ≈131 chars). Raw-completion template-leak artifacts appear in BOTH arms (incl. A), and sampled factual answers are correct in both (Canberra; 98.6 °F / 310.15 K identical). There is NO A-vs-A control capture, so intrinsic run-to-run nondeterminism cannot be separated from fusion-induced change: **quality impact is INCONCLUSIVE**. The unit test bounds the per-call perturbation at 1–2 bf16 ulp; how that propagates through greedy decode (a router/top-k amplification mechanism is plausible but not proven) is unresolved. Flag stays default-OFF.

## 6. Recommendation

Do NOT adopt for production and do not expand: +0.36% C16 / noise C256 is below the 3% bar, and the greedy quality result is inconclusive (§5). The flag (`VLLM_K3_FUSE_TAIL_NORM`, default OFF) and overlay copy stay available for reproduction. The durable finding is the root cause, not the lever: **the `rms_norm` IR op runs its eager native reference on this lane** — the workers log `Priority not set for op rms_norm, using native implementation.` (observed for rms_norm; whether other `ir.ops` also fall back was not traced). A worker-init `ir_op_priority.set_default()` may be the systemic fix, but that inference comes from a CPU-only container inspection plus the live warning, not a GPU-worker priority dump. #9 GEMM merging is rejected on analysis (§4). Note: the same-session A arm is the first post-fabric-fix baseline: 335.5 C16 / 1259.6 C256 (vs stale pre-fix 326–327 / 1199–1222).

## 7. What remains

- Model-final RMSNorm on S3 (once per step, same eager fallback, ~18 µs ≈ 0.15%): same one-line fix, not done (scope).
- Level-B folding of attn_res/small kernels into the b12x AR epilogue (est 1.05 ms C16 / 1.70 ms C256 per fuse.py): requires b12x comm-kernel epilogue surgery; FastAllReduce confirmed no conflict. Not attempted in this lever.
- Router glue chain (elementwise 1.5/2.2/1.4 + Fill + 2 pack kernels ≈ 8.7 µs/layer) and gate splitKreduce (1.8 µs/layer): each <1%/layer class; not expanded.

### Exact env used

A: `$(cat ~/int6.envstr)`; B: `+ VLLM_K3_FUSE_TAIL_NORM=1`. Both: `K3_L3V=/data/models/k3-run/overlay-vllm-l3-fusion EXEC=up2l3`, boot/bench via `~/k3-fusion-ab.sh` inside `flock /tmp/k3-lane.lock` (tmux `fuseab`). Containers `kimi-k3-qsrt-pp4` (rtx-01) and `k3-qsrt-ray-worker` (rtx-02) removed after each arm and at script end.
