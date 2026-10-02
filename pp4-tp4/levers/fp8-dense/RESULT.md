# Lever: FP8 (MXFP8) for the remaining BF16 dense weights

Date: 2026-10-01. Lane: Kimi-K3-QSRT-K2 PP4xTP4/DCP4, rtx-01 (S0,S1) + rtx-02 (S2,S3),
EXEC=up2l3, production env `~/int6.envstr` (K3_MNS=256, skinny GEMM on, b12x PCIe AR, DCP b12x).

Flag: `VLLM_K3_FP8_DENSE=gproj,latent,kvb,router,lmhead` (default OFF), overlay
`/data/models/k3-run/overlay-vllm-l3-fp8dense/linear.py` (copy + patch in this folder).

## 1. Inventory: which dense GEMMs are still BF16

Checkpoint quant config (`/data/models/Kimi-K3-QSRT-K2/config.json`,
`quantization_config`): `dense_format: mxfp8`, `ignored_layers:
["kv_b_proj","g_proj","f_a_proj","f_b_proj","b_proj"]`. So o_proj, q_b_proj,
fused_qkv_a_proj, in_proj_qkv and the shared-expert MLPs already run MXFP8 W8A8
through the b12x dense GEMM (`b12x_lib dense_gemm ... f8E4M3FN` kernels in nsys).
What is left in BF16 at decode (per TP4 rank, hidden 7168, 96 heads, qk_nope 128,
qk_rope 64, v 128, kv_lora 512, latent 3584, 896 experts):

| module | class | N | K | MB/rank | layers | decode kernel today |
|---|---|---|---|---|---|---|
| `self_attn.g_proj` (MLA output gate) | gproj | 3072 | 7168 | 44.0 | 24 | skinny (M<=16) / nvjet bf16 |
| `self_attn.in_proj_gfab` (KDA full-rank gate g+f_a+b, qkv is MXFP8) | gproj | ~3136 | 7168 | ~44.9 | 69 | skinny / nvjet bf16 |
| `mlp.routed_expert_down_proj` (latent down) | latent | 896 | 7168 | 12.8 | 93 | skinny / nvjet bf16 |
| `mlp.routed_expert_up_proj` (latent up) | latent | 7168 | 896 | 12.8 | 93 | skinny / nvjet bf16 |
| `mlp.gate` (router, TP-sharded 896/4) | router | 224 | 7168 | 3.2 | 93 | nvjet bf16 -> fp32 |
| `self_attn.kv_b_proj` | kvb | 6144 | 512 | 6.3 | 24 | **not a decode GEMM**: absorbed into W_UK_T/W_UV bmms at load (`k3_nvidia_mla.py:762 process_weights_after_loading`); only prefill runs it |
| `lm_head` | lmhead | 40960 | 7168 | 587.2 | S3 only | nvjet bf16 (8 vocab chunks) |
| `self_attn.f_b_proj` | (not selected) | 3072 | 128 | 0.8 | 69 | negligible |

Per-cycle BF16 dense time measured from the existing nsys captures
(`~/nsys6-c16`, `~/nsys6-c256`, demangled-name aggregation, S3 = the LM-head stage):

| capture | bf16 skinny | lm_head | other nvjet bf16 | total bf16 dense | cycle |
|---|---|---|---|---|---|
| C16 S3 | 1.281 ms (69 calls) | 0.393 ms (nvjet grid 10240, 1/cycle) | ~0.17 ms | ~1.85 ms | 12.540 ms (14.8%) |
| C256 S3 | 0 (M=64 > skinny max) | in nvjet pool | ~1.8 ms | ~1.9 ms | 57.055 ms (3.3%) |

## 2. SM120 FP8 GEMM paths available in the image

- `torch._scaled_mm` exists (torch 2.13.0+cu133) — not pursued: per-token fp8
  activation quant on bf16 activations, accuracy risk, and the fork already ships
  a better-fitting in-image path.
- `b12x.gemm.mxfp8_linear` (pack_weight/mm, custom-op, CUDA-graph safe) — the
  exact kernel the checkpoint-MXFP8 dense modules already run on this lane
  (`--linear-backend b12x`, `VLLM_USE_B12X_FP8_GEMM=1`). Weight: e4m3 block-32 +
  e8m0 scales; activation: bf16 -> MXFP8 inline per call (MXFP8RowsQuant).
- vLLM fork `Mxfp8OnlineLinearMethod`
  (`vllm/model_executor/layers/quantization/online/mxfp8.py`): quantizes a bf16
  weight to MXFP8 in `process_weights_after_loading` and applies via the b12x
  kernel. The lever wires this method into the overlay
  `UnquantizedLinearMethod.process_weights_after_loading` for the selected
  module classes and swaps `layer.quant_method`.


### Microbench (1 GPU, CUDA-graph replay; pool of 8 weight copies captured inside
### the graph -> cold weights, matching decode streaming; `microbench3.py`)

µs per GEMM (weight bytes GB/s in parentheses):

| shape (N,K) | M=4 bf16/mxfp8 | M=16 | M=64 |
|---|---|---|---|
| g_proj (3072,7168) 44MB | 31.6 / 27.5 (**1.15x**) | 32.1 / 25.5 (**1.26x**) | 32.5 / 25.4 (**1.28x**) |
| gfab (3136,7168) 45MB | 32.4 / 27.6 (**1.17x**) | 32.6 / 25.5 (**1.28x**) | 33.2 / 25.5 (**1.30x**) |
| latent_down (896,7168) 12.8MB | 5.3 / 24.0 (0.22x) | 5.1 / 20.9 (0.25x) | 8.1 / 20.5 (0.40x) |
| latent_up (7168,896) 12.8MB | 5.4 / 8.5 (0.64x) | 5.1 / 8.3 (0.61x) | 7.4 / 9.0 (0.83x) |
| router (224,7168) 3.2MB | 3.6 / 24.2 (0.15x) | 3.6 / 21.3 (0.17x) | 4.9 / 21.3 (0.23x) |
| **lm_head (40960,7168) 587MB** | 389 / 213 (**1.83x**) | 395 / 213 (**1.85x**) | 406 / 220 (**1.85x**) |
| o_proj (7168,3072) ref | 31.9 / 22.7 (1.40x) | 32.6 / 20.8 (1.57x) | 32.1 / 20.5 (1.57x) |

Notes: an earlier eager-mode bench (microbench.py) showed a ~190 µs host-side
floor per `mx.mm` call; inside CUDA graphs (how the lane runs) the GPU-time floor
is ~21-26 µs for small-N shapes, which still exceeds the whole bf16 GEMM for the
12.8 MB latent projections and the 3.2 MB router. cuBLASLt bf16 hits ~1.4 TB/s
cold on the big shapes; b12x MXFP8 reads ~0.52x the bytes and nets 0.9 TB/s
effective-on-bf16-bytes — a win only when the weight is large enough to amortize
the fixed cost (g_proj 44MB: yes, latent 12.8MB: no, router 3.2MB: no).

Caveat for C16: at M<=16 the live bf16 path for g_proj/gfab/latent is the CuTe
DSL skinny GEMM (nsys avg ~18.6 µs over the 69 calls/cycle mix), not cuBLASLt,
so the microbench's 1.15-1.28x for gproj at M=4/16 is measured against a slower
bf16 baseline than production. The lane A/B decides.

## 3. Implementation

Overlay copy `/data/models/k3-run/overlay-vllm-l3-fp8dense/` (rsynced to
10.10.10.2 same path): patched `linear.py` (`linear-fp8dense.patch`) plus two
new CuTe DSL files (added to its `MOUNTS.txt`). With both flags unset the code
path is byte-identical to live. Two execution paths, both default-OFF and
per-class selectable by suffix match on `layer.prefix`:

1. **`VLLM_K3_FP8_DENSE=<classes>`** — MXFP8 (block-32 e4m3 + e8m0) at load via
   the fork's `Mxfp8OnlineLinearMethod`, GEMM through `b12x.gemm.mxfp8_linear`
   (the kernel the checkpoint-MXFP8 modules already run). Covers every M.
   Weight memory roughly halves (lm_head 587->293 MB/rank).
2. **`VLLM_K3_FP8_SKINNY=<classes>`** — weight-only FP8 (e4m3, per-output-row
   fp32 scale) at load, served at M<=16 by a new CuTe DSL skinny-GEMM variant
   (`_skinny_gemm_fp8.py`, `skinny_gemm_fp8.py`; same reduction order as the
   BF16 skinny kernel, scale applied in the epilogue). The BF16 weight is kept
   for the M>16 cuBLASLt fallback, so C256 behavior is unchanged. This path
   exists because the b12x MXFP8 GEMM has a ~21-26 µs floor that loses to the
   BF16 skinny kernel for the small/medium shapes at decode M (see microbench).

Failures log and keep BF16. kv_b is excluded from the selector because MLA
decode absorbs it into the bmm weights (no decode GEMM to speed up).

## 4. Lane A/B (tmux fp8dense-ab, flock /tmp/k3-lane.lock, 2026-10-01)

decode_bench: streaming, ignore_eos, 30 s window after 10 s warm-in, built-in C4
warm pass discarded, 3 measured runs per conc (16,16,16,256,256,256). Script:
`rtx-01:~/k3-fp8dense-ab.sh` (copy in this folder).


All three arms ran in ONE session AFTER Main's rtx-02 fabric fix (16:44) — the
fix lifted the baseline (previously C16~326 / C256~1200), so no pre-16:35
number is comparable.

| arm | flags | C16 median (band) | dC16 | C256 median (band) | dC256 | probe 17*23 |
|---|---|---|---|---|---|---|
| A baseline | (live) | 335.7 (335.7-336.2) | - | 1257.4 (1250.8-1281.1) | - | PASS |
| B | `VLLM_K3_FP8_DENSE=lmhead` | 338.5 (338.1-338.7) | **+0.8%** | 1262.8 (1255.4-1274.9) | +0.4% | PASS |
| C | `VLLM_K3_FP8_DENSE=gproj,lmhead` | 342.7 (342.2-343.4) | **+2.1%** | 1259.4 (1255.4-1280.0) | +0.2% | PASS |

KV cache size identical on all arms (13,080,081 tokens); conversion log lines
(`linear.py:518 VLLM_K3_FP8_DENSE ...`) present on all 16 ranks incl. rtx-02
workers. C256 deltas are inside the run-to-run band (~2.4%) — FP8 changes
nothing measurable at M=64 (the weight streams at ~0.9-1.4 TB/s either way).

## 5. Quality

(25 prompts, greedy, A baseline vs arm; `fp8_quality_dump.py` / `fp8_quality_cmp.py`)

| arm | greedy exact-match | gsm-style correct |
|---|---|---|
| B (lmhead) | 9/25 | 12/12 = A |
| C (gproj,lmhead) | 3/25 | 12/12 = A |

Greedy divergence (first-divergence word index typically 0-30) is the expected
effect of e4m3 weight rounding; task-level correctness is unchanged.

## 6. Recommendation

- **lm_head MXFP8: +0.8% C16, ~0% C256 — below the 3% bar.** The 1.85x kernel
  speedup on a 0.39 ms GEMM nets only ~180 µs of a 12.5 ms cycle. Take it only
  if the 294 MB/rank weight saving matters; as a speed lever it does not clear
  the bar and (per user rule) is not expanded into a campaign.
- **g_proj MXFP8 adds +1.2% C16 (B->C) — also below the bar alone.**
- latent / router / kvb MXFP8: not lane-tested — microbench showed 0.15-0.83x
  (b12x MXFP8 ~21-26 µs floor vs 3.6-8 µs BF16); rejected without a campaign.
- **Skinny-FP8 (weight-only e4m3 + per-row fp32 scale, M<=16): REJECTED at
  unit level, no lane arm spent.** Kernel built and unit-tested in-image
  (1 GPU, cold-rotation CUDA-graph bench, `fp8skinny_test.py`). Accuracy is
  e4m3-typical (relerr ~2.6% vs unquantized-bf16 mm on every shape = weight
  quant noise; the 2% gate was too strict). Speed is decisive — at the decode
  microbatch M=16 the kernel is 3-7x SLOWER than BF16 skinny on every shape
  (its M=16 tiling falls off a cliff), and only at M=4 does it win on the two
  K=7168 shapes:

  | shape (N,K) | M=4 bf16/fp8sk µs | M=4 speedup | M=16 bf16/fp8sk µs | M=16 speedup |
  |---|---|---|---|---|
  | g_proj (3072,7168) | 31.7 / 17.7 | **1.79x** | 32.2 / 105.0 | 0.31x |
  | gfab (3136,7168) | 32.3 / 17.6 | **1.83x** | 32.6 / 98.7 | 0.33x |
  | latent_down (896,7168) | 5.2 / 7.4 | 0.70x | 5.1 / 35.7 | 0.14x |
  | latent_up (7168,896) | 5.4 / 9.5 | 0.57x | 5.1 / 37.2 | 0.14x |
  | router (224,7168) | 3.6 / 3.6 | 1.00x | 3.8 / 11.3 | 0.34x |

  Not adopted (C16/C256 decode runs at M=16 per rank). The M=4 win would only
  matter for a C1-C4-serving variant; not pursued.
