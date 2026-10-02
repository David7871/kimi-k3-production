# Lever: PP stage layer rebalance (VLLM_PP_LAYER_PARTITION)

Date: 2026-10-01. Lane: Kimi-K3-QSRT-K2 PP4xTP4/DCP4, rtx-01 (S0,S1) + rtx-02 (S2,S3), EXEC=up2l3, production env `~/int6.envstr` (K3_MNS=256, prefix caching ON, b12x PCIe allreduce, DCP b12x opts).

## 1. How the partition is set today

- No `VLLM_PP_LAYER_PARTITION` anywhere in `/data/models/k3-run/{scripts,env}` or `~/int6.envstr`; the head wrapper (`k3-qsrt-head-wrapper-mb.sh`) passes only `--pipeline-parallel-size 4`.
- vLLM fork default (`vllm/distributed/utils.py:get_pp_indices`): 93 layers / PP4 → base 23, remainder 1 goes to the **middle** stage → **[23,23,24,23]** (S2 gets the extra layer).
- Passthrough for an override exists: `k3-arm.sh TAG "ENV_KV ..."` builds `-e KEY=VAL` into `EXTRA_ENV_ARGS`, consumed by `docker run` in `k3-up-launch.sh` / `k3-qsrt-launch-worker-up2-l3.sh` on both hosts. Values must be space-free (comma list is fine).

## 2. Layer composition (config.json, 1-based layer ids)

93 layers = 69 KDA (linear attn) + 24 MLA (full attn at 4,8,…,92 **and 93**). Per-stage MLA/KDA:

| split | S0 | S1 | S2 | S3 (has LM head+sampling) |
|---|---|---|---|---|
| 23,23,24,23 (live) | 5/18 | 6/17 | 6/18 | **7/16** |
| 24,24,24,21 (B) | 6/18 | 6/18 | 6/18 | **6/15** |
| 23,24,24,22 (C) | 5/18 | 6/18 | 6/18 | 7/15 |

## 3. Per-stage time model from nsys (existing captures ~/nsys6-c16, ~/nsys6-c256; roof.py, exclusive ms/cycle)

C16 (cycle 12.540 ms, all stages):

| stage | busy | pp-sendrecv wait | productive | layers |
|---|---|---|---|---|
| S0 | 11.148 | 0.008 | 11.148 | 23 |
| S1 | 12.131 | 0.758 | 11.373 | 23 |
| S2 | 12.109 | 0.198 | 11.911 | 24 |
| S3 | 12.090 | 0.008 | 12.090 (incl LM head) | 23 |

C256 (cycle 54.219 ms):

| stage | busy | pp wait | productive | tp-comm | MoE |
|---|---|---|---|---|---|
| S0 | 48.456 | 0.027 | 48.456 | 6.99 | 34.4 |
| S1 | 53.522 | 2.934 | 50.588 | 7.61 | 36.1 |
| S2 | 53.498 | 0.590 | 52.908 | 11.21 | 34.3 |
| S3 | 53.518 | 0.013 | 53.518 (incl LM head) | 12.19 | 33.7 |

Derived per-layer cost (productive busy / layers):
- C16: rtx-01 ≈ 0.485-0.494 ms/layer; rtx-02 ≈ 0.496 ms/layer (comm is latency-bound, nearly equal). S3 fixed (LM head + sampling + token bcast) ≈ **0.68 ms**.
- C256: rtx-01 ≈ 2.15 ms/layer; rtx-02 ≈ 2.20 ms/layer (tp-comm per layer 0.30-0.33 ms on rtx-01 vs 0.47-0.53 on rtx-02 — the known PIX-pair PCIe penalty). S3 fixed ≈ **2.8 ms**.

Predicted new max-stage (cycle ≈ max productive + ~0.5-0.7 ms pipeline overhead):

| split | C16 pred max | C16 Δcycle | C256 pred max | C256 Δcycle |
|---|---|---|---|---|
| A 23,23,24,23 | 12.09 (S3) | 0 | 53.5 (S3) | 0 |
| B 24,24,24,21 | 11.91 (S2) | −0.2 ms (−1.5%) | 52.9 (S2) | −0.6 ms (−1.1%) |
| C 23,24,24,22 | 11.91 (S2) | −0.2 ms | 52.9 (S2) | −0.6 ms |

At C256 the win is small because MoE (~34 ms) dominates and the live imbalance is only ~5 ms spread over 4 stages; S2's 24 layers become the wall after S3 is lightened. At C16 the model says ~1.5%. **The decisive argument for B is KV capacity, not time** (below).

## 4. KV capacity interaction (the clamp)

Prior CPU-verified finding (loops/2026-09-25T1115/FINDINGS.md, reproduces live boot log "Add 3 padding layers, may waste at most 4.35% KV cache memory"): vLLM hybrid KV grouping makes per-stage slot groups of KDA(23)+MLA(24); live stage 3 has **7** MLA layers → 7-slot group → `num_blocks = min over workers` clamps **all 16 GPUs to 6/7** of the blocks their 16 GiB `--kv-cache-memory-bytes` could hold (−14.3%). Exhaustive search (18–30 layers/stage) found **[24,24,24,21] is the only 6-slot-everywhere partition** → live 13,080,081 tokens → ≈15.26M (**+16.7%**). The byte-balanced 23,24,24,22 keeps 7 slots on S3 (no KV gain). Fork knob `VLLM_K3_KV_GROUP_SIZE` unset; left alone.
Boot-safety: B lowers S3 peak GB/GPU (2 fewer layers incl. 1 MLA); S0/S1/S2 +1 layer each ≈ +0.24 GiB weights against 30-35 GiB free/GPU — fits (prior "fit: ok" gate). No pure-KDA stage, so #51764/#54793 class issues don't apply.

## 5. Research citations

- vLLM `get_pp_indices` remainder-to-middle default + `VLLM_PP_LAYER_PARTITION` override: vllm-project/vllm `vllm/distributed/utils.py`, `vllm/envs.py`.
- vLLM PR #46217 "[PP] Place remainder PP layers on the last stage(s)" (open) — argues end-loading helps prefill; notes memory trade-off on head stage; explicit partition env overrides everything. https://github.com/vllm-project/vllm/pull/46217
- SGLang Kimi-K3 production config pins `SGLANG_PP_LAYER_PARTITION=48,45` (PP2, **fewer layers on the LM-head stage**) — same direction as this lever. sgl-project/sglang `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`.
- SGLang PP docs tip: put larger partition on higher PP rank when layers don't divide (bubble reduction) — generic advice, but their own K3 recipe overrides it toward a lighter last stage. `docs/docs/advanced_features/pipeline_parallelism.mdx`.
- Hybrid-KV alignment caveats if repartitioning: vllm-project/vllm#51764 (KDA-only stage block alignment), #54793 (empty KV-group projection) — open, not triggered by B (every stage keeps ≥6 MLA layers).
- Local library: solution-map C3 conflict (03-axes.md:215), local-techniques.md:255 (24,24,24,21 CPU-VERIFIED +16.7% KV), local-techniques.md:357 (byte-balanced 23,24,24,22, ~+4% modeled C8/C16), loops/2026-09-25T1115/FINDINGS.md.

## 6. A/B protocol

Lock `flock /tmp/k3-lane.lock` on rtx-01 (tmux `rebalab`), script `~/k3-rebalance-ab.sh`: per variant boot via `EXEC=up2l3 k3-arm.sh TAG "ENV" "16,16,16,256,256,256"` (built-in C4 warm pass = discarded bench; 3 measured runs per conc → median+band), 17×23=391 chat correctness probe, KV-capacity lines captured from head log, then `docker rm -f` both hosts. A = no partition env (default 23,23,24,23); B = `VLLM_PP_LAYER_PARTITION=24,24,24,21`; optional C = 23,24,24,22 if lock time allows.

## 7. Results (measured 2026-10-01 15:00–15:41Z; A/B in tmux `rebalab` lock 14:59:47–15:16:43Z, C in tmux `rebalc` lock ~15:24–15:40Z)

decode_bench (streaming, ignore_eos, 30 s window after 10 s warm-in; 3 repeats per conc after the built-in C4 warm pass). tok/s aggregate.

| variant | split | C16 median (band) | C256 median (band) | KV tokens | probe 17×23 |
|---|---|---|---|---|---|
| A | 23,23,24,23 (live default) | 326.3 (326.2–327.4) | 1200.3 (1199.1–1221.6) | 13,080,081 | PASS |
| B | 24,24,24,21 | 322.3 (321.7–322.8) | 1197.8 (1188.6–1216.6) | **15,258,588 (+16.67%)** | PASS |
| C | 23,24,24,22 | 322.5 (321.5–322.5) | 1204.3 (1200.6–1216.8) | 13,080,081 (unchanged) | PASS |

Deltas vs A: **C16 −1.2% for both B and C** (bands disjoint from A → real small regression; B and C statistically identical), **C256 noise** (all three bands overlap: −0.2% B, +0.3% C). ms/token-equivalent at C16: 49.03 → 49.6 ms; at C256: 213.3 → ~213 ms (all). C keeping KV at 13,080,081 confirms on hardware that 23,24,24,22 stays 7-slot on S3 (min-clamp intact) — exactly as loops/2026-09-25T1115 predicted; only 24,24,24,21 lifts it. B ≈ C in speed at both concurrencies confirms the model's "S2 (24 layers) becomes the wall once S3 is lightened" prediction.

KV: B boots at 15,258,588 tokens = exactly 13,080,081 × 7/6 — the 7-slot clamp on S3 is gone, confirming the loops/2026-09-25T1115 CPU prediction on hardware. B's S3 also drops its 7th MLA layer; max concurrency per 262k request rises to 58.21x.

**Time-neutral (slightly negative), capacity-positive.** The nsys per-stage model explains why: at C16/C256 the live stage-busy spread is small (48.5–53.5 ms at C256; 11.1–12.1 ms at C16), so moving 2 layers off S3 only shifts the wall to S2 (24 layers) — predicted best-case −1.1%/−1.5%, measured −0.2%/−1.2%. With production per-token cadence scaling with the *sum* of stage times (no step overlap in this PP configuration: C16 Σ≈46.5 ms vs 49.0 ms/token; C256 Σ≈205.5 ms vs 213.3 ms/token), redistributing the same 93 layers across the same two hosts cannot reduce the total work — it can only relocate it. The measured ~0.6 ms/token regression at C16 is consistent with S0/S1 absorbing an extra MLA-heavy layer while S2 keeps 24 layers.

## 8. Recommendation

- **Adopt `VLLM_PP_LAYER_PARTITION=24,24,24,21` only if KV capacity matters** (+16.7% KV tokens, removes the 7-slot min-clamp, lowers S3 peak memory). Cost: ~1% C16 decode, nothing at C256. For long-context / prefix-cache-bound serving this is a clear win; for pure decode throughput it is a wash.
- Do **not** adopt any rebalance as a decode-speed lever: the lane is not max-stage-bound in a way layer counts can fix — MoE (~34 ms/53.5 ms at C256) dominates every stage equally. Decode-cycle savings must come from per-layer cost (MoE/GEMM/comm kernels) or step overlap (microbatching), not partition.
- Resolves library conflict C3 with hardware numbers: 24,24,24,21 = +16.7% KV at ~−1% C16 decode (C256 neutral); 23,24,24,22 = no KV change, same ~−1% C16. The "byte-balanced +4% C8/C16" model (local-techniques.md:357) does not reproduce on hardware. Pick 24,24,24,21 if KV-bound, else keep the default.

### Exact env used

A: `$(cat ~/int6.envstr)` unchanged. B: `VLLM_PP_LAYER_PARTITION=24,24,24,21` prepended to the same string. Both via `EXEC=up2l3 bash /data/models/k3-run/scripts/k3-arm.sh TAG "ENV" "16,16,16,256,256,256"` inside `flock /tmp/k3-lane.lock` (tmux `rebalab` on rtx-01; C: tmux `rebalc`). Script: `rtx-01:~/k3-rebalance-ab.sh` (copy: pp4-experiments/scratch/k3-rebalance-ab.sh). Containers `kimi-k3-qsrt-pp4` (rtx-01) and `k3-qsrt-ray-worker` (rtx-02) verified removed after each variant and at script end.
