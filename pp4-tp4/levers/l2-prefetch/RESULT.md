# Lever: L2 weight prefetch during AR/small-kernel windows (`VLLM_KIMI_L2_PREFETCH`)

Date: 2026-10-01. Lane: Kimi-K3-QSRT-K2 PP4xTP4 (rtx-01 S0/S1 + rtx-02 S2/S3, DCP on), EXEC=up2l3,
overlay `/data/models/k3-run/overlay-vllm-l3`. All runs under `flock /tmp/k3-lane.lock`, containers
removed on both hosts after each arm and at exit.

## Code audit (overlay-vllm-l3)

- `l2_prefetch.py` (`KimiDecodePrefetch`): builds bounded prefetch plans per PP-stage layer slice.
  Two hook sites per layer: (1) `RowParallelLinear` o_proj pre-all-reduce (`linear.py:2337`),
  budget 24 MiB, segments = this layer's MoE gate / routed-expert down-proj / paired decode weight /
  shared-expert weights; (2) MoE late all-reduce pre-reduce (`moe_runner.py:653`), budget 48 MiB,
  segments = next layer's attention input projections (in_proj_qkvgfab / fused_qkv_a / q_a / kv_a …).
  Hook body: `if rows <= 8: prefetcher.issue(plan, rows)` — **hard gate at ≤8 rows per step**.
- `l2_cache_hints.py` (shared with GLM-5.3): CuTe DSL kernel issuing one
  `cp.async.bulk.prefetch.L2.global.L2::cache_hint` per 4 KiB chunk (evict_last policy) on a forked
  side stream; `join()` rejoins at model forward end. Valid in FULL cudagraph capture; skipped in
  piecewise capture (this lane runs full-graph decode). Numerics untouched (cache hints only).
- Env surface in the up2l3 overlay: **only `VLLM_KIMI_L2_PREFETCH`** (`envs.py:60,676`, default 0).
  `KimiDecodePrefetch` hardcodes `persisting_l2_request="0"` → `configure_persisting_l2(device, "0")`
  → **no persisting-L2 set-aside** (confirmed in B boot log: "requested 0 MB, now 0 MB"; RTX PRO 6000
  L2 134 MB, max persisting 84 MB). The `VLLM_GLM53_L2_PREFETCH_PERSIST_MB` env exists in
  `l2_cache_hints.py` but is only consulted when request is `None`; the Kimi path never reads it.
  **There is no live persist-MB knob for the Kimi path.**
- Invocation confirmed: `prepare_decode_prefetch` called from `model.py:2597`
  (`process_weights_after_loading`, i.e. after serving-layout weight load, before graph capture).
  Arm-B boot log: "Kimi dense-weight L2 prefetch: 44/47 bounded plans, at most 8 rows" per rank and
  "CuTe kernel ready (grid=16 block=128 chunk=4096; budgets A/B/C/A_mla=20/50/15/36 MB)" ×16 ranks.
- **Old env names**: `VLLM_K3_L2_PREFETCH=1` + `VLLM_K3_L2_PREFETCH_PERSIST_MB=32` appear only in
  `env/k3-upstream.env:92-93` (not loaded by the up2l3 launch path, which uses
  `k3-qsrt-common.env` + `k3-qsrt-head/worker.env` — none contain any L2 var). `envs.py` has no
  `VLLM_K3_L2_PREFETCH*` entry → both old names are dead in this tree.
  **Recommendation: rename `VLLM_K3_L2_PREFETCH` → `VLLM_KIMI_L2_PREFETCH` in k3-upstream.env if that
  file is ever paired with the l3 overlay, and delete `VLLM_K3_L2_PREFETCH_PERSIST_MB` (no consumer;
  persist set-aside is hardcoded 0 — deliberately, see GLM note below).**

## Prior work (local library + GitHub)

- solution-map/research/local-techniques.md row "l2-prefetch": same implementation lineage
  (g0san overlay `l2-prefetch-20260903`), EXTERNAL-REPORTED: KDA in_proj 31.1→21.4 µs,
  o_proj 9.5→7.0, fused_qkv_a 23.9→20.2, ~−2 ms of a 48 ms 82k-ctx step [g0san].
- kernel-library/prs/vllm-kk-843/MANIFEST.md: PR vllm-kk-843 (CLOSED, superseded/composed into the
  production image) is the source of this overlay; "optional L2 hints" half of the "+7.72% paired
  decode" gain cited by rtx6kpro qsrt-tp9-dspark-qualification (the other half is shared-expert
  overlap, already live). 184 prepared plans, ≤8 rows, no persisting-L2 reservation.
- kernel-library/local-inference-lab__rtx6kpro/models/kimi-k3/qsrt-tp9-dspark.md: TP9 profile runs
  with `VLLM_KIMI_L2_PREFETCH=1`; controlled resident A/B attributes the earlier +7.72% to
  shared-expert/L2 combined.
- GLM-5.3 sibling (l2_cache_hints.py docstring, measured on same hardware): C1 85.1→91.1 verifier
  steps/s (+7%); a nonzero persisting-L2 set-aside *hurt* prefill (32k: 14,439→13,963 tok/s) and C12
  verifier (350.6→339.4 steps/s) while helping C1 (86.6→88.5) — hence default persist=0, which this
  overlay hardcodes.
- GitHub: `cp.async.bulk.prefetch.L2` used in NVIDIA/TensorRT-LLM `cpp/kernels/xqa/tma.h`,
  flashinfer `csrc/xqa/tma.h` and `sparse_mla_sm120/arch/cp_async.cuh` (SM120, with cache_hint),
  flashinfer `docs/mxfp4_situ_moe.rst` (bulk L2 prefetch + PDL `griddepcontrol` dependency
  prefetch on B300 decode rows, −2.1..−2.6 µs GPU span per row). No vllm-project/vllm upstream
  equivalent; no sglang decode-weight L2 prefetch found (only HiCache host-cache prefetch).
  CUTLASS SM90 TMA prefetch: NVIDIA/cutlass `include/cute/arch/copy_sm90_tma.hpp`.

## Measurement

Harness: boot both hosts via k3-qsrt-launch-{worker,head}-up2-l3.sh with EXTRA_ENV_ARGS,
readiness poll, correctness probe `17*23` (temperature 0, assert 391), warm pass + one discarded
bench, then median-of-3 measured (decode_bench.py, win=30 s warm=10 s, ignore_eos streaming).
Baseline env: `~/int6.envstr` (K3_MNS=256, PCIE allreduce b12x, DCP b12x gather/combine, fast-topk,
skinny-GEMM, prefill-fuse, …). Arm B adds only `VLLM_KIMI_L2_PREFETCH=1`.
Logs: rtx-01 `~/k3-results/k3l2-ab/` (round 1), `~/k3-results/k3l2-ab2/` (round 2).

### Round 1: C16 (+C256 check) — the ≤8-row gate means the lever is INERT here

| arm | C16 tok/s (3 runs) | median | C256 tok/s (1 run) |
|---|---|---|---|
| A (off) | 318.2 / 318.0 / 318.3 | 318.2 (band 318.0–318.3) | 1192.1 |
| B (on)  | 319.4 / 319.5 / 320.2 | 319.5 (band 319.4–320.2)  | 1208.1 |

Correctness: both arms probe PASS (`391`); no illegal-memory/EngineDead/Xid in either head log.

**Gate analysis**: the hook fires only when the per-step decode batch is ≤8 rows. This lane has
`--max-num-seqs 256` and no PP micro-batch splitting (no pp-max-micro-batch-size in wrapper/env),
so at C16 every decode step presents 16 rows → hook body skips; at C256 likewise. The +0.4% C16 and
1.3% C256 deltas are boot-to-boot variance, not the lever (C256 is a single sample per arm).
Translated to S3 cycle terms: ≤ ~0.05 ms/cycle at C16 (baseline 12.54 ms), ≤ ~0.7 ms at C256
(baseline 54.2 ms) — both within noise.

### Round 2: C8 (+C4 check) — where the lever actually engages

| arm | C8 tok/s (3 runs) | median | C4 tok/s (1 run) |
|---|---|---|---|
| A (off) | 207.6 / 207.4 / 206.9 | 207.4 (band 206.9–207.6) | 123.5 |
| B (on)  | 206.9 / 206.1 / 207.3 | 206.9 (band 206.1–207.3) | 123.0 |

Correctness: both arms probe PASS (`391`); no crash signatures. Boot log confirms the lever armed
(44/47 bounded plans per rank, CuTe kernel ready ×16 ranks); at C8/C4 the ≤8-row gate passes, so the
prefetch kernels did fire — yet the medians are statistically identical (−0.24% at C8, bands fully
overlapping; −0.4% at C4 single-sample). Per-request decode latency: 38.6 vs 38.8 ms/tok at C8
(25.9 vs 25.8 tok/s/req). In S3-cycle terms the effect is ≤ ~0.1 ms/cycle at any concurrency — null.

## Conclusion / recommendation

**No measurable effect on this lane — keep `VLLM_KIMI_L2_PREFETCH=0` (the current default).**
At C16/C256 the lever is structurally inert (≤8-row hook gate vs 16/256-row decode steps, no
micro-batch splitting in this config); at C8/C4 where it does fire, it neither helps nor hurts
(≤0.3% deltas, overlapping bands). The external +7% class gains (g0san TP9, GLM-5.3 C1 verifier)
were measured at C1 with long 82k-context attention windows and a different stage geometry; on this
PP4xTP4 lane at short context the AR/small-kernel idle windows at C8 are apparently already covered
(or too short for the 20/50/15/36 MB fills to beat GDDR7 streaming). No persist-MB variant was tried
because the Kimi path hardcodes the set-aside to 0 and the GLM sibling measured that nonzero
persisting-L2 *hurts* prefill and C12 decode on this hardware.

**Env rename verdict**: yes — in `env/k3-upstream.env:92-93`, rename `VLLM_K3_L2_PREFETCH` →
`VLLM_KIMI_L2_PREFETCH` (the only name the overlay reads) or delete the pair outright given the null
result; delete `VLLM_K3_L2_PREFETCH_PERSIST_MB=32` unconditionally — no consumer exists in the tree
(persist request is hardcoded `"0"` in `KimiDecodePrefetch`).

Exact config: baseline env = rtx-01 `~/int6.envstr`; arm B = baseline + `-e VLLM_KIMI_L2_PREFETCH=1`
on both worker (rtx-02) and head (rtx-01) launch scripts (`k3-qsrt-launch-{worker,head}-up2-l3.sh`).
Scripts: `~/k3-l2-ab.sh` (C16/C256), `~/k3-l2-ab2.sh` (C8/C4) on rtx-01, run under
`flock /tmp/k3-lane.lock`; all containers removed from both hosts at end (verified).
