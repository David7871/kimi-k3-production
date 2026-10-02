# Lever: DCP/TP collectives pack (q-replicate, early shared-expert AR, fused pair-AG)

Date: 2026-10-01. Lane: Kimi-K3-QSRT-K2 PP4xTP4/DCP4, rtx-01 (S0,S1) + rtx-02 (S2,S3), EXEC=up2l3, production env `~/int6.envstr`. **Baseline re-measured same-session** after the 16:44 rtx-02 fabric fix (P2P override + IOMMU identity + ACS clear) — all pre-16:35 numbers are stale for stage-3/4 collectives.

## 1. What is in this pack

Three independent, default-OFF levers against the decode-cycle collective inventory (~/nsys6-c256 sqlite, 64-token microbatch; roof-c16-s3.txt / roof-c256-s3.txt). Per MoE layer at C256: 3 all-reduces (AR1 latent 459 KB ~43 us, AR2 shared-expert 918 KB ~59 us of which ~48 us exposed, AR3 o_proj 918 KB ~58 us) + 2 all-gathers (AG1 routed down-proj 344 KB ~14 us, AG2 router logits fp32 172 KB ~22 us). Per MLA layer DCP adds AllGatherHeads (q gather 24->96 heads, ~195-228 us at C256 / ~19.5 us at C16) and LseReduceScatter (~244-309 us / ~25.6 us); x7 MLA layers per stage.

| arm | lever | env flag | mechanism | est. |
|---|---|---|---|---|
| B | DCP q-replicate | `VLLM_DCP_Q_REPLICATE=1 VLLM_K3_DCP_Q_REPLICATE_LAYERS=all` | replicate q heads across DCP ranks in decode -> kills AllGatherHeads entirely (b12x_mla.py:919 skips gather when qrep_decode; b12x absorb bmm auto-fallback mla_attention.py:1915). +~0.35 GB/GPU weights. | +2.6% C256, +1.1% C16 |
| C | early shared-expert AR | `VLLM_K3_SHARED_AR_EARLY=1` | issue the shared-expert AR on the aux stream immediately after the shared expert, overlapping the ~1.5 ms routed-expert kernel run; tier-1 tail skips its own AR. Bit-exact (same reduction, same order). | +2% C256, +1.8% C16 |
| D | fused pair-AG >8 rows | `VLLM_K3_PAIR_AG_MAX_TOKENS=256` | extend the b12x pair-fused all-gather path (today capped at <=8 tokens, C16 only) to 256 rows via NCCL byte-packed single AG (dcp_alltoall._fused_pair_all_gather_fallback, bit-exact); model.py branch for >8 tokens. | +0.5% C256 |

Rejected during design (kernel-library/CANDIDATES.md rules): fp8 DCP payloads (~1.5% C256, quality risk), concat AR merge (reassociation), RS+AG restructure (no byte win).

C16 note: at 4-row microbatches the ARs already go b12x (~11-15 us) and the AGs already take the pair path (<=8 tokens), so arms C/D are expected to be C256 levers; q-replicate helps both.

## 2. Implementation (all default-OFF, one overlay copy)

Overlay `/data/models/k3-run/overlay-vllm-l3-collectives` (cp -a of live `overlay-vllm-l3`, never edited), rsynced byte-identical to 10.10.10.2 same path (md5 spot-verified). MOUNTS.txt additions:

- `latent_moe_runner.py -> vllm/models/kimi_k3/nvidia/latent_moe_runner.py` (hook install + `_early_reduce` + tail skip; flag parse)
- `shared_experts.py -> vllm/model_executor/layers/fused_moe/runner/shared_experts.py` (early-AR hook in `_run_in_aux_stream`)
- `tp_projection.py -> vllm/models/kimi_k3/nvidia/tp_projection.py` (pair-AG >8 fused path)
- `model.py -> vllm/models/kimi_k3/nvidia/model.py` (>8 branch + import)

Patch copies in this folder (`patches/`). In-image import gate passed with defaults: pair_ag=8, shared_ar_early=False (inert). `python -m py_compile` + CPU import OK for all four files.

## 3. A/B protocol

`flock /tmp/k3-lane.lock` on rtx-01 (tmux `dcpc-ab`), script `~/k3-collectives-ab.sh` (copy of the rebalance A/B): per arm `EXEC=up2l3 K3_L3V=/data/models/k3-run/overlay-vllm-l3-collectives k3-arm.sh TAG "ENV" "16,16,16,256,256,256"` (built-in C4 warm pass discarded; 3 measured runs per conc -> median+band), 17x23=391 chat probe, greedy-output capture (20 fixed prompts, T=0, 48 tokens, /v1/completions) per arm for bit-exactness diff vs arm A, KV-capacity + flag lines from head log, then `docker rm -f kimi-k3-qsrt-pp4` (rtx-01) + `k3-qsrt-ray-worker` (rtx-02) after each arm.

Arms (cumulative flags, one overlay):
- **dc-a** = `$(cat ~/int6.envstr)` (same-session post-fabric-fix baseline)
- **dc-b** = A + `VLLM_DCP_Q_REPLICATE=1 VLLM_K3_DCP_Q_REPLICATE_LAYERS=all`
- **dc-c** = B + `VLLM_K3_SHARED_AR_EARLY=1`
- **dc-d** = C + `VLLM_K3_PAIR_AG_MAX_TOKENS=256`

First boot on the fixed rtx-02 fabric doubles as validation of the new state (P2P override + iommu=pt + ACS clear).

## 4. Results

TBD — awaiting lane (queued behind fp8dense-d / fastarcap).

| arm | C16 median (band) | C256 median (band) | KV tokens | 391 | greedy 20/20 |
|---|---|---|---|---|---|
| dc-a | | | | | ref |
| dc-b | | | | | |
| dc-c | | | | | |
| dc-d | | | | | |

## 5. Recommendation

TBD.

### Exact env used

TBD (arm env strings + script copy).
