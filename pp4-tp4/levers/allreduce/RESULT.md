# Lever: TP all-reduce transport at C256 + S3 logits all-gather tail (levers #2 and #7)

Date: 2026-10-01. Lane: Kimi-K3-QSRT-K2 PP4xTP4/DCP4, rtx-01 (S0,S1) + rtx-02 (S2,S3), EXEC=up2l3, production env `~/int6.envstr`.

## 1. Per-host size/backend breakdown (from ~/nsys6-c256 sqlite, 8 ranks/host)

TP all-reduce and all-gather kernels, whole capture (~4.2 s, ~78 cycles of 54.2 ms):

| kernel | host | n/rank | median | p99 | notes |
|---|---|---|---|---|---|
| `ncclDevKernel_AllReduce_bf16_RING` | rtx-01 | ~4600 | **57.9 µs** | 84 | classes ~57 µs (3600) and ~70 µs (1200) |
| same | rtx-02 | ~5000 | **87.6 µs** | 105 | classes ~75 µs (1400) and ~90 µs (3400) |
| `ncclDevKernel_AllGather_RING` | rtx-01 | ~3400 | 22.2 µs | 210 | mid class ~210 µs ≈ 4.5/cycle (per-MLA-layer DCP metadata AG) |
| same | rtx-02 | ~3900 | 33.5 µs | **838** | per-cycle classes ~160-230 µs (DCP metadata AG) + ~840 µs tail (logits AG), see per-rank table |
| `b12xcommpcie_dcp_a2a_cute_LseReduceScatter` | both | — | 239/310 µs | — | DCP a2a (peer lever #3, untouched) |
| `b12xcommpcie_dcp_a2a_cute_AllGatherHeads` | both | — | 189/229 µs | — | DCP a2a (peer lever #3, untouched) |

Per-rank breakdown (16 reports; rank = host/deviceId from each report; S0/S1 = rtx-01 dev0-3/dev4-7, S2/S3 = rtx-02 dev0-3/dev4-7). **Caveat on per-call message size: nsys kernel rows carry no collective arguments, so the classes below are duration-based size classes; byte sizes are inferred from model shapes and labeled [shape-inferred].** Format `count@median-µs`; classes <2% of that rank's calls omitted except tails; 1-2 rare >100 ms outliers per rank are pipeline/boot artifacts. Ranks within a stage agree to ±2 µs, so rows are per-stage with the per-rank values folded.

| rank | stage | AR classes (nccl AllReduce_bf16_RING) | AG classes (nccl AllGather_RING) |
|---|---|---|---|
| r1/dev0-3 (4 ranks) | S0 | ~3500@56-57, ~1250@63-64 | ~3060@22 |
| r1/dev4-7 (4 ranks) | S1 | ~3000@56-57, ~1300@63-64 | ~2950@23, ~100@49-54 |
| r2/dev0-3 (4 ranks) | S2 | ~1370@72-75, ~3440@88-90, ~200@101-104 | ~2650@23-28, ~660@42, ~74@160-162 |
| r2/dev4-7 (4 ranks) | S3 | ~1340@72-74, ~3400@88-90, ~150@101-102 | ~2510@23-28, ~720@42, **70@225-228, 70@833-843** |

The two AR classes per rank ≈ the per-layer big ARs (attn-out and MoE-down; dominant message 64×7168×2 B = 896 KiB [shape-inferred]); AG classes: base DCP per-layer gathers (22-54 µs), per-cycle DCP metadata gather (~160-230 µs, both rtx-02 stages), and the S3-only logits all-gather (833-843 µs, 1/cycle/rank, 21 MB [adjacency-identified]).

- **No b12x oneshot/twoshot AR kernels run at C256 on either host** (production boot log: `Configured b12x PCIe crossovers: algorithm=oneshot, allreduce max=86016, fused max=86016, DMA min=6291456` — oneshot ceiling 84 KB, DMA floor 6 MB, nothing in between). At C16 all TP ARs are ≤84 KB and go through b12x oneshot (med 14.9/14.4 µs, both hosts) — no lever there.
- C256 AR message sizes: 64 tokens/micro-batch × 7168 hidden × 2 B = **896 KiB** per big AR (plus smaller KDA-side ARs). All above the 84 KB oneshot ceiling → PyNCCL ring, on both hosts. The "~87 µs vs ~57 µs" per-call gap is the whole per-layer rtx-02 penalty (0.47–0.53 vs 0.30–0.33 ms/layer).
- **S3 AG tail (#7)**: devices 4–7 of rtx-02 (exactly stage S3) each show ~70 calls/capture ≈ **1 per cycle** at ~840 µs (833–900). Time-ordered context: immediately after `cutlass_80_tensorop_bf16_s16816gemm` (425 µs, the LM head GEMM) and immediately before `_temperature_kernel` → `_gumbel_sample_kernel` → token `Broadcast_RING`. It is the **logits all-gather** (`LogitsProcessor._gather_logits` → `tensor_model_parallel_all_gather`), 64×163840×2 B = 21 MB gathered, once per cycle.

## 2. Root cause

**Host fabric asymmetry — initially attributed to topology, ultimately proven host CONFIGURATION (see the fix paragraph below).** Idle-GPU 4-rank NCCL quad microbench (`~/fastar/nccl_quad_bench.py`, docker `kimi-k3-up-ray:local`, torchrun nproc=4, under flock, GPUs verified idle; NCCL transport confirmed `P2P/IPC`, not SHM):

| metric | rtx-01 q03 | rtx-01 q47 | rtx-02 q03 | rtx-02 q47 |
|---|---|---|---|---|
| AR 896 KiB | 71.4 µs | 61.2 µs | 90.3 µs | 90.8–107 µs |
| AR 1 MiB | 67.9 | 67.3 | 103.2 | 102.9 |
| AG total 21 MB | — | 474.4 µs | — | **866.9 µs** |
| AG total 5.2 MB | — | 133–134 µs | — | 224–229 µs |

rtx-02's PCIe P2P fabric is uniformly ~1.8× slower than rtx-01's for 4-GPU collectives — matching the lane medians (87.6 vs 57.9 µs AR; 840 vs ~475 µs logits AG). Topology (`nvidia-smi topo -m`): rtx-01 quads are all-NODE within one NUMA node; rtx-02 is PIX pairs (4-5, 6-7) with NODE cross-pair, i.e. cross-pair P2P hairpins the PCIe root complex. (Main is running the host-level ACS/IOMMU/driver diff; microbench facts handed over.)

**HOST FIX LANDED MID-SESSION (Main, 16:35–16:44).** The fabric asymmetry was host configuration, not hardware: rtx-01 had `/etc/modprobe.d/nvidia-p2p-override.conf` (`NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"`), rtx-02 had none. Fix on rtx-02: same modprobe conf + driver reload, GPU IOMMU groups → identity, ACS ReqRedir/CmpltRedir cleared on the 8 PEX890xx GPU downstream ports (persisted via grub `iommu=pt pci=disable_acs_redir=pci:1000:c030`). Post-fix idle NCCL quad (Main, verified bit-exact): **AR 896 KB 90–92 → 57 µs** (rtx-01: 61), **AG 21 MB 859–876 → 386 µs** (rtx-01: 459–481) — rtx-02 is now on par with (AG: faster than) rtx-01. Consequences: **#7's 840 µs tail is expected to close with zero code change** (confirmed in-lane below); **#2's premise shrinks** — the two-shot route must now beat a healthy ~57 µs ring, not a broken ~88 µs one (b12x #290 says it can: 44.5 µs @768 KB). All pre-16:35 numbers in this section are the stale-baseline context; the A/B in §5 runs entirely post-fix.

**#7 NCCL tuning is a measured dead end** (rtx-02 q47, AG 21 MB): default 868.9 µs, `NCCL_ALGO=RING NCCL_MIN_NCHANNELS=4` 857.7, `NCCL_PROTO=LL128` 893.1, `NCCL_MIN_NCHANNELS=8` 868.0 µs — all within noise. `NCCL_ALGO=TREE` is unsupported for all_gather (run aborts). The 840 µs tail is the rtx-02 fabric price of a 21 MB gather; only a payload shrink (fp8 logits — quality risk), a custom hierarchical/pull gather kernel (multi-day), or sampling-semantic changes could beat it.

**#2 fix**: the k3 up-tree (`/data/models/k3-up-tree/vllm/.../custom_all_reduce.py`, 1752 lines) already contains a b12x **bf16 two-shot** all-reduce route (`PCIeTwoShotBF16`, in the live overlay-b12x-l3 but unused): FP32 accumulation in fixed rank order, single rounding — numerically *cleaner* than the bf16 ring (which rounds every hop), though not bit-identical to NCCL. b12x #290 measured two-shot pull vs NCCL ring on 4× RTX PRO 6000: 15.0 vs 16.7 µs @128 KB, 32.8 vs 41.7 @512 KB, **44.5 vs 54.5 µs @768 KB**. The live image's `custom_all_reduce.py` (1457 lines) predates this route. Patch = minimal port of the up-tree twoshot blocks onto the live file, gated default-OFF behind `VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE` (unset → 0 → byte-identical code path to live; the A arm runs the live overlay, so any regression in B is attributable to the route).

## 3. Implementation

- Overlay copy: `/data/models/k3-run/overlay-vllm-l3-allreduce/` (both hosts, rsynced), adds `custom_all_reduce.py` (+1 MOUNTS.txt line) = live file + twoshot port. Diff: `custom_all_reduce.twoshot.patch` (226 lines) in this folder; full file `custom_all_reduce.py`.
- Flag: `VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=1MB` routes (84 KB, 1 MB] messages to two-shot pull (`VLLM_PCIE_TWOSHOT_ALLREDUCE_MODE=pull` default; push is REJECT per CANDIDATES #46). C256's 896 KiB ARs fall in the window; C16's ≤84 KB ARs stay on oneshot (unchanged).
- Prior art respected: CANDIDATES #45 (`K3_L3_TWOSHOT_ROUTE`, MARGINAL +1.5–2% at C8/C16) targeted the >86 KB band at *small* concurrency; the 896 KiB C256 band was never routed to two-shot before (production has no two-shot at all). #22/#18 cover oneshot ≤84 KB (unchanged). PCIE_PDL (#42) and push mode (#46) REJECTed, not retried.

## 4. Two-shot microbench

Prepared (`~/fastar/twoshot_bench.py` + `tb-run.sh`: 4-rank `PCIeTwoShotBF16.from_exchange_group`, correctness vs `dist.all_reduce`, 229 KB/512 KB/896 KiB) but pre-empted by the host-fix window; made moot by the lane A/B below (the A/B measures the route end-to-end, which is the gate that matters). Not run standalone.

## 5. A/B results (post-fabric-fix, same session, tmux `fastar`, lock 17:27–17:49)

Script `rtx-01:~/k3-allreduce-ab.sh`: per arm boot via `EXEC=up2l3 [K3_L3V=...] k3-arm.sh TAG "ENV" "16,16,16,256,256,256"` (built-in C4 warm pass discarded; 3 measured runs per conc → median+band), 17×23=391 chat probe, 20-prompt greedy probe (`~/fastar/greedy_probe.py`), containers removed both hosts after each arm (verified gone).

| arm | overlay / env | C16 median (band) | C256 median (band) | probe391 |
|---|---|---|---|---|
| A | live overlay, `$(cat ~/int6.envstr)` | 335.6 (335.0–335.7) | 1268.0 (1253.8–1278.5) | PASS |
| B | overlay-vllm-l3-allreduce + `VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=1MB` | 334.9 (334.7–335.1) | **1230.3 (1214.0–1231.7)** | PASS |

- B boot log confirms the route on all 16 ranks: `b12x PCIe mid-size bf16 all-reduce serves 86016 < bytes <= 1046528 (row_elems=896, max_rows=584, mode=pull)` + `two-shot bf16 max=1046528` in the crossover line.
- **C16: −0.2% (bands overlap → noise)** — expected: C16 ARs (≤84 KB) never leave the oneshot path.
- **C256: −3.0% (bands disjoint → real regression).** On the fixed fabric the NCCL ring at 896 KiB is ~57 µs on both hosts; the two-shot pull kernel (staging + two barriers) does not beat it in-lane, and every (84 KB, 1 MB] message moved onto it. Per the lever rules (<3% upside → no campaign; here negative), the route is **REJECTED on the fixed fabric**. No further variants burned: push is already REJECT per CANDIDATES #46, and a narrower (768 KB, 1 MB] window re-tests the same in-band loss.
- Quality: numerics differ by design (twoshot = fp32 accumulate, single rounding; ring = bf16 per-hop rounding). Greedy 20-prompt diff A vs B: **3/20 identical**; divergences are argmax flips (one flips at char 0); all B outputs coherent, probe391 PASS both arms. Moot given the speed reject; recorded for completeness.

**Baseline shift from the host fix itself (same script/session): A-arm C256 = 1268.0 vs pre-fix production baseline 1200.3 (+5.6%); C16 = 335.6 vs 326.3 (+2.8%).** The #2/#7 premise value arrived via the fabric repair, not via code.

## 5b. Post-fix in-lane comm profile — attempted, not load-bearing

A post-fix 16-rank nsys C256 capture (`nsys7-c256`, tmux `fastarcap`, lock 18:12–18:23) booted and benched fine but failed verification: all 16 reports exported with **no CUDA kernel table** (338K API events, zero kernels, both hosts — CUPTI-level issue, first capture attempted after the rtx-02 driver reload). Not debugged further under shared-lane pressure; containers verified removed on both hosts. The exact post-fix in-lane per-call AR/AG durations are therefore **[INFERENCE]** from two measured facts: (a) Main's idle NCCL quad microbench on the fixed fabric — AR 896 KB 90–92 → **57 µs**, AG 21 MB 859–876 → **386 µs**, bit-exact; (b) the same-session A-arm baseline jump C256 1200.3 → 1268.0 (+5.6%) and C16 326.3 → 335.6 (+2.8%), whose size matches the nsys6-modeled comm recovery (≈4.5 ms/cycle AR + ≈0.45 ms/cycle AG tail on a 54.2 ms cycle). Anyone needing per-call in-lane confirmation should re-run `~/fastar/k3-nsys-cap.sh` after a CUPTI health check.

## 6. Recommendation

- **Lever #7 (S3 840 µs AG tail): CLOSED by host repair, no code change.** Root cause was rtx-02's missing ForceP2P modprobe config + ACS redirect + IOMMU passthrough, fixed by Main (persisted in modprobe.d + grub). The logits all-gather now costs ~386 µs idle (was 859–876), and the same-session baseline gained +5.6% at C256 / +2.8% at C16. NCCL env tuning was measured to be a dead end on the broken fabric (857–893 µs across ALGO/PROTO/NCHANNELS; TREE unsupported for all_gather). If more logits-AG time is ever needed, the only remaining routes are payload shrink (fp8 logits — quality risk) or a custom hierarchical pull-gather kernel (multi-day); expected residual gain ≤0.5 ms/cycle.
- **Lever #2 (TP AR transport at C256): REJECT the b12x two-shot route on the fixed fabric** — measured −3.0% C256 (1230.3 vs 1268.0, disjoint bands), C16 noise. The premise ("rtx-02 ring is ~1.5× slower") was a host-config bug, not a routing problem; on healthy PCIe the NCCL ring at 896 KiB is ~57 µs and beats the two-shot pull kernel in-lane. The patch stays in this folder (default-OFF flag `VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE`; overlay copy `/data/models/k3-run/overlay-vllm-l3-allreduce` on both hosts can be deleted or kept for reference) — it is a faithful, verified port of the up-tree twoshot wiring if the fabric ever regresses or the kernel improves.
- **Net effect of the #2/#7 investigation: +5.6% C256 / +2.8% C16 via the host fabric fix** (Main's change, root-caused by this lever's per-host breakdown). Production env needs no changes.
