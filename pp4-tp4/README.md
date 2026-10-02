# Kimi-K3 QSRT-K2: PP4×TP4 two-host lane with Moet 2-bit routed experts

A second, measured (not production-qualified) serving topology for Kimi-K3-QSRT-K2: **16 × RTX PRO 6000
Blackwell across two 8-GPU hosts** — pipeline parallel 4 (2 stages per host)
with tensor parallel 4 and MLA decode-context parallel 4 inside each stage,
executed by Ray over a RoCE fabric. No speculative decoding, no MTP, no draft
model: this is the plain dense-decode lane, plus the **Moet 2-bit routed-expert
path** (`moe_w2_mm`) measured on top of it.

Headline numbers (legacy closed-loop decode bench, see caveats):

| config | image | C16 tok/s | C256 tok/s | KV tokens | quality |
|---|---|---|---|---|---|
| baseline (`int6.envstr`, MNS 256, Moet flag unset = same-image control) | `kimi-k3-up-ray:moet-mvp2` | 335.1 | 1268.3 | 13,080,081 | probe PASS, GSM8K-200 197/200 |
| **+ Moet 2-bit experts** | `kimi-k3-up-ray:moet-mvp2` | **372.2 (+11.1%)** | **1394.5 (+9.9%)** | 13,080,081 | probe PASS, GSM8K-200 196/200 |
| + Moet + PP rebalance 24,24,24,21 | `kimi-k3-up-ray:moet-mvp2` | 359.9 | 1341.1 | **15,258,588 (+16.7%)** | probe PASS |

All three rows were measured back to back on 2026-10-01 ~21–23Z under one
lane-lock hold, same base env, same launcher, C16×3 and C256×3 runs each
(medians above; raw files in `results/fu-off|fu-on|fu-rb/`). GSM8K scores were
recomputed from the raw `gsm8k.jsonl` files: 197/200 OFF, 196/200 ON — the ON
arm writes more tokens per answer (mean 292.8 vs 250.5 completion tokens,
+16.9%), which the closed-loop decode bench does not charge against tok/s.

## Topology

```
rtx-01 (head, 10.10.10.1)                rtx-02 (worker, 10.10.10.2)
GPU 0-3: PP stage 0, TP4/DCP4            GPU 0-3: PP stage 2, TP4/DCP4
GPU 4-7: PP stage 1, TP4/DCP4            GPU 4-7: PP stage 3, TP4/DCP4
Ray head :6379  <—— NCCL over bond0/RoCE ——>  Ray worker
```

- vLLM pipeline-parallel-size 4, tensor-parallel-size 4,
  decode-context-parallel-size 4, `--distributed-executor-backend ray`.
- PP stages 0–1 run in the head container `kimi-k3-qsrt-pp4` on rtx-01;
  stages 2–3 in `k3-qsrt-ray-worker` on rtx-02 (joined via Ray).
- `10.10.10.x` is the direct wire between the two hosts. It appears in
  `run/env/k3-qsrt-{head,worker}.env` (`VLLM_HOST_IP`, `NCCL_IB_HCA`), in the
  head wrapper / worker entry (Ray addresses), and in `run/scripts/k3-arm.sh`
  (worker ssh target). Treat it as configurable site addressing, not a
  constant.
- `K3_MNS` (`--max-num-seqs`) defaults to 256 in the head wrapper. The cap
  counts all in-flight PP micro-batches together, so MNS 16 meant only 4
  sequences per stage — raising it was the single largest throughput fix.

## Images (not published; build locally)

Both images exist only on the two serving hosts. Digests:

| tag | image ID | built |
|---|---|---|
| `kimi-k3-up-ray:local` | `sha256:c8c3c9f56a6a7482dd519797f856fa7dd445ed020f24905ae555f34d12c2a91a` | 2026-09-29 |
| `kimi-k3-up-ray:moet-mvp2` | `sha256:1475787718e29df1ab23962f6951023d2087d6e3f7230d7a00f95783ea2891d6` | 2026-10-01 |
| `kimi-k3-up-ray:moet-mvp` (Moet OFF control) | `sha256:d35a7e4a…` | 2026-10-01 |

1. `kimi-k3-up-ray:local` — `Dockerfile.k3-ray` (in this directory):
   `FROM voipmonitor/vllm@sha256:32ff80279164365b21cdb420d7a22101be704df42b66db2c17d64d5e0558f240`
   (the pinned r38 runtime; its own base digest is
   `sha256:06f67841d1ce06e4993b9e66e188759e9feac2d95bcc49364cc6395d9a0aa292`,
   version label `k3-prefill-sm120-fa4-20260901-r3`) plus
   `pip install ray[default]==2.58.0`. All model/serving bits are
   byte-identical to the pinned base. Image labels record the source
   revisions: vLLM `a3693719528811ac605dfa236321a53e5770b120`
   (github.com/voipmonitor/vllm), b12x `0d31d259c6f0e6733190a6b3af9881b874f10241`
   (github.com/voipmonitor/b12x).
2. `kimi-k3-up-ray:moet-mvp2` —
   `docker build -f configs/moet/build/Dockerfile.moet-mvp -t kimi-k3-up-ray:moet-mvp2 configs/moet/build/`
   on top of `:local`. Adds the patched `kquant_hybrid.py`, the
   `moe_w2_*` runtime modules, and 6 cubins at `/opt/moet/cubins`
   (payload: 720 KB, committed). The Moet weight packs
   (`/data/models/k3-moet-packs`, **714 GB**) are NOT shipped; regenerate with
   `configs/moet/build/convert_moet_packs.py` (driven by `conv-all.sh`,
   checkpoint ID `k3-mxfp4-moet-v1`). `moet-mvp` (the OFF control image)
   differs from `moet-mvp2` in exactly one file, `moe_w2_cubit.py` — the
   resident-fit PP fix; the v1 copy is in
   `configs/moet/build/moet-mvp-v1/`. All other Moet files and cubins are
   md5-identical between the two images. The op test is
   `configs/moet/build/moet_op_test.py`.

## Runtime layout

| path | what |
|---|---|
| `run/overlay-vllm-l3/` | 21 single-file vLLM overlays. `MOUNTS.txt` maps each file to its destination in `/opt/venv/lib/python3.12/site-packages`; launchers bind-mount each read-only and fail the boot if an overlay file lacks a MOUNTS line. |
| `run/overlay-b12x-l3/` | b12x kernel tree, bind-mounted over the image copy at `/opt/infernal-invocation/b12x/b12x`. |
| `run/scripts/` | `k3-arm.sh` (boot → ready → probe → closed-loop decode bench), the l3 head/worker launchers, the head wrapper (the actual `vllm serve` command), the worker entrypoint. |
| `run/env/` | common/head/worker container env files (RoCE/NCCL fabric settings, JIT caches, runtime tuning). |
| `run/overlays-mb/decode_bench.py` | closed-loop decode bench (see caveats). |
| `configs/` | saved configurations on top of the lane: `baseline`, `moet`, `nvfp4-kv`, `rebalance-kv`, plus `launch.sh` to boot any of them. |
| `bench/` | env strings and batch scripts for the MNS/capfix sweeps, `results.txt` (every measured number), `k3-nsys3.sh` (16-rank nsys capture). |
| `levers/` | tried-but-not-adopted levers, one `RESULT.md` each (docs only). |
| `nsys/` | nsys analysis scripts, per-stage JSON breakdowns, graphs. |
| `results/` | raw `fu-off`/`fu-on`/`fu-rb` arm output (decode.txt, probes, arm.log, head.log, GSM8K jsonl, `/metrics` snapshots) and the fu nsys per-stage JSON. |
| `MANIFEST.md5` | md5 of everything under `run/`. |
| `Dockerfile.k3-ray`, `IMAGE.txt` | base-image recipe and expected image ID. |

The launchers also bind-mount the source trees
`/data/models/k3-up-tree/{vllm,lmcache}` (at the pinned revisions above) onto
`/opt/infernal-invocation/...`, which `run/env/k3-qsrt-common.env` puts on
`PYTHONPATH`; those trees must exist on both hosts.

## Launch

On rtx-01, with the `run/` tree installed at `/data/models/k3-run` on both
hosts (see `restore.sh`, which backs up, installs, mirrors to rtx-02 over the
direct link, and verifies `MANIFEST.md5` on both):

```bash
# production baseline (image kimi-k3-up-ray:local)
bash configs/launch.sh baseline 16,256

# Moet 2-bit experts (image kimi-k3-up-ray:moet-mvp2)
bash configs/launch.sh moet 16,256

# Moet + PP rebalance (KV capacity)
MOET=1 bash configs/launch.sh rebalance-kv 16,256
```

`launch.sh` takes the lane lock (`/tmp/k3-lane.lock`), boots worker then head
via `run/scripts/k3-arm.sh` (`k3-arm-moet.sh` for Moet images, which forwards
`UP_IMAGE` to rtx-02), waits for readiness, runs a coherence probe, then the
decode bench at the requested concurrencies. Results land in
`~/k3-results/<tag>/` on rtx-01. Stop with
`docker rm -f kimi-k3-qsrt-pp4` on rtx-01 and
`ssh 10.10.10.2 docker rm -f k3-qsrt-ray-worker`.

GSM8K-200: `python3 configs/moet/scripts/k3_gsm8k.py 8001 200 out.jsonl 8`.

## Key numbers

How the baseline got here (tok/s, `bench/results.txt`; MNS = `--max-num-seqs`):

| config | C1 | C16 | C32 | C64 | C128 | C256 |
|---|---|---|---|---|---|---|
| int0-off, MNS16 | 28.6 | 294.4 | 218.7 | 184.9 | | |
| int6-best, MNS16 | 31.2 | 319.4 | 226.0 | 207.3 | | |
| + admit-cap fix, MNS16 | | 314.9 | 318.9 | 316.8 | | |
| + MNS256 | | | | | 860.1 | 1200.2 |

Saved configs on top of the lane (`configs/README.md` has the full table and
per-config notes):

| config | delta vs baseline | C16 | C256 | KV tokens |
|---|---|---|---|---|
| `baseline` | — | 326.3 | 1200.3 | 13,080,081 |
| `moet` | Moet 2-bit `moe_w2_mm` routed experts | 373.1 (+11.5% vs same-env OFF control 334.65) | see fu-on below | 13,080,081 |
| `nvfp4-kv` | `K3_KVDT=nvfp4_ds_mla` | 314.9 (older stack, no same-session control) | — | 13,080,081 (no gain) |
| `rebalance-kv` | `VLLM_PP_LAYER_PARTITION=24,24,24,21` | 322.3 (−1.2%) | 1197.8 (noise) | 15,258,588 (+16.7%) |

Moet follow-up arms (2026-10-01 21–23Z, one lock hold, medians of 3;
`results/fu-*/decode.txt`):

| arm | C16 | C256 | KV tokens | GSM8K-200 |
|---|---|---|---|---|
| fu-off (Moet flag unset, `moet-mvp2`) | 335.1 | 1268.3 | 13,080,081 | 197/200 |
| fu-on (Moet) | 372.2 (+11.1%) | 1394.5 (+9.9%) | 13,080,081 | 196/200 |
| fu-rb (Moet + 24,24,24,21) | 359.9 | 1341.1 | 15,258,588 (+16.7%) | not run |

nsys (16-rank captures, `results/nsys-json/`, analyzed with
`configs/moet/scripts/nsys_stage_cmp.py`): Moet takes one decode step from
**11.92 → 10.71 ms at C16** and **52.07 → 46.00 ms at C256** (stage 0; other
stages within noise). The win is entirely in the experts kernel: MoE experts
34.7 → 24.9 ms/step at C256, partially offset by +3.1 ms of dequant/glue
elementwise. Older captures: `nsys/k3-nsys-mns256.png` (per-stage breakdown at
MNS256/C256 — MoE experts 63% of the 54.4 ms step),
`nsys/k3-nsys-before-after.png` (MNS16 int0 vs int6-best).

## What made the difference

1. **`--max-num-seqs` 16 → 256.** The root misconfiguration: the cap counts
   all in-flight PP micro-batches, so 16 allowed only 4 sequences per stage.
   Decode is weight-bandwidth bound, so cost per token fell from 3.15 to
   0.85 ms/stage. Above 256 is blocked by the b12x MLA kernel's 1024-decode-row
   limit (`b12x_mla.py`).
2. **Balanced-PP admission-cap fix** (`run/overlay-vllm-l3/scheduler.py`). The
   cap used to count waiting requests that could not be admitted, collapsing
   micro-batches whenever the queue was non-empty (C32 → 2×8, C64 → 1×16). It
   now counts `min(waiting, max_num_seqs − running)`.
3. **int6-best kernel flags** (in `configs/baseline/int6.envstr`): fast top-k,
   skinny GEMM, prefill fuse, GC freeze, b12x DCP options. +8–11% at C1–C16.
4. **Moet 2-bit routed experts** (`VLLM_K3_MOET=1` + `configs/moet/moet.envstr`):
   +11.1% C16 / +9.9% C256 decode at matched GSM8K-200 accuracy (197/200 OFF
   vs 196/200 ON — a one-item difference), at ~5.5 GiB more
   GPU memory per device.
5. **rtx-02 host fabric repair** (ForceP2P modprobe config, IOMMU, ACS):
   +5.6% C256 / +2.8% C16 — a host fix, no code change (see
   `levers/allreduce/RESULT.md`).

Not adopted: nvfp4 `ds_mla` KV (boots and decodes correctly with the NoPE
guard in `k3_nvidia_mla.py`, but KV capacity is unchanged at 13,080,081 tokens
— the hybrid MLA+KDA page-size unification sets the token count — so it only
adds quantization noise).

## Tried-but-not-adopted levers

One line each; full write-ups in `levers/*/RESULT.md`.

| lever | outcome |
|---|---|
| `allreduce` | b12x two-shot TP all-reduce measured −3.0% C256 on the repaired fabric — rejected; the real +5.6% came from the rtx-02 host fabric fix. |
| `collectives` | DCP/TP collectives pack (q-replicate, early shared-expert AR, fused pair-AG) — investigation inconclusive, not adopted. |
| `fp8-dense` | MXFP8 for the remaining BF16 dense GEMMs: 1.8× at M=4 but 0.3× at M=16 — decode runs at M=16/rank, so not adopted. |
| `fusion` | tail-norm fusion +0.36% C16 / noise C256, below the 3% bar; root-caused an eager `rms_norm` fallback instead. |
| `l2-prefetch` | L2 weight prefetch: null result; persisting-L2 set-aside hurts on this hardware — not adopted. |
| `rebalance` | PP layer partition 24,24,24,21: adopted as a KV-capacity lever (+16.7% KV, ~−1% C16, C256-neutral), not a speed lever. |

## Caveats

- **The decode bench is the legacy closed-loop one**
  (`run/overlays-mb/decode_bench.py`): tiny prompts, `ignore_eos` streaming,
  counting SSE chunks with non-empty text inside the measurement window. It
  equals token rate only while the server streams one chunk per generated
  token. Cross-check on the fu arms via `vllm:iteration_tokens_total`
  (`results/fu-*/itok-*.txt`): mean tokens/scheduled-step during C256 is 65.0
  (fu-off) vs 64.8 (fu-on) — same scheduling, so the +9.9% is a real step-rate
  gain, not chunk inflation.
- **Cross-session comparisons.** Some table rows come from different sessions
  (marked per-row); compare only within a row's stated control. The Moet
  ON/OFF arms in `configs/moet/results/moet-on|moet-off` used two images
  (`moet-mvp2` vs `moet-mvp`) and two scripts; the fu arms used one image and
  one script and are the cleaner comparison.
- **Moet token inflation.** GSM8K completion tokens grow +16.9% with Moet on
  (292.8 vs 250.5 mean); a requests/s or latency SLA that charges for output
  tokens should re-measure end to end.
- **Spec-free export.** The shipped serve command has no
  `--speculative-config` and no draft is launched anywhere. The two inert
  remote-draft lines the live `k3-qsrt-common.env` carried
  (`VLLM_K3_DRAFT_REMOTE_ADDRESS`, `K3_PP_LAST_HOST`) were removed from this
  export, and the dropped `fu-topp*` arms were removed from
  `configs/moet/scripts/moet-followup.sh` (see "Snapshot vs live"). The head
  wrapper still has a stale DFlash comment; it is a comment only.

## Reproduce

1. Build both images on a host with the pinned base present
   (`Dockerfile.k3-ray`, then `configs/moet/build/Dockerfile.moet-mvp`).
2. Generate the Moet packs (714 GB) with
   `configs/moet/build/conv-all.sh` → `/data/models/k3-moet-packs` on both
   hosts (or on one and rsync over the direct link).
3. Install this `run/` tree at `/data/models/k3-run` on both hosts
   (`bash restore.sh` from a machine that can ssh to both; it verifies
   `MANIFEST.md5`). Put the `k3-up-tree` vLLM/LMCache sources at the pinned
   revisions on both hosts.
4. `bash configs/launch.sh moet 16,256 mytag` on rtx-01; results in
   `~/k3-results/mytag/`. Compare against `bash configs/launch.sh moet-off`
   for the same-env control.
5. Optional: `bash bench/k3-nsys3.sh` for a 16-rank nsys capture, then
   `nsys export -t sqlite` inside the image and
   `python3 configs/moet/scripts/nsys_stage_cmp.py base.json arm.json`.

## Snapshot vs live

This directory was assembled from a frozen snapshot of `/data/models/k3-run`
(2026-10-01) verified by md5 against both live hosts, plus the live files
that produced the Moet follow-up arms. 279 of 282 `run/` files are
md5-identical between snapshot and live on both hosts. The three that
differed — shipped here in their **live** version (the ones that ran the
fu-* arms) — are:

1. `run/scripts/k3-arm.sh` — live additionally passes `K3_L3V`/`K3_L3B12X`
   through to the worker's ssh invocation.
2. `run/scripts/k3-qsrt-launch-head-nospec-up2-l3.sh` — live parameterizes the
   b12x overlay path as `${K3_L3B12X:-$RUN/overlay-b12x-l3/b12x}`.
3. `run/scripts/k3-qsrt-launch-worker-up2-l3.sh` — same `K3_L3B12X`
   parameterization; shipped from rtx-02 (rtx-01's own copy of the worker
   launcher is the older hardcoded one and is never executed there).

One more live-vs-snapshot difference outside `run/`:
`configs/moet/scripts/moet-followup.sh` is shipped as the **live** version
that drove the fu-* arms, minus the three `arm fu-topp*` invocations (those
arms were dropped — incomplete probes — and their results are excluded). The
frozen snapshot had a post-run hardened variant (C256 run-count gate, GSM8K
row-count validation, INCOMPLETE marking) that never ran on hardware.

The live hosts carry additional legacy files that are intentionally **not**
shipped: older/spec-decode launcher variants (`k3-qsrt-launch-*-{mb,mp,nsys,dsmem,ima,imadsm,l2}.sh`,
draft launcher, `k3-up-launch.sh`, head wrappers other than `-mb`), legacy env
files (`k3-common.env`, `k3-head.env`, `k3-upstream.env`, `k3-worker.env`),
extra `overlays-mb/` modules from the older microbatch experiments
(`arg_utils.py`, `config_*.py`, `decode_bench_stable*`, `model_runner.py`,
`pp_utils.py`, `sched_scheduler.py`, `speculative.py`), and `*.bak*`/`*.orig*`
backups. Everything related to speculative decoding / MTP / DFlash drafts
(`configs/spec`, `overlay-vllm-dflash2`, the `fu-topp*` and `sp-*` result
arms) is excluded by design.

Deliberate deviations from the live bytes, all in the spec-free
cleanup: (1) the inert remote-draft lines
(`VLLM_K3_DRAFT_REMOTE_ADDRESS`, `K3_PP_LAST_HOST`, plus their comment) were
removed from `run/env/k3-qsrt-common.env`; (2) the three dropped
`arm fu-topp*` invocations were removed from
`configs/moet/scripts/moet-followup.sh` as noted above;
(3) `configs/moet/scripts/moet-nsys.sh` — the script that produced the
shipped `results/nsys-json/fuoff|fuon-*.json` captures — was rewritten to run
only the fuoff/fuon captures (the live copy also captured topp95, ran
decode-only fu-topp90/80 arms, and waited on the topp95 arm of the follow-up
batch); (4) `configs/moet/scripts/moet_fu_summary.py` had its `ARMS` list
restricted to the three shipped arms; (5) in `run/scripts/k3-arm.sh` and
`configs/moet/scripts/k3-arm-moet.sh` the two `$(case ...)` command
substitutions were rewritten as plain `case` statements assigning `W`/`H` —
identical behavior, but the old form does not parse under bash 3.2 (stock
macOS), which broke `bash -n` verification off the GPU hosts.
