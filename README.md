# Kimi-K3 Production Serving Stack (9× RTX PRO 6000 Blackwell)

Production Docker Compose configuration for serving **Kimi-K3-QSRT-K2** — a 2bpw
QSRT-quantized build of Kimi K3 (57B) — on a single 9-GPU Blackwell host with a
1,048,576-token (1M) context window, remote DFlash speculative decoding, and a
two-tier (L1 SHM + L2 disk) LMCache prefix cache.

This repository is the sanitized export of the live production directory. All
credentials are placeholders (`CHANGE_ME`); internal host paths are kept as
shipped (`/home/g0san`, `/mnt/models`, `/mnt/lmcache-l2`) and must be adapted to
your host.

## Architecture

- **Target engine**: vLLM fork (`local-inference-lab/vllm`,
  `dev/infernal-invocation` branch) with b12x GPU kernels
  (`local-inference-lab/b12x`). TP8 + DCP8 across 8 RTX PRO 6000 Blackwell
  Max-Q GPUs.
- **Draft engine**: remote DFlash draft (fixed K=3) on a 9th GPU (RTX PRO 6000
  Workstation Edition), connected over ZMQ (`tcp://127.0.0.1:8092` shipped).
- **Quantization**: QSRT MoE expert layers with W4A16 for both prefill and
  decode; F32/BF16/F8_E4M3 non-expert layers. Checkpoint:
  [`lukealonso/Kimi-K3-QSRT-K2`](https://huggingface.co/lukealonso/Kimi-K3-QSRT-K2)
  (~700 GB, 106 safetensors shards). QSRT codec research:
  [`pedapudi/qsrt`](https://github.com/pedapudi/qsrt).
- **Prefix cache**: LMCache multi-process connector in `engine_driven` mode.
  L1 is an 80 GiB shared-memory pool; L2 is a disk-backed store on a RAID0 XFS
  volume. Chunk size 1536 tokens. L1 eviction (watermark 0.90) flushes keys to
  L2 synchronously.
- **KV capacity**: ~2.13 GiB per rank → ~2.18M-token global DCP capacity
  (supports the full 1M-token context window with headroom).

## Container images

| Role | Image | Notes |
|------|-------|-------|
| Target (TP8/DCP8) | `ghcr.io/myshytf/kimi-k3:production-r29-dflash-20260904` | ~48 GB uncompressed |
| Draft (DFlash) | `ghcr.io/myshytf/kimi-k3:draft-bf16-cudagraph` | BF16 draft, CUDA-graph mode |

The images contain the vLLM/b12x/LMCache stack and JIT toolchains; model
weights are **not** baked in and must be mounted (see compose files).

## Repository layout

- `compose/` — top-level compose profiles (TP8/TP9, DFlash/DSpark variants,
  rollback profiles). Historical variants are kept for reference.
- `candidates/all-compatible-20260829/runtime/` — the **base runtime** of the
  live profile (`compose.tp8-dflash-flashkda-320k.yml` plus serve scripts).
- `candidates/*/compose.*.override.yml` — layered override files applied on top
  of the base. The live stack combines the base with the overrides listed in
  the running container's `com.docker.compose.project.config_files` label
  (36 overrides as of the 2026-09-04 export).
- `bin/` — entrypoint wrappers and runtime patch scripts applied before the
  packaged launcher runs (LMCache heartbeat registration, prefetch policy,
  namespace handling, etc.).
- `overlays/` — source patches bind-mounted into the container over the
  installed vLLM/b12x/LMCache trees. Compiled artifacts (`*.so` > 10 MB) and
  nested `.git` directories were removed from this export; rebuild them from
  the corresponding sources or request them separately.
- `deploy-bundle/`, `docker-r36-qsrt-b12x-compat/`, `Dockerfile.sched1536-pr227` —
  image build contexts.
- `docs/ACTIVE-PRODUCTION.md` — snapshot of the active profile and its
  qualification evidence as of 2026-08-30.

## Quick start (outline)

1. Pull the images above (`docker pull ghcr.io/...`).
2. Download the checkpoint: `hf download lukealonso/Kimi-K3-QSRT-K2
   --local-dir /mnt/models/Kimi-K3-QSRT-K2` (gated — provide `HF_TOKEN`).
3. Fetch the DFlash draft weights into the path referenced as `/draft-model`.
4. Provision an 80 GiB+ SHM segment (`/dev/shm` or tmpfs) and an L2 disk
   directory; point the compose `volumes` at them.
5. Set `VLLM_API_KEY` and any other `CHANGE_ME` placeholders.
6. Launch with the base compose plus the desired overrides, e.g.:
   ```bash
   docker compose -p kimi-k3-production \
     -f candidates/all-compatible-20260829/runtime/compose.tp8-dflash-flashkda-320k.yml \
     -f candidates/<topic>/compose.<topic>.override.yml ... \
     up -d
   ```

## Qualified behavior (evidence snapshot)

Latest qualification on the live stack (decode: 2026-09-03; prefill A/B:
2026-08-29; sustained, `openai_continuous_usage`-sourced aggregates):

- Decode: 62.9 tok/s at concurrency 1 (ITL 15.5 ms), 136.9 tok/s at
  concurrency 4, 130.9 tok/s at concurrency 8; ctx0 single-stream ITL series
  peaked at 75.2 tok/s.
- 32K cold prefill: ~888 tok/s.
- First-token logprob fingerprints verified identical across restarts
  (mid2k/long12k/long24k).
- LMCache L2 at export time: ~2.2 TB across ~145K objects with live
  store/lookup traffic and L1→L2 eviction flush verified in server logs.

The earlier 320K-context profile snapshot with its qualification runs is kept
in `docs/ACTIVE-PRODUCTION.md` (2026-08-30) for reference.

## Limitations

- Several patches are SM120 (Blackwell) specific; other architectures need
  source-matched re-qualification.
- The PCIe topology assumptions (Gen4 PLX switches behind Gen5-capable GPUs)
  require host-side link-speed pinning; see host notes in `bin/` and compose
  comments. Not container-managed.
- Host paths, NIC names, and IP addresses are shipped verbatim from the
  production host; adapt before use.
- This export excludes logs, benchmark raw data, and compiled kernel binaries.

## Licenses and attribution

vLLM, LMCache, and derivative patches retain their upstream licenses
(Apache-2.0 where applicable). QSRT checkpoint and codec research are
third-party works — see the linked repositories for their terms.
