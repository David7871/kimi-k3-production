# Kimi-K3 production deployment bundle

The image `ghcr.io/myshytf/kimi-k3:all-compatible-20260901-r6-deploy`
contains the Kimi-K3 TP8/DCP8 target runtime, LMCache integration, DFlash
remote-draft integration, and deployment assets under `/opt/kimi-k3-deploy`.
Model weights and persistent cache data are host-managed and are not included.

## Runtime behavior

- The OpenAI-compatible API serves `claude-mythos-5` on host port 8090.
- Eight RTX PRO 6000 Blackwell GPUs run the target model. A ninth GPU runs the
  DFlash draft model through ports 8091 and 8092.
- The target accepts contexts up to 1,048,576 tokens. Scheduling uses
  1,536-token target chunks, four sequence slots, and three speculative tokens.
- LMCache uses an 80 GiB shared-memory L1 arena and a host-mounted L2 directory.
  The L1 arena is recreated on process start because its key and allocation
  index is process-local. L2 objects persist across restarts.
- An async KV load performs full-input admission once. After the transfer
  completes, scheduling resumes under the normal per-step capacity check.
- The source checkout can load the matching wheel-provided `lmcache.c_ops`
  extension, so block transfer uses the native pointer interface.

## Included assets

- `compose/`: the three-file Compose merge for target, draft, LMCache, model
  paths, and the `claude-mythos-5` served name.
- `bin/`: entrypoints, capacity guards, and runtime patch validators.
- `overlays/`: B12X and vLLM source overlays used by the Compose mounts.
- `autotune/`: PCIe IPC all-reduce configuration for an eight-GPU target.
- `runtime/`: runtime specification and the target launch script.

## Host requirements

- Nine compatible NVIDIA GPUs: eight target ranks and one draft rank.
- `/dev/shm` capacity of at least 84 GiB. The supplied profile uses 96 GiB to
  provide an 80 GiB L1 arena plus a 4 GiB safety reserve.
- Target weights at `/mnt/models/Kimi-K3-QSRT-K2`.
- Draft weights at `/mnt/models/Kimi-K3-DFlash-stock`.
- A writable LMCache L2 path. The supplied profile uses
  `/mnt/lmcache-l2/kimik3/l2-mythos5-dflash-stock-mamba1536-chunk1536-v1`.

## Extract and launch

```bash
docker pull ghcr.io/myshytf/kimi-k3:all-compatible-20260901-r6-deploy
docker pull ghcr.io/myshytf/kimi-k3:draft-bf16-cudagraph

bundle_container=$(docker create \
  ghcr.io/myshytf/kimi-k3:all-compatible-20260901-r6-deploy)
docker cp "$bundle_container:/opt/kimi-k3-deploy" ./kimi-k3-deploy
docker rm "$bundle_container"

export DEPLOY_ROOT=$PWD/kimi-k3-deploy
export MODELS_ROOT=/mnt/models
export L2_ROOT=/mnt/lmcache-l2

docker compose \
  -f "$DEPLOY_ROOT/compose/compose.tp8-dflash-flashkda-320k-w4a8.yml" \
  -f "$DEPLOY_ROOT/compose/compose.mxfp8-entry.override.yml" \
  -f "$DEPLOY_ROOT/compose/compose.mythos5-dflash-stock-fine1536.override.yml" \
  up -d
```

Change `SERVED_MODEL_NAME`, `MODELS_ROOT`, or `L2_ROOT` only when the receiving
environment requires different identifiers or mount paths. Cache geometry
changes require a distinct L1 name and L2 directory.
