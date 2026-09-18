# Inspect the exact deployed runtime

Status: **qualified** for runtime-file identity and CPU launch checks.
The image packages the inspected runtime's base, both launchers, Python KDA
overlay, and native KDA extension. It is suitable for extracting the files and
comparing them with the full Git repositories; no model download or GPU is
needed for inspection.

## Pull and verify

```bash
IMAGE=ghcr.io/myshytf/kimi-k3@sha256:856320d49eb35e2a0652348b644813bac42ff23e2b7a9af193390c388e115e5d
docker pull "$IMAGE"
docker run --rm --network none --entrypoint /opt/venv/bin/python \
  "$IMAGE" /recipe/verify-image.py
```

Expected result: 3,846 checked files, no failures, no unexpected runtime files.
[image-lock.json](image-lock.json) pins the published manifest;
[source-lock.json](source-lock.json) pins sources, model revisions, and binaries.

The original production base is also available separately:

```bash
docker pull ghcr.io/myshytf/kimi-k3@sha256:9a0a5818b664e263ca68aba772f76ae2b2948e97e42d6bbfd0c3859507e76b30
```

That base alone lacks the active bind-mounted KDA overlay and current launch
scripts. The complete snapshot above includes them.

## Extract without starting a model

Run in a directory with room for the source trees and compiled extensions:

```bash
mkdir -p runtime-extract
snapshot_container=$(docker create --entrypoint /bin/true "$IMAGE")
docker cp "$snapshot_container:/opt/infernal-invocation" runtime-extract/
docker cp "$snapshot_container:/opt/k3kda" runtime-extract/
docker cp "$snapshot_container:/opt/k3ext" runtime-extract/
docker cp "$snapshot_container:/runtime" runtime-extract/
docker cp "$snapshot_container:/recipe" runtime-extract/
docker rm "$snapshot_container"
```

Other prebuilt dependencies, including the Python environment and custom NCCL,
remain in the image and can be copied in the same way. Clone the actual Git
repositories using [GIT-AND-PRS.md](GIT-AND-PRS.md) to inspect commit history;
image source directories are installed runtime trees, not Git clones.

## What was observed and what was packaged?

The original container used the base image with these additional runtime mounts:

| Container path | Exported file |
| --- | --- |
| `/runtime/serve_tp9.sh` | [runtime/serve_tp9.sh](runtime/serve_tp9.sh) |
| `/runtime/lmcache-mp-wrapper.sh` | [runtime/lmcache-mp-wrapper.sh](runtime/lmcache-mp-wrapper.sh) |
| `/opt/infernal-invocation/vllm/vllm/models/kimi_k3/nvidia/kda.py` | [overlays/kda.py](overlays/kda.py) |
| `/opt/k3kda/_C_k3kda.abi3.so` | [overlays/_C_k3kda.abi3.so](overlays/_C_k3kda.abi3.so) |

The target and draft checkpoints are identified by repository, revision,
metadata hashes, and payload sizes. Their weights are not included. The observed
container's `/cache` and `/lmcache/l2` data are not included; its schedule file
and marker state are recorded in [cache-controls.json](observed/cache-controls.json).
The actual environment is [environment.json](observed/environment.json).

The snapshot image retains the original `serve_tp9.sh` entrypoint. The optional
Compose recipe selects an added `entrypoint.sh` that checks model metadata and
initializes the schedule before invoking that unchanged launcher. Other added
helpers perform validation and SHM preflight. These additions do not replace
any inference source or native kernel.

This export does not preserve a live process, GPU memory, driver-injected files,
request logs, JIT cache payloads, or KV data. The configured 1M context ceiling
has not been newly qualified, and the packaged image has not been booted as a
replacement for production. See [validation.json](validation.json).
