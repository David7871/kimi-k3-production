# Kimi-K3 QSRT-K2: Docker and serving recipe

This recipe reconstructs the runtime inspected on 2026-09-18: nine RTX PRO
6000 Blackwell GPUs, TP9 / DCP9, DFlash2 on those same nine ranks, and LMCache
with host-memory and disk tiers. It serves the published QSRT-K2 checkpoint.
There is no offline quantization or calibration step in this serving recipe.

Status: **implemented**. See [validation.json](validation.json) for build,
file-identity, and CPU launch checks. The running deployment has prior serving
qualification; the newly packaged image has not replaced it or received a
separate full-model serving run.

## 1. Hardware and host requirements

The reference host has:

| Resource | Observed / configured value |
| --- | --- |
| GPU | 8 × RTX PRO 6000 Blackwell Max-Q + 1 × Workstation Edition; nominal 96 GB each |
| GPU architecture | SM120; standalone KDA extension compiled for `sm_120a` |
| CPU | Ryzen 9 7950X, 16 cores / 32 threads |
| Host memory | 128 GB installed; about 124 GiB visible, with swap provisioned |
| NVIDIA driver | `595.99.02` observed on the host |
| Container toolchain | CUDA 13.3.1; PyTorch package `2.13.0`; Triton `3.7.1+gitf797708c.nv26.7` |
| Shared memory | Host IPC; `/dev/shm` sized to 96 GiB |
| LMCache L1 | 48 GiB pool, plus at least 4 GiB free reserve at startup |
| LMCache L2 | Configured ceiling 2457.6 GiB; a dedicated disk directory |
| Model storage | 757,524,534,968 target safetensors bytes + 7,385,765,464 draft bytes, plus metadata |

Use Linux x86-64, Docker Engine with Compose v2 / BuildKit, NVIDIA Container
Toolkit, Python 3 on the host, and the Hugging Face `hf` CLI. Budget model
storage separately from Docker layers, build cache, compilation cache, and L2
cache. The configured L2 ceiling does not reserve disk space.

The image's `CUDA_DRIVER_VERSION` environment value is inherited container
metadata; it is not the host's loaded driver. The table records the actual
`nvidia-smi` result, not a minimum-driver compatibility claim.

Verify GPU visibility and topology before configuring this recipe:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi topo -m
df -h /dev/shm
```

The PCIe rank layout is part of this deployment. The default physical GPU
order is `1,2,3,4,5,6,7,8,0`. The two collective groups are **TP ranks**
`0,1,2,3,8` and `4,5,6,7`, corresponding to the reference host's two switches.
Match `K3_GPU_ORDER` and `K3_PCIE_GROUPS` to your hardware. This TP9 profile is
**unsupported** as a drop-in recipe for 8 GPUs, DGX Spark, another architecture,
or a topology without the required peer access. Those need their own configuration
and qualification.

Host CPU affinity is disabled by default in `.env.example` for portability.
Set `K3_CPU_PIN=1` to reproduce the reference 7950X layout after reviewing the
core mapping in [runtime/serve_tp9.sh](runtime/serve_tp9.sh).

## 2. Download the exact checkpoints

From the repository root:

```bash
cd production
cp .env.example .env
```

The following examples use the paths in `.env.example`. Create directories
that the Docker daemon can access, then use the same absolute paths in `.env`:

```bash
sudo install -d -m 0755 -o "$(id -u)" -g "$(id -g)" \
  /srv/models/Kimi-K3-QSRT-K2 /srv/models/Kimi-K3-DFlash2 \
  /srv/kimi-k3/cache /srv/kimi-k3/lmcache-l2

hf download lukealonso/Kimi-K3-QSRT-K2 \
  --revision 3b98114115f1d41ce7963ba346c3fca19918b0bd \
  --local-dir /srv/models/Kimi-K3-QSRT-K2

hf download lightseekorg/kimi-k3-dflash2 \
  --revision e77935fb4804e17eb55085bffd045eae1d779769 \
  --local-dir /srv/models/Kimi-K3-DFlash2
```

If Hugging Face requires access authorization, complete it for those models
and authenticate with `hf auth login` on the host. The serving container does
not need the download credential. Keep the whole target directory, including
its tokenizer, processor, config, index, and QSRT metadata files.

Model weights are external read-only bind mounts at `/model` and
`/draft-model`. The build context excludes model directories and caches.
Startup checks config/manifest hashes, shard presence, counts, and byte sizes;
it does not rehash all weight tensors. Both upstream `config.json` files at the pinned revisions
were compared with the inspected production files and matched exactly.

## 3. Pull the exact runtime snapshot

```bash
./run.sh pull
./run.sh check
```

The published image is `ghcr.io/myshytf/kimi-k3:runtime-20260918`.
[image-lock.json](image-lock.json) pins its immutable digest. To pull that exact
manifest directly:

```bash
docker pull ghcr.io/myshytf/kimi-k3@sha256:856320d49eb35e2a0652348b644813bac42ff23e2b7a9af193390c388e115e5d
```

`./run.sh pull` also follows that immutable digest and assigns the Compose
image name locally; it does not resolve a potentially moved registry tag.

The image uses the deployed base
`ghcr.io/myshytf/kimi-k3@sha256:9a0a5818b664e263ca68aba772f76ae2b2948e97e42d6bbfd0c3859507e76b30`
and includes the exact active KDA Python overlay, compiled extension, and
original launch scripts copied from the running container. It verifies 3,846
runtime files against their deployed hashes. The native extension is included
as the actual served binary; no recompilation is needed.

To reconstruct the snapshot from that public base and the supplied files:

```bash
./run.sh build
./run.sh check
```

Neither build nor check requires a GPU. The Dockerfile retains the base's
prebuilt vLLM, FlashKDA, FlashInfer, LMCache, custom NCCL, and other dependencies.
It does not claim to rebuild the whole toolchain from upstream source. Full
Git histories, source commits, native source provenance, and PR mappings are
in [GIT-AND-PRS.md](GIT-AND-PRS.md).

`./run.sh check` validates the image and actual launch arguments on CPU,
including TP9/DCP9, three speculative tokens, precision settings, and the
4,624-token budget. It does not load a model or start a cache server.
Do not replace the base with a stock vLLM image or upgrade packages in place:
package versions alone do not identify this runtime.

## 4. Configure and start

Edit `.env` for model/cache paths, GPU topology, CPU affinity, cache capacity,
and any occupied ports. Use a new writable `CACHE_DIR` and L2 directory for
this recipe. Default API/LMCache/LMCache HTTP ports are 8090 / 5555 / 8088.

With `ipc: host`, Docker's container `shm_size` does not enlarge the host's
`/dev/shm`. Provision the host tmpfs before startup. The preflight needs at
least **52 GiB free** for the default pool and reserve. It checks mappings in
the host PID namespace and only cleans an unmapped pool with this recipe's
exact `LMCACHE_SHM_NAME`.

```bash
./run.sh up
./run.sh logs
```

The first startup can take several minutes for loading, JIT compilation, and
CUDA graph capture. The entrypoint creates `/cache/ubatch-mode` containing:

```text
overlap moesplit=1 offset=2
```

This file is required for the production prefill schedule. A different existing
mode or a cache-control disable marker causes startup to stop, so an old cache
directory cannot silently change the profile.

The engine listens on **127.0.0.1:8090** with an OpenAI-compatible API and
served model name **Kimi-K3**. Production's authenticated llmconduit gateway
is a separate service and is not required to use the local engine. Preserve
loopback binding and put your own authenticated gateway in front if needed.

## 5. Verify generation

Health and model enumeration:

```bash
curl --fail http://127.0.0.1:8090/health
curl --fail http://127.0.0.1:8090/v1/models
```

Then verify that streaming completes, including a terminal `data: [DONE]`:

```bash
curl --fail --no-buffer --max-time 300 \
  http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Kimi-K3","messages":[{"role":"user","content":"Reply with a short greeting."}],"temperature":0,"max_tokens":64,"stream":true}'
```

A health response alone does not establish working generation. Repeat a
representative long prompt to check the prefix-cache path and inspect logs:

```bash
docker compose -p kimi-k3-qsrt exec kimi tail -n 80 /tmp/lmcache-mp-8090.log
curl --fail http://127.0.0.1:8090/metrics
```

Changing `PORT` or `LMCACHE_HTTP_PORT` in `.env` requires matching your diagnostic
URLs. The default LMCache log filename remains `/tmp/lmcache-mp-8090.log`.
The new namespace intentionally starts cold; old KV entries from another
precision/runtime profile are not evidence of a valid cache hit here.

## 6. Exact serving settings

The complete environment is in [compose.yaml](compose.yaml). The verbatim
production launch script is [runtime/serve_tp9.sh](runtime/serve_tp9.sh), and
[runtime/lmcache-mp-wrapper.sh](runtime/lmcache-mp-wrapper.sh) adds the actual
LMCache connector configuration and owns the cache-server process.

| Setting | Value |
| --- | --- |
| Target format | QSRT K2; `qsrt_sqg_e4m3`, `sqg_xor_cheb_t12`, `qsrt_atoms_v2` |
| Expert payload | 2 trellis bits/weight; all 82,432 routed experts QSRT; no X4T experts |
| Dense projections | Existing MXFP8 hybrid W8A16; model dtype BF16 |
| Target KV / draft KV | `fp8_ds_mla` / `fp8` |
| Parallelism | `--tensor-parallel-size 9 --decode-context-parallel-size 9` |
| Backends | `--moe-backend b12x --linear-backend b12x --attention-backend B12X_MLA --kda-prefill-backend flashkda` |
| Loading | `--load-format instanttensor` target; safetensors draft |
| Speculation | DFlash2, 3 proposals, TP9, probabilistic draft sampling, block rejection |
| Context ceiling | 1,048,576 tokens |
| GPU KV allocation | 3,435,973,836 bytes per rank |
| Scheduler | 4 requests; both batched/scheduled token limits 4,624 |
| Alignment | Mamba block 4,608; prefix match unit 1,536; LMCache chunk 4,608 |
| Graphs | `FULL_AND_PIECEWISE`; capture sizes 1, 2, 4, 8, 16 |
| Vision | Enabled; same target checkpoint includes the vision tower |
| LMCache transfer | `engine_driven`, separate object groups, recompute on load failure |
| LMCache L1 policy | LRU; watermark 0.80; eviction ratio 0.15; backup flush every second |
| LMCache restore | Server pins 4 chunks; worker restore window 4 chunks |
| Decode KDA | Standalone fused extension enabled through `VLLM_K3_KDA_SPEC_FUSED_LIB` |

K2 describes the expert codec, not a uniform two-bit format for every model
tensor. The target's nested `quant_method: modelopt` / `quant_algo: NVFP4`
metadata is interpreted by this fork together with its QSRT metadata; it is
not an instruction to convert the model to a different quantizer.

The 4,624-token scheduler budget is deliberate: one aligned 4,608-token prefill
must coexist with three decode requests, each potentially scheduling four
tokens (one plus three proposals). A 4,608-token budget can starve that prefill.
The wrapper passes the restore-window value in `kv_connector_extra_config`,
so spawned workers receive it; an environment variable alone is insufficient.

For an expanded target command without loading weights:

```bash
docker compose -p kimi-k3-qsrt run --rm --no-deps kimi --print-command
```

The LMCache wrapper appends `--kv-transfer-config` at actual startup. Its
connector and server flags are recorded in the wrapper, not hidden in another
repository or generated by a missing local script.

## 7. Restart and stop

```bash
./run.sh restart
./run.sh down
```

`restart` stops this recipe's model owner before rerunning the named-pool
preflight and recreating services. A bare `docker restart` can leave an orphaned
48 GiB pool. Do not delete a pool mapped by a live process. `down` leaves model
files and both disk-backed cache directories in place.

## Source identity and validation limits

Full source/model identities and the original inspected image are recorded in
[source-lock.json](source-lock.json). The observed deployment used image
`sha256:9a0a5818b664e263ca68aba772f76ae2b2948e97e42d6bbfd0c3859507e76b30`
plus the KDA Python/native overlays and two launcher bind mounts. The Dockerfile
here consolidates those files into the published snapshot image; it does not claim the new image
has that same container-image digest.

The recipe's deliberate deployment differences are host paths, project/pool
names, an isolated cache namespace, a profiler output directory, and disabled
host-specific CPU pinning by default. Precision and inference flags follow the
inspected runtime. The entrypoint additionally validates model metadata and
initializes the schedule file before invoking the unchanged production scripts.
The exact explicit-environment differences are recorded in
[recipe-differences.json](observed/recipe-differences.json).

The deployed fused KDA implementation has prior kernel-level float64 comparison
and serving checks. It is **not bit-identical** to the preceding three-kernel
chain: a retained 64k greedy generation diverged near a tie. The prior report
records smaller kernel-level numerical error, but no full-model high-precision
reference adjudicated that divergence. Keep caches isolated across that change.

The 1,048,576-token value is a configured ceiling, not a fresh end-to-end
qualification at that length. Earlier restore checks reached 320k; the KDA
package's retained runs cover smaller contexts. New hardware, topology, model
revisions, or precision settings need separate qualification.

This publication performs no production restart, full-model load, cache reset,
or new performance run. [validation.json](validation.json) distinguishes what
was checked during packaging from the prior serving evidence.
