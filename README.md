# Kimi-K3 QSRT-K2: exact runtime, Git history, and recipe

This repository shares the runtime actually inspected on the production host:
**TP9 / DCP9 on 9 × RTX PRO 6000 Blackwell, colocated DFlash2, and LMCache**.
It includes the active overlays and compiled kernel, the launch configuration,
and links to the full Git histories and integration PRs.

- **[Runtime image and inspection instructions](production/SNAPSHOT.md)**
- **[Full Git repositories and PR mapping](production/GIT-AND-PRS.md)**
- **[Docker + serving recipe](production/README.md)**

## Get the exact runtime

```bash
docker pull ghcr.io/myshytf/kimi-k3@sha256:856320d49eb35e2a0652348b644813bac42ff23e2b7a9af193390c388e115e5d
```

Convenience tag: `ghcr.io/myshytf/kimi-k3:runtime-20260918`.
The image contains the original deployed base plus the actual KDA Python/native
overlays and launch scripts, copied without recompiling the kernel. Model
weights and cache data are mounted separately.

Status: **qualified** for runtime-file identity and CPU launch checks.
All **3,846 checked runtime files** match the deployed snapshot; all **3,782
Git-tracked runtime files** match the pinned repository commits. This does not
claim a fresh full-model qualification of the packaged image. Production was
not restarted to create this export. See [validation.json](production/validation.json).

## Which Git commits are deployed?

| Component | Full Git repository and pinned source |
| --- | --- |
| vLLM | [myshytf/vllm](https://github.com/myshytf/vllm), [`a369371952`](https://github.com/myshytf/vllm/commit/a3693719528811ac605dfa236321a53e5770b120): production `25ddae8773` plus the active KDA integration |
| B12X | [myshytf/b12x](https://github.com/myshytf/b12x), [`0d31d259c6`](https://github.com/myshytf/b12x/commit/0d31d259c6f0e6733190a6b3af9881b874f10241) |
| LMCache | [myshytf/LMCache](https://github.com/myshytf/LMCache), [`98c8be0a`](https://github.com/myshytf/LMCache/commit/98c8be0a655bdf5411a8a2335f21cb44a76083d7) |

[GIT-AND-PRS.md](production/GIT-AND-PRS.md) provides full-history clone commands,
current PR integration details, and the wider LIL contribution inventory.
It explicitly records the scheduler cherry-pick and the B12X communication
files integrated by content rather than by merging PR history.

The previously shared `kimi-k3-production-20260917` snapshot predates the
current KDA overlay and LMCache resident-L1 prefetch fix. Use
[source-lock.json](production/source-lock.json) and
[image-lock.json](production/image-lock.json) for immutable identities.

## Run it

```bash
git clone https://github.com/myshytf/kimi-k3-production.git
cd kimi-k3-production/production
cp .env.example .env
# Follow README.md to download the pinned models and configure paths/topology.
./run.sh pull
./run.sh check
./run.sh up
./run.sh logs
```

[compose.yaml](production/compose.yaml) is a single self-contained profile.
[Dockerfile](production/Dockerfile) can reconstruct the snapshot from the
public deployed base and supplied overlays. The recipe serves the existing
[lukealonso/Kimi-K3-QSRT-K2](https://huggingface.co/lukealonso/Kimi-K3-QSRT-K2)
checkpoint with
[lightseekorg/kimi-k3-dflash2](https://huggingface.co/lightseekorg/kimi-k3-dflash2).
It performs no new quantization.

The observed raw container environment and mount roles are retained under
[production/observed/](production/observed/). Portable recipe differences,
including the isolated cache namespace and CPU pinning default, are documented.
A 16-GPU installation needs its own launch/topology choices; the source and
binary snapshot can be inspected independently of this TP9 recipe.


## PP4×TP4 two-host lane (16 GPUs, no speculative decoding)

[pp4-tp4/](pp4-tp4/) documents a second, measured (not production-qualified)
serving topology: pipeline parallel 4 × tensor parallel 4 (with MLA
decode-context parallel 4) across two 8-GPU hosts — no DFlash, no MTP, no
draft model — plus the optional **Moet 2-bit routed-expert** path. It ships
the exact vLLM/b12x overlay trees, launchers, env files, image recipes, and
raw benchmark output. Headline: 372 tok/s at C16 (+11.1%) and 1394 tok/s at
C256 (+9.9%) with Moet on vs the same-image control, GSM8K-200 197/200 OFF vs
196/200 ON, and an optional PP layer rebalance that adds +16.7% KV-cache
capacity at a C16/C256 cost (360/1341 tok/s). See
[pp4-tp4/README.md](pp4-tp4/README.md).

## Historical files and attribution

`compose/`, `candidates/`, `overlays/`, `bin/`, and older build directories are
historical exports. The current recipe does not depend on their TP8/remote-draft
override chains. [docs/ACTIVE-PRODUCTION.md](docs/ACTIVE-PRODUCTION.md) points to
the current serving contract and its validation limits.

QSRT checkpoint creation and codec research belong to
[local-inference-lab/qsrt](https://github.com/local-inference-lab/qsrt).
Upstream code, binary dependencies, and model artifacts retain their own
licenses; consult the linked repositories and model cards.
