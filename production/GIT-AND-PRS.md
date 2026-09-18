# Git history and PR mapping for the deployed runtime

Status: **qualified** for the file comparisons described below. The immutable
commits in [source-lock.json](source-lock.json) identify the runtime source.
The public forks retain Git history; the commands below clone normal Git
repositories with their history, not source archives.

## Clone the exact source histories

Run in a new directory:

```bash
git clone https://github.com/myshytf/vllm.git
git -C vllm checkout --detach a3693719528811ac605dfa236321a53e5770b120

git clone https://github.com/myshytf/b12x.git
git -C b12x checkout --detach 0d31d259c6f0e6733190a6b3af9881b874f10241

git clone https://github.com/myshytf/LMCache.git
git -C LMCache checkout --detach 98c8be0a655bdf5411a8a2335f21cb44a76083d7
```

| Repository | Published branch containing the pinned tip | Pinned tip |
| --- | --- | --- |
| [myshytf/vllm](https://github.com/myshytf/vllm) | `feat/kda-spec-fused-decode-20260918` | `a3693719528811ac605dfa236321a53e5770b120` |
| [myshytf/b12x](https://github.com/myshytf/b12x) | `production/kimi-k3-20260917` | `0d31d259c6f0e6733190a6b3af9881b874f10241` |
| [myshytf/LMCache](https://github.com/myshytf/LMCache) | `production/kimi-k3-20260917` | `98c8be0a655bdf5411a8a2335f21cb44a76083d7` |

The earlier shared tag `kimi-k3-production-20260917` is a baseline. In
particular, the original LMCache tag identified `6ce828ec6f`; use the full
current commit above. A branch name or the installed Python package version
is not an immutable runtime identity.

## Current integration map

PRs below link to the **personal fork** unless the row explicitly says LIL.
The pinned branch already contains the integrated code. This table is for
review and tracing, not a list of patches to apply again.

### vLLM

The original running image has production base `25ddae8773`. Its one active
Python overlay, `vllm/models/kimi_k3/nvidia/kda.py`, matches `a369371952`.
The snapshot image bakes that overlay and the separately pinned KDA binary.

| PR / integration | Role in the current runtime |
| --- | --- |
| [#7](https://github.com/myshytf/vllm/pull/7) | Earlier recurrent-tail cache work; the current lineage contains later integration revisions rather than that PR head verbatim |
| [#8](https://github.com/myshytf/vllm/pull/8) | Compact QSRT prefill scratch; head `b6d4495a9a` is an ancestor |
| [#9](https://github.com/myshytf/vllm/pull/9) | Packed KV transport; head `9b0f61960f` is an ancestor |
| [#10](https://github.com/myshytf/vllm/pull/10) | Reuse restored draft KV and recurrent boundary checkpoints; head `8971e6e1b` is an ancestor |
| [#11](https://github.com/myshytf/vllm/pull/11) | Request-endpoint cache; head `935ba01342` is an ancestor; later commits contain the deployed corrections |
| [#12](https://github.com/myshytf/vllm/pull/12) | MXFP8 dense W8A16 kernel selection and large-M path; head `1a99b2371e` is an ancestor |
| [#13](https://github.com/myshytf/vllm/pull/13) | Exact split-prefill scheduling; head `33e2741c76` is an ancestor |
| [#15](https://github.com/myshytf/vllm/pull/15) | Expert-range overlap, shared-expert slot lifetime, and production qualification; head `2bd40e4513` is an ancestor |
| [#14](https://github.com/myshytf/vllm/pull/14) | Scheduler load-failure fix: `b1d4790869` cherry-picked as `25ddae8773`; stable patch IDs match |
| [#17](https://github.com/myshytf/vllm/pull/17) | Fused KDA speculative-decode source and dispatch; pinned head `a369371952` |

The KDA integration history is `cabc252b1f` → `a5672a8425` → `7415df4f96` →
`a369371952`. The compiled extension's source is
[`csrc/libtorch_stable/kimi_k3/fused_kda_spec_decode_kernel.cu`](https://github.com/myshytf/vllm/blob/a3693719528811ac605dfa236321a53e5770b120/csrc/libtorch_stable/kimi_k3/fused_kda_spec_decode_kernel.cu).
Its exact deployed binary is [overlays/_C_k3kda.abi3.so](overlays/_C_k3kda.abi3.so);
[overlays/README.md](overlays/README.md) records its build provenance.

[#16](https://github.com/myshytf/vllm/pull/16) is supporting cache-residency
benchmark/qualification work. Its PR head is not an ancestor of the pinned
runtime; do not treat it as an additional runtime patch.

### B12X

| PR / integration | Role in the current runtime |
| --- | --- |
| [#6](https://github.com/myshytf/b12x/pull/6) | Rotation scratch reuse; later integration retains the deployed scratch contract |
| [#7](https://github.com/myshytf/b12x/pull/7) | Two-switch DCP gather relay; incorporated by file-content integration, not by merging this PR's history |
| [#8](https://github.com/myshytf/b12x/pull/8) | Fused-MoE launch controls and ring options; head `c0c304ead7` is an ancestor |
| [#9](https://github.com/myshytf/b12x/pull/9) | Retained FC2 outputs and stable route ordering; head `85223507ea` is an ancestor |
| [#10](https://github.com/myshytf/b12x/pull/10) | Static TP9 peer addresses; head `b6c8bde010` is an ancestor |
| [#11](https://github.com/myshytf/b12x/pull/11) | Inline reference-preserving FC2 group preparation; head `7b01728256` is an ancestor |
| [#12](https://github.com/myshytf/b12x/pull/12) | Carried FC2 task cursor; head `0fa8f11314` is an ancestor |
| [`0d31d259c6`](https://github.com/myshytf/b12x/commit/0d31d259c6f0e6733190a6b3af9881b874f10241) | Final integration of the four served PCIe communication files |

The last commit sets `_dcp_a2a_cute.py`, `_switch_gather.py`,
`pcie_dcp_a2a.py`, and `pcie_dma.py` to their served versions from
`65a593e506`, combining the relay with the ring options. All four Git blob
IDs match that source. PR #7's head will not appear as an ancestor in
`git log`; that is an explicit source-integration boundary.

### LMCache

The sequence from the baked DFlash2 base is:

| PR / integration | Commit and role |
| --- | --- |
| [#5](https://github.com/myshytf/LMCache/pull/5) | `212d96c1d4`: presence-only lookup for local hits |
| [LIL #68](https://github.com/local-inference-lab/LMCache/pull/68) | `656c98ed24`: bounded emergency-flush budget |
| [LIL #69](https://github.com/local-inference-lab/LMCache/pull/69) | `a07e0799e7`: windowed restore |
| [#6](https://github.com/myshytf/LMCache/pull/6) | `74e7344652`: exact per-worker read-lock release |
| [`3e381944ba`](https://github.com/myshytf/LMCache/commit/3e381944ba) + `6ce828ec6f` | Mixed-tier restore, failed-admission recovery, and fixture correction |
| [#7](https://github.com/myshytf/LMCache/pull/7) | `98c8be0a`: resident-L1 prefetch fix and fixed lookup-kind selection |

## Full PR inventory, including LIL contributions

[pr-inventory.json](pr-inventory.json) records all 135 PRs returned for author
`myshytf` across these six repositories at export time, with URLs, full head
commits, base branches, captured state, and head-ancestry checks. It includes
older alternatives and research-only work so they can be inspected without
confusing them with the selected runtime.

- [LIL vLLM contributions](https://github.com/local-inference-lab/vllm/pulls?q=is%3Apr+author%3Amyshytf)
- [LIL B12X contributions](https://github.com/local-inference-lab/b12x/pulls?q=is%3Apr+author%3Amyshytf)
- [LIL LMCache contributions](https://github.com/local-inference-lab/LMCache/pulls?q=is%3Apr+author%3Amyshytf)

An `ancestor` result identifies history inclusion, not whether a runtime flag
selects that feature. `not_ancestor` does not prove the feature is absent:
some changes were cherry-picked, rebased, or carried through a source snapshot.
`not_in_local_object_store` means that head's ancestry was not checked.
The pinned source, actual settings, and file comparisons define this export.

Prebuilt extensions inherited from the base image were not all rebuilt at the
latest source commits. Their deployed bytes are pinned by the image and runtime
manifest. The tracked-file comparison covers the runtime package sources;
it does not assert that every inherited binary was compiled from those tips.

To inspect a PR whose head is outside the selected ancestry, fetch its Git ref:

```bash
git -C b12x fetch origin pull/7/head:refs/remotes/origin/pr/7
git -C b12x log --oneline --graph --decorate --all
```

## Evidence of agreement with the running container

- [git-vs-runtime.json](observed/git-vs-runtime.json): 3,782 tracked runtime
  files compared by Git blob identity against the three pinned commits;
  no mismatches.
- [runtime-manifest.sha256](runtime-manifest.sha256): 3,842 runtime files,
  including built files inherited from the image. The image verifier also
  checks both launchers and both standalone native extensions: 3,846 files.
- [integration-check.json](observed/integration-check.json): matching scheduler
  cherry-pick patch IDs and the four B12X communication-file blob IDs.
- [environment.json](observed/environment.json), [launch.json](observed/launch.json),
  and [cache-controls.json](observed/cache-controls.json): observed effective
  container settings and external control-file state.
- [writable-layer-audit.json](observed/writable-layer-audit.json): filtered source/
  library changes outside the image. The listed KDA extension is included in
  the overlay manifest. Generated bytecode, request logs, and cache payloads
  are not runtime source exports.

This is a code, binary, and configuration snapshot. It does not serialize
live processes, GPU memory, driver injection, compiled cache state, or KV data.
