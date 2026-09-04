# Kimi-K3 L2 weight prefetch for decode (vLLM branch `agent/kimi-k3-l2-prefetch-20260903`)

Port of the GLM-5.3 L2 weight prefetch (local-inference-lab/vllm#586).
A CuTe kernel issues `cp.async.bulk.prefetch.L2` (evict_last) over the next
dense projection weights on a side stream inside the idle windows of every
decoder layer, so the decode GEMMs read them from L2. Cache hints only;
numerics unchanged. `VLLM_K3_L2_PREFETCH=0` disables it.

Files (mounted by `candidates/k3-decode2-20260903/compose.l2-prefetch.override.yml`):

- `l2_prefetch.py` — new module `vllm/models/kimi_k3/nvidia/l2_prefetch.py`
  (GLM module with `VLLM_K3_L2_PREFETCH*` environment names).
- `model.py` — `_KimiL2PrefetchPlanner` (per-layer A/B/C plans, hook
  installation), the successor chain, the window-C hook in `KimiMoE`, the
  side-stream join at the end of the model forward. Branch file; the
  served model.py equals the branch base.
- `mla.py` — the served pipeline `mla.dma4.py` plus the window-A hook after
  the first projection. Branch file (the L2 branch stacks on the gather
  pipeline branch).
- `kda.py` — the served (image) kda.py plus the window-A hook after the
  input projection and the window-B hook before
  `reduce_kimi_full_width_projection`. The image's kda.py is newer than the
  fork branches (flashkda checkpoint state), so the branch file is not
  mountable; this copy carries the same two hunks (`kda_l2.patch`).
- `linear.py` — `RowParallelLinear.forward` pre-reduce hook (MLA o_proj,
  dense down_proj). Branch file; the served linear.py equals the base.

Measured on one RTX PRO 6000 with the served MXFP8 dense GEMM at M=4 (cold
L2 -> prefetched): KDA in_proj 22.7 MB 31.1 -> 21.4 us, o_proj 7.6 MB
9.5 -> 7.0 us, fused_qkv_a 11.4 MB 23.9 -> 20.2 us. The 82k-context decode
step spends 9.6 ms of 48 ms in dense GEMMs (4.9 ms MXFP8 DenseGemmKernel,
1.6 ms 128x8x128 split-K GEMM); expected about -2 ms per step.
Status: tests pass (17 env/persisting-L2 tests, stub planner with CUDA
graph); not deployed.
