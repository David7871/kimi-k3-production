# Pipelined DCP context-KV gather (vLLM PR #588)

Files mounted by `candidates/k3-longctx-20260902/compose.dcp-gather-pipeline.override.yml`:

- `mla.py` — `vllm/models/kimi_k3/nvidia/mla.py` from branch
  `agent/kimi-k3-dcp-gather-pipeline-20260902-pr` (identical to the branch
  file). The chunked-context loop publishes window i + 1 through
  `DCPKVGatherPipeline` on a side stream while window i is projected and
  attended; `VLLM_K3_DCP_GATHER_PIPELINE=0` restores the serial loop.
- `dcp_utils.py` — the branch file plus the side-extension loader hunks of
  `overlays/dcp-gather-rotate-20260902/dcp_utils.py` (`VLLM_K3_DCP_GATHER_ROTATE_LIB`,
  ubatch slot for `VLLM_K3_UBATCH_PREFILL`). Adds `kv_gather_slots()`
  (`VLLM_DCP_KV_GATHER_SLOTS`, default 3), `DCPKVGatherPipeline`,
  `get_dcp_kv_gather_pipeline`, `MLADCPKVGather.kv_gather_slots`.
- `envs.py` — the served file plus `VLLM_DCP_KV_GATHER_SLOTS` and
  `VLLM_K3_DCP_GATHER_PIPELINE`.
- `_C_k3ext.abi3.so` — the rotated-destination peer gather extension rebuilt
  from `src/k3ext_dcp_gather.cu` (`build.sh`, inside the production image)
  with the slot checks generalized to the workspace's slot count
  (`received_kv.size(0)` slots, `0 <= buffer_slot < slots`). Kernel body
  unchanged from `overlays/dcp-gather-rotate-20260902`. Served since the
  2026-09-02 13:34 UTC boot (md5 b59797b6...).
- `_C_k3ext.blockcap.abi3.so` — the same source plus a launch-grid bound:
  `VLLM_K3_DCP_GATHER_MAX_BLOCKS` (load time) and
  `VLLM_K3_DCP_GATHER_MAX_BLOCKS_FILE` (first integer of the file, re-read
  at most once per second; a tuning knob that needs no restart), clamped to
  [world_size, 1024]. Built with `OUT=_C_k3ext.blockcap.abi3.so ./build.sh`;
  not served.

Protocol: window w occupies slot `w % S`; the side stream publishes w after
the compute stream released window `w - (S - 1)`; the gather's all-rank
rendezvous guarantees every peer released `w - S` before any rank
publishes w. Planes are byte-identical to the serial schedule.

Status: research-qualified on one device (pipeline ordering test, workspace
geometry tests); the four-rank ring test runs in the deployment window
(`candidates/k3-longctx-20260902/deploy.sh`). Not deployed.
