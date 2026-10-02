# nvfp4-kv: NVFP4 MLA KV cache

- Delta: `K3_KVDT=nvfp4_ds_mla`, which `k3-arm.sh` passes as `--kv-cache-dtype`. It needs the NoPE guard in `overlay-vllm-l3/k3_nvidia_mla.py`, which is already in the snapshot.
- Launch: `bash ../launch.sh nvfp4-kv`.
- Measured once, 2026-10-01 02:11 (`results/int7b-best-nvfp4`, script `k3-batch-int2.sh`). The boot log confirms `kv_cache_dtype=nvfp4_ds_mla`. Output was coherent. C1/C8/C16 = 30.8/202.1/314.9 tok/s, but on the older stack2 config (pre-fabric-fix, MNS 16) with no fp8 control in the same session, so the speed cannot be compared.
- KV capacity: **13,080,081 tokens, identical to fp8.** The hybrid MLA+KDA page-size unification ("Padding mamba page size by 4.15%") appears to set the token count, so smaller MLA entries free nothing. This is inferred from the logs and not traced in code.
- Verdict: not recommended as-is, because it adds quantization noise for no capacity gain. It would only pay off if attention pages were decoupled from the KDA state page size.
