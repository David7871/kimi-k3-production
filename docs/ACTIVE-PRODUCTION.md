# Active Kimi-K3 production profile

Active since 2026-08-30 KST:

- Compose: `/home/g0san/kimi-k3-production/candidates/all-compatible-20260829/runtime/compose.tp8-dflash-flashkda-320k-w4a8.yml`
- Image: `kimi-k3-production-issue75-dspark@sha256:087276f1ec791909e71055d655eeadb3b35c246e46f2be21af9d210aa970a60b`
- KDA prefill: FlashKDA checkpoint ABI with native SM120 TensorMap workspace
  loads
- QSRT MoE: W4A8-MX prefill with coupled H512/H128 boundaries; W4A16 decode
- Maximum context: 320,000 tokens
- KV cache: 1.75 GiB/rank, 1,081,344-token global DCP capacity
- Scheduler budgets: target 3,080; draft input 3,080
- Speculation: remote DFlash, fixed `K=3`
- Restart policy: `unless-stopped`

Qualification on the active profile:

- 8k standalone prefill: 1,765 tok/s (5,417 prompt tokens in 3.069 s),
  versus the 957 tok/s FlashKDA baseline (+84%, 1.84x)
- Normal completion: exact `W4A8-OK 42`
- JSON schema: exact `{"answer":7}`
- W4A8 decode rejected: 41.7 tok/s versus 49.7 tok/s with W4A16
- Fixed-K3 steady proposal cycle: 46.99 ms / 21.28 cycles/s; target-side
  entry drain 39.82 ms and remote draft RPC 6.83 ms
- Dynamic workspace planning covers every M16/M64 token band; an exact
  828-token request passes after the 1,791-tile / 5,373-task fix
- W4A16 Trellis direct micro rejected: 18.86 versus 21.28 cycles/s; production
  keeps `B12X_W4A16_TRELLIS_MICRO=0`
- Final free VRAM: 376-514 MiB/rank; no allocator or EngineCore errors after
  the final rollback boot

Review PRs:

- B12X: https://github.com/myshytf/b12x/pull/1
- vLLM: https://github.com/myshytf/vllm/pull/1

Rollback to the previous FlashKDA/W4A16-only profile:

```bash
docker compose -p kimi-k3-production \
  -f /home/g0san/kimi-k3-production/candidates/all-compatible-20260829/runtime/compose.tp8-dflash-flashkda-320k.yml \
  up -d --no-deps --force-recreate kimi-k3-production
```

The original immutable fine3072 image remains the second-level rollback at
`/home/g0san/kimi-k3-production/compose.tp8-dflash-fine3072.yml`.
