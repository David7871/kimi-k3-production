# Production deployment result

Qualified runtime:

- Image: `kimi-k3-production-issue75-dspark@sha256:087276f1ec791909e71055d655eeadb3b35c246e46f2be21af9d210aa970a60b`
- Compose: `compose.tp8-dflash-flashkda-320k.yml`
- KDA prefill backend: FlashKDA checkpoint ABI
- SM120 runtime: native TensorMap workspace loads; no bulk-copy syscall
- Maximum context: 320,000 tokens
- Fixed KV cache: 1.75 GiB/rank, 625,777-token capacity
- Target/draft budgets: 3,080 / 3,080
- Status after qualification: healthy, restart 0, OOM false

Active patches include vLLM #52388, #52789, #53614, #53080, #54348, the
target-cache correctness follow-up, and LMCache #44101.

Earlier FlashKDA activation failed because CUDA 13.3 lowered K2's six
`SM90_BULK_COPY_G2S` operations to
`__cuda_syscall_cp_async_bulk_unicast`. Loading that SM120 runtime reserved
about 3.8 GiB per rank. The r5 binary replaces those operations with native
TensorMap loads and retains only the two production checkpoint ABI variants.

The full 372K/2.0 GiB r5 trial started and completed its first FlashKDA request,
but was rejected after rank 0 logged an allocator OOM warning: a 264,241,152
byte request had only 263,585,792 bytes physically free. The engine survived,
but the margin was not production-safe. Reducing both the served ceiling and
the explicit KV reservation produced the active 320K/1.75 GiB profile.

The primary rollback compose is
`compose.tp8-dflash-compatible-sm120-triton.yml`; the original immutable
rollback is `/home/g0san/kimi-k3-production/compose.tp8-dflash-fine3072.yml`.

Live qualification:

- Kernel binary SHA-256:
  `c0b440ab87e6cfec16b23138f95a67d8fe9142836c040a56c41a574dc7d27ed1`.
- `cuobjdump`: no `__cuda_syscall`; racecheck: 0 hazards.
- FP32 reference: V-split/default, checkpoint/no-checkpoint, T=3,080 with a
  3,072 checkpoint all passed at relative RMSE 0.0044-0.0056.
- Normal response: exact marker, `finish_reason=stop`.
- JSON schema: `{"answer":7}`, no FSM error.
- Four simultaneous requests: 4/4 HTTP 200 and stop.
- Interrupted 18K-class prefill: client disconnected at 1.2 s; scheduler was
  already running=0/waiting=0 at the first 0.08-second observation; LMCache
  pending/in-flight counters were zero; immediate retry completed in 1.52 s.
- 16,104-token cold prefill: 21.0 s and stop; local checkpoint/prefix replay:
  6.56 s with identical visible output.
- 32,092-token cold prefill: 36.28 s, approximately 884.6 prompt tok/s versus
  the prior Triton candidate's 888 tok/s.
- Production counters after qualification: error=0, restart=0, OOM=false.

The two r5 cutovers triggered the existing L2 process-generation policy. It
removed 416 files / 6,900,940,800 bytes, then 32 files / 530,841,600 bytes.
