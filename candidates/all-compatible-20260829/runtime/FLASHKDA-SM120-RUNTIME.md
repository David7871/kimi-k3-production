# FlashKDA SM120 runtime fix

Root cause:

- The generic checkpoint ABI was not responsible for the 3.8 GiB allocation.
- K1 alone reserved about 122 MiB. K2 alone reproduced the full allocation.
- Both V-split and default recurrence paths reproduced it.
- K2's cubin uniquely referenced
  `__cuda_syscall_cp_async_bulk_unicast`; the reference came from six
  `SM90_BULK_COPY_G2S` workspace restores.

Implementation:

- Keep only the two production ABI variants: FP32 state in/out, int32 varlen,
  checkpoint off/on.
- Preserve K1's raw swizzled workspace images.
- Load those images as unswizzled `uint32` TensorMap tiles in K2. This emits
  native SM120 TensorMap operations and keeps all six loads in the existing
  three-stage TMA transaction barrier.

Validation:

- Binary SHA-256:
  `c0b440ab87e6cfec16b23138f95a67d8fe9142836c040a56c41a574dc7d27ed1`
- `cuobjdump`: no `__cuda_syscall` symbol; local memory and stack are zero.
- Racecheck: 0 hazards after the TensorMap implementation. An intermediate
  cooperative-vector prototype was rejected after racecheck found pipeline
  publication hazards.
- Naive FP32 reference:
  - H12, T97 V-split: output/state/checkpoint passed.
  - H12, eight sequences, default K2: passed.
  - H12, T3080 with checkpoint at 3072: passed.
- Reusable output/final/checkpoint/workspace buffers were bitwise stable over
  repeated calls.
- Direct warm benchmark at T3080 with checkpoints: 0.1294 ms/call, versus
  0.1354 ms/call for the old bulk-copy build.

The qualified live profile is `compose.tp8-dflash-flashkda-320k.yml`.
Rollback remains `compose.tp8-dflash-compatible-sm120-triton.yml`.
