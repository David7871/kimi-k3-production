# Exact overlays copied from the running container

Status: **qualified** for byte identity with the inspected deployment.

| File | Container destination | SHA-256 |
| --- | --- | --- |
| `kda.py` | `/opt/infernal-invocation/vllm/vllm/models/kimi_k3/nvidia/kda.py` | `522b540ea859c97a7f9ec78689112cdea86861774fee04b91aa04b40957931a9` |
| `_C_k3kda.abi3.so` | `/opt/k3kda/_C_k3kda.abi3.so` | `fa546e7ae1da450ff2d3b6515a598920d95ead8a628c766fa5b88ba56efc4749` |

The Python file matches vLLM commit
`a3693719528811ac605dfa236321a53e5770b120`. The extension source is that commit's
`csrc/libtorch_stable/kimi_k3/fused_kda_spec_decode_kernel.cu`, using the
adjacent `csrc/libtorch_stable/torch_utils.h`. The standalone build changes
only the include path from `../torch_utils.h` to `torch_utils.h` and names the
source `src/k3kda_spec_decode_v4.cu`. The resulting source SHA-256 is
`b9ae6d4c60cc1e577f7dfb4328c49c567af65bf09f8cf49af6bc133f5850ae23`.

Build provenance: the deployed CUDA 13.3 / PyTorch 2.13 image, `nvcc -O3
-std=c++17 -Xcompiler -fPIC -shared`, `compute_120a/sm_120a`,
`TORCH_TARGET_VERSION=0x020B000000000000ULL`, `USE_CUDA`,
`Py_LIMITED_API=0x03090000`, and links to `torch`, `torch_cpu`, `torch_cuda`,
`c10`, `c10_cuda`, and `cudart`. No fast-math compiler flag was used.

The snapshot deliberately distributes the already-deployed ELF. A new
compilation is not assumed byte-identical; use the recorded hash when the
requirement is to inspect exactly the binary that was served. The source and
integration history remain available in the full vLLM fork and PR #17.
These files retain the vLLM project's Apache-2.0 licensing and attribution.
