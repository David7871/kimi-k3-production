#!/bin/bash
# Build the standalone _C_k3ext extension (rotated DCP peer KV gather) inside
# the production image so it links against the exact PyTorch 2.13 / CUDA 13.3
# runtime. Output: _C_k3ext.abi3.so next to this script.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=${IMG:-kimi-k3-production-issue75-dspark:k3-prefill-sm120-fa4-20260901-r3}
docker run --rm -v "$HERE":/ext -w /ext --entrypoint bash "$IMG" -c '
set -euo pipefail
TORCH=$(/opt/venv/bin/python -c "import torch, os; print(os.path.dirname(torch.__file__))")
PYINC=$(/opt/venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()[\"include\"])")
# The 2.13 stable string-conversion hotfix that the vLLM build applies is
# only needed for std::string IValues; this op passes tensors and ints.
nvcc -O3 -std=c++17 -Xcompiler -fPIC -shared \
  -gencode arch=compute_120,code=sm_120 \
  -DTORCH_TARGET_VERSION=0x020B000000000000ULL -DUSE_CUDA -DPy_LIMITED_API=0x03090000 \
  -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$PYINC" -I/ext/src \
  src/k3ext_dcp_gather.cu -o _C_k3ext.abi3.so \
  -L"$TORCH/lib" -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart -lcublas \
  -Xlinker -rpath -Xlinker "$TORCH/lib"
/opt/venv/bin/python - <<PY
import torch, vllm._C  # noqa: F401  (base ops first, as in serving)
torch.ops.load_library("/ext/_C_k3ext.abi3.so")
print("loaded:", torch.ops._C_k3ext.direct_dcp_kv_gather_rotated)
PY
'
ls -la "$HERE"/_C_k3ext.abi3.so
