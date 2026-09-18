#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --print-command ]]; then
  exec /bin/bash /runtime/serve_tp9.sh "$@"
fi
python3 /runtime/check-models.py
python3 - <<'PY'
import os
from pathlib import Path

# This file is part of the schedule, not a disposable compilation cache.
path = Path(os.environ['VLLM_K3_UBATCH_MODE_FILE'])
mode = 'overlap moesplit=1 offset=2\n'
path.parent.mkdir(parents=True, exist_ok=True)
try:
    with path.open('x') as output:
        output.write(mode)
except FileExistsError:
    if path.read_text().strip() != mode.strip():
        raise SystemExit(f'{path} must contain {mode.strip()!r}; use a fresh CACHE_DIR')
for name in ['VLLM_K3_REQUEST_ENDPOINT_CACHE_DISABLE_FILE',
             'VLLM_K3_REQUEST_ENDPOINT_CACHE_DEBUG_DISABLE_FILE',
             'VLLM_K3_REQUEST_ENDPOINT_CACHE_TORCH_COPY_FILE',
             'VLLM_K3_DRAFT_REUSE_RESTORED_KV_DISABLE_FILE']:
    marker = os.environ.get(name, '')
    if marker and Path(marker).exists():
        raise SystemExit(f'{marker} changes the pinned runtime; use a fresh CACHE_DIR')
PY
exec /bin/bash /runtime/serve_tp9.sh "$@"
