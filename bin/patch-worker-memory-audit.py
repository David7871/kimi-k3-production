#!/usr/bin/env python3
"""Device-memory audit hooks for the vLLM GPU worker (env-gated).

With K3_MEMORY_AUDIT=1 every worker records the CUDA caching allocator's
history from device init on (allocation stacks included) and installs a
SIGUSR1 handler that writes, per rank, the live-allocation snapshot
(`torch.cuda.memory._snapshot()`), `torch.cuda.memory_stats()` and
`torch.cuda.mem_get_info()` to K3_MEMORY_AUDIT_DIR
(default /cache/memory-audit). Sending SIGUSR1 to a worker process at any
time after warm-up therefore yields the full attribution of PyTorch device
memory on that rank; the difference between the device's used memory and
PyTorch's reserved memory is the non-PyTorch share (CUDA context, NCCL,
IPC slabs allocated through the runtime API).

Analyze with research/vram-audit-20260902/analyze_snapshot.py.
Marker: kimi-k3-memory-audit. Idempotent.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_GPU_WORKER",
        "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu_worker.py",
    )
)
MARKER = "kimi-k3-memory-audit"

OLD = '''    def init_device(self):
        if self.device_config.device_type == "cuda":
'''

NEW = '''    def _k3_memory_audit_install(self):
        """kimi-k3-memory-audit: allocator history + SIGUSR1 snapshot dump."""
        import os as _os
        import pickle as _pickle
        import signal as _signal
        import time as _time

        import torch as _torch

        if _os.environ.get("K3_MEMORY_AUDIT", "0") != "1":
            return
        out_dir = _os.environ.get("K3_MEMORY_AUDIT_DIR", "/cache/memory-audit")
        try:
            _torch.cuda.memory._record_memory_history(
                max_entries=int(_os.environ.get("K3_MEMORY_AUDIT_MAX_ENTRIES", "400000"))
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.warning("memory audit: history recording unavailable: %s", exc)

        def _dump(signum, frame):
            try:
                _os.makedirs(out_dir, exist_ok=True)
                rank = getattr(self, "rank", _os.environ.get("RANK", "x"))
                stamp = _time.strftime("%Y%m%d-%H%M%S")
                free, total = _torch.cuda.mem_get_info()
                payload = {
                    "rank": rank,
                    "time": stamp,
                    "mem_get_info": {"free": int(free), "total": int(total)},
                    "memory_stats": dict(_torch.cuda.memory_stats()),
                    "snapshot": _torch.cuda.memory._snapshot(),
                }
                path = _os.path.join(out_dir, f"rank{rank}-{stamp}.pickle")
                with open(path, "wb") as fh:
                    _pickle.dump(payload, fh)
                logger.info("memory audit: wrote %s (free %.2f GiB of %.2f)",
                            path, free / 2**30, total / 2**30)
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.warning("memory audit: dump failed: %s", exc)

        _signal.signal(_signal.SIGUSR1, _dump)
        logger.info("memory audit armed (SIGUSR1 -> %s)", out_dir)

    def init_device(self):
        self._k3_memory_audit_install()
        if self.device_config.device_type == "cuda":
'''


def main() -> int:
    text = PATH.read_text()
    if MARKER in text:
        print(f"gpu_worker already patched: {MARKER}")
        return 0
    if text.count(OLD) != 1:
        raise SystemExit(f"anchor not found exactly once in {PATH}")
    PATH.write_text(text.replace(OLD, NEW))
    print(f"gpu_worker patched: {MARKER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
