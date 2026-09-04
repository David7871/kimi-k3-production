#!/usr/bin/env python3
"""Gate torch profiling to rank 0 (host-RAM safety).

2026-08-28 04:0x lesson: profile stop serializes each rank's multi-hundred-MB
trace in host RAM simultaneously; on the 124 GB host (48 GB LMCache L1
pinned) all eight ranks exporting at once OOM-killed dbus and froze the
desktop. TP ranks are symmetric for step-anatomy purposes, so with
K3_PROFILE_RANK0_ONLY=1 only rank 0 records and exports (~1/8 the spike).
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_GPU_WORKER",
        "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu_worker.py",
    )
)
MARKER = "kimi-k3-profile-rank0-only"

OLD = '''    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # Check if profiling is enabled
        if self.profiler_config is None or self.profiler_config.profiler is None:
'''

NEW = '''    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # kimi-k3-profile-rank0-only: eight ranks exporting traces at once
        # OOM-killed the 124 GB host (2026-08-28); rank 0 suffices for
        # step anatomy on symmetric TP ranks.
        import os as _os

        if _os.environ.get("K3_PROFILE_RANK0_ONLY") == "1" and self.rank != 0:
            return
        # Check if profiling is enabled
        if self.profiler_config is None or self.profiler_config.profiler is None:
'''


def main() -> None:
    text = PATH.read_text()
    if MARKER in text:
        print(f"gpu_worker already patched: {MARKER}")
        return
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(f"ERROR: expected exactly one anchor, found {count}")
    text = text.replace(OLD, NEW, 1)
    compile(text, str(PATH), "exec")
    PATH.write_text(text)
    print(f"gpu_worker patched: {MARKER}")


if __name__ == "__main__":
    main()
