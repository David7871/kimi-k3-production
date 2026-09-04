#!/usr/bin/env python3
"""Gate DSpark grouped context-cache ops on actual kernel availability.

The standalone draft image was built for SM86 (RTX 3090): its compiled
``_C_cache_ops`` extension predates ``concat_and_cache_mla_grouped``. The
source gates the native (pure-PyTorch) context path on device capability
``< 9`` only, so running the same image on an SM 12.0 RTX PRO 6000 selects
the grouped custom op and crashes with AttributeError at startup.

Extend the gate: use the native path when the grouped op is missing from
the loaded extension, regardless of capability. On the 3090 the branch is
unchanged (capability check short-circuits first).
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_DSPARK_MLA",
        "/opt/k3-dspark-3090/source/vllm/models/kimi_k3/nvidia/dspark_mla.py",
    )
)
MARKER = "grouped-op-availability-gate"

OLD = '''def _needs_ampere_context_fallback(tensor: torch.Tensor) -> bool:
    """Return whether DSpark context ops need their SM86-safe path."""
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] < 9
'''

NEW = '''def _needs_ampere_context_fallback(tensor: torch.Tensor) -> bool:
    """Return whether DSpark context ops need their SM86-safe path."""
    if tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] < 9:
        return True
    # grouped-op-availability-gate: this runtime may run on newer devices
    # (SM >= 9) whose compiled _C_cache_ops predates the grouped kernels.
    # The native path is correct everywhere; the grouped op is only an
    # optimization, so fall back whenever it is absent.
    return not hasattr(torch.ops._C_cache_ops, "concat_and_cache_mla_grouped")
'''

FUSED_PATH = Path(
    os.environ.get(
        "K3PATCH_DSPARK_FUSED",
        "/opt/k3-dspark-3090/source/vllm/models/kimi_k3/nvidia/ops/"
        "fused_mla_key_concat_kv_cache.py",
    )
)

FUSED_OLD = '''def _needs_ampere_fallback(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] < 9
'''

FUSED_NEW = '''def _needs_ampere_fallback(tensor: torch.Tensor) -> bool:
    if tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] < 9:
        return True
    # grouped-op-availability-gate: the compiled _C extension in this image
    # predates the fused decode/insert kernels; use the native path whenever
    # the fused op is absent (SM >= 9 devices with the SM86-era build).
    return not hasattr(
        torch.ops._C, "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert"
    )
'''


def _apply(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if MARKER in text:
        print(f"{label} already patched: {MARKER}")
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"ERROR: expected exactly one anchor in {path} ({label}), found {count}"
        )
    text = text.replace(old, new, 1)
    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"{label} patched: {MARKER}")


def main() -> None:
    _apply(PATH, OLD, NEW, "dspark_mla")
    _apply(FUSED_PATH, FUSED_OLD, FUSED_NEW, "fused_mla_key_concat_kv_cache")


if __name__ == "__main__":
    main()
