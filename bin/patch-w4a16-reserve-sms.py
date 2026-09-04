#!/usr/bin/env python3
"""Reserve SMs for concurrent kernels next to the persistent fused MoE grid.

The W4A16 fused FC1+FC2 launch is a cooperative persistent grid of
sms x blocks_per_sm CTAs whose prefill plan pins the whole register file of
every SM, so no other kernel gets an SM while it runs (measured on GPU 0,
2026-09-02: a one-block probe issued alongside starts only when the MoE
CTAs retire). With B12X_W4A16_RESERVE_SMS=N the persistent cap becomes
(sms - N) x blocks_per_sm, leaving N SMs for the small kernels of an
overlapped communication stream (the DMA ring's flag waits and adds) at a
~N/sms cost in MoE throughput. Unset or 0: unchanged.

Applies to the image's b12x/moe/_shared/kernels/w4a16/kernel.py
(_w4a16_fused_persistent_grid_x). Marker: kimi-k3-w4a16-reserve-sms.
Idempotent; composes with the split-K and block-48 register patches.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_W4A16_KERNEL",
        "/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a16/kernel.py",
    )
)
MARKER = "kimi-k3-w4a16-reserve-sms"

OLD = '''    cap = int(sms) * int(fused.blocks_per_sm)
    if _w4a16_small_m_splitk_enabled():
'''

NEW = '''    # kimi-k3-w4a16-reserve-sms: keep B12X_W4A16_RESERVE_SMS SMs free of the
    # persistent grid so concurrently launched one-block kernels (an
    # overlapped communication stream) can run while the MoE executes.
    reserved_sms = max(0, int(os.getenv("B12X_W4A16_RESERVE_SMS", "0") or 0))
    cap = max(1, (int(sms) - reserved_sms)) * int(fused.blocks_per_sm)
    if _w4a16_small_m_splitk_enabled():
'''


def main() -> int:
    text = PATH.read_text()
    if MARKER in text:
        print(f"w4a16 kernel already patched: {MARKER}")
        return 0
    if text.count(OLD) != 1:
        raise SystemExit(f"anchor not found exactly once in {PATH}")
    if "\nimport os\n" not in text:
        raise SystemExit("kernel.py does not import os at module level")
    PATH.write_text(text.replace(OLD, NEW))
    print(f"w4a16 kernel patched: {MARKER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
