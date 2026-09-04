#!/usr/bin/env python3
"""Gate the W4A16 stripe split-K schedule to decode-sized launches.

B12X_W4A16_SMALL_M_SPLITK=1 measured on the TP8 fine3072 profile
(2026-08-28): decode MoE kernel 235.2 -> 198.5 us (-15.6%, step -2.9 ms,
decode +1.4 tok/s) but PREFILL regressed 902 -> 782 tok/s because the flag
also strips the large-M prefill launches where the whole-tile wave schedule
is already grid-saturated and the stripe finalize is pure overhead.

Apply the stripe only when the launch's size_m <= 16 (decode/verify
shapes; prefill buckets are 512+). size_m already participates in the
kernel cache key, so per-M variants compile independently.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_W4A16_KERNEL",
        "/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a16/kernel.py",
    )
)
MARKER = "kimi-k3-splitk-decode-only"

OLD = "        self.small_m_splitk = _w4a16_small_m_splitk_enabled()\n"
NEW = (
    "        # kimi-k3-splitk-decode-only: stripe only decode-sized launches;\n"
    "        # large-M prefill keeps the whole-tile wave schedule (stripe\n"
    "        # finalize regressed prefill 902->782 tok/s on 2026-08-28).\n"
    "        self.small_m_splitk = (\n"
    "            _w4a16_small_m_splitk_enabled() and int(size_m) <= 16\n"
    "        )\n"
)


def main() -> None:
    text = PATH.read_text()
    if MARKER in text:
        print(f"w4a16 kernel already patched: {MARKER}")
        return
    count = text.count(OLD)
    if count != 2:
        raise SystemExit(
            f"ERROR: expected exactly two flag anchors in {PATH}, found {count}"
        )
    text = text.replace(OLD, NEW)
    compile(text, str(PATH), "exec")
    PATH.write_text(text)
    print(f"w4a16 kernel patched: {MARKER} (both classes)")


if __name__ == "__main__":
    main()
