#!/usr/bin/env python3
"""Add the 48-row trellis route-block register entry to the W4A16 kernel table.

The W4A16 kernel refuses any (cta_threads, m_blocks, n_blocks, k_blocks,
uses_m_block_8) specialization without a measured register count. The
48-row route block at the pinned 128x128 CTA tile is (256, 3, 8, 8, False),
which the served b12x tree (f25c8bdd base) lacks; upstream b12x da8f2cb
(2026-09-01) added it at the architectural cap of 255 registers, noting
that the 32-row block is already above the one-CTA-per-SM threshold so the
cap yields the same launch contract a measured value would.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_W4A16_KERNEL",
        "/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a16/kernel.py",
    )
)
MARKER = "(256, 3, 8, 8, False): 255,"
ANCHOR = "    (256, 4, 8, 8, False): 255,\n"
NEW = (
    "    # 48-row trellis route block at the pinned 128x128 tile (upstream\n"
    "    # b12x da8f2cb): the architectural cap gives the same one-CTA-per-SM\n"
    "    # launch contract as the block-32 entry.\n"
    "    (256, 3, 8, 8, False): 255,\n"
)


def main() -> None:
    text = PATH.read_text()
    if MARKER in text:
        print("w4a16 kernel already has the block-48 register entry")
        return
    count = text.count(ANCHOR)
    if count != 1:
        raise SystemExit(f"ERROR: expected one block-64 register anchor in {PATH}, found {count}")
    text = text.replace(ANCHOR, ANCHOR + NEW)
    compile(text, str(PATH), "exec")
    PATH.write_text(text)
    print("w4a16 kernel patched: block-48 register entry added")


if __name__ == "__main__":
    main()
