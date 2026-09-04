#!/usr/bin/env python3
"""Allow the remote Kimi target to parse DFlash2 draft metadata.

The TP8 target never constructs or executes the draft model; it only parses
its config to determine auxiliary taps and speculative geometry before talking
to the standalone GPU over the remote protocol.  Register DFlash2DraftModel as
the existing DFlash draft interface here so ModelConfig validation succeeds.
The standalone container owns the real DFlash2 implementation and weights.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(
    os.environ.get(
        "K3PATCH_TARGET_VLLM_ROOT",
        "/opt/infernal-invocation/vllm/vllm",
    )
)
REGISTRY = ROOT / "model_executor/models/registry.py"
MARKER = "k3-remote-dflash2-metadata"
ANCHOR = '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),\n'
ADDITION = (
    '    "DFlash2DraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),  '
    "# k3-remote-dflash2-metadata\n"
)


def main() -> None:
    if not REGISTRY.is_file():
        raise SystemExit(f"target vLLM registry is missing: {REGISTRY}")
    source = REGISTRY.read_text()
    if MARKER in source:
        print(f"{REGISTRY}: DFlash2 metadata alias already registered")
        return
    if ANCHOR not in source:
        raise SystemExit(f"DFlashDraftModel registry anchor not found: {REGISTRY}")
    source = source.replace(ANCHOR, ANCHOR + ADDITION, 1)
    compile(source, str(REGISTRY), "exec")
    REGISTRY.write_text(source)
    print(f"{REGISTRY}: DFlash2 remote metadata alias registered")


if __name__ == "__main__":
    main()
