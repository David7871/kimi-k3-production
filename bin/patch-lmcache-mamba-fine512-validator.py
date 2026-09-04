#!/usr/bin/env python3
"""Validate one-step fine recurrent cadence against a larger physical page.

The deployed vLLM keeps Kimi-K3's attention page at 1,536 tokens while an
explicit ``--mamba-block-size`` defines the logical recurrent checkpoint
cadence.  The existing coarse-cadence patch accepts only a cadence that is a
multiple of the physical page.  A 512-token cadence is the inverse compatible
case: 1,536 is exactly three 512-token checkpoints.

This profile also caps ``max_num_batched_tokens`` at 520 and
``max_num_scheduled_tokens`` at 512.  The scheduler therefore advances at most
one logical recurrent checkpoint per request, while the padded physical page
layout remains unchanged.  Accept a cadence when either size divides the
other; reject non-integral geometry exactly as before.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_LMCACHE_MP",
        "/opt/infernal-invocation/lmcache/lmcache/integration/vllm/"
        "lmcache_mp_connector.py",
    )
)

MARKER = "kimi-k3-integral-fine-cadence-step-alignment"

OLD = '''    if getattr(vllm_config.cache_config, "mamba_cache_mode", "none") != "align":
        return
    block_size = vllm_config.cache_config.block_size
    max_batched = vllm_config.scheduler_config.max_num_batched_tokens
'''

NEW = '''    if getattr(vllm_config.cache_config, "mamba_cache_mode", "none") != "align":
        return
    block_size = vllm_config.cache_config.block_size
    # kimi-k3-integral-fine-cadence-step-alignment: an explicit recurrent
    # cadence may be either an integral multiple OR divisor of the physical
    # attention page.  vLLM schedules on the recurrent MambaSpec cadence while
    # the state page stays allocation-compatible through physical padding.
    # Non-integral pairs retain the conservative physical-page validator.
    _cadence = getattr(vllm_config.cache_config, "mamba_block_size", None)
    if _cadence and (
        _cadence % block_size == 0 or block_size % _cadence == 0
    ):
        block_size = _cadence
    max_batched = vllm_config.scheduler_config.max_num_batched_tokens
'''

COARSE = '''    _cadence = getattr(vllm_config.cache_config, "mamba_block_size", None)
    if _cadence and _cadence % block_size == 0:
        block_size = _cadence
'''

FINE = '''    # kimi-k3-integral-fine-cadence-step-alignment
    _cadence = getattr(vllm_config.cache_config, "mamba_block_size", None)
    if _cadence and (
        _cadence % block_size == 0 or block_size % _cadence == 0
    ):
        block_size = _cadence
'''


def main() -> None:
    text = PATH.read_text()
    if MARKER in text:
        print(f"lmcache_mp_connector already patched: {MARKER}")
        return
    original_count = text.count(OLD)
    coarse_count = text.count(COARSE)
    if original_count == 1 and coarse_count == 0:
        text = text.replace(OLD, NEW, 1)
    elif original_count == 0 and coarse_count == 1:
        # The qualified image already contains the coarse-cadence patch.
        # Upgrade only its cadence predicate and retain the surrounding
        # validator text byte-for-byte.
        text = text.replace(COARSE, FINE, 1)
    else:
        raise SystemExit(
            "ERROR: expected exactly one original or coarse validator anchor "
            f"in {PATH}, found original={original_count} coarse={coarse_count}"
        )
    compile(text, str(PATH), "exec")
    PATH.write_text(text)
    print(f"lmcache_mp_connector patched: {MARKER}")


if __name__ == "__main__":
    main()
