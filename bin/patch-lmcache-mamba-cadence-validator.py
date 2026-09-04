#!/usr/bin/env python3
"""Validate LMCache mamba step alignment against the snapshot cadence.

``validate_mamba_step_alignment`` enforces that every mid-prefill scheduler
step advances exactly one recurrent-state block, so no snapshot boundary is
skipped (a skipped boundary is stored as null-block garbage under a valid
key — the 2026-08-21 poisoning class). It checks
``cache_config.block_size`` — the PHYSICAL attention block.

This fork's align mode explicitly supports a snapshot cadence coarser than
the physical block (``interface.py``: "permits a hybrid model to align
recurrent checkpoints with a coarser attention or DCP boundary while
retaining smaller physical attention pages"): ``cache_config.
mamba_block_size`` is the cadence, the Mamba group spec carries it, and the
scheduler's block-aligned split advances in cadence units. With cadence C =
k * physical_block, steps of C tokens end exactly on snapshot boundaries
and the invariant holds — but the validator still compares max batched
tokens against the physical block and refuses.

Validate against the cadence when it is a whole multiple of the physical
block; otherwise keep the physical-block check unchanged. Profiles where
cadence == block (every qualified 1536 profile) are byte-identical. The
TP9 fine4608 production ran cadence 4608 / block 1536 / pmu 1536 all night;
this patch admits the same geometry with cadence-sized steps.
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

MARKER = "kimi-k3-cadence-step-alignment"

OLD = '''    if getattr(vllm_config.cache_config, "mamba_cache_mode", "none") != "align":
        return
    block_size = vllm_config.cache_config.block_size
    max_batched = vllm_config.scheduler_config.max_num_batched_tokens
'''

NEW = '''    if getattr(vllm_config.cache_config, "mamba_cache_mode", "none") != "align":
        return
    block_size = vllm_config.cache_config.block_size
    # kimi-k3-cadence-step-alignment: align mode snapshots at scheduler-step
    # ends, and the scheduler splits mid-prefill steps on the RECURRENT
    # cadence (the Mamba group block, lcm'd in Scheduler.__init__), which
    # this fork allows to be a whole multiple of the physical attention
    # block. Steps of one cadence block end exactly on snapshot boundaries,
    # so the no-skipped-boundary invariant must be checked against the
    # cadence, not the physical page size. Cadences that are not a whole
    # multiple of the physical block keep the strict physical-block check.
    _cadence = getattr(vllm_config.cache_config, "mamba_block_size", None)
    if _cadence and _cadence % block_size == 0:
        block_size = _cadence
    max_batched = vllm_config.scheduler_config.max_num_batched_tokens
'''


def main() -> None:
    text = PATH.read_text()
    if MARKER in text:
        print(f"lmcache_mp_connector already patched: {MARKER}")
        return
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(
            f"ERROR: expected exactly one validator anchor in {PATH}, found {count}"
        )
    text = text.replace(OLD, NEW, 1)
    compile(text, str(PATH), "exec")
    PATH.write_text(text)
    print(f"lmcache_mp_connector patched: {MARKER}")


if __name__ == "__main__":
    main()
