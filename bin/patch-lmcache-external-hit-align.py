#!/usr/bin/env python3
"""Floor LMCache external hits to the scheduler hit-alignment grid.

Local prefix-cache hits in the hybrid (KDA + MLA) coordinator only land on
multiples of ``scheduler_block_size`` (12,288 here), and the coordinator also
floors block caching to that grid.  An LMCache restore lands on the 3,072
chunk grid, so 3 out of 4 restores put the request's only recurrent-state
block just ABOVE the cache-blocks floor: the state is never hashed, every
following turn misses locally, re-retrieves the same head from LMCache, and
recomputes the tail — forever.

Flooring the admitted external hit to the same grid makes the restore
boundary a legal local-hit boundary: the restored KDA state gets cached at
load completion and the next turn resumes locally.

Gated by LMCACHE_EXTERNAL_HIT_ALIGN (tokens); unset or 0 disables (stock
behavior).
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_LMCACHE_MP_CONNECTOR",
        "/opt/infernal-invocation/vllm/vllm/distributed/kv_transfer/"
        "kv_connector/v1/lmcache_mp_connector.py",
    )
)
MARKER = "kimi-k3-external-hit-align"

OLD = '''        assert (
            ret % (self.scheduler_adapter.num_blocks_per_chunk() * self.vllm_block_size)
            == 0
        )

        # Update num stored blocks for the tracker
'''

NEW = '''        assert (
            ret % (self.scheduler_adapter.num_blocks_per_chunk() * self.vllm_block_size)
            == 0
        )

        # kimi-k3-external-hit-align: admit external hits only at the local
        # hit-alignment grid so the restored boundary (and its recurrent
        # state) is cacheable and later turns can resume locally.
        align = getattr(self, "_external_hit_align", None)
        if align is None:
            align = int(os.environ.get("LMCACHE_EXTERNAL_HIT_ALIGN", "0") or 0)
            self._external_hit_align = align
        if align > 0:
            ret = ret // align * align
            if ret == 0:
                return 0, False

        # Update num stored blocks for the tracker
'''


def main() -> None:
    text = PATH.read_text()
    if MARKER in text:
        print(f"lmcache_mp_connector already patched: {MARKER}")
        return
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(f"ERROR: expected exactly one anchor, found {count}")
    text = text.replace(OLD, NEW, 1)
    compile(text, str(PATH), "exec")
    PATH.write_text(text)
    print(f"lmcache_mp_connector patched: {MARKER}")


if __name__ == "__main__":
    main()
