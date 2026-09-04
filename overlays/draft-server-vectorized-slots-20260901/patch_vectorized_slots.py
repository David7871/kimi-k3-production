#!/usr/bin/env python3
"""Vectorize the per-token cache-slot computation in the draft server's
`_append_context` (marker: kimi-k3-draft-vectorized-slots).

`cache_slot(state, position)` is integer arithmetic
(`(1 + slot*bpr + (pos // bs) % bpr) * bs + pos % bs`); the server evaluated
it in a 1,536-iteration Python generator under the server lock for every
context ingest. The same integers are now produced with torch int64 ops on
the CPU positions tensor, and the slot mapping is built once instead of from
a Python list.
"""
import sys, py_compile
from pathlib import Path
path = Path(sys.argv[1]); src = path.read_text()
MARKER = "kimi-k3-draft-vectorized-slots"
if MARKER in src:
    print("already patched"); sys.exit(0)
old = '''                state.committed_end = int(req_positions[-1]) + 1
                slots.extend(
                    self.allocator.cache_slot(state, int(position))
                    for position in req_positions
                )
            offset += count
        slot_mapping = torch.tensor(slots, dtype=torch.int64, device=self.device)
'''
new = f'''                state.committed_end = int(req_positions[-1]) + 1
                # {MARKER}: same integers as allocator.cache_slot, vectorized.
                bs = self.allocator.block_size
                bpr = self.allocator.blocks_per_request
                phys = 1 + state.slot * bpr + (req_positions // bs) % bpr
                slots.append(phys * bs + req_positions % bs)
            offset += count
        slot_mapping = (
            torch.cat(slots).to(self.device, non_blocking=True)
            if slots
            else torch.empty(0, dtype=torch.int64, device=self.device)
        )
'''
assert src.count(old) == 1, "anchor"
src = src.replace(old, new)
old2 = "        slots: list[int] = []\n        offset = 0\n        for state, count in zip(states, context_counts):\n            req_positions = positions_cpu[offset : offset + count]\n"
assert src.count(old2) == 1, "anchor2"
src = src.replace(old2, "        slots: list[torch.Tensor] = []  # " + MARKER + "\n        offset = 0\n        for state, count in zip(states, context_counts):\n            req_positions = positions_cpu[offset : offset + count]\n")
path.write_text(src); py_compile.compile(str(path), doraise=True); print("patched + compiled:", path)
