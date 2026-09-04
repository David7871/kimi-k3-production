#!/usr/bin/env python3
"""Let a prefix reconnect start from whatever the projected cache retains.

Both reconnect sites compute the restore origin purely from the rolling
window::

    restore_start = block_align_down(max(0, prefix_end - window_size))

For every conversation shorter than the window (372,736 tokens here) that
evaluates to 0, so the source must retain projected context all the way back
to position 0.  A request that was itself cold-bootstrapped from a
cache-restored prefix starts its context at that prefix position instead, so
``has_range(0, prefix_end)`` is false and the reconnect is refused -- the
follow-up turn cold-bootstraps again, and every later turn inherits the same
condition.  In production that path never recovers: 100% of turns bootstrap.

Clamp the restore origin up to the cache's retained start (block-aligned up,
so the first restored block is whole).  The draft then resumes with the
history the source actually holds, which is strictly more than a fresh
bootstrap at ``prefix_end`` would give it, and the target verifies every
proposal, so acceptance can only improve.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_DSPARK_RPC",
        "/opt/k3-dspark-3090/source/vllm/entrypoints/k3_dspark_rpc.py",
    )
)
MARKER = "reconnect-partial-window"

RESTORE_OLD = '''        restore_start = max(0, prefix_end - self.allocator.window_size)
        restore_start = (
            restore_start // self.allocator.block_size * self.allocator.block_size
        )
        if not context_cache.has_range(restore_start, prefix_end):
'''

RESTORE_NEW = '''        restore_start = max(0, prefix_end - self.allocator.window_size)
        restore_start = (
            restore_start // self.allocator.block_size * self.allocator.block_size
        )
        # reconnect-partial-window: a source that was cold-bootstrapped from a
        # cache-restored prefix retains context only from that position on.
        # Start where the cache actually begins (rounded up to a whole block)
        # instead of refusing the reconnect outright.
        if restore_start < context_cache.start_position:
            block_size = self.allocator.block_size
            restore_start = (
                (context_cache.start_position + block_size - 1)
                // block_size
                * block_size
            )
        if not context_cache.has_range(restore_start, prefix_end):
'''

RECONNECT_OLD = '''            context_cache = state.context_cache
            restore_start = max(0, prefix_end - self.allocator.window_size)
            restore_start = (
                restore_start // self.allocator.block_size * self.allocator.block_size
            )
            if (
'''

RECONNECT_NEW = '''            context_cache = state.context_cache
            restore_start = max(0, prefix_end - self.allocator.window_size)
            restore_start = (
                restore_start // self.allocator.block_size * self.allocator.block_size
            )
            # reconnect-partial-window: mirror the clamp applied by
            # _restore_projected_context so this guard accepts the same range.
            if context_cache is not None and restore_start < (
                context_cache.start_position
            ):
                block_size = self.allocator.block_size
                restore_start = (
                    (context_cache.start_position + block_size - 1)
                    // block_size
                    * block_size
                )
            if (
'''


def _apply(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if MARKER in text and old not in text:
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
    _apply(PATH, RESTORE_OLD, RESTORE_NEW, "_restore_projected_context")
    _apply(PATH, RECONNECT_OLD, RECONNECT_NEW, "reconnect")


if __name__ == "__main__":
    main()
