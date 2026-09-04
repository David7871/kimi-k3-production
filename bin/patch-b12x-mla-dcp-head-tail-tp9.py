#!/usr/bin/env python3
"""Allow non-multiple-of-eight DCP-gathered query heads in B12X_MLA.

``_kernel_query_heads`` already rounds the effective head count up to the
eight-row kernel tile and the decode path already owns padded query/output
buffers sized from that rounded count (``dense_mla_padded_q/_output``), with
the pad rows zero-filled per batch and the output/LSE sliced back to
``effective_heads`` BEFORE the DCP LSE reduction. The only thing preventing
TP9/DCP9 from using the stock virtual-TP head padding (96 -> 99, local 11)
is a validation ``raise`` for ``dcp_size > 1`` — a host-side choice, not a
kernel limit. Removing it makes the gathered count 99 run as 104 kernel rows
with 5 masked tail rows, instead of forcing local 16 / 144 padded heads
(+50% MLA decode rows and +50% q/o projection padding).

Second hunk: in the DCP path the gathered query is already a view of
``dense_mla_padded_q``, so the padded-copy branch would perform an
overlapping self-copy, which torch rejects. Guard the copy with a
``data_ptr`` identity check; the tail zero-fill is kept unconditional.

TP8/DCP8 (96 % 8 == 0) and every DCP=1 path are byte-identical.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_B12X_MLA",
        "/opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/b12x_mla.py",
    )
)

TILE_MARKER = "kimi-k3-dcp-head-tile-tail"
COPY_MARKER = "kimi-k3-dcp-padded-self-copy-guard"

TILE_OLD = '''    effective_heads = local_heads * dcp_size
    if dcp_size > 1 and effective_heads % _B12X_QUERY_HEAD_TILE:
        raise ValueError(
            "B12X_MLA requires a multiple of eight query heads after DCP "
            f"gather, got local={local_heads}, DCP={dcp_size}, "
            f"effective={effective_heads}."
        )
    return (
'''

TILE_NEW = '''    effective_heads = local_heads * dcp_size
    # kimi-k3-dcp-head-tile-tail: a DCP-gathered head count that does not
    # close an eight-head tile is padded up to the tile, exactly like the
    # DCP=1 path. The decode path zero-fills the pad rows per batch and
    # slices output and LSE back to effective_heads before the DCP LSE
    # reduction, so pad rows never reach the combine.
    return (
'''

COPY_OLD = '''                padded_q[:, :effective_heads].copy_(q)
                padded_q[:, effective_heads:].zero_()
                q = padded_q
'''

COPY_NEW = '''                # kimi-k3-dcp-padded-self-copy-guard: after a DCP gather the
                # query is already a leading view of padded_q; an overlapping
                # self-copy is rejected by torch, and the data is in place.
                if q.data_ptr() != padded_q.data_ptr():
                    padded_q[:, :effective_heads].copy_(q)
                padded_q[:, effective_heads:].zero_()
                q = padded_q
'''


def apply(text: str, marker: str, old: str, new: str, what: str) -> str:
    if marker in text:
        print(f"b12x_mla already patched: {marker}")
        return text
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"ERROR: expected exactly one {what} anchor in {PATH}, found {count}"
        )
    print(f"b12x_mla patched: {marker}")
    return text.replace(old, new, 1)


def main() -> None:
    text = PATH.read_text()
    text = apply(text, TILE_MARKER, TILE_OLD, TILE_NEW, "head-tile")
    text = apply(text, COPY_MARKER, COPY_OLD, COPY_NEW, "padded-copy")
    compile(text, str(PATH), "exec")
    PATH.write_text(text)

    # Pure-math self-check of the padding arithmetic without importing torch:
    # extract _kernel_query_heads into an isolated namespace.
    ns: dict = {}
    start = text.index("def _kernel_query_heads")
    end = text.index("\ndef ", start + 1)
    fn_src = "_B12X_QUERY_HEAD_TILE = 8\n" + text[start:end]
    exec(compile(fn_src, "<self-check>", "exec"), ns)  # noqa: S102
    fn = ns["_kernel_query_heads"]
    expect = {
        (12, 8): 96,   # qualified TP8/DCP8: unchanged
        (11, 9): 104,  # TP9/DCP9 stock padding: 99 -> one masked tile
        (16, 9): 144,  # legacy TP9 over-padded layout still valid
        (11, 1): 16,   # DCP=1 keeps existing tile rounding
    }
    for (local, dcp), want in expect.items():
        got = fn(local, dcp)
        if got != want:
            raise SystemExit(
                f"self-check failed: heads({local},{dcp}) = {got}, want {want}"
            )
    print("b12x_mla head-tile self-check passed:", expect)


if __name__ == "__main__":
    main()
