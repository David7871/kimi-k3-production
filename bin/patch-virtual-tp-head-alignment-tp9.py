#!/usr/bin/env python3
"""Kimi-K3 TP9 virtual-TP fixes (DSpark MXFP8 dense alignment only).

RETIRED HUNK (2026-08-28): the kimi-k3-dcp-head-alignment hunk that raised
MLA local head alignment to 8 (local 16 / padded 144 at TP9/DCP9) is no
longer applied. patch-b12x-mla-dcp-head-tail-tp9.py teaches B12X_MLA to pad
a non-multiple-of-eight gathered head count to the kernel tile (99 -> 104
rows, 5 masked tail rows), so the stock planner padding 96 -> 99 (local 11)
loads and runs. This removes +50% MLA decode rows and +50% q/o projection
padding, and restores the fused KDA decode eligibility concern to moot.

The DSpark draft MXFP8 dense-axis alignment hunk below is still required.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_VIRTUAL_TP",
        "/opt/infernal-invocation/vllm/vllm/config/virtual_tp.py",
    )
)
# The DSpark draft's dense MLP is quantized MXFP8 at load time through the
# speculative config, so its checkpoint config carries no dense_format hint
# and the stock plan aligns the dense axis to 1. TP9 then produces
# input_size_per_partition 14337/9 = 1593, violating MXFP8's 32-divisibility.
# Raise the draft dense alignment floor to 32. TP8 keeps 14336 (1792 local,
# already 32-aligned), so qualified behavior is unchanged.
DSPARK_OLD = """        "dense_intermediate_size": _make_virtual_axis(
            _require_int_attr(text_config, "intermediate_size"),
            tp_size,
            _get_dense_linear_local_alignment(model_config),
        ),
        "vocab_size": _make_virtual_vocab_axis(
            _require_int_attr(text_config, "vocab_size"), tp_size
        ),
    }
"""

DSPARK_NEW = """        "dense_intermediate_size": _make_virtual_axis(
            _require_int_attr(text_config, "intermediate_size"),
            tp_size,
            # kimi-k3-dspark-mxfp8-dense-alignment: the draft is quantized
            # MXFP8 via the speculative config, which requires every local
            # input extent to close 32-channel blocks.
            max(32, _get_dense_linear_local_alignment(model_config)),
        ),
        "vocab_size": _make_virtual_vocab_axis(
            _require_int_attr(text_config, "vocab_size"), tp_size
        ),
    }
"""


def main() -> None:
    text = PATH.read_text()
    dspark_marker = "kimi-k3-dspark-mxfp8-dense-alignment"
    if dspark_marker in text:
        print(f"virtual_tp already patched: {dspark_marker}")
    else:
        count = text.count(DSPARK_OLD)
        if count != 1:
            raise SystemExit(
                f"ERROR: expected exactly one dspark dense anchor in {PATH}, "
                f"found {count}"
            )
        text = text.replace(DSPARK_OLD, DSPARK_NEW, 1)
        compile(text, str(PATH), "exec")
        PATH.write_text(text)
        print(f"virtual_tp patched: {dspark_marker}")


if __name__ == "__main__":
    main()
