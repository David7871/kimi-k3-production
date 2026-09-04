#!/usr/bin/env python3
"""Decouple Kimi-K3 KDA head padding from the MLA head padding.

The stock apply step copies the MLA attention padded head count into
``linear_attn_config["num_heads"]``. At TP9/DCP9 the MLA axis must pad to
144 heads (local 16) for the B12X_MLA DCP-gather rule, but KDA linear
attention has no DCP gather: it only needs ``num_heads % tp_size == 0``.
Inheriting 144 inflates each rank's recurrent (conv/KDA) state by 4/3,
which pushes the computed attention block size from 1536 to 2048 and breaks
the qualified fine-object cadence.

Pad KDA separately with local alignment 4: TP9 pads 96 -> 108 (local 12 —
the same local geometry as qualified TP8, keeping the fused KDA decode
kernel eligible and the mamba page size unchanged). TP sizes that divide 96
(TP8) produce padded == original == 96, so qualified behavior is unchanged.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_VIRTUAL_TP",
        "/opt/infernal-invocation/vllm/vllm/config/virtual_tp.py",
    )
)
MARKER = "kimi-k3-kda-separate-padding"

OLD = """    if _is_kimi_k3_config(model_config):
        linear_attn_config = getattr(
            model_config.hf_text_config, "linear_attn_config", None
        )
        if isinstance(linear_attn_config, dict) and "num_heads" in linear_attn_config:
            linear_attn_config["original_num_heads"] = linear_attn_config["num_heads"]
            linear_attn_config["num_heads"] = attention_axis["padded_size"]
"""

NEW = """    if _is_kimi_k3_config(model_config):
        linear_attn_config = getattr(
            model_config.hf_text_config, "linear_attn_config", None
        )
        if isinstance(linear_attn_config, dict) and "num_heads" in linear_attn_config:
            # kimi-k3-kda-separate-padding: KDA has no DCP head gather, so it
            # does not inherit the MLA eight-head alignment. Pad KDA heads to
            # the smallest multiple of TP with local alignment 4, keeping the
            # recurrent state (and therefore the mamba page size) minimal.
            kda_axis = _make_virtual_axis(
                int(linear_attn_config["num_heads"]),
                int(attention_axis["tp_size"]),
                4,
            )
            linear_attn_config["original_num_heads"] = linear_attn_config["num_heads"]
            linear_attn_config["num_heads"] = kda_axis["padded_size"]
"""


def main() -> None:
    text = PATH.read_text()
    if MARKER in text:
        print(f"virtual_tp already patched: {MARKER}")
        return
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(
            f"ERROR: expected exactly one KDA apply anchor in {PATH}, found {count}"
        )
    text = text.replace(OLD, NEW, 1)
    compile(text, str(PATH), "exec")
    PATH.write_text(text)
    print(f"virtual_tp patched: {MARKER}")


if __name__ == "__main__":
    main()
