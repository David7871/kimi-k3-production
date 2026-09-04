#!/usr/bin/env python3
"""Register the DFlash draft with B12X virtual-TP padding (TP9 support).

``DFlashDraftModel`` is a Qwen3-shaped six-layer SWA draft (32 query heads,
8 KV heads, dense 14336). Colocated speculative decoding shards it with the
target's TP group, and TP9 fails SpeculativeConfig validation ("heads (32)
must be divisible by tensor parallel size (9)") because no virtual-TP plan
covers the architecture — unlike the K3 DSpark draft, which has a dedicated
planner. Add a coupled-GQA plan: q 32 -> 36 (local 4), KV 8 -> 9 (local 1,
integer 4:1 ratio preserved), dense 14336 and vocab padded like the DSpark
plan. Zero-filled tail shards follow the standard B12X virtual-TP loader
contract; the four q heads riding the zero KV head produce zeros that the
zero o_proj columns discard.

Even TP sizes that divide 32/8 produce padded == original, so TP8 remote
and standalone DFlash behavior is unchanged.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_VIRTUAL_TP",
        "/opt/infernal-invocation/vllm/vllm/config/virtual_tp.py",
    )
)

MARKER = "kimi-k3-dflash-draft-virtual-tp"

# 1. Plan builder + apply + detection, appended after the DSpark builder.
BUILDER_OLD = '''def _build_gqa_gdn_moe_virtual_tp_plan(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> dict[str, dict[str, int] | str]:
    """Build a plan for models combining GQA, GDN, and routed experts.
'''

BUILDER_NEW = '''_KIMI_K3_DFLASH_PROFILE = "kimi-k3-dflash"


def _is_dflash_draft_config(model_config: ModelConfig) -> bool:
    # kimi-k3-dflash-draft-virtual-tp
    for config in _iter_virtual_tp_configs(model_config):
        architectures = getattr(config, "architectures", None) or ()
        if "DFlashDraftModel" in architectures:
            return True
    return False


def _build_dflash_draft_virtual_tp_plan(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> dict[str, dict[str, int] | str]:
    """Coupled-GQA zero-tail padding for the Qwen3-shaped DFlash draft."""
    text_config = model_config.hf_text_config
    tp_size = parallel_config.tensor_parallel_size
    attention_axis, kv_axis = _make_coupled_virtual_axes(
        _require_int_attr(text_config, "num_attention_heads"),
        _require_int_attr(text_config, "num_key_value_heads"),
        tp_size,
        ratio_key="q_heads_per_kv",
        allow_secondary_replication=True,
    )
    return {
        "sharding": _VIRTUAL_TP_PLAN_KIND_B12X_PADDED,
        "model_type": _KIMI_K3_DFLASH_PROFILE,
        "attention_heads": attention_axis,
        "kv_heads": kv_axis,
        "dense_intermediate_size": _make_virtual_axis(
            _require_int_attr(text_config, "intermediate_size"),
            tp_size,
            _get_dense_linear_local_alignment(model_config),
        ),
        "vocab_size": _make_virtual_vocab_axis(
            _require_int_attr(text_config, "vocab_size"), tp_size
        ),
    }


def _apply_dflash_draft_virtual_tp_plan(
    model_config: ModelConfig,
    plan: dict[str, dict[str, int] | str],
) -> None:
    configs = tuple(_iter_virtual_tp_configs(model_config))
    for axis_name, attr in (
        ("attention_heads", "num_attention_heads"),
        ("kv_heads", "num_key_value_heads"),
        ("dense_intermediate_size", "intermediate_size"),
    ):
        _apply_virtual_axis_to_config_attr(configs, plan, axis_name, attr)
    for config in configs:
        setattr(config, VIRTUAL_TP_PLAN_ATTR, plan)

    model_config.model_arch_config = model_config.get_model_arch_config()

    changes = []
    for axis_name, label in (
        ("attention_heads", "attention heads"),
        ("kv_heads", "KV heads"),
        ("dense_intermediate_size", "dense intermediate size"),
        ("vocab_size", "vocab storage size"),
    ):
        axis = _require_axis(plan, axis_name)
        if axis["original_size"] != axis["padded_size"]:
            changes.append(f"{label} {axis['original_size']} -> {axis['padded_size']}")
    logger.warning(
        "Automatically enabled B12X virtual TP padding for the DFlash "
        "draft: %s.",
        ", ".join(changes),
    )


def _build_gqa_gdn_moe_virtual_tp_plan(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> dict[str, dict[str, int] | str]:
    """Build a plan for models combining GQA, GDN, and routed experts.
'''

# 2. Build dispatch.
DISPATCH_OLD = '''    if _is_kimi_k3_dspark_config(model_config):
        return _build_kimi_k3_dspark_virtual_tp_plan(model_config, parallel_config)
    if _is_minimax_m3_config(model_config):
'''

DISPATCH_NEW = '''    if _is_kimi_k3_dspark_config(model_config):
        return _build_kimi_k3_dspark_virtual_tp_plan(model_config, parallel_config)
    if _is_dflash_draft_config(model_config):
        return _build_dflash_draft_virtual_tp_plan(model_config, parallel_config)
    if _is_minimax_m3_config(model_config):
'''

# 3. Apply dispatch.
APPLY_OLD = '''    if plan.get("model_type") == _KIMI_K3_DSPARK_PROFILE:
        _apply_kimi_k3_dspark_virtual_tp_plan(model_config, plan)
        return
'''

APPLY_NEW = '''    if plan.get("model_type") == _KIMI_K3_DSPARK_PROFILE:
        _apply_kimi_k3_dspark_virtual_tp_plan(model_config, plan)
        return
    if plan.get("model_type") == _KIMI_K3_DFLASH_PROFILE:
        _apply_dflash_draft_virtual_tp_plan(model_config, plan)
        return
'''

# 4. Supported-config gate.
SUPPORTED_OLD = '''        _get_virtual_tp_profile(model_config) == _GQA_GDN_MOE_PROFILE
        or _is_kimi_k3_dspark_config(model_config)
'''

SUPPORTED_NEW = '''        _get_virtual_tp_profile(model_config) == _GQA_GDN_MOE_PROFILE
        or _is_kimi_k3_dspark_config(model_config)
        or _is_dflash_draft_config(model_config)
'''


SPECULATIVE_PATH = Path(
    os.environ.get(
        "K3PATCH_SPECULATIVE",
        "/opt/infernal-invocation/vllm/vllm/config/speculative.py",
    )
)

# 5. SpeculativeConfig only applies draft virtual-TP for mtp/dspark; dflash
# returns early and verify_with_parallel_config sees the raw 32 heads.
METHOD_GATE_OLD = """        if (
            self.method not in ("mtp", "dspark")
            or self.draft_model_config is None
"""

METHOD_GATE_NEW = """        if (
            self.method not in ("mtp", "dspark", "dflash")
            or self.draft_model_config is None
"""


def apply(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"ERROR: expected exactly one {what} anchor in {PATH}, found {count}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    text = PATH.read_text()
    if MARKER in text:
        print(f"virtual_tp already patched: {MARKER}")
    else:
        text = apply(text, BUILDER_OLD, BUILDER_NEW, "builder")
        text = apply(text, DISPATCH_OLD, DISPATCH_NEW, "build-dispatch")
        text = apply(text, APPLY_OLD, APPLY_NEW, "apply-dispatch")
        text = apply(text, SUPPORTED_OLD, SUPPORTED_NEW, "supported-gate")
        compile(text, str(PATH), "exec")
        PATH.write_text(text)
        print(f"virtual_tp patched: {MARKER}")

    spec = SPECULATIVE_PATH.read_text()
    if METHOD_GATE_NEW in spec:
        print("speculative.py already patched: dflash draft virtual-TP gate")
    else:
        count = spec.count(METHOD_GATE_OLD)
        if count != 1:
            raise SystemExit(
                "ERROR: expected exactly one method-gate anchor in "
                f"{SPECULATIVE_PATH}, found {count}"
            )
        spec = spec.replace(METHOD_GATE_OLD, METHOD_GATE_NEW, 1)
        compile(spec, str(SPECULATIVE_PATH), "exec")
        SPECULATIVE_PATH.write_text(spec)
        print("speculative.py patched: dflash draft virtual-TP gate")


if __name__ == "__main__":
    main()
