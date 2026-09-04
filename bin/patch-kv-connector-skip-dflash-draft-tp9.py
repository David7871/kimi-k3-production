#!/usr/bin/env python3
"""Exclude colocated DFlash draft KV layers from KV-connector registration.

With colocated DFlash the draft's six SWA layers join the unified KV cache,
and page-size parity padding gives their cache tensors an interior-padded
stride. LMCache's fine-object format detection correctly rejects that layout
(`attempt_permute_to_contiguous_view: interior-dim padding is not
supported`), killing engine init.

Draft KV must not be cached anyway: it is ephemeral speculation state that
the DFlash proposer rebuilds from the target's aux hidden states, and the
remote-DFlash production never exposed it to LMCache either (the draft
lived in a separate process). Filter attention layers whose module class
name starts with ``DFlash`` out of the dict passed to
``register_kv_caches``; after an LMCache restore the draft simply re-warms
its 4096-token window while block rejection sampling preserves output
correctness. Non-DFlash stacks register an unchanged dict.
"""

import os
from pathlib import Path

PATH = Path(
    os.environ.get(
        "K3PATCH_KV_CONNECTOR",
        "/opt/infernal-invocation/vllm/vllm/v1/worker/gpu/kv_connector.py",
    )
)

MARKER = "kimi-k3-skip-dflash-draft-kv"

OLD = '''        self.kv_connector = get_kv_transfer_group()
        # Register kv caches with KV Connector if applicable.
        # TODO: support cross_layers_kv_cache
        # (see https://github.com/vllm-project/vllm/pull/27743)
        self.kv_connector.register_kv_caches(kv_caches_dict)
'''

NEW = '''        self.kv_connector = get_kv_transfer_group()
        # Register kv caches with KV Connector if applicable.
        # TODO: support cross_layers_kv_cache
        # (see https://github.com/vllm-project/vllm/pull/27743)
        # kimi-k3-skip-dflash-draft-kv: colocated DFlash draft SWA layers
        # carry interior-padded cache views that the LMCache fine-object
        # format rejects, and draft KV is ephemeral speculation state with
        # no restore value (the proposer rebuilds context K/V from target
        # aux hidden states). Keep them out of connector registration.
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )

        draft_kv_names = {
            name
            for name, layer in get_layers_from_vllm_config(
                vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).items()
            if type(layer).__name__.startswith("DFlash")
        }
        registered_kv_caches = {
            name: tensor
            for name, tensor in kv_caches_dict.items()
            if name not in draft_kv_names
        }
        skipped = sorted(set(kv_caches_dict) - set(registered_kv_caches))
        if skipped:
            print(
                "kv_connector: skipping DFlash draft KV layers for the KV "
                f"connector: {skipped}",
                flush=True,
            )
        self.kv_connector.register_kv_caches(registered_kv_caches)
'''


LMCACHE_PATH = Path(
    os.environ.get(
        "K3PATCH_LMCACHE_MP",
        "/opt/infernal-invocation/lmcache/lmcache/integration/vllm/"
        "lmcache_mp_connector.py",
    )
)

# The connector (both scheduler and worker roles) walks
# kv_cache_config.kv_cache_groups for group edits, group-info creation, and
# object planning; groups listing the excluded draft layers would KeyError
# against the filtered tensor dict. Prune whole DFlash draft groups from the
# stored config right after it is captured. The draft group is appended
# after the target groups, so earlier group indices are unchanged.
LM_OLD = """        super().__init__(vllm_config, role, kv_cache_config)

        # Fail fast, before the server handshake below.
        validate_mamba_step_alignment(vllm_config)
"""

LM_NEW = """        super().__init__(vllm_config, role, kv_cache_config)

        # kimi-k3-skip-dflash-draft-kv: colocated DFlash draft SWA layers
        # are excluded from KV-connector registration (ephemeral speculation
        # state with interior-padded cache views). Prune their groups from
        # this role's config view so group walks, edits, and object planning
        # never reference them.
        _spec_cfg = getattr(vllm_config, "speculative_config", None)
        _cfg = getattr(self, "_kv_cache_config", None)
        if (
            _cfg is not None
            and _spec_cfg is not None
            and getattr(_spec_cfg, "method", None) == "dflash"
        ):
            import dataclasses as _dataclasses

            _kept = [
                _g
                for _g in _cfg.kv_cache_groups
                if not (
                    _g.layer_names
                    and all(
                        _n.startswith("dflash_model.") for _n in _g.layer_names
                    )
                )
            ]
            if len(_kept) != len(_cfg.kv_cache_groups):
                logger.info(
                    "Pruned %d DFlash draft KV cache group(s) from the "
                    "LMCache connector view.",
                    len(_cfg.kv_cache_groups) - len(_kept),
                )
                self._kv_cache_config = _dataclasses.replace(
                    _cfg, kv_cache_groups=_kept
                )

        # Fail fast, before the server handshake below.
        validate_mamba_step_alignment(vllm_config)
"""


def main() -> None:
    text = PATH.read_text()
    if MARKER in text:
        print(f"kv_connector already patched: {MARKER}")
    else:
        count = text.count(OLD)
        if count != 1:
            raise SystemExit(
                f"ERROR: expected exactly one register anchor in {PATH}, found {count}"
            )
        text = text.replace(OLD, NEW, 1)
        compile(text, str(PATH), "exec")
        PATH.write_text(text)
        print(f"kv_connector patched: {MARKER}")

    lm = LMCACHE_PATH.read_text()
    if "kimi-k3-skip-dflash-draft-kv" in lm:
        print("lmcache_mp_connector already patched: draft group prune")
    else:
        count = lm.count(LM_OLD)
        if count != 1:
            raise SystemExit(
                "ERROR: expected exactly one connector-init anchor in "
                f"{LMCACHE_PATH}, found {count}"
            )
        lm = lm.replace(LM_OLD, LM_NEW, 1)
        compile(lm, str(LMCACHE_PATH), "exec")
        LMCACHE_PATH.write_text(lm)
        print("lmcache_mp_connector patched: draft group prune")


if __name__ == "__main__":
    main()
