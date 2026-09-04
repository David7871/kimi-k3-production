#!/usr/bin/env python3
"""Backport LMCache PR #4219 semantics to the issue-75 source tree.

Kimi-K3 is a hybrid MLA + recurrent/linear-attention model. The MLA cache is
replicated across TP ranks, but recurrent state is rank-sharded. Treating the
whole model as pure MLA makes the multiprocess adapter expose kv_tp_size=1 and
can restore rank 0's recurrent shard on every rank. The current issue-75 tree
predates LMCache PR #4219, so apply its `mla_only` classifier while retaining
this tree's DCP-aware `ParallelStrategy(use_mla=...)` API.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

ROOT = Path(
    os.environ.get(
        "K3_LMCACHE_SOURCE_ROOT",
        "/opt/infernal-invocation/lmcache/lmcache",
    )
)
UTILS = ROOT / "integration/vllm/utils.py"
CONNECTORS = [
    ROOT / "integration/vllm/lmcache_mp_connector.py",
    ROOT / "integration/vllm/lmcache_mp_connector_0180.py",
    ROOT / "integration/vllm/lmcache_mp_connector_0201.py",
]


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    count = text.count(old)
    if count == 0:
        return False
    if count != 1:
        raise SystemExit(f"{path}: expected one occurrence, found {count}: {old!r}")
    path.write_text(text.replace(old, new, 1))
    return True


def patch_utils() -> bool:
    text = UTILS.read_text()
    if "def mla_only(" in text:
        return False
    anchor = '''def mla_enabled(model_config: "ModelConfig") -> bool:
    return (
        hasattr(model_config, "use_mla")
        and isinstance(model_config.use_mla, bool)
        and model_config.use_mla
    )


'''
    addition = '''def _use_multiple_attentions(model_config: "ModelConfig") -> bool:
    """Return whether vLLM classifies the model as attention/recurrent hybrid."""
    return bool(getattr(model_config, "is_hybrid", False))


def mla_only(model_config: "ModelConfig") -> bool:
    """Return whether every cached tensor is replicated across TP ranks.

    Pure MLA state is replicated and can use LMCache's rank-sharing shortcut.
    Hybrid MLA + Mamba/KDA state is not: its recurrent tensors are rank-sharded.
    """
    return mla_enabled(model_config) and not _use_multiple_attentions(model_config)


'''
    if text.count(anchor) != 1:
        raise SystemExit(f"{UTILS}: mla_enabled anchor drifted")
    UTILS.write_text(text.replace(anchor, anchor + addition, 1))
    return True


def patch_connector(path: Path) -> bool:
    changed = False
    if path.name == "lmcache_mp_connector.py":
        changed |= replace_once(
            path,
            "from lmcache.integration.vllm.utils import mla_enabled, vllm_layout_hints",
            "from lmcache.integration.vllm.utils import mla_only, vllm_layout_hints",
        )
    else:
        changed |= replace_once(
            path,
            "from lmcache.integration.vllm.utils import mla_enabled",
            "from lmcache.integration.vllm.utils import mla_only",
        )
    if path.name == "lmcache_mp_connector_0180.py":
        changed |= replace_once(
            path,
            "use_mla = mla_enabled(vllm_config.model_config)",
            "use_mla = mla_only(vllm_config.model_config)",
        )
    else:
        changed |= replace_once(
            path,
            "use_mla=mla_enabled(vllm_config.model_config),",
            "use_mla=mla_only(vllm_config.model_config),",
        )
    return changed


def verify() -> None:
    paths = [UTILS, *CONNECTORS]
    for path in paths:
        ast.parse(path.read_text(), filename=str(path))
    utils_text = UTILS.read_text()
    if utils_text.count("def mla_only(") != 1:
        raise SystemExit(f"{UTILS}: mla_only contract missing or duplicated")
    for path in CONNECTORS:
        text = path.read_text()
        unsafe_forms = (
            "use_mla=mla_enabled(vllm_config.model_config)",
            "use_mla = mla_enabled(vllm_config.model_config)",
        )
        if any(form in text for form in unsafe_forms):
            raise SystemExit(f"{path}: hybrid-unsafe mla_enabled strategy remains")
        safe_forms = (
            "use_mla=mla_only(vllm_config.model_config)",
            "use_mla = mla_only(vllm_config.model_config)",
        )
        if not any(form in text for form in safe_forms):
            raise SystemExit(f"{path}: mla_only strategy wiring missing")


changed = [str(UTILS)] if patch_utils() else []
for connector in CONNECTORS:
    if patch_connector(connector):
        changed.append(str(connector))
verify()
print(
    "LMCache PR #4219 hybrid MLA classifier: "
    + ("patched " + ", ".join(changed) if changed else "already present")
)
