#!/usr/bin/env python3
"""Fail-closed contract for the r36 QSRT B12X compatibility child."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import torch

EXPECTED_TREE = "6e7a050c63068d827ced00032215f6ea9e1d49467a75a341023cf11537d32988"
ROOTS = [
    Path("/opt/venv/lib/python3.12/site-packages/b12x"),
    Path("/opt/infernal-invocation/b12x/b12x"),
]


def tree_hash(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = 0
    size = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        files += 1
        size += len(payload)
    return digest.hexdigest(), files, size


hashes = []
for root in ROOTS:
    value, files, size = tree_hash(root)
    assert value == EXPECTED_TREE, (root, value)
    hashes.append({"root": str(root), "sha256": value, "files": files, "bytes": size})

from b12x.moe import fused_moe  # noqa: E402

signature = inspect.signature(fused_moe.plan_weights)
for name in ("qsrt_storage_format", "qsrt_profile"):
    assert name in signature.parameters, signature

plan = fused_moe.plan_weights(
    quant_modes="w4a16",
    source_format="qsrt_sqg_e4m3",
    activation="situ",
    params_dtype=torch.bfloat16,
    num_experts=896,
    hidden_size=7168,
    intermediate_size=384,
    w13_layout="w13",
    trellis_bits=2,
    trellis_tile_config=(128, 128, 128, 128),
    qsrt_storage_format="qsrt_atoms_v2",
    qsrt_profile="k2_coupled_h512_h128",
)
assert plan.qsrt_storage_format == "qsrt_atoms_v2"
assert plan.qsrt_profile == "k2_coupled_h512_h128"
assert plan.trellis_tile_config == (128, 128, 128, 128)

route_pack = ROOTS[0] / "moe/_shared/kernels/w4a16/route_pack.py"
route_text = route_pack.read_text()
assert "STABLE_ROUTE" in route_text

kquant = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/kquant_hybrid.py"
)
kquant_text = kquant.read_text()
assert 'qsrt_storage_format="qsrt_atoms_v2"' in kquant_text
assert "qsrt_profile=metadata_v2.profile" in kquant_text

print(
    json.dumps(
        {
            "status": "ok",
            "trees": hashes,
            "plan_signature": str(signature),
            "qsrt_storage_format": plan.qsrt_storage_format,
            "qsrt_profile": plan.qsrt_profile,
            "trellis_tile_config": plan.trellis_tile_config,
            "stable_route_pack": True,
        },
        indent=2,
        default=str,
        sort_keys=True,
    )
)
