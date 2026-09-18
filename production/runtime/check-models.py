#!/usr/bin/env python3
"""Check checkpoint metadata and shard presence; never load weight tensors."""

import hashlib
import json
from pathlib import Path

lock = json.loads(Path("/recipe/source-lock.json").read_text())
for role, directory in [("target", "/model"), ("draft", "/draft-model")]:
    expected = lock["models"][role]
    root = Path(directory)
    config = root / "config.json"
    if not config.is_file():
        raise SystemExit(f"{role}: missing {config}; download the pinned checkpoint")
    if hashlib.sha256(config.read_bytes()).hexdigest() != expected["config_sha256"]:
        raise SystemExit(f"{role}: config.json differs from the pinned checkpoint")
    shards = list(root.glob("*.safetensors"))
    if (len(shards), sum(p.stat().st_size for p in shards)) != (
        expected["safetensors_files"],
        expected["safetensors_bytes"],
    ):
        raise SystemExit(f"{role}: incomplete or different safetensors payload")
    if role == "target":
        manifest = root / "qsrt-manifest.json"
        if (
            not manifest.is_file()
            or hashlib.sha256(manifest.read_bytes()).hexdigest()
            != expected["qsrt_manifest_sha256"]
        ):
            raise SystemExit("target: missing or different QSRT manifest")
        index = json.loads((root / "model.safetensors.index.json").read_text())
        missing = [
            name
            for name in set(index["weight_map"].values())
            if not (root / name).is_file()
        ]
        if missing:
            raise SystemExit(f"target: missing indexed shards: {missing}")
        for name in ["tokenizer_config.json", "preprocessor_config.json"]:
            if not (root / name).is_file():
                raise SystemExit(f"target: missing {name}")
    print(f"{role}: pinned metadata, shard count and byte count match")
print("Weight contents are not rehashed by this startup check.")
