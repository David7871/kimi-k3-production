#!/usr/bin/env python3
"""Check this image against the captured production runtime without GPUs."""

import hashlib
import json
from pathlib import Path

recipe = Path("/recipe")
lock = json.loads((recipe / "source-lock.json").read_text())
root = Path("/opt/infernal-invocation")
expected = {}
for line in (recipe / "runtime-manifest.sha256").read_text().splitlines():
    digest, relative = line.split("  ", 1)
    expected[str(root / relative)] = digest
expected.update(lock["runtime_files"])
expected.update(lock["native_extensions"])
failures = []
for filename, digest in expected.items():
    path = Path(filename)
    if not path.is_file():
        failures.append({"path": filename, "error": "missing"})
        continue
    with path.open("rb") as data:
        actual = hashlib.file_digest(data, "sha256").hexdigest()
    if actual != digest:
        failures.append({"path": filename, "expected": digest, "actual": actual})
extra = []
for package in ["vllm/vllm", "b12x/b12x", "lmcache/lmcache"]:
    for path in (root / package).rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        if str(path) not in expected and path.suffix not in {".md", ".rst", ".txt"}:
            extra.append(str(path))
report = {
    "status": "qualified" if not failures and not extra else "unsupported",
    "check": "production runtime file identity",
    "checked_files": len(expected),
    "failures": failures,
    "extra_runtime_files": extra,
}
print(json.dumps(report, indent=2))
raise SystemExit(bool(failures or extra))
