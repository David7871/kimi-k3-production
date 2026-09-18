#!/usr/bin/env python3
"""Validate image identity and the rendered launch command without GPU access."""

import json
import shlex
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
compose = [
    "docker",
    "compose",
    "-p",
    "kimi-k3-qsrt",
    "--env-file",
    str(root / ".env"),
    "-f",
    str(root / "compose.yaml"),
]
config = json.loads(subprocess.check_output(compose + ["config", "--format", "json"]))
service = config["services"]["kimi"]
environment = service["environment"]
image = service["image"]
subprocess.run(
    [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "/opt/venv/bin/python",
        image,
        "/recipe/verify-image.py",
    ],
    check=True,
)
command = ["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/bash"]
for key, value in environment.items():
    command += ["-e", f"{key}={value}"]
# Avoid writing to any host cache while printing the command.
command += [
    "-e",
    "VLLM_PROFILER_DIR=/tmp/recipe-check",
    image,
    "/runtime/entrypoint.sh",
    "--print-command",
]
arguments = shlex.split(subprocess.check_output(command, text=True))


def value(flag):
    return arguments[arguments.index(flag) + 1]


spec = json.loads(value("--speculative-config"))
batch, scheduled = map(
    int, [value("--max-num-batched-tokens"), value("--max-num-scheduled-tokens")]
)
block, seqs = map(int, [value("--mamba-block-size"), value("--max-num-seqs")])
assert (value("--tensor-parallel-size"), value("--decode-context-parallel-size")) == (
    "9",
    "9",
)
assert (block, seqs, spec["num_speculative_tokens"]) == (4608, 4, 3)
assert batch == scheduled == 4624
assert scheduled >= block + (seqs - 1) * (spec["num_speculative_tokens"] + 1)
assert value("--kv-cache-dtype") == "fp8_ds_mla"
assert value("--kv-cache-memory-bytes") == "3435973836"
assert spec["kv_cache_dtype"] == "fp8" and spec["model"] == "/draft-model"
assert int(environment["LMCACHE_RESTORE_PIN_LIMIT_CHUNKS"]) == 4
assert int(environment["LMCACHE_RESTORE_WINDOW_CHUNKS"]) == 4
print(
    json.dumps(
        {
            "status": "qualified",
            "check": "CPU launch validation",
            "image": image,
            "tp": 9,
            "dcp": 9,
            "token_budget": scheduled,
            "required_mixed_budget": block
            + (seqs - 1) * (spec["num_speculative_tokens"] + 1),
            "model_loaded": False,
        },
        indent=2,
    )
)
