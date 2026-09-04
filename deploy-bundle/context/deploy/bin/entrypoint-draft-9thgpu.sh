#!/usr/bin/env bash
# Draft entrypoint for the dedicated 9th GPU (SM 12.0): apply the grouped-op
# availability gate before launching the standalone DSpark runtime. All CLI
# arguments are forwarded unchanged.
set -euo pipefail

python3 /usr/local/bin/patch-dspark-standalone-grouped-op-gate.py
python3 /usr/local/bin/patch-dspark-reconnect-partial-window.py

# DFlash2 drafter support: registers DFlash2DraftModel and routes its selector
# through the existing --method dflash standalone protocol. Skipped when the
# patch is not mounted.
if [[ -f /usr/local/bin/patch-dspark-standalone-dflash2.py ]]; then
  python3 /usr/local/bin/patch-dspark-standalone-dflash2.py
fi

exec /opt/venv/bin/python -m vllm.entrypoints.k3_dspark_standalone "$@"
