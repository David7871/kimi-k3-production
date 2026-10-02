#!/usr/bin/env bash
# rtx-02 (worker) container entrypoint: join the rtx-01 Ray cluster.
set -euo pipefail

rm -rf /tmp/ray

echo "[k3-qsrt-worker] joining ray cluster at 10.10.10.1:6379"
exec /opt/venv/bin/ray start --address=10.10.10.1:6379 \
  --node-ip-address=10.10.10.2 --disable-usage-stats --block
