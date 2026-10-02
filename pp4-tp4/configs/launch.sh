#!/usr/bin/env bash
# Boot one saved K3 PP4xTP4 config on rtx-01 (head) + rtx-02 (worker) and run the decode bench.
# Run ON rtx-01, after restore.sh has installed the lane snapshot.
# Usage: bash launch.sh <baseline|moet|moet-off|moet-off2|nvfp4-kv|rebalance-kv> [concurrencies, default 16,256] [tag]
# Compose variants by setting the env vars below yourself, e.g. MOET=1 for moet+rebalance.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CFG=${1:?config}; CONC=${2:-16,256}; TAG=${3:-cfg-$CFG-$(date +%m%d-%H%M)}
ENV=$(cat "$HERE/baseline/int6.envstr")
IMG=kimi-k3-up-ray:local; ARM=/data/models/k3-run/scripts/k3-arm.sh
case $CFG in
  baseline) ;;
  moet)         MOET=1 ;;
  moet-off)     IMG=kimi-k3-up-ray:moet-mvp; ARM="$HERE/moet/scripts/k3-arm-moet.sh" ;;  # the first measured control: v1 image, flag unset
  moet-off2)    IMG=kimi-k3-up-ray:moet-mvp2; ARM="$HERE/moet/scripts/k3-arm-moet.sh" ;;  # README headline baseline (fu-off): same image as moet, flag unset
  nvfp4-kv)     ENV="$ENV $(cat "$HERE/nvfp4-kv/nvfp4.envstr")" ;;
  rebalance-kv) ENV="$ENV $(cat "$HERE/rebalance-kv/rebalance.envstr")" ;;
  *) echo "unknown config $CFG"; exit 2 ;;
esac
if [ "${MOET:-0}" = 1 ]; then
  ENV="$ENV $(cat "$HERE/moet/moet.envstr")"; IMG=kimi-k3-up-ray:moet-mvp2
  ARM="$HERE/moet/scripts/k3-arm-moet.sh"   # forwards UP_IMAGE to the rtx-02 worker
fi
echo "config=$CFG image=$IMG tag=$TAG"; echo "env=$ENV"
exec 9>/tmp/k3-lane.lock; flock 9
EXEC=up2l3 UP_IMAGE=$IMG bash "$ARM" "$TAG" "$ENV" "$CONC"
echo "results: ~/k3-results/$TAG/  (stop: docker rm -f kimi-k3-qsrt-pp4; ssh 10.10.10.2 docker rm -f k3-qsrt-ray-worker)"
