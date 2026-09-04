#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT=/home/g0san/kimi-k3-production
RUNTIME="$ROOT/candidates/all-compatible-20260829/runtime"
BASE="$RUNTIME/compose.tp8-dflash-flashkda-320k-w4a8.yml"
MXFP8="$ROOT/research/mxfp8-entry-20260830/compose.mxfp8-entry.override.yml"
PROFILE="$RUNTIME/compose.mythos5-dflash-stock-fine1536.override.yml"
compose=(docker compose --project-name kimi-k3-production --project-directory "$RUNTIME" -f "$BASE" -f "$MXFP8" -f "$PROFILE")

[[ "${1:-}" == '--execute' ]] || {
  echo 'Refusing rollback without --execute' >&2
  exit 2
}

echo 'Stopping Mythos Kimi services'
sudo -n systemctl stop llmconduit-kimi.service >/dev/null 2>&1 || true
sudo -n systemctl disable llmconduit-kimi.service >/dev/null 2>&1 || true
"${compose[@]}" stop -t 120 kimi-k3-production kimi-k3-draft >/dev/null 2>&1 || true

echo 'Restoring GLM-5.3'
docker start glm53-r28 >/dev/null
ready=0
for attempt in $(seq 1 180); do
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:5317/health 2>/dev/null || true)"
  if [[ "$code" == 200 ]]; then ready=1; break; fi
  sleep 5
done
[[ "$ready" == 1 ]] || { echo 'GLM rollback readiness timeout' >&2; exit 1; }

sudo -n systemctl enable llmconduit.service >/dev/null
sudo -n systemctl start llmconduit.service
[[ "$(systemctl is-active llmconduit.service)" == active ]] || { echo 'GLM gateway restore failed' >&2; exit 1; }
echo 'MYTHOS5_ROLLBACK_OK service=glm-5.3 gateway=5318'
