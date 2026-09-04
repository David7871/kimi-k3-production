#!/bin/bash
# Recreate the Kimi-K3 production service with a decided LMCache cache namespace.
#
# Usage: k3-compose-up.sh [-f compose.yml ...]   (run from the runtime directory)
#
# 1. Renders the merged compose configuration of the given -f files.
# 2. Computes the configuration fingerprint (image id, every bind-mounted file's
#    content, model directories, engine environment) with
#    bin/k3_cache_namespace.py and resolves the namespace against
#    NAMESPACE.json in the L2 directory: an accepted fingerprint maps to the
#    production namespace, anything else to an isolated iso-<fingerprint>
#    namespace whose entries production can never restore.
# 3. Appends candidates/k3-cache-namespace-20260902/compose.cache-namespace.override.yml
#    (LMCACHE_CACHE_NAMESPACE from K3_CACHE_NAMESPACE, LMCACHE_REQUIRE_CACHE_NAMESPACE=1,
#    the namespace-aware connector overlay) and recreates the service.
#
# Environment:
#   TARGET_SERVICE   service to recreate (default kimi-k3-production)
#   K3_L2_DIR        host L2 directory holding NAMESPACE.json
#                    (default: the service's /lmcache/l2 bind source)
#   K3_NAMESPACE_OVERRIDE  force a namespace (e.g. iso-experiment-x) instead of
#                    resolving it; never use it to force the production namespace.
#   K3_DRY_RUN=1     print the fingerprint and namespace and exit without recreating.
set -euo pipefail
ROOT=/home/g0san/kimi-k3-production
SERVICE="${TARGET_SERVICE:-kimi-k3-production}"
NS_OVERRIDE_FILE="$ROOT/candidates/k3-cache-namespace-20260902/compose.cache-namespace.override.yml"
FILES=("$@")
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "usage: $0 -f compose.yml [-f override.yml ...]" >&2
  exit 2
fi
ALL=("${FILES[@]}" -f "$NS_OVERRIDE_FILE")

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
# The namespace variable must exist for `config` to render; the placeholder is
# replaced below once the real value is known.
K3_CACHE_NAMESPACE=pending docker compose -p kimi-k3-production "${ALL[@]}" config --format json > "$tmp/config.json"

l2_dir="${K3_L2_DIR:-}"
if [[ -z "$l2_dir" ]]; then
  l2_dir="$(python3 - "$tmp/config.json" "$SERVICE" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
for vol in cfg["services"][sys.argv[2]].get("volumes", []):
    if vol.get("type") == "bind" and vol.get("target") == "/lmcache/l2":
        print(vol["source"]); break
PY
)"
fi
if [[ -z "$l2_dir" || ! -d "$l2_dir" ]]; then
  echo "ERROR: cannot determine the L2 directory (set K3_L2_DIR)" >&2
  exit 2
fi

fp="$(python3 "$ROOT/bin/k3_cache_namespace.py" fingerprint --compose-config "$tmp/config.json" --service "$SERVICE" --l2-dir "$l2_dir" --manifest-out "$tmp/manifest.json")"
if [[ -n "${K3_NAMESPACE_OVERRIDE:-}" ]]; then
  if [[ "$K3_NAMESPACE_OVERRIDE" == "legacy" ]]; then
    echo "ERROR: the production namespace can only be selected by an accepted fingerprint (use k3_cache_namespace.py accept)" >&2
    exit 2
  fi
  namespace="$K3_NAMESPACE_OVERRIDE"
else
  namespace="$(python3 "$ROOT/bin/k3_cache_namespace.py" resolve --compose-config "$tmp/config.json" --service "$SERVICE" --l2-dir "$l2_dir")"
fi
echo "config fingerprint: $fp"
echo "cache namespace:    $namespace"
if [[ "$namespace" != legacy ]]; then
  echo "NOTE: this configuration is not accepted for the production cache; its entries are isolated." >&2
  echo "      After qualifying it: python3 $ROOT/bin/k3_cache_namespace.py accept --fingerprint $fp --l2-dir $l2_dir --note '<evidence>'" >&2
fi
mkdir -p "$ROOT/qualification-namespace" && cp "$tmp/manifest.json" "$ROOT/qualification-namespace/manifest-$fp.json" 2>/dev/null || true

export K3_CACHE_NAMESPACE="$namespace"
docker compose -p kimi-k3-production "${ALL[@]}" config --services >/dev/null
if [[ "${K3_DRY_RUN:-0}" == "1" ]]; then
  docker compose -p kimi-k3-production "${ALL[@]}" config | grep -E 'LMCACHE_CACHE_NAMESPACE|LMCACHE_REQUIRE_CACHE_NAMESPACE'
  echo "dry run: not recreating $SERVICE"
  exit 0
fi
exec docker compose -p kimi-k3-production "${ALL[@]}" up -d --no-deps --force-recreate "$SERVICE"
