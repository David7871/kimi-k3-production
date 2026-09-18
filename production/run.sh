#!/usr/bin/env bash
set -euo pipefail
recipe_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$recipe_dir"
[[ -f .env ]] || { echo 'Copy .env.example to .env and set host paths first.' >&2; exit 2; }
compose=(docker compose -p kimi-k3-qsrt --env-file .env -f compose.yaml)
case "${1:-}" in
  pull)
    locked_image=$(python3 -c 'import json; print(json.load(open("image-lock.json"))["image"])')
    local_image=$("${compose[@]}" config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["kimi"]["image"])')
    docker pull "$locked_image"
    docker tag "$locked_image" "$local_image"
    ;;
  build) "${compose[@]}" build kimi ;;
  check)
    "${compose[@]}" config --quiet
    python3 check-launch.py
    ;;
  up)
    "${compose[@]}" config --quiet
    "${compose[@]}" up -d
    ;;
  restart)
    "${compose[@]}" config --quiet
    "${compose[@]}" stop -t 180 kimi
    # Re-run the preflight after the old owner has released its named pool.
    "${compose[@]}" up -d --force-recreate
    ;;
  logs) "${compose[@]}" logs -f --tail 100 kimi ;;
  down) "${compose[@]}" down --timeout 180 ;;
  *) echo 'Usage: ./run.sh {pull|build|check|up|restart|logs|down}' >&2; exit 2 ;;
esac
