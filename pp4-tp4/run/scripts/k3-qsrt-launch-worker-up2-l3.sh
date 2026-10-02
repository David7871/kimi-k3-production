#!/usr/bin/env bash
# rtx-02 (10.10.10.2) worker for the L3 integration overlay:
# overlay-b12x-l3/b12x (imadsm b12x + default-off B12X_TWOSHOT_MODE /
# B12X_PCIE_PDL gates) and every single-file vLLM mount listed in
# overlay-vllm-l3/MOUNTS.txt. With no extra env this boots the same code paths
# as up2-imadsm.
set -euo pipefail
set -f  # MOUNTS.txt words are data, never globs

IMAGE=${UP_IMAGE:-kimi-k3-up-ray:local}
RUN=${K3_RUN:-/data/models/k3-run}
SP=/opt/venv/lib/python3.12/site-packages
L3B12X="${K3_L3B12X:-$RUN/overlay-b12x-l3/b12x}"
L3V="${K3_L3V:-$RUN/overlay-vllm-l3}"
MOUNTS="$L3V/MOUNTS.txt"

die() { echo "FATAL(up2-l3 worker): $*" >&2; exit 1; }

# ---- resolve the single-file vLLM mounts from the shared MOUNTS.txt ---------
# Format (append-only, owned by whoever adds a file): "<basename> -> <dest
# relative to site-packages>"; "#" lines and blank lines are ignored. A line
# that cannot be resolved fails the launch instead of silently not mounting.
[ -f "$MOUNTS" ] || die "$MOUNTS is missing"
[ -d "$L3B12X" ] || die "$L3B12X is missing"

# Historical destinations, so a bare "<basename>" line still resolves.
fallback_dest() {
  case "$1" in
    gpu_model_runner.py) echo "vllm/v1/worker/gpu_model_runner.py" ;;
    dcp_utils.py)        echo "vllm/v1/attention/ops/dcp_utils.py" ;;
    envs.py)             echo "vllm/envs.py" ;;
    *)                   echo "" ;;
  esac
}

VLLM_MOUNTS=()
COVERED=()
PRETTY=()
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%$'\r'}"
  case "$line" in ''|'#'*) continue ;; esac
  tokens=()
  for tok in $line; do
    [ "$tok" = "->" ] && continue
    tokens+=("$tok")
  done
  src="${tokens[0]:-}"
  dest="${tokens[1]:-}"
  [ -n "$src" ] || die "unparsable MOUNTS.txt line: $line"
  [ -n "$dest" ] || dest="$(fallback_dest "$src")"
  [ -n "$dest" ] || die "MOUNTS.txt line has no destination and no fallback: $line"
  case "$dest" in /*) ;; *) dest="$SP/$dest" ;; esac
  [ -f "$L3V/$src" ] || die "$L3V/$src listed in MOUNTS.txt does not exist"
  VLLM_MOUNTS+=(-v "$L3V/$src:$dest:ro")
  COVERED+=("$src")
  PRETTY+=("$L3V/$src -> $dest")
done < "$MOUNTS"
set +f  # the completeness check below needs the *.py glob to expand

# Every overlay file must be mounted: a file dropped into overlay-vllm-l3
# without a MOUNTS.txt line would otherwise be silently ignored.
for f in "$L3V"/*.py; do
  [ -e "$f" ] || continue
  base="${f##*/}"
  listed=0
  for c in ${COVERED[@]+"${COVERED[@]}"}; do
    [ "$c" = "$base" ] && { listed=1; break; }
  done
  [ "$listed" = 1 ] || die "$base is in overlay-vllm-l3 but has no MOUNTS.txt line; append '$base -> <dest relative to site-packages>'"
done

# ---- opt-in PCIe two-shot route (default OFF) ------------------------------
# The image's own vllm/distributed/device_communicators/custom_all_reduce.py
# has no two-shot route at all, so b12x's PCIeTwoShotBF16 is never constructed
# and B12X_TWOSHOT_MODE would be inert. The k3-up-tree copy is the only one
# with _initialize_pcie_twoshot(); mount it over site-packages to make the
# pull-vs-push A/B reachable. Unset => no extra mount => exactly the
# up2-imadsm mount set.
TWOSHOT_ROUTE_DEST="$SP/vllm/distributed/device_communicators/custom_all_reduce.py"
TWOSHOT_ROUTE_SRC=${K3_L3_TWOSHOT_ROUTE_SRC:-/data/models/k3-up-tree/vllm/distributed/device_communicators/custom_all_reduce.py}
if [ -n "${K3_L3_TWOSHOT_ROUTE:-}" ]; then
  case "${K3_L3_TWOSHOT_ROUTE}" in
    1|true|yes|on) ;;
    *) die "K3_L3_TWOSHOT_ROUTE must be 1/true/yes/on or unset, got '${K3_L3_TWOSHOT_ROUTE}'" ;;
  esac
  route_covered=0
  for c in ${COVERED[@]+"${COVERED[@]}"}; do
    [ "$c" = "custom_all_reduce.py" ] && { route_covered=1; break; }
  done
  if [ "$route_covered" = 1 ]; then
    echo "up2-l3 worker: K3_L3_TWOSHOT_ROUTE set but overlay-vllm-l3/custom_all_reduce.py is already in MOUNTS.txt; the overlay file wins"
  else
    [ -f "$TWOSHOT_ROUTE_SRC" ] || die "K3_L3_TWOSHOT_ROUTE=1 but $TWOSHOT_ROUTE_SRC is missing"
    VLLM_MOUNTS+=(-v "$TWOSHOT_ROUTE_SRC:$TWOSHOT_ROUTE_DEST:ro")
    PRETTY+=("$TWOSHOT_ROUTE_SRC -> $TWOSHOT_ROUTE_DEST (K3_L3_TWOSHOT_ROUTE)")
  fi
fi

echo "up2-l3 worker: ${#PRETTY[@]} single-file vLLM mount(s) from $MOUNTS"
printf '  %s\n' ${PRETTY[@]+"${PRETTY[@]}"}

docker rm -f k3-qsrt-ray-worker 2>/dev/null || true

exec docker run -d \
  --name k3-qsrt-ray-worker \
  --restart no \
  --gpus all \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device=/dev/infiniband/uverbs0 \
  --device=/dev/infiniband/rdma_cm \
  --device=/dev/infiniband/umad0 \
  --security-opt label=disable \
  -v /data/models:/data/models:ro \
  -v /data/models/k3-jit-cache-up2:/cache/jit:rw \
  -v "$RUN/scripts":/k3run:ro \
  -v /data/models/k3-up-tree/vllm/vllm:/opt/infernal-invocation/vllm/vllm:ro \
  -v "$L3B12X":/opt/infernal-invocation/b12x/b12x:ro \
  -v /data/models/k3-up-tree/lmcache/lmcache:/opt/infernal-invocation/lmcache/lmcache:ro \
  ${VLLM_MOUNTS[@]+"${VLLM_MOUNTS[@]}"} \
  --env-file "$RUN/env/k3-qsrt-common.env" \
  ${EXTRA_ENV_ARGS:-} \
  --env-file "$RUN/env/k3-qsrt-worker.env" \
  --entrypoint /k3run/k3-qsrt-worker-entry.sh \
  "$IMAGE"
