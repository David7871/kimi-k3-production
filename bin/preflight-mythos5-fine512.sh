#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT=/home/g0san/kimi-k3-production
RUNTIME="$ROOT/candidates/all-compatible-20260829/runtime"
BASE="$RUNTIME/compose.tp8-dflash-flashkda-320k-w4a8.yml"
MXFP8="$ROOT/research/mxfp8-entry-20260830/compose.mxfp8-entry.override.yml"
PROFILE="$RUNTIME/compose.mythos5-dflash-stock-fine1536.override.yml"
TARGET_IMAGE='kimi-k3-production-issue75-dspark@sha256:087276f1ec791909e71055d655eeadb3b35c246e46f2be21af9d210aa970a60b'
TARGET_IMAGE_ID='sha256:087276f1ec791909e71055d655eeadb3b35c246e46f2be21af9d210aa970a60b'
DRAFT_IMAGE='kimi-k3-dspark-3090:bf16-cudagraph'
DRAFT_IMAGE_ID='sha256:f344aea65eaca451633db830bd854c505b32eb3f5db6d69ea381611645497160'
DRAFT_MODEL=/mnt/models/Kimi-K3-DFlash-stock
DRAFT_SHA='0f0bd39d19c2dd0249de4f8216180f9729963402f902d50f19a156a8d358b733'
L2=/mnt/lmcache-l2/kimik3/l2-mythos5-text-prob-dflash-w32768-mamba1536-chunk1536-kv2176m-v4

die() {
  echo "MYTHOS5_PREFLIGHT_ERROR: $*" >&2
  exit 1
}

for command in docker jq curl nvidia-smi sha256sum; do
  command -v "$command" >/dev/null || die "missing command: $command"
done
for path in "$BASE" "$MXFP8" "$PROFILE" "$DRAFT_MODEL/config.json" "$DRAFT_MODEL/model.safetensors" "$L2"; do
  [[ -e "$path" ]] || die "missing path: $path"
done

compose=(docker compose --project-name kimi-k3-production --project-directory "$RUNTIME" -f "$BASE" -f "$MXFP8" -f "$PROFILE")
"${compose[@]}" config --quiet
effective="$("${compose[@]}" config --format json)"
jq -e '
  .services["kimi-k3-production"].environment as $t |
  .services["kimi-k3-draft"] as $d |
  ($t.SERVED_MODEL_NAME == "claude-mythos-5") and
  ($t.MAMBA_BLOCK_SIZE == "1536") and
  ($t.PREFIX_MATCH_UNIT == "1536") and
  ($t.LMCACHE_CHUNK_SIZE == "1536") and
  ($t.MAX_NUM_BATCHED_TOKENS == "1536") and
  ($t.MAX_NUM_SCHEDULED_TOKENS == "1536") and
  ($t.MAX_MODEL_LEN == "1048576") and
  ($t.ENABLE_VISION == "0") and
  ($t.KV_CACHE_MEMORY_BYTES == "2281701376") and
  ($t.SPECULATIVE_METHOD == "dflash") and
  ($t.DRAFT_SAMPLE_METHOD == "probabilistic") and
  ($t.LMCACHE_L1_INIT_GB == "80") and
  ($t.LMCACHE_L1_GB == "80") and
  ($t.LMCACHE_L2_GB == "2048") and
  ($t.LMCACHE_MAX_CPU_WORKERS == "16") and
  ($d.environment.VLLM_K3_DRAFT_PREFIX_CACHE_TOKENS == "32768") and
  (($d.command | index("dflash")) != null) and
  (($d.command | index("dspark")) == null) and
  ($d.command[17] == "1048576") and
  ($d.command[19] == "1536") and
  ($d.command[25] == "8.0") and
  ($d.command[27] == "32768") and
  (any($d.volumes[]; .target == "/draft-model" and .source == "/mnt/models/Kimi-K3-DFlash-stock")) and
  (any($d.volumes[]; .target == "/opt/k3-dspark-3090/source/vllm/entrypoints/k3_dspark_rpc.py" and (.source | endswith("/overlays/vllm-live-base/vllm/entrypoints/k3_dspark_rpc.py")))) and
  (any(.services["kimi-k3-production"].volumes[]; .target == "/draft-model" and .source == "/mnt/models/Kimi-K3-DFlash-stock")) and
  (any(.services["kimi-k3-production"].volumes[]; .target == "/lmcache/l2" and (.source | contains("mamba1536-chunk1536"))))
' <<<"$effective" >/dev/null || die "effective Compose policy mismatch"

[[ "$(docker image inspect "$TARGET_IMAGE" --format '{{.Id}}' 2>/dev/null || true)" == "$TARGET_IMAGE_ID" ]] || die "target image mismatch"
[[ "$(docker image inspect "$DRAFT_IMAGE" --format '{{.Id}}' 2>/dev/null || true)" == "$DRAFT_IMAGE_ID" ]] || die "draft image mismatch"
[[ "$(jq -r '.architectures | join(",")' "$DRAFT_MODEL/config.json")" == DFlashDraftModel ]] || die "stock draft architecture mismatch"
[[ "$(jq -r '.dflash_config.block_size' "$DRAFT_MODEL/config.json")" == 16 ]] || die "stock DFlash block metadata mismatch"
[[ "$(sha256sum "$DRAFT_MODEL/model.safetensors" | awk '{print $1}')" == "$DRAFT_SHA" ]] || die "stock DFlash hash mismatch"

grep -Fq 'dflash_logits_bf16_v1' "$ROOT/overlays/vllm-live-base/vllm/entrypoints/k3_dspark_rpc.py" || die "draft logits RPC patch missing"
grep -Fq 'dflash_logits_bf16_v1' "$ROOT/overlays/vllm-live-base/vllm/v1/worker/gpu/spec_decode/dspark/remote_speculator.py" || die "target probabilistic patch missing"
grep -Fq 'upstream_model: claude-mythos-5' /home/g0san/llmconduit-kimi/config.yaml || die "Kimi gateway model mismatch"
grep -Fq '  claude-mythos-5:' /home/g0san/llmconduit-kimi/config.yaml || die "Kimi gateway profile missing"
grep -Fq '    native_vision: false' /home/g0san/llmconduit-kimi/config.yaml || die "Kimi gateway native vision is enabled"
sudo -n true >/dev/null 2>&1 || die "passwordless sudo required"

sudo -n test -w "$L2" || die "LMCache namespace is not writable: $L2"
available="$(df -B1 --output=avail "$L2" | tail -n 1 | tr -d ' ')"
required="$(( 2112 * 1024 * 1024 * 1024 ))"
(( available >= required )) || die "LMCache filesystem lacks 2,112 GiB free"
shm_size="$(df -B1 --output=size /dev/shm | tail -n 1 | tr -d ' ')"
(( shm_size >= 84 * 1024 * 1024 * 1024 )) || die "/dev/shm is smaller than the 80+4 GiB contract"

[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" == 9 ]] || die "expected nine GPUs"
for index in 1 2 3 4 5 6 7 8; do
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$index" | tr -d ' ')"
  [[ "$cc" == 12.0 ]] || die "GPU $index is not SM120"
done

glm_state="$(docker inspect glm53-r28 --format '{{.State.Status}}' 2>/dev/null || true)"
[[ "$glm_state" == running || "$glm_state" == exited ]] || die "unexpected GLM container state: $glm_state"
[[ "$(docker inspect kimi-k3-production --format '{{.State.Status}}' 2>/dev/null || true)" == exited ]] || die "old Kimi target is not stopped"
draft_state="$(docker inspect kimi-k3-draft --format '{{.State.Status}}' 2>/dev/null || true)"
[[ "$draft_state" == running || "$draft_state" == exited ]] || die "unexpected Kimi draft state: $draft_state"

echo "MYTHOS5_PREFLIGHT_OK model=claude-mythos-5 vision=disabled sampling=probabilistic physical=1536 chunk=1536 draft_sha=$DRAFT_SHA"
