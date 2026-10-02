#!/usr/bin/env bash
: "${K3_EXTRA_ARGS_CSV:=}"
# rtx-01 (head): Ray head + wait for 16 GPUs + QSRT engine PP4 x TP4/DCP4
# with remote DFlash spec decode. Direct serve (qsrt.sh owns spec config
# and cannot express the remote-draft pairing).
set -euo pipefail

RAY=/opt/venv/bin/ray
PY=/opt/venv/bin/python

rm -rf /tmp/ray

echo "[k3-qsrt-head] starting ray head on 10.10.10.1:6379"
"$RAY" start --head --node-ip-address=10.10.10.1 --port=6379 \
  --disable-usage-stats --include-dashboard=false

echo "[k3-qsrt-head] waiting for 16 ray GPUs (2 nodes)..."
ok=0
for i in $(seq 1 180); do
  gpus=$("$PY" - <<'EOF' 2>/dev/null || echo 0
import ray
ray.init(address="auto", ignore_reinit_error=True)
print(int(ray.cluster_resources().get("GPU", 0)))
EOF
)
  echo "[k3-qsrt-head] ray cluster GPUs: ${gpus} (attempt ${i})"
  if [ "${gpus}" -ge 16 ]; then ok=1; break; fi
  sleep 5
done
if [ "${ok}" != 1 ]; then
  echo "[k3-qsrt-head] TIMEOUT waiting for 16 ray GPUs" >&2
  exit 1
fi

echo "[k3-qsrt-head] launching QSRT PP4xTP4/DCP4 + remote DFlash"
exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve \
  /data/models/Kimi-K3-QSRT-K2 \
  --served-model-name Kimi-K3-QSRT-K2 \
  --trust-remote-code \
  --language-model-only \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 4 \
  --decode-context-parallel-size 4 \
  --dcp-kv-cache-interleave-size 1 \
  --mamba-block-size 1536 \
  --mamba-cache-mode align \
  --prefix-match-unit 256 \
  --load-format instanttensor \
  --moe-backend b12x \
  --linear-backend b12x \
  --attention-backend B12X_MLA \
  --additional-config '{"kda_prefill_backend":"triton"}' \
  --kv-cache-dtype ${K3_KVDT:-fp8} \
  --kv-cache-memory-bytes 17179869184 \
  --max-model-len 262144 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs ${K3_MNS:-256} \
  --gpu-memory-utilization 0.72 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}' \
  --generation-config vllm \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  ${K3_DCAR:---disable-custom-all-reduce} \
  --distributed-executor-backend ray ${K3_EXTRA_ARGS:-} ${K3_EXTRA_ARGS_CSV//,/ }
