#!/usr/bin/env bash
set -x
S2=$(cat ~/stack2.envstr)
B="$S2 VLLM_PP_BALANCED_ADMIT=1 B12X_SM_RESERVE=8 VLLM_K3_FAST_TOPK=1 VLLM_K3_SKINNY_GEMM=1 VLLM_K3_SKINNY_GEMM_USE_TABLE=0 VLLM_K3_PREFILL_FUSE=1 VLLM_K3_GC_FREEZE=1 VLLM_K3_DCP_B12X=1 VLLM_K3_DCP_COMBINE_B12X=1 VLLM_K3_DCP_PAIR_FUSED_AG=1"
docker rm -f kimi-k3-qsrt-pp4 k3-qsrt-ray-worker; ssh 10.10.10.2 "docker rm -f kimi-k3-qsrt-pp4 k3-qsrt-ray-worker"; sleep 5
EXEC=up2l3 timeout 2400 bash /data/models/k3-run/scripts/k3-arm.sh nsys3-mns256 "$B K3_MNS=256 K3_EXTRA_ARGS_CSV=--ray-workers-use-nsight,--profiler-config.profiler=cuda" 1
docker exec kimi-k3-qsrt-pp4 rm -rf /tmp/nsys-root; ssh 10.10.10.2 docker exec k3-qsrt-ray-worker rm -rf /tmp/nsys-root
cd /data/models/k3-run/overlays-mb
(timeout 110 python3 decode_bench.py 8001 256 60 10 > ~/nsys3-bench.txt &)
sleep 40; curl -sXPOST localhost:8001/start_profile; sleep 1.5; curl -sXPOST localhost:8001/stop_profile
sleep 150
docker exec kimi-k3-qsrt-pp4 bash -c "ls -la /tmp/nsys-root; nsys sessions list"
ssh 10.10.10.2 "docker exec k3-qsrt-ray-worker ls -la /tmp/nsys-root"
echo NSYS3_CAPTURED
