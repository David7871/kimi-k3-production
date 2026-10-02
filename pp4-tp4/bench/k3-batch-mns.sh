#!/usr/bin/env bash
S2=$(cat ~/stack2.envstr)
MB="$S2 VLLM_PP_BALANCED_ADMIT=1 B12X_SM_RESERVE=8"
B="$MB VLLM_K3_FAST_TOPK=1 VLLM_K3_SKINNY_GEMM=1 VLLM_K3_SKINNY_GEMM_USE_TABLE=0 VLLM_K3_PREFILL_FUSE=1 VLLM_K3_GC_FREEZE=1 VLLM_K3_DCP_B12X=1 VLLM_K3_DCP_COMBINE_B12X=1 VLLM_K3_DCP_PAIR_FUSED_AG=1"
cleanup(){ docker rm -f kimi-k3-qsrt-pp4 k3-qsrt-ray-worker >/dev/null 2>&1; timeout 60 ssh 10.10.10.2 "docker rm -f kimi-k3-qsrt-pp4 k3-qsrt-ray-worker" >/dev/null 2>&1; sleep 5; }
run(){ cleanup; echo "### $1 start $(date -Is)"; EXEC=up2l3 timeout 2400 bash /data/models/k3-run/scripts/k3-arm.sh "$1" "$2" "$3"; echo "### $1 rc=$? $(date -Is)"; grep -hE "^C[0-9]+ decode|probe:|DIED|TIMEOUT" ~/k3-results/$1/arm.log | cut -c1-160; docker logs kimi-k3-qsrt-pp4 > ~/k3-results/$1/head-full.log 2>&1; }
run mns32-capfix "$B K3_MNS=32" 16,32,64
run mns64-capfix "$B K3_MNS=64" 32,64,128
cleanup
