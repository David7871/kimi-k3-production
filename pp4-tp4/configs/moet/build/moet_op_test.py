#!/usr/bin/env python3
"""K3-Moet MVP single-GPU op test (GPU window item a).

Loads the Moet pack for one real layer/rank, runs moe_w2_forward on random
activations, and compares against an f32 reference computed from the
official MXFP4 experts (dequant -> SiTU(beta=4, linear_beta=25) -> down),
for the same routing. Reports worst/mean rel error per token count.

Run inside kimi-k3-up-ray:moet-mvp with ONE GPU visible:
  docker run --rm --gpus '"device=0"' --network none \
    --entrypoint /opt/venv/bin/python \
    -v /data/models:/data/models -v /data/models/k3-moet-work:/work \
    -e VLLM_K3_MOET=1 -e VLLM_K3_MOET_PACKS=/data/models/k3-moet-packs \
    -e VLLM_MOE_W2=1 -e VLLM_MOE_W2_CUBIT_DIR=/opt/moet/cubins \
    -e VLLM_MOE_W2_KS=3584,768 -e VLLM_MOE_W2_CKPT_ID=k3-mxfp4-moet-v1 \
    -e VLLM_MOE_W2_NUM_LAYERS=93 -e VLLM_MOE_W2_DELTA=0 \
    kimi-k3-up-ray:moet-mvp /work/moet_op_test.py --layer 1 --rank 0
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert_moet_packs import (  # noqa: E402
    E, H, I, TP, IR, mxfp4_dequant)
from moe_w2_planes import mxfp4_to_codes, reference_dequant  # noqa: E402

# Kernel-correctness gate: kernel vs the pack's own 2-bit dequant. The
# vs-MXFP4 number is the (expected, ~0.6) quantization loss, reported only.
KERNEL_TOL = 0.05

SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0


def situ(gate, up):
    return (SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
            * (SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tokens", default="1,4,16,64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="/data/models/moonshotai-Kimi-K3")
    args = ap.parse_args()

    from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
    from vllm.model_executor.layers.quantization.utils import (
        moe_w2_planes_cache as pc)

    # No distributed init in this standalone probe: stand in for the lane's
    # TP4 identity so the pack rank dir/meta resolve.
    pc._tp_ids = lambda: (TP, args.rank)
    moe_w2_cubit.k3_moet_bootstrap(os.environ["VLLM_K3_MOET_PACKS"])
    assert moe_w2_cubit._ensure_ready(), "cubin load failed"
    moe_w2_cubit._require_kernels(H, IR, need_w4=False)

    # Load pack layer -> GPU (bypasses the layer object; registers _LAYERS).
    key = args.layer
    sizes = pc.expected_sizes(E, 2 * IR, H, H, IR, want_fp4=False)
    cached = pc.try_load(key, sizes)
    assert cached is not None, f"pack miss for layer {key} rank {args.rank}"
    dev = torch.device("cuda")
    moe_w2_cubit._LAYERS[key] = dict(
        planes13=cached["planes13"].view(E, -1).to(dev),
        sc13=cached["sc13"].view(E, -1).to(dev),
        planes2=cached["planes2"].view(E, -1).to(dev),
        sc2=cached["sc2"].view(E, -1).to(dev),
        N13=2 * IR, K13=H, N2=H, K2=IR, E=E, tl_idx=key,
        activation="situ", swiglu_limit=None,
        swiglu_alpha=1.0, swiglu_beta=0.0,
        situ_beta=SITU_BETA, situ_linear_beta=SITU_LINEAR_BETA,
    )
    del cached
    torch.cuda.empty_cache()

    # f32 reference experts for the sampled routing, from MXFP4 source.
    from safetensors import safe_open
    idx = json.load(open(os.path.join(args.source,
                                      "model.safetensors.index.json")))
    wm = idx["weight_map"]
    handles = {}

    def get(name):
        fn = wm[name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(args.source, fn),
                                    framework="pt", device="cpu")
        return handles[fn].get_tensor(name)

    r = args.rank
    pfx = (f"language_model.model.layers.{args.layer}"
           ".block_sparse_moe.experts.")

    def ref_expert(e):
        r0, r1 = r * IR, (r + 1) * IR
        out = {}
        for m in ("w1", "w3", "w2"):
            p = get(pfx + f"{e}.{m}.weight_packed")
            s = get(pfx + f"{e}.{m}.weight_scale")
            mx = mxfp4_dequant(p, s)
            pk = reference_dequant(mxfp4_to_codes(p), s)
            if m == "w2":
                out[m] = (mx[:, r0:r1], pk[:, r0:r1])   # [3584, 768]
            else:
                out[m] = (mx[r0:r1], pk[r0:r1])         # [768, 3584]
        return out

    g = torch.Generator().manual_seed(args.seed)
    top_k = 16
    worst = 0.0
    for T in [int(t) for t in args.tokens.split(",")]:
        x = torch.randn(T, H, generator=g).bfloat16().cuda()
        # top_k distinct experts per token (a global randperm breaks for
        # T * top_k > E).
        ids = torch.stack([torch.randperm(E, generator=g)[:top_k]
                           for _ in range(T)]).to(torch.int32).cuda()
        w = torch.softmax(torch.randn(T, top_k, generator=g),
                          dim=1).float().cuda()
        got = moe_w2_cubit.moe_w2_forward(x, w, ids, key).float().cpu()
        used = sorted(set(ids.cpu().reshape(-1).tolist()))
        ref_e = {e: ref_expert(e) for e in used}
        xf = x.float().cpu()
        refs = {0: torch.zeros(T, H), 1: torch.zeros(T, H)}  # 0=mxfp4 1=pack
        idl = ids.cpu().long()
        wl = w.cpu()
        for t in range(T):
            for k in range(top_k):
                d = ref_e[int(idl[t, k])]
                for j in (0, 1):
                    h = situ(xf[t] @ d["w1"][j].T, xf[t] @ d["w3"][j].T)
                    refs[j][t] += wl[t, k] * (h @ d["w2"][j].T)
        for j, tag in ((1, "kernel_rel_rms"), (0, "quant_rel_rms")):
            ref = refs[j]
            err = got - ref
            rel = (err.norm() / ref.norm().clamp_min(1e-12)).item()
            per_tok = err.norm(dim=1) / ref.norm(dim=1).clamp_min(1e-12)
            print(f"T={T:4d}  {tag}={rel:.4e}  worst_tok={per_tok.max():.4e} "
                  f"mean_tok={per_tok.mean():.4e}", flush=True)
            if j == 1:
                worst = max(worst, rel)
    print(f"KERNEL_GATE worst={worst:.4e} tol={KERNEL_TOL} "
          f"{'PASS' if worst < KERNEL_TOL else 'FAIL'}", flush=True)
    sys.exit(0 if worst < KERNEL_TOL else 2)


if __name__ == "__main__":
    main()
