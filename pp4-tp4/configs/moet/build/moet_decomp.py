#!/usr/bin/env python3
"""CPU decomposition of the op-test 0.63 discrepancy (layer 1, rank 0).

A = f32 MoE forward on MXFP4-dequant weights (op-test reference math)
B = same forward on pack-dequant weights (reference_dequant of
    mxfp4_to_codes(...) — byte-identical content to the packs)
If rel(A,B) ~ 0.63, the 0.63 is the true 2-bit-snap error through SiTU
(accuracy limit). If ~0.42, the kernel path diverges from the pack.
Also reports weight-space rel_rms for the same experts (golden cross-check)
and a gate/up-swapped variant to rule out ordering.
"""
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_moet_packs import (E, H, IR, mxfp4_dequant, mxfp4_to_codes,  # noqa
                                reference_dequant)

SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
LAYER, RANK = 1, 0


def situ(gate, up):
    return (SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
            * (SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA)))


def main():
    src = "/data/models/moonshotai-Kimi-K3"
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    handles = {}

    def get(name):
        fn = wm[name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(src, fn), framework="pt",
                                    device="cpu")
        return handles[fn].get_tensor(name)

    pfx = (f"language_model.model.layers.{LAYER}"
           ".block_sparse_moe.experts.")
    r0, r1 = RANK * IR, (RANK + 1) * IR

    def experts_mx(e):
        w1 = mxfp4_dequant(get(pfx + f"{e}.w1.weight_packed"),
                           get(pfx + f"{e}.w1.weight_scale"))
        w3 = mxfp4_dequant(get(pfx + f"{e}.w3.weight_packed"),
                           get(pfx + f"{e}.w3.weight_scale"))
        w2 = mxfp4_dequant(get(pfx + f"{e}.w2.weight_packed"),
                           get(pfx + f"{e}.w2.weight_scale"))
        return w1[r0:r1], w3[r0:r1], w2[:, r0:r1]

    def experts_pk(e):
        w1p, w1s = get(pfx + f"{e}.w1.weight_packed"), get(
            pfx + f"{e}.w1.weight_scale")
        w3p, w3s = get(pfx + f"{e}.w3.weight_packed"), get(
            pfx + f"{e}.w3.weight_scale")
        w2p, w2s = get(pfx + f"{e}.w2.weight_packed"), get(
            pfx + f"{e}.w2.weight_scale")
        w13p = torch.cat((w1p[r0:r1], w3p[r0:r1]), 0)
        w13s = torch.cat((w1s[r0:r1], w3s[r0:r1]), 0)
        w2pr = w2p[:, RANK * (IR // 2):(RANK + 1) * (IR // 2)]
        w2sr = w2s[:, RANK * (IR // 32):(RANK + 1) * (IR // 32)]
        w13 = reference_dequant(mxfp4_to_codes(w13p), w13s)
        w2 = reference_dequant(mxfp4_to_codes(w2pr), w2sr)
        return w13[:IR], w13[IR:], w2  # gate, up, down

    g = torch.Generator().manual_seed(0)
    T, top_k = 4, 16
    x = torch.randn(T, H, generator=g).float()
    ids = torch.stack([torch.randperm(E, generator=g)[:top_k]
                       for _ in range(T)]).long()
    w = torch.softmax(torch.randn(T, top_k, generator=g), dim=1).float()

    refA = torch.zeros(T, H)
    refB = torch.zeros(T, H)
    refS = torch.zeros(T, H)  # swapped gate/up on pack weights
    wrels = []
    used = sorted(set(ids.reshape(-1).tolist()))
    cache = {}
    for e in used:
        g1, u1, d1 = experts_mx(e)
        g2, u2, d2 = experts_pk(e)
        cache[e] = (g1, u1, d1, g2, u2, d2)
        for tag, a, b in (("gate", g1, g2), ("up", u1, u2), ("down", d1, d2)):
            wrels.append(((b - a).norm() / a.norm().clamp_min(1e-12)).item())
    print(f"weight-space rel_rms over {len(used)} experts: "
          f"mean={sum(wrels) / len(wrels):.4f} "
          f"min={min(wrels):.4f} max={max(wrels):.4f}")
    for t in range(T):
        for k in range(top_k):
            e = int(ids[t, k])
            g1, u1, d1, g2, u2, d2 = cache[e]
            refA[t] += w[t, k] * (situ(x[t] @ g1.T, x[t] @ u1.T) @ d1.T)
            refB[t] += w[t, k] * (situ(x[t] @ g2.T, x[t] @ u2.T) @ d2.T)
            refS[t] += w[t, k] * (situ(x[t] @ u2.T, x[t] @ g2.T) @ d2.T)
    relAB = ((refB - refA).norm() / refA.norm()).item()
    relAS = ((refS - refA).norm() / refA.norm()).item()
    print(f"OUTPUT rel(A,B) pack-vs-mxfp4 = {relAB:.4f}")
    print(f"OUTPUT rel(A,swap-gate/up)   = {relAS:.4f}  (ordering probe)")


if __name__ == "__main__":
    main()
