#!/usr/bin/env python3
"""CPU check of convert_moet_packs.scale_search_codes on real K3 experts.

1. d restricted to {0} must reproduce the fixed LUT codes and scales exactly.
2. Full search must never be worse than the LUT on any block, and reports the
   rel-RMS error vs the original MXFP4 (LUT vs search) per matrix.
Usage: test_scale_search.py LAYER N_EXPERTS
"""
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_moet_packs as C  # noqa: E402

torch.set_num_threads(4)
ROOT = "/data/models/moonshotai-Kimi-K3"
wm = json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
L, NE = int(sys.argv[1]), int(sys.argv[2])


def get(n):
    with safe_open(f"{ROOT}/{wm[n]}", "pt") as f:
        return f.get_tensor(n)


def rel(codes, sc, p, s):
    ref = C.mxfp4_dequant(p, s).double()
    got = C.reference_dequant(codes, sc).double()
    return float((got - ref).norm() / ref.norm())


tot = {}
for e in range(NE):
    for w in ("w1", "w3", "w2"):
        pfx = f"language_model.model.layers.{L}.block_sparse_moe.experts.{e}.{w}"
        p, s = get(pfx + ".weight_packed"), get(pfx + ".weight_scale")
        lut = C.mxfp4_to_codes(p)
        C.SS_SHIFTS = range(0, 1)
        c0, s0 = C.scale_search_codes(p, s)
        s_u8 = s if s.dtype == torch.uint8 else s.view(torch.uint8)
        assert torch.equal(c0, lut), "d=0 codes != LUT"
        assert torch.equal(s0, s_u8), "d=0 scales changed"
        C.SS_SHIFTS = range(-3, 3)
        c1, s1 = C.scale_search_codes(p, s)
        ref = C.mxfp4_dequant(p, s).view(p.shape[0], -1, 32).double()
        e_lut = ((C.reference_dequant(lut, s_u8).view_as(ref).double() - ref) ** 2).sum(-1)
        e_ss = ((C.reference_dequant(c1, s1).view_as(ref).double() - ref) ** 2).sum(-1)
        assert bool((e_ss <= e_lut * (1 + 1e-9) + 1e-30).all()), "search worse on a block"
        a, b = rel(lut, s_u8, p, s), rel(c1, s1, p, s)
        shifts = torch.bincount((s1.int() - s_u8.int() + 3).flatten(), minlength=6).tolist()
        tot.setdefault(w, []).append((a, b))
        print(f"L{L} e{e} {w}: lut={a:.4f} search={b:.4f} shift_hist(d=-3..2)={shifts}")
for w, v in tot.items():
    print(f"MEAN {w}: lut={sum(x for x, _ in v) / len(v):.4f} "
          f"search={sum(y for _, y in v) / len(v):.4f}")
print("PASS")
