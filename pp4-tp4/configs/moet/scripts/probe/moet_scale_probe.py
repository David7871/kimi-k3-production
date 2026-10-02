"""Weight-level error vs original MXFP4 for Moet 2-bit grid {-4,-1,1,4}*2^e.

A: current converter (fixed LUT at original block scale == nearest rounding at d=0)
B: per-block-32 best power-of-two scale shift d in [-3,2], nearest rounding
"""
import json, sys, torch
from safetensors import safe_open

torch.set_num_threads(4)
ROOT = "/data/models/moonshotai-Kimi-K3"
idx = json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
LEVELS = torch.tensor([1.0, 4.0])


def load(name):
    with safe_open(f"{ROOT}/{idx[name]}", "pt") as f:
        return f.get_tensor(name)


def decode(prefix):
    p = load(prefix + ".weight_packed").view(torch.uint8)
    s = load(prefix + ".weight_scale")
    s = s.view(torch.uint8).to(torch.int32) if s.dtype != torch.uint8 else s.to(torch.int32)
    nib = torch.stack((p & 0xF, p >> 4), dim=-1).flatten(-2).long()
    val = E2M1[nib & 7] * torch.where((nib & 8) > 0, -1.0, 1.0)
    N, K = val.shape
    vb = val.view(N, K // 32, 32)
    return vb, s  # value in units of 2^(s-127)


def quant_err(vb, d):
    # grid in original-scale units: {1,4} * 2^d, sign kept
    a = vb.abs()
    lv = LEVELS * (2.0 ** d)
    q = torch.where((a - lv[0]).abs() <= (a - lv[1]).abs(), lv[0], lv[1])
    return ((q - a) ** 2)  # per element, original-scale units


def run(prefix):
    vb, s = decode(prefix)
    scale2 = (2.0 ** (s.float() - 127)) ** 2  # [N, K/32]
    ref = ((vb ** 2).sum(-1) * scale2).sum()
    errs = torch.stack([quant_err(vb, d).sum(-1) for d in range(-3, 3)])  # [6,N,B]
    a = (errs[3] * scale2).sum()
    best = errs.min(0)
    b = (best.values * scale2).sum()
    hist = torch.bincount(best.indices.flatten(), minlength=6).tolist()
    return (a / ref).sqrt().item(), (b / ref).sqrt().item(), hist


layers = [int(x) for x in sys.argv[1].split(",")]
experts = range(int(sys.argv[2]))
for L in layers:
    for w in ("w1", "w3", "w2"):
        ra = rb = 0.0
        H = [0] * 6
        for e in experts:
            a, b, h = run(f"language_model.model.layers.{L}.block_sparse_moe.experts.{e}.{w}")
            ra += a; rb += b; H = [x + y for x, y in zip(H, h)]
        n = len(experts)
        print(f"L{L} {w}: A_fixedLUT={ra/n:.4f} B_scalesearch={rb/n:.4f} d-hist(-3..2)={H}", flush=True)
