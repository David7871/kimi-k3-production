#!/usr/bin/env python3
"""Offline converter: Kimi-K3 official MXFP4 routed experts -> Moet 2-bit
planes packs (moe_w2_planes_cache layout), per TP4 rank, for VLLM_K3_MOET=1.

CPU-only. Reuses Moet's quantizer (moe_w2_planes.py, vendored verbatim
next to this script) so the pack bytes are exactly what an in-vLLM requant
would have produced — boot never requants (_consume_planes_cache hit).

Layout written:
  $OUT/v2/<meta_key>/tp4-rank{r}/layer{L}.{planes13,sc13,planes2,sc2}.bin
  $OUT/v2/<meta_key>/tp4-rank{r}/meta.json
  $OUT/k3-moet-packs.json            (sidecar: config, stats, per-layer log)

TP sharding matches the lane's loader (hybrid_weight_loader): w1/w3 shard
along rows (gate -> top half of w13, up -> bottom), w2 shards along K cols;
e8m0 block-32 scales slice on the same axes (32-col aligned).
"""
import argparse
import json
import hashlib
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_w2_planes import (  # noqa: E402  (Moet, vendored verbatim)
    MOE_W2_QUANTIZER_ABI,
    _E2M1_VALS,
    mxfp4_to_codes,
    mxfp4_to_nibbles,
    pack_fragment_major,
    pack_scales,
    reference_dequant,
    zero_mode,
)

E = 896
H = 3584            # routed_expert_hidden_size (w13 contraction K)
I = 3072            # moe_intermediate_size (global)
TP = 4
IR = I // TP        # 768 per rank
N13, K13 = 2 * IR, H      # 1536, 3584
N2, K2 = H, IR            # 3584, 768
MOE_LAYERS = list(range(1, 93))   # layer 0 is dense

PART_SIZES = {
    "planes13": E * N13 * K13 // 4,
    "sc13": E * N13 * K13 // 32,
    "planes2": E * N2 * K2 // 4,
    "sc2": E * N2 * K2 // 32,
}


def meta_for(rank: int, ckpt_id: str) -> dict:
    return dict(
        version=2,
        quantizer_abi=MOE_W2_QUANTIZER_ABI,
        ckpt_id=ckpt_id,
        world=TP,
        rank=rank,
        zero_mode=zero_mode(),
        fp4_split=False,
    )


def meta_key(meta: dict) -> str:
    return hashlib.sha256(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]


def mxfp4_dequant(w_packed: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """[N, K/2] u8 + [N, K/32] u8 e8m0 -> f32 [N, K] (reference)."""
    nib = mxfp4_to_nibbles(w_packed)
    vals = _E2M1_VALS[nib.long()].float()
    scale = torch.pow(
        torch.tensor(2.0), s.to(torch.int32) - 127).float()
    return (vals.view(vals.shape[0], vals.shape[1] // 32, 32)
            * scale.view(scale.shape[0], scale.shape[1], 1)
            ).view(vals.shape)

# Opt-in (--scale-search): per block-32, pick the power-of-two scale shift d
# in SS_SHIFTS that minimizes L2 error of the sign-symmetric grid
# {-4,-1,1,4} * 2^(s+d) against the original MXFP4 block. d=0 reproduces the
# fixed LUT exactly (e2m1 {0..2} -> 1, {3,4,6} -> 4), so the result is never
# worse than the default per block. Scales stay UE8M0 bytes (s+d), which the
# kernel consumes verbatim, so no kernel/runtime change is needed.
SCALE_SEARCH = False
SS_SHIFTS = range(-3, 3)


def scale_search_codes(w_packed: torch.Tensor, s: torch.Tensor):
    """[N, K/2] u8 + [N, K/32] e8m0 -> ([N, K] u8 codes, [N, K/32] u8 scales)."""
    s8 = (s if s.dtype == torch.uint8 else s.view(torch.uint8)).to(torch.int32)
    nib = mxfp4_to_nibbles(w_packed).long()
    N, K = nib.shape
    a = _E2M1_VALS.abs().float()[nib].view(N, K // 32, 32)
    neg = ((nib & 8) > 0).view(N, K // 32, 32)
    best_err = best_d = None
    for d in SS_SHIFTS:
        lo, hi = 2.0 ** d, 4.0 * 2.0 ** d
        e = torch.minimum((a - lo) ** 2, (a - hi) ** 2).sum(-1)
        ok = (s8 + d >= 0) & (s8 + d <= 254)
        e = torch.where(ok, e, torch.full_like(e, float("inf")))
        if best_err is None:
            best_err, best_d = e, torch.full_like(s8, d)
        else:
            better = e < best_err
            best_err = torch.where(better, e, best_err)
            best_d = torch.where(better, torch.full_like(s8, d), best_d)
    scale = torch.exp2(best_d.float()).unsqueeze(-1)
    big = (a - 4.0 * scale).abs() < (a - scale).abs()
    codes = torch.where(neg, torch.where(big, 0, 1), torch.where(big, 3, 2))
    return codes.view(N, K).to(torch.uint8), (s8 + best_d).to(torch.uint8)



def convert_layer(layer: int, opener, weight_map: dict, out: str,
                  ckpt_id: str, golden: int, log) -> dict:
    t0 = time.time()
    pfx = f"language_model.model.layers.{layer}.block_sparse_moe.experts."

    def get(name):
        return opener(weight_map[name]).get_tensor(name)

    stats = {"layer": layer, "golden": []}
    # Accumulate per-rank packed planes.
    rank_parts = {
        r: {p: [] for p in PART_SIZES} for r in range(TP)
    }
    for e in range(E):
        w1p = get(f"{pfx}{e}.w1.weight_packed")   # [3072, 1792]
        w1s = get(f"{pfx}{e}.w1.weight_scale")    # [3072, 112]
        w3p = get(f"{pfx}{e}.w3.weight_packed")
        w3s = get(f"{pfx}{e}.w3.weight_scale")
        w2p = get(f"{pfx}{e}.w2.weight_packed")   # [3584, 1536]
        w2s = get(f"{pfx}{e}.w2.weight_scale")    # [3584, 96]
        for r in range(TP):
            r0, r1 = r * IR, (r + 1) * IR
            w13p = torch.cat((w1p[r0:r1], w3p[r0:r1]), dim=0)  # [1536, 1792]
            w13s = torch.cat((w1s[r0:r1], w3s[r0:r1]), dim=0)  # [1536, 112]
            w2pr = w2p[:, r * (IR // 2):(r + 1) * (IR // 2)]   # [3584, 384]
            w2sr = w2s[:, r * (IR // 32):(r + 1) * (IR // 32)]  # [3584, 24]
            if SCALE_SEARCH:
                codes13, sc13 = scale_search_codes(w13p, w13s)
                codes2, sc2 = scale_search_codes(w2pr, w2sr)
            else:
                codes13, sc13 = mxfp4_to_codes(w13p), w13s
                codes2, sc2 = mxfp4_to_codes(w2pr), w2sr
            rank_parts[r]["planes13"].append(pack_fragment_major(codes13))
            rank_parts[r]["sc13"].append(pack_scales(sc13))
            rank_parts[r]["planes2"].append(pack_fragment_major(codes2))
            rank_parts[r]["sc2"].append(pack_scales(sc2))
            if e < golden and r == 0:
                ref13 = mxfp4_dequant(w13p, w13s)
                got13 = reference_dequant(codes13, sc13)
                ref2 = mxfp4_dequant(w2pr, w2sr)
                got2 = reference_dequant(codes2, sc2)
                for tag, ref, got in (("w13", ref13, got13),
                                      ("w2", ref2, got2)):
                    err = (got.double() - ref.double())
                    rel = (err.norm() / ref.double().norm().clamp_min(1e-12))
                    stats["golden"].append({
                        "expert": e, "matrix": tag,
                        "rel_rms": round(float(rel), 6),
                        "max_abs": round(float(err.abs().max()), 6),
                    })
        if e % 128 == 0:
            log(f"  layer {layer} expert {e}/{E} "
                f"({time.time() - t0:.0f}s)")
    # Write per-rank part files.
    sizes = {}
    for r in range(TP):
        meta = meta_for(r, ckpt_id)
        d = os.path.join(out, "v2", meta_key(meta), f"tp{TP}-rank{r}")
        os.makedirs(d, exist_ok=True)
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp):
            with open(mp, "w") as f:
                json.dump(meta, f)
        for part, chunks in rank_parts[r].items():
            buf = torch.stack(chunks).numpy().tobytes()
            assert len(buf) == PART_SIZES[part], (part, len(buf))
            sizes[part] = len(buf)
            path = os.path.join(d, f"layer{layer}.{part}.bin")
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(buf)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
    stats["seconds"] = round(time.time() - t0, 1)
    stats["bytes_per_rank"] = sum(sizes.values())
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/data/models/moonshotai-Kimi-K3")
    ap.add_argument("--out", default="/data/models/k3-moet-packs")
    ap.add_argument("--layers", default="",
                    help="e.g. '1' or '1-92'; default all 92 MoE layers")
    ap.add_argument("--ckpt-id", default="k3-mxfp4-moet-v1")
    ap.add_argument("--golden", type=int, default=4,
                    help="experts per layer to golden-check (rank 0)")
    ap.add_argument("--scale-search", action="store_true",
                    help="per-block-32 power-of-two scale shift search; "
                         "needs its own --out and --ckpt-id")
    args = ap.parse_args()
    global SCALE_SEARCH
    SCALE_SEARCH = args.scale_search
    if SCALE_SEARCH and (args.ckpt_id == "k3-mxfp4-moet-v1"
                         or os.path.abspath(args.out)
                         == "/data/models/k3-moet-packs"):
        ap.error("--scale-search writes different bytes: give it a new "
                 "--ckpt-id and --out (sidecar layers_done is per --out)")

    torch.set_num_threads(int(os.getenv("CONVERT_THREADS", "16")))

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    idx = json.load(open(os.path.join(args.source,
                                      "model.safetensors.index.json")))
    weight_map = idx["weight_map"]
    layers = MOE_LAYERS
    if args.layers:
        if "-" in args.layers:
            a, b = args.layers.split("-")
            layers = list(range(int(a), int(b) + 1))
        else:
            layers = [int(x) for x in args.layers.split(",")]

    # Sanity: tensor names/shapes for layer 1 expert 0.
    from safetensors import safe_open
    handles = {}

    def handle(fn):
        if fn not in handles:
            handles[fn] = safe_open(
                os.path.join(args.source, fn), framework="pt", device="cpu")
        return handles[fn]

    pfx = "language_model.model.layers.1.block_sparse_moe.experts.0."
    for name, want in (
            ("w1.weight_packed", (3072, 1792)),
            ("w1.weight_scale", (3072, 112)),
            ("w2.weight_packed", (3584, 1536)),
            ("w2.weight_scale", (3584, 96)),
            ("w3.weight_packed", (3072, 1792)),
            ("w3.weight_scale", (3072, 112))):
        full = pfx + name
        h = handle(weight_map[full])
        sl = h.get_slice(full)
        assert tuple(sl.get_shape()) == want, (full, sl.get_shape())
    log("source tensor names/shapes verified (w1=gate, w3=up, w2=down; "
        "mxfp4 e2m1 packed + e8m0 block-32 scales)")

    os.makedirs(args.out, exist_ok=True)
    sidecar_path = os.path.join(args.out, "k3-moet-packs.json")
    sidecar = {}
    if os.path.exists(sidecar_path):
        sidecar = json.load(open(sidecar_path))
    sidecar.update({
        "kind": "k3-moet-planes-packs",
        "source": args.source,
        "ckpt_id": args.ckpt_id,
        "format": "moe_w2_planes_cache v2",
        "quantizer_abi": MOE_W2_QUANTIZER_ABI,
        "geometry": {"E": E, "H": H, "I": I, "TP": TP,
                     "N13": N13, "K13": K13, "N2": N2, "K2": K2},
        "part_sizes": PART_SIZES,
        "moe_layers": [layers[0], layers[-1]],
        "scale_search": (list(SS_SHIFTS) if SCALE_SEARCH else None),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    done = sidecar.get("layers_done", {})
    t_all = time.time()
    for layer in layers:
        if str(layer) in done:
            log(f"layer {layer} already done, skipping")
            continue
        st = convert_layer(layer, handle, weight_map, args.out,
                           args.ckpt_id, args.golden, log)
        done[str(layer)] = st
        sidecar["layers_done"] = done
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=1)
        log(f"layer {layer} DONE in {st['seconds']}s "
            f"({st['bytes_per_rank'] / 2**30:.2f} GiB/rank); "
            f"golden: {st['golden'][:2]}")
    dt = time.time() - t_all
    log(f"ALL DONE: {len(layers)} layers in {dt:.0f}s "
        f"({dt / max(len(layers), 1):.1f}s/layer)")


if __name__ == "__main__":
    main()
