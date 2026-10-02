# Usage: fuse.py roof-cXX-s3.txt -> critical-path time of small kernels / gaps per layer and per S3 cycle, plus fusion savings.
# Input: median per-layer kernel traces printed by roof.py (start/dur in us relative to the previous main MoE end).
import re, sys
txt = open(sys.argv[1]).read()
LAYERS = {"KDA": 16, "MLA": 7}            # S3 layers per cycle
FIX = 4.4                                  # fitted per-GEMM fixed cost (us)

def klass(c, n, d):
    if "b12xmoe" in n: return "moe"
    if c == "tp-comm": return "comm"
    if "kda_decode" in n or "MLADecode" in n or "b12xatte" in n: return "attn"
    if "_attn_res" in n: return "attnres"
    if c == "dense-gemm" and "splitK" not in n and d >= 4.0: return "gemm"
    return "small"                         # norms, elementwise, act-quant (<4us cutlass), router glue, split-K reduce

PRI = ["moe", "gemm", "attn", "comm", "attnres", "small"]
res = {}
for kind, body in re.findall(r"== (\w+) layer:.*?\n((?:\s+\d+\s+start.*\n)+)", txt):
    ks = []
    for m in re.finditer(r"\s+(\d+)\s+start\s+([\d.]+)\s+dur\s+([\d.]+) us\s+(\S+)\s+(\S+)", body):
        i, s, d, c, n = int(m[1]), float(m[2]), float(m[3]), m[4], m[5]
        ks.append((s, s + d, klass(c, n, d), n, d))
    t0, t1 = 0.0, max(e for s, e, *_ in ks)
    # exclusive sweep on a 0.1 us grid
    pts = sorted({t0, t1} | {s for s, *_ in ks} | {e for _, e, *_ in ks})
    excl = dict.fromkeys(PRI + ["gap"], 0.0); timeline = []
    for a, b in zip(pts, pts[1:]):
        act = [k for s, e, k, *_ in ks if s < b and e > a]
        lab = min(act, key=PRI.index) if act else "gap"
        excl[lab] += b - a; timeline.append((a, b, lab))
    # runs of {small, attnres, gap} between big kernels (moe/gemm/attn/comm)
    runs, cur = [], None
    for a, b, lab in timeline:
        if lab in ("small", "attnres", "gap"):
            if cur is None: cur = [a, b]
            else: cur[1] = b
        elif cur: runs.append(cur); cur = None
    if cur: runs.append(cur)
    l1 = l2 = 0.0; nrun = 0
    for a, b in runs:
        mem = [(k, d) for s, e, k, n, d in ks if s >= a - 0.05 and e <= b + 0.05 and k in ("small", "attnres")]
        smalls = [d for k, d in mem if k == "small"]
        if not mem: continue
        nrun += 1
        # L1: chain of elementwise/norm/quant kernels -> one kernel (cost = longest member), attn_res kept, one launch gap left
        keep = sum(d for k, d in mem if k == "attnres") + (max(smalls) if smalls else 0) + 0.5 * (1 + sum(1 for k, _ in mem if k == "attnres"))
        l1 += max(0.0, (b - a) - keep) if len(smalls) >= 2 else 0.0
        # L2: everything in the run folded into neighbouring GEMM / all-reduce prologue/epilogue, 0.5 us residual
        l2 += max(0.0, (b - a) - 0.5)
    ngemm = sum(1 for s, e, k, *_ in ks if k == "gemm")
    res[kind] = dict(wall=t1, excl=excl, l1=l1, l2=l2, runs=nrun, ngemm=ngemm, nk=len(ks))

print(sys.argv[1])
for kind, r in res.items():
    e = r["excl"]
    print("  %s layer %.1f us, %d kernels, %d weight GEMMs: critical path  moe %.1f  gemm %.1f  attn %.1f  comm %.1f  attn_res %.1f  small %.1f  gaps %.1f" % (
        kind, r["wall"], r["nk"], r["ngemm"], e["moe"], e["gemm"], e["attn"], e["comm"], e["attnres"], e["small"], e["gap"]))
    print("     %d small-kernel runs; L1 saves %.1f us/layer, L2 saves %.1f us/layer" % (r["runs"], r["l1"], r["l2"]))
cyc = lambda key: sum(LAYERS[k] * (res[k][key] if key in res[k] else 0) for k in res) / 1e3
cx = lambda key: sum(LAYERS[k] * res[k]["excl"][key] for k in res) / 1e3
print("  per S3 cycle (ms): small %.2f  attn_res %.2f  gaps %.2f  -> L1 %.2f  L2 %.2f  GEMM fixed-cost bound %.2f" % (
    cx("small"), cx("attnres"), cx("gap"), cyc("l1"), cyc("l2"), sum(LAYERS[k] * res[k]["ngemm"] for k in res) * FIX / 1e3))
