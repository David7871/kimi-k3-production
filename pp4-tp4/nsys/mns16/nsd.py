"""Cross-rank nsys analyzer: per-rank busy breakdown + aligned ASCII timeline."""
import sqlite3, sys, glob, os, collections

import re
files = sorted(f for f in glob.glob(sys.argv[1] + "/*/*.sqlite")
               if re.search(sys.argv[4] if len(sys.argv) > 4 else ".", f))
WIN_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0
RES = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5  # ms per char


def cat(name):
    n = name.lower()
    if "broadcast" in n: return "B"
    if "sendrecv" in n: return "P"
    if "nccl" in n: return "T"  # TP collectives
    if "allreduce" in n or "oneshot" in n or "twoshot" in n: return "A"
    return "C"


ranks = []
for f in files:
    c = sqlite3.connect(f)
    try:
        base = c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    except Exception:
        base = 0
    host = os.path.basename(os.path.dirname(f))
    ks = c.execute("select k.start,k.end,k.streamId,k.deviceId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id").fetchall()
    if not ks:
        continue
    dev = collections.Counter(k[3] for k in ks).most_common(1)[0][0]
    rt = c.execute("select r.start,r.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id").fetchall()
    # NVTX ranges if any
    try:
        nv = c.execute("select start,end,text from NVTX_EVENTS where end is not null").fetchall()
    except Exception:
        nv = []
    ranks.append(dict(f=f, host=host, dev=dev, base=base, ks=ks, rt=rt, nv=nv))

# global aligned window: middle of the overlapping capture
starts = [r["base"] + min(k[0] for k in r["ks"]) for r in ranks]
ends = [r["base"] + max(k[1] for k in r["ks"]) for r in ranks]
lo, hi = max(starts), min(ends)
T0 = lo + (hi - lo) / 2 - WIN_MS * 1e6 / 2
T1 = T0 + WIN_MS * 1e6
print("overlap %.1f ms; window %.1f ms" % ((hi - lo) / 1e6, WIN_MS))

prio = "BPTAC"  # later wins when drawing
for r in sorted(ranks, key=lambda r: (r["host"], r["dev"])):
    busy = collections.Counter()
    ncols = int(WIN_MS / RES)
    lane = ["."] * ncols
    comp_union = []
    for st, en, sid, dev, name in r["ks"]:
        a, b = r["base"] + st, r["base"] + en
        if b < T0 or a > T1: continue
        a, b = max(a, T0), min(b, T1)
        k = cat(name)
        busy[k] += b - a
        if k == "C" or k == "A": comp_union.append((a, b))
        for i in range(int((a - T0) / 1e6 / RES), min(ncols, int((b - T0) / 1e6 / RES) + 1)):
            cur = lane[i]
            if cur == "." or prio.index(k) > prio.index(cur if cur in prio else "B"):
                lane[i] = k
    # union of compute (multi-stream overlap)
    comp_union.sort(); u = 0; ce = 0
    for a, b in comp_union:
        if b <= ce: continue
        u += b - max(a, ce); ce = b
    launches = [r["base"] + s for s, e, n in r["rt"] if n.startswith("cudaGraphLaunch") and T0 <= r["base"] + s <= T1]
    hl = ["."] * ncols
    for s in launches:
        hl[min(ncols - 1, int((s - T0) / 1e6 / RES))] = "L"
    # long host API calls (>0.5ms)
    longapi = collections.Counter()
    for s, e, n in r["rt"]:
        if T0 <= r["base"] + s <= T1 and e - s > 0.5e6:
            longapi[n.split("_v")[0]] += (e - s) / 1e6
    tag = "%s d%d" % (r["host"][-6:], r["dev"])
    print("\n%-12s compute-union %4.1f%%  |  " % (tag, 100 * u / (T1 - T0)) +
          " ".join("%s=%.1fms" % (k, v / 1e6) for k, v in sorted(busy.items())) +
          "  launches=%d" % len(launches) +
          ("  longAPI " + ",".join("%s:%.1f" % kv for kv in longapi.most_common(3)) if longapi else ""))
    print("  GPU  " + "".join(lane))
    print("  host " + "".join(hl))
print("\nlegend: C compute  A custom allreduce  T nccl TP coll  P PP sendrecv  B token broadcast  . idle  L cudaGraphLaunch; %.2fms/char" % RES)
