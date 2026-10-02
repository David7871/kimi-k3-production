# Usage: roof.py NSYS_DIR STAGE -> per-layer median kernel trace (KDA / MLA layers), non-layer kernels per cycle,
# exclusive (priority-attributed, overlap-free) time per category per cycle; mean over the 4 TP ranks of STAGE.
import sys, os, collections, statistics, sqlite3, glob
D, ST = sys.argv[1], int(sys.argv[2])
src = open(os.path.expanduser("~/nsys3/timeline.py")).read()
sys.argv = [sys.argv[0], D, "/dev/null", "1", "x"]
exec(src[:src.index("# window")])
LO = max(r["ks"][0][0] for r in ranks); HI = min(r["ks"][-1][1] for r in ranks)
SP = HI - LO; LO += 0.1 * SP; HI -= 0.1 * SP
for r in ranks: r["ks"] = sorted(r["ks"])
ranks.sort(key=lambda r: (r["stage"], r["dev"]))
ends = sorted(e for s, e, n in ranks[0]["ks"] if "Broadcast" in n and LO <= e <= HI)
bnd = []
for e in ends:
    if not bnd or e - bnd[-1] > 1e6: bnd.append(e)
W0, W1, NCYC = bnd[0], bnd[-1], len(bnd) - 1
print("%s S%d: %d cycles, %.3f ms/cycle" % (D, ST, NCYC, (W1 - W0) / 1e6 / NCYC))

PRI = ["moe-experts", "dense-gemm", "kda/linear-attn", "attn/mla", "tp-comm", "norm/elementwise", "moe-topk/router", "other", "pp-sendrecv"]
excl = collections.Counter(); types = collections.defaultdict(list); extra = collections.Counter(); extra_n = collections.Counter()
for r in [x for x in ranks if x["stage"] == ST]:
    ks = [(max(s, W0), min(e, W1), n) for s, e, n in r["ks"] if e > W0 and s < W1 and "Broadcast" not in n]
    # exclusive time: sweep, attribute each instant to the highest-priority running category
    ev = []
    for s, e, n in ks:
        c = cat(n) if "SendRecv" not in n else "pp-sendrecv"
        p = PRI.index(c) if c in PRI else len(PRI) - 2
        ev += [(s, 1, p), (e, -1, p)]
    ev.sort(); act = collections.Counter(); t = W0
    for x, d, p in ev:
        if act:
            excl[PRI[min(k for k, v in act.items() if v > 0)]] += x - t
        t = x; act[p] += d
        if act[p] == 0: del act[p]
    # layer segmentation on the main (longest-total) b12xmoe kernel
    tot = collections.Counter()
    for s, e, n in ks:
        if "b12xmoe" in n: tot[n] += e - s
    main = tot.most_common(1)[0][0]
    mends = [e for s, e, n in ks if n == main]
    inlayer = set()
    for a, b in zip(mends, mends[1:]):
        seg = [(s, e, n) for s, e, n in ks if a < e <= b]
        if b - a > 5e6: continue  # crosses a pipeline wait
        names = tuple(n for s, e, n in seg)
        kind = "MLA" if any("MLADecode" in n or "b12xatte" in n for n in names) else ("KDA" if any("kda_decode" in n for n in names) else "?")
        types[kind].append((names, [(s - a, e - s) for s, e, n in seg], b - a))
        for s, e, n in seg: inlayer.add((s, n))
    for s, e, n in ks:
        if (s, n) not in inlayer: extra[n] += e - s; extra_n[n] += 1

nr = 4
print("\nexclusive ms/cycle (overlap-free; priority %s)" % " > ".join(PRI))
for c in PRI: print("  %-18s %7.3f" % (c, excl[c] / nr / 1e6 / NCYC))
print("  %-18s %7.3f" % ("SUM busy", sum(excl.values()) / nr / 1e6 / NCYC))
for kind, L in sorted(types.items()):
    sig, cnt = collections.Counter(x[0] for x in L).most_common(1)[0]
    rows = [x for x in L if x[0] == sig]
    print("\n== %s layer: %.1f layers/cycle/rank, modal signature in %d/%d, median wall %.1f us" % (
        kind, len(L) / nr / NCYC, len(rows), len(L), statistics.median(x[2] for x in rows) / 1e3))
    for i, n in enumerate(sig):
        st = statistics.median(x[1][i][0] for x in rows) / 1e3; du = statistics.median(x[1][i][1] for x in rows) / 1e3
        print("  %2d  start %7.1f  dur %7.1f us  %-16s %s" % (i, st, du, cat(n), n[:70]))
print("\nkernels outside layer segments (ms/cycle, calls/cycle)")
for n, t in extra.most_common(12):
    print("  %7.3f ms %5.1f calls  %-16s %s" % (t / nr / 1e6 / NCYC, extra_n[n] / nr / NCYC, cat(n), n[:70]))
f = sorted(glob.glob(D + "/rtx-02/*.sqlite"))[0]
c = sqlite3.connect(f)
for pat in ("%topk%", "%b12xmoe%"):
    for row in c.execute("select s.value, k.gridX, k.gridY, k.gridZ, count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName "
                         "where s.value like ? group by 1,2,3,4 order by 5 desc limit 4", (pat,)):
        print("GRID", row[0][:50], row[1:])
