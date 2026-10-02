import sqlite3, glob, sys, re, collections, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import importlib.util as _u, os as _o; _k=_o.path.expanduser("~/nsys3/k3cmp.py"); _s=_u.spec_from_file_location("k3c",_k); exec(open(_k).read().split("def analyze")[0])
COL = {"moe-experts":"#d62728","dense-gemm":"#1f77b4","tp-comm":"#ff7f0e","moe-topk/router":"#9467bd",
       "norm/elementwise":"#8c564b","attn/mla":"#2ca02c","pp-sendrecv":"#e377c2","kda/linear-attn":"#17becf",
       "pp-token-bcast":"#cccccc","other":"#7f7f7f"}
d, out, win_ms, title = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
ranks = []
for f in sorted(glob.glob(d + "/*/*.sqlite")):
    c = sqlite3.connect(f)
    if not c.execute("select name from sqlite_master where name=\x27CUPTI_ACTIVITY_KIND_KERNEL\x27").fetchone(): continue
    t0 = c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    dev = c.execute("select deviceId from CUPTI_ACTIVITY_KIND_KERNEL limit 1").fetchone()[0]
    ks = c.execute("select k.start+?, k.end+?, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start", (t0, t0)).fetchall()
    host = "rtx-01" if "rtx-01" in f else "rtx-02"
    stage = (0 if dev < 4 else 1) if host == "rtx-01" else (2 if dev < 4 else 3)
    ranks.append(dict(host=host, dev=dev, stage=stage, ks=ks))
# cross-host clock offset: stage1 (rtx-01) and stage2 (rtx-02) SendRecv kernels are the same transfers -> their ends coincide
def sr_ends(rs): return np.array(sorted(e for r in rs for s, e, n in r["ks"] if "SendRecv" in n))
a = sr_ends([r for r in ranks if r["stage"] == 1]); b = sr_ends([r for r in ranks if r["stage"] == 2])
off = 0
if len(a) and len(b):
    diffs = (a[None, ::max(1, len(a)//400)] - b[::max(1, len(b)//400), None]).ravel()
    diffs = diffs[np.abs(diffs) < 5e9]
    h, edges = np.histogram(diffs, bins=np.arange(diffs.min(), diffs.max() + 2e4, 2e4))
    off = edges[h.argmax()] + 1e4
for r in ranks:
    if r["host"] == "rtx-02": r["ks"] = [(s + off, e + off, n) for s, e, n in r["ks"]]
# window: middle of the time span common to all ranks
# reference window = time span of the majority host (rtx-02); drop ranks that recorded a different window
ref = [r for r in ranks if r["host"] == "rtx-02"] or ranks
lo = max(r["ks"][0][0] for r in ref); hi = min(r["ks"][-1][1] for r in ref)
cand = [r for r in ranks if r["ks"][0][0] < hi and r["ks"][-1][1] > lo]
lo = max(r["ks"][0][0] for r in cand); hi = min(r["ks"][-1][1] for r in cand)
w0 = (lo + hi) / 2 - win_ms * 0.5e6; w1 = w0 + win_ms * 1e6
dropped = [r for r in ranks if r["ks"][0][0] > w0 or r["ks"][-1][1] < w1]
ranks = [r for r in ranks if r not in dropped]
if dropped: title += "  (%d ranks recorded a different time window, omitted)" % len(dropped)
ranks.sort(key=lambda r: (r["stage"], r["dev"]))
fig, ax = plt.subplots(figsize=(22, 0.55 * len(ranks) + 2.5))
seen = set()
for i, r in enumerate(ranks):
    segs = collections.defaultdict(list)
    for s, e, n in r["ks"]:
        if e < w0 or s > w1: continue
        k = cat(n); segs[k].append(((max(s, w0) - w0) / 1e6, (min(e, w1) - max(s, w0)) / 1e6))
    for k in ["pp-token-bcast"] + [x for x in COL if x != "pp-token-bcast"]:
        if segs[k]:
            hgt = 0.35 if k == "pp-token-bcast" else 0.8
            ax.broken_barh(segs[k], (i - hgt / 2, hgt), facecolors=COL[k], label=k if k not in seen else None, linewidth=0)
            seen.add(k)
ax.set_yticks(range(len(ranks))); ax.set_yticklabels(["S%d %s gpu%d" % (r["stage"], r["host"], r["dev"]) for r in ranks], fontsize=9)
for y in range(len(ranks) - 1):
    if ranks[y]["stage"] != ranks[y + 1]["stage"]: ax.axhline(y + 0.5, color="k", lw=0.8)
ax.invert_yaxis(); ax.set_xlim(0, win_ms); ax.set_xlabel("time (ms) from t0; rtx-02 clock shifted %+.3f ms to align SendRecv with rtx-01" % (off / 1e6))
ax.xaxis.set_major_locator(plt.MultipleLocator(5 if win_ms <= 60 else 10)); ax.grid(axis="x", alpha=0.3)
import datetime; ax.set_title("%s   |   t0 = %s UTC" % (title, datetime.datetime.fromtimestamp(w0 / 1e9, datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3]))
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=10, fontsize=9)
plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.savefig(out, dpi=100)
print(out, "ranks", len(ranks), "offset_ms %.3f" % (off / 1e6))
