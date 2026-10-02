# Usage: gen_html.py NSYS_DIR out.html "TITLE"
# Interactive, scrollable/zoomable timeline of all ranks on one clock (rtx-02 aligned to rtx-01 via SendRecv).
import sys, os, base64, json, array
d, OUT_HTML, TITLE = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(os.path.expanduser("~/nsys3/timeline.py")).read()
sys.argv = [sys.argv[0], d, "/dev/null", "1", TITLE]
exec(src[:src.index("# window")])          # loads ranks (host, dev, stage, ks) + clock offset `off`, defines cat/COL

CATNAMES = list(COL)
MERGE_NS = 3000                            # merge same-category kernels separated by < 3 us
lo = max(r["ks"][0][0] for r in ranks)     # common span of all ranks
hi = min(r["ks"][-1][1] for r in ranks)
ranks.sort(key=lambda r: (r["stage"], r["dev"]))

def enc(a): return base64.b64encode(a.tobytes()).decode()
rows = []
for r in ranks:
    lanes = {"main": ([], [], [], []), "wait": ([], [], [], [])}
    for s, e, n in sorted(r["ks"]):
        if e < lo or s > hi: continue
        c = cat(n); ln = lanes["wait" if c == "pp-token-bcast" else "main"]
        S, E, C, N = ln
        if S and C[-1] == CATNAMES.index(c) and s - E[-1] < MERGE_NS:
            E[-1] = max(E[-1], e); N[-1] += 1
        else:
            S.append(s); E.append(e); C.append(CATNAMES.index(c)); N.append(1)
    row = dict(label="S%d %s gpu%d" % (r["stage"], r["host"], r["dev"]), stage=r["stage"])
    for k, (S, E, C, N) in lanes.items():
        row[k] = dict(s=enc(array.array("i", [int((x - lo) / 100) for x in S])),     # 100 ns units
                      d=enc(array.array("i", [max(1, int((e - s) / 100)) for s, e in zip(S, E)])),
                      c=enc(array.array("B", C)), n=enc(array.array("H", [min(x, 65535) for x in N])))
    rows.append(row)
    print(row["label"], "segments", len(lanes["main"][0]), file=sys.stderr)

import datetime
meta = dict(title=TITLE, t0=datetime.datetime.fromtimestamp(lo / 1e9, datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3],
            span_ms=(hi - lo) / 1e6, offset_ms=off / 1e6, cats=CATNAMES, colors=[COL[c] for c in CATNAMES])
html = open(os.path.expanduser("~/nsys3/timeline_template.html")).read()
html = html.replace("/*DATA*/", "const META=%s;\nconst ROWS=%s;" % (json.dumps(meta), json.dumps(rows)))
open(OUT_HTML, "w").write(html)
print(OUT_HTML, "%.1f MB" % (len(html) / 1e6), "span %.0f ms" % meta["span_ms"])
