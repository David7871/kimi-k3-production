#!/usr/bin/env python3
"""Per-stage ms/step by kernel category from ~/nsys3/k3cmp.py JSON output.

Usage: nsys_stage_cmp.py BASE.json [ARM.json ...]
Each JSON is k3cmp.py NSYS_DIR OUT.json for one capture (16 ranks). Ranks are
averaged within a PP stage (rtx-01 dev 0-3 = S0, 4-7 = S1; rtx-02 = S2, S3).
With more than one file, prints each arm's per-category delta vs BASE.
"""
import json
import statistics as st
import sys


def stage(r):
    dev = r["dev"][0]
    return (0 if dev < 4 else 1) + (2 if "/rtx-02/" in r["file"] else 0)


def load(path):
    per = {}
    for r in json.load(open(path)):
        per.setdefault(stage(r), []).append(r)
    out = {}
    for s, rs in sorted(per.items()):
        cats = {}
        for r in rs:
            for k, v in r["cats"].items():
                cats.setdefault(k, []).append(v)
        out[s] = dict(n=len(rs), period=st.mean(r["period"] for r in rs),
                      busy=st.mean(r["busy"] for r in rs),
                      cats={k: sum(v) / len(rs) for k, v in cats.items()})
    return out


files = sys.argv[1:]
data = [load(f) for f in files]
names = [f.rsplit("/", 1)[-1].removesuffix(".json") for f in files]
cats = sorted({k for d in data for s in d.values() for k in s["cats"]},
              key=lambda k: -sum(s["cats"].get(k, 0) for s in data[0].values()))
for s in sorted(data[0]):
    print(f"== stage S{s} (ranks: {', '.join(str(d[s]['n']) for d in data if s in d)})")
    print(f"{'ms/step':18s}" + "".join(f"{n:>16s}" for n in names)
          + "".join(f"{'d ' + n:>16s}" for n in names[1:]))
    rows = [("step period", "period"), ("busy %", "busy")] + [(k, k) for k in cats]
    for label, key in rows:
        vals = [(d[s][key] if key in ("period", "busy") else d[s]["cats"].get(key, 0.0))
                if s in d else float("nan") for d in data]
        line = f"{label:18s}" + "".join(f"{v:16.2f}" for v in vals)
        line += "".join(f"{v - vals[0]:+16.2f}" for v in vals[1:])
        print(line)
