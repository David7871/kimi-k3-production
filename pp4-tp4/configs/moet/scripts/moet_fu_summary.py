#!/usr/bin/env python3
"""Summarize moet-followup.sh arms: decode medians, KV, tokens/step during
C256, GSM8K accuracy + completion tokens, paired comparison vs fu-off."""
import json
import os
import re
import statistics as st

R = os.path.expanduser("~/k3-results")
ARMS = ["fu-off", "fu-on", "fu-rb"]  # the dropped fu-topp* arms are excluded from this export
NO_GSM = {"fu-rb"}          # moet-followup.sh runs GSM8K with gsm=0 here
RUNS = {16: 3, 256: 3}      # decode.txt runs an arm must have
GSM_N = 200


def decode(tag):
    out = {}
    try:
        for line in open(f"{R}/{tag}/decode.txt"):
            m = re.match(r"C(\d+) decode=([\d.]+)", line)
            if m:
                out.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    except OSError:
        pass
    return out


def kv(tag):
    try:
        for line in open(f"{R}/{tag}/head.log"):
            m = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)
            if m:
                return m.group(1)
    except OSError:
        pass
    return "?"


def hist(path):
    b, s, c = {}, 0.0, 0.0
    for line in open(path):
        m = re.match(r'vllm:iteration_tokens_total_bucket\{.*le="([^"]+)".*\} ([\d.e+]+)', line)
        if m:
            b[m.group(1)] = b.get(m.group(1), 0.0) + float(m.group(2))
        elif line.startswith("vllm:iteration_tokens_total_sum"):
            s += float(line.split()[-1])
        elif line.startswith("vllm:iteration_tokens_total_count"):
            c += float(line.split()[-1])
    return b, s, c


def itok(tag):
    try:
        b0, s0, c0 = hist(f"{R}/{tag}/itok-pre256.txt")
        b1, s1, c1 = hist(f"{R}/{tag}/itok-post256.txt")
    except OSError:
        return "n/a"
    n = c1 - c0
    if n <= 0:
        return "no steps"
    keys = sorted(b1, key=lambda k: float("inf") if k == "+Inf" else float(k))
    cum = {k: b1[k] - b0.get(k, 0.0) for k in keys}
    parts, prev = [], 0.0
    for k in keys:
        if cum[k] - prev > 0:
            parts.append(f"<={k}:{(cum[k] - prev) / n:.0%}")
        prev = cum[k]
    return f"steps={n:.0f} mean={(s1 - s0) / n:.1f} tok/step [{' '.join(parts)}]"


def gsm(tag):
    try:
        return [json.loads(l) for l in open(f"{R}/{tag}/gsm8k.jsonl")]
    except (OSError, ValueError):
        return None


def gsm_errors(g):
    """Rows with no server response (request error / dead engine)."""
    return sum(r.get("finish_reason") is None
               or str(r.get("tail", "")).startswith("ERROR:") for r in g)


def gsm_complete(g):
    return g is not None and len(g) == GSM_N and gsm_errors(g) == 0


def probe(tag):
    """True/False = 391 found or not; None = no usable response."""
    try:
        m = json.load(open(f"{R}/{tag}/probe391.json"))["choices"][0]["message"]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    return "391" in (m.get("content") or "") + (m.get("reasoning_content") or "")


def missing(tag, d, g, p):
    out = [f"C{c} runs {len(d.get(c, []))}/{n}"
           for c, n in RUNS.items() if len(d.get(c, [])) != n]
    if p is None:
        out.append("probe391 missing")
    if tag not in NO_GSM and not gsm_complete(g):
        out.append("GSM8K missing" if g is None else
                   f"GSM8K rows {len(g)}/{GSM_N} errors={gsm_errors(g)}")
    return out


base = gsm("fu-off")
for t in ARMS:
    d, g, p = decode(t), gsm(t), probe(t)
    miss = missing(t, d, g, p)
    status = "INCOMPLETE: " + "; ".join(miss) if miss else "OK"
    print(f"{t:10s} {status} | probe391={'-' if p is None else 'PASS' if p else 'FAIL'}")
    fmt = lambda c: (f"{st.median(d[c]):.1f} ({min(d[c]):.1f}-{max(d[c]):.1f})"
                     if d.get(c) else "-")
    print(f"{'':10s} C16 {fmt(16):22s} C256 {fmt(256):24s} KV {kv(t)}")
    print(f"{'':10s} C256 {itok(t)}")
    if g:
        toks = [r.get("completion_tokens") or 0 for r in g]
        line = (f"{'':10s} GSM8K {sum(r['ok'] for r in g)}/{len(g)} "
                f"tok_mean={st.mean(toks):.1f} tok_median={st.median(toks):.0f} "
                f"truncated={sum(r.get('finish_reason') == 'length' for r in g)}")
        if t != "fu-off" and not (gsm_complete(g) and gsm_complete(base)):
            line += " | vs off: skipped (arm or fu-off GSM8K incomplete)"
        elif t != "fu-off":
            n = GSM_N
            b = sum(g[i]["ok"] and not base[i]["ok"] for i in range(n))
            c = sum(base[i]["ok"] and not g[i]["ok"] for i in range(n))
            both = [i for i in range(n) if g[i]["ok"] and base[i]["ok"]
                    and g[i].get("completion_tokens") and base[i].get("completion_tokens")]
            ratio = (sum(g[i]["completion_tokens"] for i in both)
                     / max(1, sum(base[i]["completion_tokens"] for i in both)))
            line += (f" | vs off: arm_only={b} off_only={c} "
                     f"tokens_ratio(both-correct n={len(both)})={ratio:.3f}")
        print(line)
