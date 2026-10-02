import sqlite3, sys, re, glob, statistics, collections, json
CATS = [
 ("pp-token-bcast", r"broadcast"),
 ("pp-sendrecv", r"sendrecv"),
 ("tp-comm", r"b12xcomm|allgather|reducescatter|allreduce|oneshot|twoshot|dcp_a2a|lse_reduce"),
 ("moe-experts", r"b12xmoe|moe_w2"),
 ("moe-topk/router", r"topk|sigmoid|router|grouped_topk|gating|moe_align|expert_tokens|per_token_group_quant|_desc_build"),
 ("attn/mla", r"mla|attn|flash|fmha|paged|dcp"),
 ("kda/linear-attn", r"mamba|kda|ssm|conv1d|chunk|gated_delta|recurrent"),
 ("dense-gemm", r"gemm|nvjet|cutlass|mma|matmul|skinny|vllmmode|b12x_lib|gemv"),
 ("norm/elementwise", r"norm|elementwise|triton|reduce|copy|fill|index|scatter|gather|cat"),
]
def cat(n):
    l = n.lower()
    for c, p in CATS:
        if re.search(p, l): return c
    return "other"
def analyze(f):
    c = sqlite3.connect(f)
    if not c.execute("select name from sqlite_master where name='CUPTI_ACTIVITY_KIND_KERNEL'").fetchone(): return None
    dev = c.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    ks = c.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start").fetchall()
    gl = [r[0] for r in c.execute("select r.start from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where s.value like 'cudaGraphLaunch%' order by r.start")]
    lo, hi = ks[0][0], ks[-1][1]; span = hi - lo
    a, b = lo + span * 0.1, lo + span * 0.9
    gl = [t for t in gl if a <= t <= b]
    per = statistics.median([y - x for x, y in zip(gl, gl[1:])]) / 1e6 if len(gl) > 2 else float("nan")
    steps = max(1, len(gl) - 1) if len(gl) > 2 else 1
    w = (gl[-1] - gl[0]) if len(gl) > 2 else (b - a)
    acc = collections.Counter(); names = collections.Counter(); busy = []
    for s, e, n in ks:
        s2, e2 = max(s, gl[0] if len(gl) > 2 else a), min(e, gl[-1] if len(gl) > 2 else b)
        if e2 <= s2: continue
        k = cat(n); acc[k] += e2 - s2; names[(k, n[:60])] += e2 - s2
        if k != "pp-token-bcast": busy.append((s2, e2))
    busy.sort(); u = 0; cs = ce = None
    for s, e in busy:
        if ce is None or s > ce:
            if ce is not None: u += ce - cs
            cs, ce = s, e
        else: ce = max(ce, e)
    if ce is not None: u += ce - cs
    return dict(file=f, dev=[d[0] for d in dev], period=per, steps=steps,
                busy=100 * u / w, idle_ms=(w - u) / 1e6 / steps,
                cats={k: v / 1e6 / steps for k, v in acc.items()},
                top=[(k, n, v / 1e6 / steps) for (k, n), v in names.most_common(14)])
out = [r for r in (analyze(f) for f in sorted(glob.glob(sys.argv[1] + "/*/*.sqlite"))) if r]
json.dump(out, open(sys.argv[2], "w"), indent=1)
for r in out:
    print(r["file"].split("/")[-2], r["file"][-11:-7], "dev", r["dev"], "period %.2fms busy %.1f%% idle %.2fms" % (r["period"], r["busy"], r["idle_ms"]),
          " ".join("%s=%.2f" % (k, v) for k, v in sorted(r["cats"].items(), key=lambda x: -x[1])))
