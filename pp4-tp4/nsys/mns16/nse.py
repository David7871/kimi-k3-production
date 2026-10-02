import sqlite3, sys, re, collections, statistics

def bucket(n):
    l = n.lower()
    rules = [
        ("nccl-broadcast(tokens)", "broadcast"), ("nccl-sendrecv(PP)", "sendrecv"),
        ("nccl-allgather", "allgather"), ("nccl-reducescatter", "reducescatter"),
        ("nccl-allreduce", "ncclallreduce"), ("nccl-other", "nccl"),
        ("custom-allreduce(pcie/b12x)", "allreduce|oneshot|twoshot|pcie_ar|cross_device|custom_ar"),
        ("moe-b12x", "b12xmoe|moe|topk|expert|router|sigmoid"),
        ("attn/mla", "mla|attn|flash|fmha|paged|dcp"),
        ("mamba/kda/linear-attn", "mamba|kda|ssm|conv1d|chunk|gated_delta|recurrent"),
        ("gemm", "gemm|nvjet|cutlass|mma|matmul|sm120"),
        ("norm/elementwise/triton", "norm|elementwise|triton|reduce|copy|fill|index|scatter|gather|cat"),
    ]
    for name, pat in rules:
        if re.search(pat, l): return name
    return "other"

for f in sys.argv[1:]:
    c = sqlite3.connect(f)
    ks = c.execute("select k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start").fetchall()
    rt = c.execute("select r.start,r.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id order by r.start").fetchall()
    lo, hi = ks[0][0], ks[-1][1]
    # steady region: middle 80%
    a0, a1 = lo + (hi - lo) * 0.1, lo + (hi - lo) * 0.9
    W = a1 - a0
    tot = collections.Counter(); names = collections.Counter()
    for st, en, sid, n in ks:
        if en < a0 or st > a1: continue
        d = min(en, a1) - max(st, a0)
        b = bucket(n); tot[b] += d; names[(b, n[:50])] += d
    # compute-union excluding waiting nccl broadcast/sendrecv
    iv = sorted((max(st, a0), min(en, a1)) for st, en, sid, n in ks if en > a0 and st < a1 and not re.search("broadcast|sendrecv", n.lower()))
    u = 0; ce = 0; gaps = []
    for s, e in iv:
        if s > ce and ce: gaps.append(s - ce)
        if e <= ce: continue
        u += e - max(s, ce); ce = e
    L = [s for s, e, n in rt if n.startswith("cudaGraphLaunch") and a0 <= s <= a1]
    Ld = [(e - s) for s, e, n in rt if n.startswith("cudaGraphLaunch") and a0 <= s <= a1]
    per = [(L[i + 1] - L[i]) / 1e6 for i in range(len(L) - 1)]
    api = collections.Counter()
    for s, e, n in rt:
        if a0 <= s <= a1: api[n.split("_v")[0]] += e - s
    print("=" * 100); print(f, "steady window %.0fms" % (W / 1e6))
    print("busy(non-wait kernels union) %.1f%%   launches %d  period median %.2fms p90 %.2fms   graphLaunch host cost median %.2fms max %.2fms" % (
        100 * u / W, len(L), statistics.median(per), sorted(per)[int(.9 * len(per))], statistics.median(Ld) / 1e6, max(Ld) / 1e6))
    big = [g for g in gaps if g > 50e3]
    print("idle gaps >50us: n=%d total %.1fms (%.1f%%)  median %.0fus max %.0fus" % (len(big), sum(big) / 1e6, 100 * sum(big) / W, statistics.median(big) / 1e3 if big else 0, max(big) / 1e3 if big else 0))
    print("per-step (ms, per graph-launch period):")
    n = max(1, len(L))
    for b, v in tot.most_common():
        print("   %-30s %6.2f ms/step  %5.1f%% of wall" % (b, v / 1e6 / n, 100 * v / W))
    print("top kernels:")
    for (b, nm), v in names.most_common(14):
        print("   %6.2f ms/step  %-26s %s" % (v / 1e6 / n, b, nm))
    print("host API top:", ", ".join("%s %.1fms/step" % (k, v / 1e6 / n) for k, v in api.most_common(6)))
