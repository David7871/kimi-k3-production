import json, statistics as st, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
cats=["moe-experts","dense-gemm","tp-comm","moe-topk/router","norm/elementwise","attn/mla","pp-sendrecv","kda/linear-attn","other"]
col=dict(zip(cats,["#d62728","#1f77b4","#ff7f0e","#9467bd","#8c564b","#2ca02c","#e377c2","#17becf","#7f7f7f"]))
def rows(p):
    out=[]
    for r in json.load(open(p)):
        h="rtx-01" if "rtx-01" in r["file"] else "rtx-02"; dv=r["dev"][0]
        stg={("rtx-01",0):0,("rtx-01",1):1,("rtx-02",0):2,("rtx-02",1):3}[(h,dv//4)]
        out.append((stg,h,dv,r))
    return sorted(out,key=lambda x:(x[0],x[2]))
fig,(a1,a2)=plt.subplots(1,2,figsize=(20,9),gridspec_kw=dict(width_ratios=[3,2]))
R=rows("/tmp/mns256.json"); lab=[]
for i,(stg,h,dv,r) in enumerate(R):
    left=0
    for k in cats:
        v=r["cats"].get(k,0); a1.barh(i,v,left=left,color=col[k],label=k if i==0 else None); left+=v
    idle=r["idle_ms"]; a1.barh(i,idle,left=left,color="#dddddd",hatch="//",label="GPU idle" if i==0 else None)
    a1.text(left+idle+0.4,i,"step %.1f ms, busy %.0f%%"%(r["period"],r["busy"]),va="center",fontsize=8)
    lab.append("S%d %s gpu%d"%(stg,h,dv))
a1.set_yticks(range(len(R))); a1.set_yticklabels(lab,fontsize=8); a1.invert_yaxis()
for y in (3.5,7.5,11.5): a1.axhline(y,color="k",lw=0.6)
a1.set_xlim(0,80); a1.set_xlabel("GPU ms per micro-batch step (~64 seqs/micro-batch)")
a1.set_title("MNS256 @ C256 — all 16 ranks (PP stage S0..S3)")
# per-token comparison stage 2/3 (only stages with full old data)
O=rows("/tmp/new.json"); groups=[]
for stg in (2,3):
    o=[r for s,h,d,r in O if s==stg]; n=[r for s,h,d,r in R if s==stg]
    groups.append(("S%d MNS16\n(4/mb)"%stg,o,4)); groups.append(("S%d MNS256\n(64/mb)"%stg,n,64))
for i,(name,rs,m) in enumerate(groups):
    left=0
    for k in cats:
        v=st.mean(r["cats"].get(k,0) for r in rs)/m; a2.barh(i,v,left=left,color=col[k]); left+=v
    per=st.mean(r["period"] for r in rs)/m
    a2.text(left+0.03,i,"%.2f ms/token"%per,va="center",fontsize=9)
a2.set_yticks(range(len(groups))); a2.set_yticklabels([g[0] for g in groups],fontsize=9); a2.invert_yaxis()
a2.set_xlabel("GPU ms per token per stage"); a2.set_xlim(0,4.3)
a2.set_title("Cost per token: MNS16 vs MNS256 (rtx-02 stages)")
fig.legend(*a1.get_legend_handles_labels(),loc="lower center",ncol=10,fontsize=9)
fig.suptitle("Kimi-K3 PP4xTP4 nsys, int6-best + admit-cap fix — 1200 tok/s @ C256 (unprofiled) vs 317 @ MNS16",fontsize=13)
plt.tight_layout(rect=(0,0.04,1,0.97)); plt.savefig("k3-nsys-mns256.png",dpi=105)
