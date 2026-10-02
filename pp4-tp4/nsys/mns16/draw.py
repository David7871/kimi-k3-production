import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
cats=["moe-experts","dense-gemm","tp-comm","moe-topk/router","norm/elementwise","attn/mla","pp-sendrecv","kda/linear-attn","other"]
col=dict(zip(cats,["#d62728","#1f77b4","#ff7f0e","#9467bd","#8c564b","#2ca02c","#e377c2","#17becf","#7f7f7f"]))
def load(p):
    d=json.load(open(p)); out=[]
    for r in d:
        host="r1" if "rtx-01" in r["file"] else "r2"
        out.append((host,r["dev"][0],r["period"],r["cats"]))
    return sorted(o for o in out if o[0]=="r2")
fig,axes=plt.subplots(1,2,figsize=(18,8),sharey=False)
for ax,(tag,p) in zip(axes,(("BEFORE (pre-opt)","/tmp/base.json"),("AFTER int6-best","/tmp/new.json"))):
    rows=load(p); labels=[]
    for i,(h,dv,per,c) in enumerate(rows):
        left=0
        for k in cats:
            v=c.get(k,0); ax.barh(i,v,left=left,color=col[k],label=k if i==0 else None); left+=v
        ax.barh(i,max(per-left,0),left=left,color="#eeeeee",label="idle/gap" if i==0 else None)
        ax.text(per+0.1,i,"%.2f ms"%per,va="center",fontsize=8); labels.append("%s gpu%d"%(h,dv))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels,fontsize=8); ax.set_title(tag); ax.set_xlabel("GPU ms per decode step (C16)"); ax.set_xlim(0,16); ax.invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=10,fontsize=9)
fig.suptitle("Kimi-K3 PP4xTP4 per-rank kernel time per step - rtx-02 = PP stages 2 (gpu0-3) & 3 (gpu4-7); bars sum >period where streams overlap; label = measured step period")
plt.tight_layout(rect=(0,0.05,1,1)); plt.savefig("k3-before-after.png",dpi=110)
