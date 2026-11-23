"""Exception interference and retrieval after discarding original events."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loki_memory.lifelong import LifelongMemory
from .reference import ROOT


def run(seed):
    rng=np.random.default_rng(seed)
    x=rng.random((64,12)).astype("f4"); y=rng.normal(0,.04,(64,8)).astype("f4")
    order=rng.integers(0,64,4096); order[0]=0;order[1:]=1+order[1:]%63
    y[0]=.85
    keys=x[order]+rng.normal(0,.002,(4096,12))
    effects=y[order]+rng.normal(0,.005,(4096,8)); actions=order%4
    result={}
    for name,flags in [('full',[1,1,1,1]),('no_surprise',[1,1,1,0]),('no_exception',[1,0,1,1]),('no_merge',[0,1,1,1])]:
        m=LifelongMemory();m.flags[:]=flags;m.learn(keys,actions,effects)
        pred=m.predict(keys[0],actions[0],exceptions=True)
        result[name]=dict(rare_mse=float(np.mean((pred-effects[0])**2)),
             ordinary_mse=float(np.mean([(m.predict(x[i],i%4)-y[i])**2 for i in range(1,64)])),
             occupied_exceptions=int((m.exmass>0).sum()))
    return result


if __name__=="__main__":
    rows=[run(s) for s in range(10111,10123)]
    (ROOT/"measurements/v10_ablation.json").write_text(json.dumps(rows,indent=2))
    data=json.loads((ROOT/"measurements/v10_stream.json").read_text())["stream"]
    fig,ax=plt.subplots(figsize=(6,3.3))
    ax.plot([r["events"] for r in data],[r["bytes"]/1024 for r in data],"o-",label="Complete v10 snapshot")
    ax.axhline(8192,linestyle="--",color="gray",label="8 MiB limit")
    ax.set(xscale="log",yscale="log",ylim=(128,13000),xlabel="Completed events",ylabel="Persistent KiB")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"examples/v10_growth.png",dpi=150)
