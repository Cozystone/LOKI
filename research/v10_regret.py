import json
import numpy as np
from .reference import ROOT
from loki_memory.lifelong import LifelongMemory


def run(seed):
    rng=np.random.default_rng(seed)
    keys=rng.random((40,12));keys[1:20]=keys[0]+rng.normal(0,.04,(19,12))
    outcome=rng.normal(0,.23,(40,8))
    idx=np.r_[0,rng.integers(1,40,5000)]
    a=np.zeros(len(idx),int)
    rows={}
    for name,exception,policy in (("regret",1,2),("surprise",1,1),("no_exception",0,1)):
        m=LifelongMemory();m.flags[1]=exception;m.flags[3]=policy
        m.learn(keys[idx],a,outcome[idx])
        err=[float(np.mean((m.predict(k,0,exceptions=True)-v)**2)) for k,v in zip(keys,outcome)]
        rows[name]=[err[0],np.mean(err[1:])]
    return rows


if __name__=="__main__":
    r=[run(s) for s in range(10301,10313)]
    (ROOT/"measurements/v10_regret.json").write_text(json.dumps(r,indent=2))
    for k in r[0]:print(k,np.mean([x[k] for x in r],0))
