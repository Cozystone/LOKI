from copy import deepcopy
import numpy as np
import torch
from loki_memory import LokiMemory
from .dark import compound
from .reference import mse,setup,write
from .v3_common import prior_version,compare


def score(seed,models,trained,ref):
    frames,actions = compound(seed,368)
    inner = deepcopy(trained)
    ref.train_inner(inner,[(frames[:129],actions[:128])],64,seed)
    models = deepcopy(models)
    for m in models.values():
        m.cut()
        m.observe(frames[0])
    outputs = {key:[] for key in (*models,"inner","last")}
    h = inner.zero(1)
    incoming = torch.tensor(frames[:1])
    was_dark = False
    last = frames[0]
    with torch.no_grad():
        for i,a in enumerate(actions):
            dark = i>=128 and (i-128)%60>=40
            h,pred = inner.step(h,incoming,torch.tensor([a]),torch.tensor([float(was_dark)]))
            target = frames[i+1]
            if dark:
                values = {name:m.advance(int(a)) for name,m in models.items()}
                values.update(inner=pred.numpy().ravel().clip(0,1),last=last)
                for name,y in values.items():
                    outputs[name].append(mse(y,target))
            else:
                last = target
                for m in models.values():
                    m.observe(target,int(a))
            incoming = torch.zeros_like(incoming) if dark else torch.tensor(target[None])
            was_dark = dark
    return dict(seed=seed,whole={k:float(np.mean(v)) for k,v in outputs.items()},
                late={k:float(np.array(v).reshape(4,20)[:,-5:].mean()) for k,v in outputs.items()},
                horizon={k:np.array(v).reshape(4,20).mean(axis=0).tolist() for k,v in outputs.items()})


def run(seeds=range(3101,3113),file="v3_dark"):
    _,_,_,ref,_ = setup()
    from atanor_os.emerge.inner import Inner
    torch.manual_seed(803)
    torch.set_num_threads(1)
    inner = Inner(96,4,h=32)
    training = [compound(s,208) for s in range(1,13)]
    ref.train_inner(inner,training,500,803)
    models = {"v2":prior_version()(96),"v3":LokiMemory(96)}
    for frames,acts in training:
        for m in models.values():
            m.cut()
            m.observe(frames[0])
        for target,a in zip(frames[1:],acts):
            for m in models.values():
                m.observe(target,int(a))
    rows = [score(s,models,inner,ref) for s in seeds]
    gates = {k:compare(rows,k,("inner","v2","last"),.1,10) for k in ("whole","late")}
    write(file,dict(rows=rows,gates=gates,passed=all(v["passed"] for g in gates.values() for v in g.values())))


if __name__=="__main__":
    run()
