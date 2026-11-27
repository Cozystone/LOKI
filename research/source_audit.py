"""python -m research.source_audit"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from loki_memory.source_readout import SourceReadout


def run(seed):
    rng=np.random.default_rng(seed)
    cue=torch.tensor(rng.random(4)*.2,dtype=torch.float32)
    d=torch.tensor([.8,.2])
    changes=torch.tensor([[-.6,0],[-.6,0]])
    models=[SourceReadout(),SourceReadout(contexts=0)]
    curves=[]
    context_return=[]
    elapsed=[]
    for contextual in (False,True):
        models=[SourceReadout(),SourceReadout(contexts=0)]
        phases=[]
        for stage,count in enumerate([40,40,60]):
            rows=[]
            for t in range(count):
                location=torch.tensor([rng.uniform(3,6),rng.uniform(-1,1)],dtype=torch.float32)
                noise=torch.tensor(rng.normal(size=(2,2)),dtype=torch.float32)
                positions=torch.stack([location*(2 if stage==1 else 1),location])+noise*torch.tensor([[.15],[.7]])
                for model in models:
                    begin=time.perf_counter_ns()
                    model.offer(cue+(.7 if contextual and stage==1 else 0),positions,changes)
                    score,s,_=model.advice(d)
                    chosen=int(score.argmax())
                    if model is models[0]:
                        rows.append([chosen,float(s[0]/s.sum())])
                    if stage==2 and t==0 and contextual:
                        context_return.append(chosen)
                    # Independent contact anchor; both were claims about the sole spring.
                    own=location+torch.tensor(rng.normal(0,.05,2),dtype=torch.float32)
                    model.verify(own,torch.ones(2,dtype=torch.bool),chosen,d,(d+changes[chosen]).clamp(0,1))
                    elapsed.append((time.perf_counter_ns()-begin)/1e6)
            phases.append(rows)
        if not contextual:
            curves=phases
    model=SourceReadout()
    pos=torch.tensor([[-3.,0.],[3.,0.]])
    model.offer(cue,pos,torch.tensor([[-.6,0.],[0.,-.6]]))
    first=[int(model.advice(state)[0].argmax()) for state in (d,d.flip(0))]
    model.offer(cue,pos,changes)
    prior=model.advice(d)[1].tolist()
    for _ in range(256):
        model.verify()
    unchanged=model.advice(d)[1].tolist()==prior
    old=int(model.advice(d)[0].argmax())
    model.verify(attempted=old,before=d,after=d,adequate_miss=True)
    new=int(model.advice(d)[0].argmax())
    return dict(seed=seed,curves=curves,context_return=context_return,need_choices=first,
                unobserved_unchanged=unchanged,after_miss=model.advice(d)[1].tolist(),
                first=old,next=new,midpoint_distance=float((pos-pos.mean(0)).norm(dim=1).min()),
                bytes=sum(t.numel()*t.element_size() for t in model.buffers()),
                p95_ms=float(np.percentile(elapsed,95)))


def delay(rows,target):
    for i in range(len(rows)-2):
        if all(r[0]==target for r in rows[i:i+3]):
            return i  # completed confirmations before the first of three decisions
    return None


if __name__=='__main__':
    torch.set_num_threads(1)
    data=[run(i) for i in range(93000,93024)]
    for stage,target in [(1,1),(2,0)]:
        delays=[delay(row['curves'][stage],target) for row in data]
        print(stage,delays)
    print('weights', [np.mean([r['curves'][phase][-1][1] for r in data]) for phase in range(3)])
    print('return',np.sum([r['context_return'] for r in data],axis=0),'p95 range',
          min(r['p95_ms'] for r in data),max(r['p95_ms'] for r in data),data[0]['bytes'])
    print('controls',all(r['need_choices']==[0,1] and r['unobserved_unchanged'] and
          r['next']!=r['first'] and r['midpoint_distance']==3 for r in data))
    Path('measurements/source_audit.json').write_text(json.dumps(data,indent=2)+'\n')
