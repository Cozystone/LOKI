"""Registered event-level fork assay. Truth and candidate manufacture stay here."""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from loki_memory.somatic_readout import SomaticReadout


def traverse(before, relief):
    d = before.clone()
    prefix = None
    for t in range(29):
        d.add_(.01)
        if t==19:
            prefix=d.clone()
        if t==23:
            d[relief]=(d[relief]-.55).clamp_min(0)
    return d,prefix,bool((d<1).all())


def life(seed):
    rng=np.random.default_rng(seed)
    gen=torch.Generator().manual_seed(seed)
    actions=torch.tensor(rng.choice(90,2,replace=False))
    cue=torch.tensor(rng.uniform(.05,.35,34),dtype=torch.float32)
    flipped=seed%2
    fast=SomaticReadout(34,2)
    shuffled=copy.deepcopy(fast)
    empty=copy.deepcopy(fast)
    base=torch.full((2,),.62)
    trials=rng.permutation(np.arange(48)%2)
    labels=rng.permutation(trials)
    for arm,label in zip(trials,labels):
        for model,out in ((fast,arm),(shuffled,label)):
            end,_,_=traverse(base,out^flipped)
            model.remember(cue,int(actions[arm]),base,end)
    rows=[]
    ms=[]
    for need in (0,1):
        d=torch.full((2,),.25)
        d[need]=.72
        outcomes=[traverse(d,arm^flipped) for arm in (0,1)]
        assert torch.equal(outcomes[0][1],outcomes[1][1])
        for name,model,mode in [('adaptive',fast,'adaptive'),('fast',fast,'fast'),
                               ('slow',fast,'slow'),('erased',empty,'fast'),
                               ('shuffle',shuffled,'fast'),('uniform',None,None)]:
            if model is None:
                arm=int(rng.integers(2))
                conf,call=0,False
            else:
                start=time.perf_counter_ns()
                a,conf,call=model.choose(cue,actions,d,mode,gen)
                ms.append((time.perf_counter_ns()-start)/1e6)
                arm=int(a==actions[1])
            rows.append(dict(mode=name,need=need,first=bool((arm^flipped)==need),
                             survived=outcomes[arm][2],confidence=conf,call=call))
    curves={}
    for phase in ('familiar','new','hidden'):
        model=copy.deepcopy(fast)
        observations=[]
        for t in range(16):
            current=cue+.55 if phase=='new' else cue
            truth=flipped^(phase!='familiar')
            d=base.clone()
            need=t%2
            d[need]=.71
            a,conf,call=model.choose(current,actions,d,'adaptive',gen)
            arm=int(a==actions[1])
            end,_,alive=traverse(d,arm^truth)
            model.remember(current,a,d,end)
            observations.append(dict(call=call,confidence=conf,correct=(arm^truth)==need,
                                     survived=alive))
        curves[phase]=observations
    return dict(seed=seed,probes=rows,shifts=curves,read_ms=ms,
                bytes=sum(t.numel()*t.element_size() for t in fast.buffers()))


if __name__=='__main__':
    torch.set_num_threads(1)
    data=[life(s) for s in range(8710,8734)]
    result={}
    for m in ('adaptive','fast','slow','erased','shuffle','uniform'):
        rows=[p for r in data for p in r['probes'] if p['mode']==m]
        result[m]=[sum(p[k] for p in rows) for k in ('first','survived','call')]
    for phase in ('familiar','new','hidden'):
        result[phase]=[
            [np.mean([r['shifts'][phase][t][k] for r in data for t in range(i,i+4)])
             for k in ('call','confidence','correct')] for i in (0,4,8,12)]
    result['read_p95_ms']=np.percentile([m for r in data for m in r['read_ms']],95)
    result['bytes']=data[0]['bytes']
    print(result)
    Path('measurements/somatic.json').write_text(json.dumps(dict(summary=result,lives=data),indent=2)+'\n')
