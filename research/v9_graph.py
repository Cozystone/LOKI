"""Fresh-seed durable episode, source, concept transfer and cost tests."""
import json
from pathlib import Path
import tempfile
from time import perf_counter_ns

import numpy as np

from loki_memory import GraphMemory
from loki_memory.episodic import latent
from .reference import ROOT


def run(seed):
    rng=np.random.default_rng(seed)
    frame=np.full((16,16,3),.5,np.float32)
    action=np.full(8,.5,np.float32)
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/"cortex"
        m=GraphMemory.create(path,"A")
        start=path.stat().st_size
        events=[]
        rates=[]
        for batch in range(12):
            t=perf_counter_ns()
            with m.wake() as w:
                for j in range(16):
                    i=batch*16+j
                    pre=rng.uniform(.2,.8,8).astype("f4")
                    post=pre.copy()
                    post[0]+=.12 if pre[1]>.5 else -.12
                    events.append(pre)
                    assert w.finish_action(w.begin_action(frame,pre,action),frame,post)
            rates.append((perf_counter_ns()-t)/16e6)
        late=m.begin_action(frame,pre,action)
        m=GraphMemory.resume(path,m.transfer("B"),"B")
        assert not m.finish_action(late,frame,post)
        assert m.encounter(frame,"B",1) is None
        reads=[]
        for i,pre in enumerate(events):
            t=perf_counter_ns()
            e=m.encounter(frame,"A",i+1)
            reads.append((perf_counter_ns()-t)/1e6)
            assert e["id"]==i+1 and np.max(abs(e["before"][12:]-pre))<=.5/255+1e-6
        # Deliberate ambiguous query exactly on the decision boundary is not labelled.
        ok=0
        for _ in range(64):
            pre=rng.uniform(.2,.8,8).astype("f4")
            x=latent(frame,pre)
            candidates=m.outcomes(x,action)
            _,_,y=max(candidates,key=lambda r:r[1])
            ok+=bool((y[12]>x[12])==(pre[1]>.5))
        for _ in range(2):
            assert m.finish_action(m.begin_action(frame,post,action),frame,post)
        first=m.encounter(frame,"B",1)
        assert first["previous"] is None and first["following"]==194
        m.cut()
        assert m.finish_action(m.begin_action(frame,post,action),frame,post)
        assert m.encounter(frame,"B",3)["previous"] is None
        return dict(ordinal_correct=192,effect_correct=ok,trials=64,
                    disk_growth=path.stat().st_size-start,write_ms=np.median(rates),
                    read_ms=np.median(reads),state_bytes=m.state_bytes)


if __name__=="__main__":
    rows=[run(s) for s in range(9701,9705)]
    print(rows)
    (ROOT/"measurements/v9_graph.json").write_text(json.dumps(rows,indent=2)+"\n")
