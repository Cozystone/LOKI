"""Sealed two-body camera emulation. No hardware or ATANOR policy is simulated."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from loki_memory.portable import PortableMemory
from loki_memory.reverse import ReverseMemory
from .reference import ROOT


def scene(rng):
    y,x = np.mgrid[-1:1:16j,-1:1:16j]
    a = rng.uniform(.15,.7,(1,1,3)) + np.zeros((16,16,3))
    for _ in range(7):
        u,v = rng.uniform(-.85,.85,2)
        bump = np.exp(-((x-u)**2+(y-v)**2)/rng.uniform(.035,.3))
        a += bump[:,:,None]*rng.uniform(-.55,.55,3)
    return a.clip(0,1).astype(np.float32)


def camera(x,rng):
    dy,dx=rng.choice([-1,1],2)
    pad=np.pad(x,((1,1),(1,1),(0,0)),mode="edge")
    shifted=pad[1+dy:17+dy,1+dx:17+dx]
    return (shifted*rng.uniform(.83,1.15)+rng.uniform(-.07,.07)+rng.normal(0,.012,x.shape)).clip(0,1).astype(np.float32)


def run(seed):
    rng=np.random.default_rng(seed)
    keys=np.stack([scene(rng) for _ in range(16)])
    original=np.stack([scene(rng) for _ in keys])
    foreign=np.stack([camera(x,rng) for x in keys])
    past=original.copy()
    past[8:]=[camera(x,rng) for x in original[8:]]
    signals=rng.random((16,8),dtype=np.float32)
    rates=[]
    with TemporaryDirectory() as directory:
        path=Path(directory)/"being.db"
        m=PortableMemory.create(path,"laptop")
        v6=ReverseMemory()
        def train(indices,cues):
            for i in indices:
                m.cut()
                assert not m.observe(past[i],signals[i])
                t=perf_counter()
                assert m.observe(cues[i])
                rates.append(1000*(perf_counter()-t))
                v6.cut()
                v6.observe(past[i],[.5,.5])
                v6.observe(cues[i],[.5,.5],0)
        def switch(dest,cue):
            before=m.recall(cue)
            ident=m.transfer(dest)
            other=PortableMemory.resume(path,ident,dest)
            after=other.recall(cue)
            assert all(np.array_equal(a,b) for a,b in zip(before,after))
            try:
                m.observe(cue)
            except RuntimeError:
                pass
            else:
                raise AssertionError("stale writer")
            assert not other.observe(cue)
            assert np.array_equal(before[0],other.recall(cue)[0])
            return other
        def score(indices,cues):
            rows=[]
            for i in indices:
                pic,body,support=m.recall(cues[i])
                dist=((past-pic)**2).mean(axis=(1,2,3))
                old=v6.recall(cues[i])
                rows.append([int(dist.argmin()==i),dist[i],np.mean((past[i]-old)**2),
                             support,np.mean((body-signals[i])**2),float(support>=.65),float(support>=.65 and dist.argmin()==i)])
            return np.mean(rows,axis=0).tolist()
        train(range(8),keys)
        m=switch("glasses",keys[0])
        abroad=score(range(8),foreign)
        train(range(8,16),foreign)
        m=switch("laptop",foreign[8])
        returned=score(range(8,16),keys)
        retained=score(range(8),keys)
        novel=[m.recall(scene(rng))[2] for _ in keys]
        twins=[m.recall((x+.02*rng.normal(size=x.shape)).clip(0,1))[2] for x in keys]
        # A visual twin's hidden history is independent of the familiar object's.
        return dict(seed=seed,outbound=abroad,inbound=returned,retained=retained,
                    novel_accept=float(np.mean(np.array(novel)>=.65)),twin_accept=float(np.mean(np.array(twins)>=.65)),
                    write_p95_ms=float(np.percentile(rates,95)),persistent_bytes=m.state_bytes,db_bytes=path.stat().st_size)


if __name__=="__main__":
    rows=[run(s) for s in range(7701,7713)]
    avg={k:np.mean([r[k] for r in rows],axis=0).tolist() for k in rows[0] if k!="seed"}
    gates={k:bool(avg[k][0]>=.8 and avg[k][1]<=.9*avg[k][2]) for k in ("outbound","inbound")}
    gates.update({k:avg[k]<=.1 for k in ("novel_accept","twin_accept")})
    result=dict(columns=["top1","mse","v6_mse","support","aux_mse","accepted","accepted_correct"],means=avg,gates=gates,rows=rows)
    (ROOT/"measurements/v7_roundtrip.json").write_text(json.dumps(result,indent=2)+"\n")
    print(avg,gates)
