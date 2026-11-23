from collections import defaultdict
import json
import numpy as np
from .reference import ROOT
from .v6_maze import Maze
from .v9_route import run as v9_return
from .v10_stream import LifelongMemory


def descriptor(w,p):
    return w.view(p).reshape(2,8,4,4,3).mean((1,3)).ravel().astype("f4")


def run(seed):
    rng=np.random.default_rng(seed); w=Maze(seed); warm=Maze(seed+5000); m=LifelongMemory()
    def acquire(world,n):
        ds={(*xy,h):descriptor(world,(*xy,h)) for xy in world.edges for h in range(4)}
        poses=list(ds)
        for _ in range(n):
            p=poses[rng.integers(len(poses))]; a=int(rng.integers(4))
            m.transition(ds[p],a,ds[world.step(p,a)])
        return ds
    acquire(warm,600)
    # Cross-room prediction with no adaptation, nearest experienced visual state.
    err=[]
    poses=[(*xy,h) for xy in w.edges for h in range(4)]
    ds={p:descriptor(w,p) for p in poses}
    for p in poses:
        i=m.locate(ds[p])
        for a in range(4):
            weights=m.links[i,a]
            pred=weights@m.views/weights.sum() if weights.sum() else ds[p]
            y=ds[w.step(p,a)]
            err.append([np.mean((pred-y)**2),np.mean((ds[p]-y)**2)])
    # Separate local acquisition, isolated one-step encounters. No map is given to memory.
    m=LifelongMemory(); acquire(w,1000)
    p=w.start; route=[p]
    for a in w.guide:
        q=w.step(p,a);m.transition(ds[p],a,ds[q]);p=q;route.append(p)
    def travel(p,g,mask=False,random=False):
        first=ds[p].copy()
        if mask: first[:6]=first.mean()
        i=m.locate(first)
        recovered=poses[np.argmin([np.sum((ds[q]-m.views[i])**2) for q in poses])]==p
        for j in range(32):
            if p[:2]==g[:2]:break
            acts=m.plan(first if j==0 else ds[p],ds[g]) if not random else []
            if not acts and not random: break
            p=w.step(p,int(rng.integers(4)) if random else acts[0])
        return [int(recovered),int(p[:2]==g[:2])]
    shifted=(*sorted(w.edges[p[:2]])[0],p[2])
    out=[travel((*p[:2],(p[2]+h)%4),w.start) for h in (0,1,-1,2)]
    out += [travel(shifted,w.start),travel(p,w.start,True)]
    # Endpoints never visited on the single contiguous acquisition route; other
    # training episodes contain just one action. Require >=2 translations.
    unseen=[q for q in poses if all(q[:2]!=r[:2] for r in route)]
    pairs=[(q,g) for q in unseen for g in unseen if abs(q[0]-g[0])+abs(q[1]-g[1])>=2]
    q,g=pairs[rng.integers(len(pairs))]
    novel=travel(q,g)[1]; chance=travel(q,g,random=True)[1]
    m.links.fill(0)
    return dict(recovery=out,novel=novel,chance=chance,no_temporal_links=travel(q,g)[1],
                v9=v9_return(seed),transfer=np.mean(err,0).tolist())


if __name__=="__main__":
    rows=[run(s) for s in range(10201,10225)]
    (ROOT/"measurements/v10_world.json").write_text(json.dumps(rows,indent=2))
