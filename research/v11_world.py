import argparse
import copy
import json
import numpy as np
from .reference import ROOT
from .v10_world import Maze,descriptor
from loki_memory.belief import BeliefModel


def measure(seed,rev):
    rng=np.random.default_rng(seed)
    def sample(world):
        poses=[(*xy,h) for xy in sorted(world.edges) for h in range(4)]
        return poses,{p:descriptor(world,p) for p in poses}
    m=BeliefModel(rev);warm=Maze(seed+5000)
    ps,ds=sample(warm)
    for _ in range(600):
        p=ps[rng.integers(len(ps))];a=rng.integers(4)
        m.fit(ds[p],a,ds[warm.step(p,a)],topology=False)
    w=Maze(seed);ps,ds=sample(w);loss=[]
    for p in ps:
        for a in range(4):
            y=ds[w.step(p,a)]
            loss.append([np.mean((z-y)**2) for z in (m.forecast(ds[p],a,np.zeros(4)),ds[p])])
    for _ in range(1000):
        p=ps[rng.integers(len(ps))];a=rng.integers(4)
        m.fit(ds[p],a,ds[w.step(p,a)],adapt=False)
    p=w.start;guide=[p]
    for a in w.guide:
        q=w.step(p,a);m.fit(ds[p],a,ds[q],adapt=False);p=q;guide.append(p)
    def travel(p,g,mask=False,jump=None,block=None):
        model=copy.deepcopy(m)
        q,t=model.policy(ds[g]);b=model.infer(ds[p]); trace=[p];masses=[];bad=0;delay=None
        for j in range(32):
            if p[:2]==g[:2]:break
            a=model.choose(b,q,t)
            nxt=w.step(p,a)
            if block and {p[:2],nxt[:2]}==block:nxt=p
            if jump is not None and j==0:nxt=jump
            prior=b@t[a];evidence=ds[nxt].copy()
            if mask and j==0:evidence[:6]=0
            idx=int(model.emission_cost(ds[nxt]).argmin());mass=float(prior[idx]);masses.append(mass)
            if mass<.05:
                bad+=1
                if delay is None:delay=j
                if rev>=1:
                    model.fit(ds[p],a,evidence,adapt=False)
                    q,t=model.policy(ds[g])
                    if len(prior)!=model.n:prior=np.pad(prior,(0,model.n-len(prior)))
            b=model.infer(evidence,prior);p=nxt;trace.append(p)
        return dict(success=int(p[:2]==g[:2]),steps=len(trace)-1,bad=bad,delay=delay,
                    mass=float(np.mean(masses)) if masses else None,trace=trace)
    starts=[(*p[:2],(p[2]+k)%4) for k in (0,1,-1,2)]+[(*sorted(w.edges[p[:2]])[0],p[2]),p]
    out=[travel(s,w.start,mask=i==5) for i,s in enumerate(starts)]
    unseen=[q for q in ps if q[:2] not in [r[:2] for r in guide]]
    pairs=[(a,b) for a in unseen for b in unseen if abs(a[0]-b[0])+abs(a[1]-b[1])>=2]
    a,g=pairs[rng.integers(len(pairs))]
    novel=travel(a,g)
    kidnapped=[travel(p,w.start,jump=s) for s in starts[1:5]]
    return dict(seed=seed,transfer=np.mean(loss,0).tolist(),occupied=m.n,recovery=[r['success'] for r in out],
                novel=novel,kidnap=[(r['delay'],r['success']) for r in kidnapped],rotated=out[3],
                # A spanning tree loses all paths between endpoints across a cut edge.
                tree_edges=sum(map(len,w.edges.values()))//2,tree_vertices=len(w.edges))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("rev",type=int);ap.add_argument("--dev",action="store_true");args=ap.parse_args()
    start=(11001 if args.dev else 11101)+args.rev*100
    rows=[measure(s,args.rev) for s in range(start,start+(1 if args.dev else 24))]
    print(np.sum([r['recovery'] for r in rows],0),sum(r['novel']['success'] for r in rows),np.mean([r['transfer'] for r in rows],0))
    if not args.dev:(ROOT/f"measurements/v11_world_{args.rev}.json").write_text(json.dumps(rows,indent=1))
