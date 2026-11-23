"""Chronological reconstruction, forward dynamics and physically unprobed maze return."""
import itertools
import json
import tempfile
from pathlib import Path
import numpy as np

from loki_memory import EpisodeMemory, Transition
from loki_memory.episodic import EpisodeField, latent
from .reference import ROOT
from .v6_maze import Maze, prepare, voyage


def history(n,seed):
    rng=np.random.default_rng(seed)
    flat=lambda: rng.random((2,2,3),dtype=np.float32).repeat(8,0).repeat(8,1)
    cue=flat()
    s=np.full(8,.5,np.float32)
    expected=[]
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"m"
        m=EpisodeMemory.create(p,"laptop")
        for i in range(n):
            f=flat()
            expected.append(latent(f,s))
            assert m.finish_action(m.begin_action(f,s,s),cue,s)
        common=latent(m.recall(Transition(cue,np.stack([s]*3)))[0],s)
        before=m.event(3)[0]
        m=EpisodeMemory.resume(p,m.transfer("glasses"),"glasses")
        assert np.array_equal(before,m.event(3)[0])
        predictions=np.stack([m.event(i+1)[0] for i in range(n)])
        distances=((predictions[:,None]-expected)**2).mean(2)
        third=m.occurrences(latent(cue,s),"laptop",window=n)
        return [np.mean((predictions-expected)**2),np.mean((common-expected)**2),
                np.mean(distances.argmin(1)==np.arange(n)),len(third),int(third[2:3]==[3]),
                len(m.occurrences(latent(cue,s),"glasses",window=n))]


def dynamics(seed):
    rng=np.random.default_rng(seed)
    f=EpisodeField()
    def step(x,a):
        return np.clip(.5+.85*(x-.5)+np.resize((a-.5)*(.15+.25*(x[:8]-.5)),20),0,1)
    def sample(n): return rng.random(n,dtype=np.float32)
    x=sample(20)
    for i in range(2200):
        if i%10==0: x=sample(20)
        a=sample(8); y=step(x,a)
        f.record(x,a,y,"body"); x=y
    err=[]
    for _ in range(40):
        x=sample(20); acts=[sample(8) for _ in range(8)]; y=x.copy()
        row=[]
        for j,a in enumerate(acts,1):
            y=step(y,a)
            if j in (1,4,8):
                row.append([np.mean((f.rollout(x,acts[:j])-y)**2),np.mean((x-y)**2)])
        err.append(row)
    return np.mean(err,0)


def maze(seed):
    w=Maze(seed)
    old,sh,poses=prepare(w,seed)
    def state(p): return latent(w.view(p),np.full(8,.5,np.float32))
    actions=np.eye(8,dtype=np.float32)[:4]
    f=EpisodeField()
    rng=np.random.default_rng(seed)
    warm=Maze(seed+5000); p=warm.start
    for _ in range(96):
        a=int(rng.choice(warm.actions(p))); q=warm.step(p,a)
        f.record(latent(warm.view(p),np.zeros(8)),actions[a],latent(warm.view(q),np.zeros(8)),"maze"); p=q
    for p,a,q in zip(poses,w.guide,poses[1:]):
        f.record(state(p),actions[a],state(q),"maze")
    p=poses[-1]
    for j in range(2*len(w.guide)):
        goal=f.event(97)[0]
        cur=state(p)
        plans=itertools.product(w.actions(p),range(4))
        a,_=min(plans,key=lambda seq:np.mean((f.rollout(cur,actions[list(seq)])-goal)**2))
        p=w.step(p,a)
        if p[:2]==poses[0][:2]: break
    return [int(p[:2]==poses[0][:2]),voyage(w,poses,old,sh,"model",seed,False)[0]["success"]]


if __name__=="__main__":
    result=dict(state_bytes=EpisodeField().data.nbytes+3136,
        history={n:np.mean([history(n,s) for s in range(9601,9605)],0).tolist() for n in (8,32,96,192)},
        rollout=np.mean([dynamics(s) for s in range(9601,9605)],0).tolist(),
        maze=[maze(s) for s in range(7201,7225)])
    print(result)
    (ROOT/"measurements/v9.json").write_text(json.dumps(result,indent=2)+"\n")
