"""Delayed nonlinear dynamics. Every model sees the same physical probe and lag."""
import json
import numpy as np
from .reference import ROOT
from .v10_stream import Samples
from loki_memory.lifelong import LifelongMemory
from loki_memory.episodic import EpisodeField


def run(seed):
    rng=np.random.default_rng(seed)
    m=LifelongMemory(); plain=LifelongMemory(); plain.flags[0]=0
    nohistory=LifelongMemory(); nohistory.flags[2]=0
    v9=EpisodeField(); fifo=Samples(512,False,rng);reservoir=Samples(512,True,rng)
    def initial(): return rng.uniform(.1,.9,8)
    def advance(x,a,h):
        return np.clip(x+.12*np.sin(5*x+1.57*a)+.11*np.roll(h,1).repeat(2)-.15*(x-.5),0,1)
    def command(a): return np.eye(4)[a]
    def state9(x,h):return np.resize(np.r_[x,h],20).astype('f4')
    def action9(a):return np.resize(command(a),8).astype('f4')
    def feat(x,a,h):return np.r_[1,x,command(a),h]
    P=np.eye(17)*10; W=np.zeros((17,8))
    xx=[]; aa=[]; yy=[]
    x=initial();h=command(0)
    for t in range(1500):
        if t%16==0:x=initial()
        a=int(rng.integers(4));y=advance(x,a,h)
        xx.append(np.r_[x,h]);aa.append(a);yy.append(y-x)
        z=feat(x,a,h);v=P@z;g=v/(1+z@v)
        W+=np.outer(g,y-x-z@W);P-=np.outer(g,v)
        v9.record(state9(x,h),action9(a),state9(y,command(a)),"rig")
        x,h=y,command(a)
    X,A,Y=map(np.asarray,(xx,aa,yy))
    for model in (m,plain,nohistory,fifo,reservoir):model.learn(X,A,Y)
    # VQ quantizes sensory context only, then averages effects by cluster/action.
    code=X[:48].copy();avg=np.zeros((48,4,8));count=np.zeros((48,4))
    for key,a,delta in zip(X,A,Y):
        i=np.argmin(np.sum((code-key)**2,1))
        count[i,a]+=1;avg[i,a]+=(delta-avg[i,a])/count[i,a]
    def vq(k,a):return avg[np.argmin(np.sum((code-k)**2,1)),a]
    methods={'v10':m.predict,'no_entity':lambda k,a:m.predict(k,a,entities=False),
    'prototype_only':lambda k,a:m.predict(k,a,shared=False),'no_merge':plain.predict,
    'no_sensorimotor':nohistory.predict,'VQ':vq,'FIFO':fifo.predict,'reservoir':reservoir.predict,
    'RLS':lambda k,a:feat(k[:8],a,k[8:])@W,'v9':lambda k,a:v9.rollout(state9(k[:8],k[8:]),[action9(a)])[:8]-k[:8],
    'hold':lambda k,a:np.zeros(8)}
    scores={k:[] for k in methods}
    for _ in range(24):
        x=initial(); h=command(1); pred={k:x.copy() for k in methods}
        for j in range(8):
            a=int(rng.integers(4)); x=advance(x,a,h)
            for k,fn in methods.items():
                last=pred[k]; pred[k]=np.clip(last+fn(np.r_[last,h],a),0,1)
                if j in (0,3,7):scores[k].append(np.mean((pred[k]-x)**2))
            h=command(a)
    return {k:np.mean(np.reshape(v,(-1,3)),0).tolist() for k,v in scores.items()}


if __name__=="__main__":
    rows=[run(s) for s in range(10111,10123)]
    (ROOT/"measurements/v10_dynamics.json").write_text(json.dumps(rows,indent=2))
