"""Bounded predictive coalescence. Arrays contain only inferred statistics."""
from collections import deque
import os
from pathlib import Path
import tempfile

import numpy as np
from numba import njit


@njit
def nearest(rows, weights, x):
    idx, best = -1, 1e20
    for i in range(len(weights)):
        if weights[i]>0:
            d=0.
            for j in range(len(x)):
                d+=(rows[i,j]-x[j])**2
            if d<best:
                idx,best=i,d
    return idx,best


@njit
def stream(keys, acts, effects, centers, mass, means, count, laws, lawmass, effectlaw,
           exkey, exmass, exval, flags):
    """Inputs are chronological; the caller can discard a chunk after this returns."""
    for t in range(len(acts)):
        x=keys[t].copy()
        if not flags[2]: x[8:]=0.
        a=acts[t]
        i,d=nearest(centers,mass,x)
        if i<0 or d>.055:
            empty=np.where(mass==0)[0]
            if len(empty):
                i=empty[0]
            elif flags[0]:
                # Coalesce the most alike occupied prototypes, carrying all effect sums.
                best=1e20
                u,v=0,1
                for j in range(len(mass)):
                    for k in range(j):
                        dist=np.sum((centers[j]-centers[k])**2)
                        if dist<best: best,u,v=dist,j,k
                total=mass[u]+mass[v]
                centers[u]=(centers[u]*mass[u]+centers[v]*mass[v])/total
                mass[u]=total
                for b in range(count.shape[1]):
                    n=count[u,b]+count[v,b]
                    if n: means[u,b]=(means[u,b]*count[u,b]+means[v,b]*count[v,b])/n
                    count[u,b]=n
                mass[v]=0
                count[v]=0
                means[v]=0
                i=v
        y=effects[t]
        prediction=means[i,a].copy()
        err=np.mean((y-prediction)**2)
        if flags[3]==2:
            z=np.concatenate((x,np.array([float(a)])))
            reused,distance=nearest(exkey,exmass,z)
            if reused>=0 and distance<.003:
                saved=err-np.mean((y-exval[reused])**2)
                exmass[reused]=min(4096.,max(1.,exmass[reused]+saved))
        mass[i]+=1
        centers[i]+=(x-centers[i])/mass[i]
        count[i,a]+=1
        means[i,a]+=(y-means[i,a])/min(128.,count[i,a])
        # Predict effects across entities having similar observed probe responses.
        h=x[8:]
        k,dist=nearest(laws[:,a],lawmass[:,a],h)
        if k<0 or dist>.035:
            free=np.where(lawmass[:,a]==0)[0]
            if len(free): k=free[0]
        # law prototypes are action-specific (one motor per column in a batch).
        lawmass[k,a]+=1
        laws[k,a]+=(h-laws[k,a])/lawmass[k,a]
        effectlaw[k,a]+=(y-effectlaw[k,a])/min(128.,lawmass[k,a])
        if flags[1] and (err>.0225 or not flags[3]):
            z=np.concatenate((x,np.array([float(a)])))
            k,dist=nearest(exkey,exmass,z)
            if k<0 or dist>.003 or np.mean((exval[k]-y)**2)>.01:
                free=np.where(exmass==0)[0]
                if len(free): k=free[0]
                else:
                    best=1e20
                    u,v=0,1
                    for j in range(len(exmass)):
                        for l in range(j):
                            delta=np.sum((exkey[j]-exkey[l])**2)+np.sum((exval[j]-exval[l])**2)
                            if flags[3]==2:
                                delta*=exmass[j]*exmass[l]/(exmass[j]+exmass[l])
                            if delta<best: best,u,v=delta,j,l
                    total=exmass[u]+exmass[v]
                    exkey[u]=(exkey[u]*exmass[u]+exkey[v]*exmass[v])/total
                    exval[u]=(exval[u]*exmass[u]+exval[v]*exmass[v])/total
                    exmass[u]=total
                    exmass[v]=0
                    k=v
            exmass[k]+=1
            exkey[k]+=(z-exkey[k])/exmass[k]
            exval[k]+=(y-exval[k])/exmass[k]



class LifelongMemory:
    """Standalone local numerical memory. Keys = eight sensory and four intervention features."""
    shapes=((48,12),(48,),(48,4,8),(48,4),(16,4,4),(16,4),(16,4,8),
            (16,13),(16,),(16,8),(128,24),(128,),(128,4,128))
    def __init__(self):
        self.arrays=[np.zeros(s,"<f4") for s in self.shapes]
        (self.centers,self.mass,self.means,self.count,self.laws,self.lawmass,self.effectlaw,
         self.exkey,self.exmass,self.exval,self.views,self.visits,self.links)=self.arrays
        self.flags=np.ones(4,np.uint8)

    def learn(self,keys,actions,effects):
        x=np.asarray(keys,np.float32); a=np.asarray(actions); y=np.asarray(effects,np.float32)
        if x.shape!=(len(a),12) or y.shape!=(len(a),8) or a.ndim!=1 or a.dtype.kind not in "iu" or np.any((a<0)|(a>=4)) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("invalid sensorimotor batch")
        stream(x,a,y,*self.arrays[:10],self.flags)

    def predict(self,x,a,entities=True,shared=True,exceptions=False):
        x=np.asarray(x,np.float32).copy()
        if not self.flags[2]: x[8:]=0
        if exceptions:
            j,d=nearest(self.exkey,self.exmass,np.r_[x,float(a)])
            if j>=0 and d<.003: return self.exval[j].copy()
        i,d=nearest(self.centers,self.count[:,a],x)
        if entities and i>=0 and (d<.055 or not shared): return self.means[i,a].copy()
        j,_=nearest(self.laws[:,a],self.lawmass[:,a],x[8:])
        return self.effectlaw[j,a].copy() if shared and j>=0 else np.zeros(8)

    def locate(self,x):
        return nearest(self.views,self.visits,x)[0]

    def node(self,x):
        i,d=nearest(self.views,self.visits,x)
        free=np.flatnonzero(self.visits==0)
        if (i<0 or d>1e-8) and len(free): i=free[0]
        self.visits[i]+=1
        self.views[i]+=(x-self.views[i])/self.visits[i]
        return i

    def transition(self,x,a,y):
        i=self.node(x); j=self.node(y)
        self.links[i,a,j]+=1

    def plan(self,x,goal):
        start=self.locate(x); end=self.locate(goal)
        queue=deque([(start,[])])
        seen={start}
        while queue:
            i,path=queue.popleft()
            if i==end: return path
            for a,weights in enumerate(self.links[i]):
                j=int(weights.argmax())
                if weights[j]>0 and j not in seen:
                    seen.add(j); queue.append((j,path+[a]))
        return []

    def save(self,path):
        path=Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
            try:
                f.write(b"LOKI10A1"+self.flags.tobytes())
                for x in self.arrays: f.write(x.tobytes())
                f.flush(); os.fsync(f.fileno())
            except BaseException:
                f.close(); os.unlink(f.name); raise
        try: os.replace(f.name,path)
        finally: Path(f.name).unlink(missing_ok=True)

    @classmethod
    def load(cls,path):
        m=cls()
        with open(path,"rb") as f:
            if f.read(8)!=b"LOKI10A1": raise ValueError("wrong format")
            m.flags[:]=np.frombuffer(f.read(4),np.uint8)
            for x in m.arrays:
                x[:]=np.frombuffer(f.read(x.nbytes),"<f4").reshape(x.shape)
            if f.read(1): raise ValueError("unexpected tail")
        return m
