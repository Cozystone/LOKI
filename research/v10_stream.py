"""Fixed-capacity controls and generated chronological input; identity is evaluator-only."""
from pathlib import Path
import json
import tempfile
import numpy as np
from .reference import ROOT
from loki_memory.lifelong import LifelongMemory


class Samples:
    def __init__(self,n,reservoir,rng):
        self.rows=np.zeros((n,21),np.float32)
        self.priority=np.full(n,-1.)
        self.reservoir,self.rng=reservoir,rng
        self.n=0

    def learn(self,x,a,y):
        records=np.column_stack((x,a,y))
        if self.reservoir:
            ranks=np.r_[self.priority,self.rng.random(len(x))]
            idx=np.argpartition(ranks,-len(self.rows))[-len(self.rows):]
            self.rows[:]=np.vstack((self.rows,records))[idx]
            self.priority=ranks[idx]
        else:
            self.rows[:]=np.vstack((self.rows,records))[-len(self.rows):]
        self.n=min(len(self.rows),self.n+len(x))

    def predict(self,x,a):
        rows=self.rows if self.n==len(self.rows) else self.rows[self.priority>=0] if self.reservoir else self.rows[-self.n:]
        d=np.sum((rows[:,:12]-x)**2,1)+(rows[:,12]!=a)*4
        return rows[d.argmin(),13:]


def nearest_center(m,k):
    return np.argmin(np.where(m.mass>0,((m.centers-k)**2).sum(1),np.inf))


def stress():
    rng=np.random.default_rng(10101); m=LifelongMemory()
    size=12+sum(x.nbytes for x in m.arrays)
    fifo=Samples(size//84,False,rng); reservoir=Samples(size//92,True,rng)
    probes=rng.uniform(.1,.8,(12,12)).astype("f4")
    probes[:,0]=.1
    signature=rng.uniform(-.12,.12,(12,4,8)).astype("f4")
    rare=probes[0].copy(); rare[:2]=1
    ordinary=None
    rows=[]
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/"life"
        for batch in range(1000):
            ids=rng.integers(0,12,1000); a=rng.integers(0,4,1000)
            x=probes[ids]+rng.normal(0,.002,(1000,12))
            sign=1 if batch<500 else -1
            x[:,0]=.1 if sign==1 else .8
            y=signature[ids,a]*sign+rng.normal(0,.005,(1000,8))
            if batch==0:
                x[0]=rare; a[0]=0; y[0]=.85
                ordinary=(x[1].copy(),a[1],y[1].copy())
            m.learn(x,a,y); fifo.learn(x,a,y); reservoir.learn(x,a,y)
            if batch+1 in (1,10,100,1000):
                m.save(path); m=LifelongMemory.load(path)
                metrics={}
                for name,predict in [('v10',lambda x,a:m.predict(x,a,exceptions=True)),('fifo',fifo.predict),('reservoir',reservoir.predict)]:
                    laws=[]
                    for mode in (1,-1):
                        errors=[]
                        for i,key in enumerate(probes):
                            key=key.copy(); key[0]=.1 if mode==1 else .8
                            for act in range(4): errors.append(np.mean((predict(key,act)-mode*signature[i,act])**2))
                        laws.append(np.mean(errors))
                    metrics[name]=dict(exception=float(np.mean((predict(rare,0)-.85)**2)),old_new=laws,
                        ordinary=float(np.mean((predict(*ordinary[:2])-ordinary[2])**2)))
                fake=rare.copy(); fake[4]+=.3
                rows.append(dict(events=(batch+1)*1000,bytes=path.stat().st_size,metrics=metrics,
                     known_entities=int((m.mass>0).sum()),false_exception=bool(m.predict(fake,0,exceptions=True).mean()>.5),
                     entity_probe_accuracy=np.mean([np.argmin(np.sum((probes[:,1:]-m.centers[nearest_center(m,k),1:])**2,1))==i for i,k in enumerate(probes)])))
    return rows


def identity(seed):
    rng=np.random.default_rng(seed); m=LifelongMemory(); blind=LifelongMemory(); blind.flags[2]=0
    col=rng.uniform(.2,.8,(8,3)); patterns=rng.normal(0,.07,(8,12,12,3))
    laws=rng.uniform(-.4,.4,(4,4)); labels=np.arange(16)%4
    rng.shuffle(labels)
    def obs(i,abroad):
        im=np.rot90(col[i//2]+patterns[i//2],int(rng.integers(4)))
        im=np.roll(im,rng.integers(-3,4,2),axis=(0,1))*rng.uniform(.92,1.08)
        if abroad: im[:3]=.4
        v=np.r_[im.mean((0,1)),im.std((0,1)),im.mean(),im.std()]
        probe=rng.choice([-1,1]); delta=probe*laws[labels[i]]+rng.normal(0,.009,4)
        return np.r_[v,probe*delta].astype("f4")
    train=np.stack([obs(i,False) for i in range(16)])
    y=np.tile(laws[labels],(1,2))
    for model in (m,blind): model.learn(train,np.zeros(16,int),y)
    result={}
    for name,model in [('full',m),('no_history',blind)]:
        def ident(x):
            xx=x.copy()
            if not model.flags[2]: xx[8:]=0
            return np.argmin(np.sum((model.centers[model.mass>0]-xx)**2,1))
        ids=[ident(x) for x in train]
        tp=fp=fn=tn=0
        for i in range(16):
            j=ident(obs(i,True))
            for k in range(16):
                match=j==ids[k]
                tp+=int(match and k==i); fp+=int(match and k!=i)
                fn+=int(not match and k==i); tn+=int(not match and k!=i)
        result[name]=[tp/max(1,tp+fp),tp/(tp+fn),fp/(fp+tn),fn/(tp+fn)]
    # New appearances: transition target differs from observed probe channels.
    def effect(k): return np.resize(np.sin(4*laws[k][::-1])*.2,8)
    m=LifelongMemory(); samples=Samples(24,False,rng)
    x=np.stack([obs(i,False) for i in range(16)]); a=np.zeros(16,int)
    y=np.stack([effect(k) for k in labels]); m.learn(x,a,y); samples.learn(x,a,y)
    errors=[]
    for i in range(16):
        xx=obs(i,True); xx[:8]=rng.random(8)
        target=effect(labels[i])
        errors.append([np.mean((p-target)**2) for p in (m.predict(xx,0,entities=False),samples.predict(xx,0),m.predict(xx,0,shared=False),np.zeros(8))])
    result["new_entity_mse"]=np.mean(errors,0).tolist()
    return result


if __name__=="__main__":
    result=dict(stream=stress(),entity=[identity(s) for s in range(10111,10123)])
    (ROOT/"measurements/v10_stream.json").write_text(json.dumps(result,indent=2,default=float))
