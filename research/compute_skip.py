"""python -m research.compute_skip"""
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from loki_memory.certified_read import CertifiedRead


def assay(seed, world):
    rng=np.random.default_rng(seed)
    keys=rng.uniform(.2,.8,(128,240))
    mapping=rng.integers(2,size=128)
    if world=='peripheral':
        keys[1::2]=keys[::2]
        keys[::2,-1]=.1
        keys[1::2,-1]=.9
        mapping[1::2]=1-mapping[::2]
    mem=CertifiedRead(128,240,2,2)
    change=np.eye(2)*-.12
    for i in rng.permutation(128):
        for action in range(2):
            mem.learn(keys[i],action,change[action^mapping[i]])
    n=1200
    ids=rng.integers(128,size=n)
    if world=='stable':
        ids=np.repeat(ids[:60],20)
    if world=='peripheral':
        ids=np.repeat(ids[:60]//2*2,20)+np.arange(n)%2
    cues=keys[ids]+rng.normal(0,.0004,(n,240))
    drain=rng.uniform(.008,.034,(n,2))
    modes=['full','certified','sample']
    rng.shuffle(modes)
    results={}
    for mode in modes:
        agent=copy.deepcopy(mem)
        d=np.array([.4,.45])
        queue=np.zeros((3,2))
        acts=bytearray()
        lat=[]
        whole=0.
        relief=0.
        begin=time.process_time_ns()
        for t,cue in enumerate(cues):
            d=np.clip(d+drain[t],0,1)
            restored=np.minimum(d,queue[t%3])
            relief+=restored.sum()
            d-=restored
            start=time.perf_counter_ns()
            a=agent.choose(cue,d,mode)
            lat.append(time.perf_counter_ns()-start)
            queue[t%3]=-change[a^mapping[ids[t]]]
            acts.append(a)
            whole+=1-np.mean(d*d)
        results[mode]=dict(cpu_ms=(time.process_time_ns()-begin)/1e6,p95_ms=np.percentile(lat,95)/1e6,
                 reads=agent.reads,comparisons=agent.reads*128*240,
                 relief=relief,mean_integrity=whole/n,final=d.tolist(),actions=hashlib.sha256(acts).hexdigest())
    return results


def sleep_test():
    # Same observed body residuals assigned to the same cues and actions.
    rng=np.random.default_rng(339)
    ix=rng.integers(128,size=8192)
    gains=rng.uniform(-.2,.2,(8192,2))
    rows={}
    for chunk in (1,32):
        totals=np.zeros((128,2))
        counts=np.zeros(128,dtype=int)
        pending_i=np.empty(chunk,dtype=int)
        pending_v=np.empty((chunk,2))
        wake=sleep=0
        for t,(i,v) in enumerate(zip(ix,gains)):
            start=time.perf_counter_ns()
            j=t%chunk
            pending_i[j]=i
            pending_v[j]=v
            wake+=time.perf_counter_ns()-start
            if j+1==chunk:
                start=time.perf_counter_ns()
                np.add.at(totals,pending_i,pending_v)
                np.add.at(counts,pending_i,1)
                sleep+=time.perf_counter_ns()-start
        rows[chunk]=dict(wake_ms=wake/1e6,commit_ms=sleep/1e6,bytes=pending_i.nbytes+pending_v.nbytes,
                         digest=hashlib.sha256(totals.tobytes()+counts.tobytes()).hexdigest())
    return rows


if __name__=='__main__':
    rows=[dict(seed=s,**{w:assay(s,w) for w in ('stable','rapid','peripheral')}) for s in range(94100,94124)]
    out=dict(platform=platform.platform(),cpu=platform.processor(),numpy=np.__version__,lives=rows,sleep=sleep_test())
    Path('measurements/compute_skip.json').write_text(json.dumps(out,indent=2)+'\n')
    for w in ('stable','rapid','peripheral'):
        for m in ('full','certified','sample'):
            v=[r[w][m] for r in rows]
            print(w,m,[round(np.mean([x[k] for x in v]),5) for k in ('reads','cpu_ms','p95_ms','mean_integrity')],
                  sum(x['actions']==r[w]['full']['actions'] for x,r in zip(v,rows)))
    print(out['sleep'],out['cpu'])
