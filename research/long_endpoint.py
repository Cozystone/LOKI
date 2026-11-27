"""Paired guided single-body tests. Training floor and trajectory live only in the harness."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import platform
import time

import numpy as np
import torch

from loki_memory.embodied_choice import EmbodiedChoice
from loki_memory.endpoint_memory import EndpointMemory
from research.life_choice import MOVES, step


class Roll20:
    """Empirical one-beat dynamics and second-order continuation of the guiding behaviour."""
    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.follow = defaultdict(Counter)
        self.effects = defaultdict(list)

    def fit(self, rows):
        prev = None
        for p, a, q, d, e in rows:
            self.transitions[p,a][q] += 1
            self.follow[prev,p][a] += 1
            self.effects[p,a].append(np.maximum(0,(d-e).numpy()))
            prev = p
        self.gain = {k: np.mean(v,axis=0) for k,v in self.effects.items()}

    def choose(self, p, d, rate):
        best, move = -1, 0
        for s, first in self.transitions:
            if s != p:
                continue
            x, a = p, first
            improvement = 0
            for t in range(1,21):
                nxt = self.transitions[x,a].most_common(1)[0][0]
                need = np.clip(d.numpy()+rate*t,0,1)
                gain = self.gain[x,a]
                improvement += np.sum(need**2-np.maximum(0,need-gain)**2)/t
                following = self.follow[x,nxt]
                if gain.any() or not following:
                    break
                x, a = nxt, following.most_common(1)[0][0]
            if improvement > best:
                best, move = improvement, first
        return move


def run(seed, double, extended=False):
    rng = np.random.default_rng(seed)
    limb = rng.permutation(np.arange(90)%4)
    sensors = rng.permutation(4)
    basis = rng.uniform(-.13,.13,(2,30))
    way = [(0,0),(0,-2),(4,-2),(4,0)]
    if double:
        way += [(6,0),(6,-3),(10,-3),(10,0)]
    if extended:
        way[-2:] = [(26,-3),(26,0)]
    route = [way[0]]
    for endpoint in way[1:]:
        while route[-1] != endpoint:
            inc = np.sign(np.subtract(endpoint,route[-1]))
            route.append(tuple(np.add(route[-1],inc)))
    other = [(-x,-y) for x,y in route]
    floor = set(route+other)
    length = len(route)-1
    rate = np.array([.46,.08],dtype='f')/length
    fast = int(extended and seed % 2)
    if fast:
        rate = rate[::-1].copy()
    def features(cell):
        return (torch.tensor([float(step(cell,i,floor)!=cell) for i in sensors]),
                torch.tensor(np.array(cell)@basis,dtype=torch.float32))
    inputs = {p:features(p) for p in floor}
    # Anonymous observation tuples for the empirical comparator.
    tokens = {p:tuple(torch.cat(v).tolist()) for p,v in inputs.items()}
    one = EmbodiedChoice(4,2)
    memory, scrambled = EndpointMemory(), EndpointMemory()
    walk = [route[0]]
    for visit in range(24):
        arm = (route,other)[visit%2]
        if visit:
            walk.extend(reversed((other,route)[visit%2][:-1]))
        walk.extend(arm[1:])
    d = torch.zeros(2)
    records, raw, awake, replay = [], [], [], []
    def learn(agent,p,a,d,e,measure=False):
        t = time.perf_counter_ns()
        relief = agent.observe(*inputs[p],a,d,e)
        if measure:
            awake.append((time.perf_counter_ns()-t)/1e6)
        if relief.any():
            t = time.perf_counter_ns()
            agent.consolidate(relief)
            if measure:
                replay.append((time.perf_counter_ns()-t)/1e6)
    for p,q in zip(walk,walk[1:]):
        motion = MOVES.index(tuple(np.subtract(q,p)))
        a = int(rng.choice(np.flatnonzero(limb==motion)))
        e = (d+torch.from_numpy(rate)).clamp_max(1)
        for j,arm in enumerate((route,other)):
            if q == arm[-1]:
                e[j] = 0
        one.experience(*inputs[p],d,a,inputs[q][1],e)
        learn(memory,p,a,d,e,True)
        raw.append((p,a,d,e))
        records.append((tokens[p],a,tokens[q],d,e))
        d = e
    for i in rng.permutation(len(raw)):
        learn(scrambled,*raw[i])
    fixed = Roll20()
    fixed.fit(records)
    def trial(model,channel=0,anticipate=False,wait=False):
        cell = route[0]
        deficit = torch.zeros(2)
        if not anticipate:
            deficit[channel] = .48
        first = None
        for tick in range(1,4*length+10):
            if wait and deficit.max()<.75 and first is None:
                nxt = cell
            else:
                if model == '20':
                    a = fixed.choose(tokens[cell],deficit,memory.drift.numpy())
                elif model == 'one':
                    a = one.choose(*inputs[cell],deficit)
                else:
                    t = time.perf_counter_ns()
                    a = (memory if model=='end' else scrambled).choose(*inputs[cell],deficit)
                    if model=='end':
                        awake_choice.append((time.perf_counter_ns()-t)/1e6)
                nxt = step(cell,limb[a],floor)
                if first is None:
                    first = nxt==(route,other)[channel][1]
            deficit += torch.from_numpy(rate)
            if nxt==(route,other)[channel][-1]:
                return dict(arrived=True,dead=False,first=bool(first),ticks=tick)
            if deficit.max()>=1:
                return dict(arrived=False,dead=True,first=bool(first),ticks=tick)
            cell = nxt
        return dict(arrived=False,dead=False,first=bool(first),ticks=tick)
    awake_choice = []
    trials = {name:trial(name) for name in ('end','one','20','shuffled')}
    trials.update(swap=trial('end',1),future=trial('end',fast,anticipate=True),
                  delayed=trial('end',fast,anticipate=True,wait=True))
    full_ticks = []
    if extended:
        for name in ('one','20','shuffled'):
            trials[name+'_swap'] = trial(name,1)
        sense,path = inputs[route[0]]
        for _ in range(256):
            t = time.perf_counter_ns()
            action = memory.choose(sense,path,d)
            memory.observe(sense,path,action,d,d)
            full_ticks.append((time.perf_counter_ns()-t)/1e6)
    return dict(full_p95=float(np.percentile(full_ticks,95)) if full_ticks else None,trials=trials,bytes=sum(b.numel()*b.element_size() for b in memory.buffers()),
                update_p95=float(np.percentile(awake,95)),read_p95=float(np.percentile(awake_choice,95)),
                sleep_p95=float(np.percentile(replay,95)),length=length)


def blur(seed):
    # Independent repeated noisy 1-D plant with two coordinates (place and deficit).
    rng = np.random.default_rng(seed)
    rows = []
    for h in (10,30,100,300):
        t = np.arange(h+1)
        mean = np.stack([np.sin(t/42),.3+.5*(1-np.exp(-t/80))],axis=1)
        traces = mean + rng.normal(0,[.04,.013],(24,h+1,2))
        train,test = traces[:12],traces[12:]
        design = np.column_stack([train[:,:-1].reshape(-1,2),np.ones(12*h)])
        coef = np.linalg.lstsq(design,train[:,1:].reshape(-1,2),rcond=None)[0]
        pred = test[:,0].copy()
        for _ in range(h):
            pred = np.column_stack([pred,np.ones(len(test))])@coef
        direct = train[:,-1].mean(axis=0)
        rows.append(dict(h=h,roll=np.abs(pred-test[:,-1]).mean(axis=0).tolist(),
                         end=np.abs(direct-test[:,-1]).mean(axis=0).tolist()))
    return rows


if __name__=='__main__':
    torch.set_num_threads(1)
    data = [dict(seed=s,single=run(s,False),double=run(s,True),blur=blur(s))
            for s in range(8300,8324)]
    Path('measurements/long_endpoint.json').write_text(json.dumps(dict(
        platform=platform.platform(),torch=torch.__version__,lives=data),indent=2)+'\n')
    for geom in ('single','double'):
        for m in data[0][geom]['trials']:
            print(geom,m,[sum(r[geom]['trials'][m][k] for r in data) for k in ('arrived','dead','first')])

