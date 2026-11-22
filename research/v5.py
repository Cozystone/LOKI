"""Sealed v5 law/exception, interference, capacity and RoomBody-renderer tests."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
import numpy as np
from loki_memory import LokiMemory
from loki_memory.cortex import LongTermCortex
from .reference import ATANOR,ROOT,write,mse


class Buffer:
    """Quantized recent associations. Same action, nearest cue; return cue plus learned change."""
    def __init__(self,width,budget):
        n = (budget-16)//(2*width+1)
        self.keys = np.zeros((n,width),np.uint8)
        self.values = np.zeros((n,width),np.int8)
        self.actions = np.zeros(n,np.uint8)
        self.count = 0

    def add(self,x,a,y):
        i = self.count%len(self.keys)
        self.keys[i] = np.rint(x*255)
        self.values[i] = np.rint((y-x)*127)
        self.actions[i] = a
        self.count += 1

    def predict(self,x,a,b):
        ids = np.flatnonzero(self.actions[:min(self.count,len(self.keys))]==a)
        if not len(ids):
            return b
        dist = ((self.keys[ids].astype(np.float32)/255-x)**2).sum(1)
        return (x+self.values[ids[dist.argmin()]]/127).clip(0,1)


class Trial:
    def __init__(self,core,sizes):
        self.core = core
        self.fields = {f"r{n}":LongTermCortex(core.width,n) for n in sizes}
        self.fields["all"] = LongTermCortex(core.width,16,selective=False)
        self.fifos = {k:Buffer(core.width,f.state_bytes) for k,f in self.fields.items() if k!="all"}
        self.priority = Buffer(core.width,self.fields["r16"].state_bytes)
        self.fields["wave_off"] = LongTermCortex(core.width,16,modes=0)
        self.fields["no_sleep"] = LongTermCortex(core.width,16)
        self.writes = {k:0 for k in self.fields}
        self.mask = np.ones(core.width,bool)
        self.count = 0

    def train(self,x,a,y,b,integrity):
        for k,f in self.fields.items():
            self.writes[k] += f.imprint(x,a,y,b,integrity_change=integrity)
        for fifo in self.fifos.values():
            fifo.add(x,a,y)
        if abs(integrity)>=.08 or mse(y,b)>.12**2:
            self.priority.add(x,a,y)
        self.count += 1
        if self.count%256==0:
            self.sleep()

    def sleep(self):
        for name,c in self.fields.items():
            if name!="no_sleep":
                c.sleep()

    def predictions(self,x,a):
        b = self.core.recall(x,self.mask,[a],carry=False)
        result = dict(v4=b,last=x)
        result.update({k:c.predict(x,a,b) for k,c in self.fields.items()})
        result.update({"fifo_"+k:c.predict(x,a,b) for k,c in self.fifos.items()})
        result["priority_fifo"] = self.priority.predict(x,a,b)
        return result


def summarize(records):
    return {k:float(np.mean([v[k] for v in records])) for k in records[0]}


def errors(pred,y):
    return {k:mse(v,y) for k,v in pred.items()}


def structured(seed):
    rng = np.random.default_rng(seed)
    encoder = rng.normal(size=(6,96)).astype(np.float32)/2
    motor = rng.normal(size=(5,6,96)).astype(np.float32)/3
    trial = Trial(LokiMemory(96),[16])
    probes = {k:[] for k in ("early","middle","late","exception","novel")}
    for i in range(24256):
        latent = rng.uniform(-1,1,6).astype(np.float32)
        latent[0] = (latent[0]+(-.75 if i<8000 else .75 if i<16000 else 0))/1.75
        x = (.5+.32*np.tanh(latent@encoder)).astype(np.float32)
        a = int(rng.integers(5))
        shock = rng.random()<.018
        delta = .17*np.tanh(latent@motor[a]) if a else np.zeros(96,np.float32)
        if shock:
            delta += rng.choice([-.19,.19],96).astype(np.float32)
        y = (x+delta).clip(0,1)
        if i>=24000:
            probes["novel"].append((x,a,y))
            continue
        if shock and len(probes["exception"])<160:
            probes["exception"].append((x,a,y))
        for k,start in (("early",0),("middle",11920),("late",23840)):
            if start<=i<start+160:
                probes[k].append((x,a,y))
        trial.core.cut()
        trial.core.observe(x)
        base = trial.core.imagine([a])
        trial.train(x,a,y,base,.2 if shock else 0.)
        trial.core.observe(y,a)
    trial.sleep()
    scores = {}
    for k,examples in probes.items():
        rows=[]
        for x,a,y in examples:
            cue = (x+rng.normal(0,.008,x.shape)).clip(0,1).astype(np.float32)
            rows.append(errors(trial.predictions(cue,a),y))
        scores[k] = summarize(rows)
    return dict(seed=seed,scores=scores,writes=trial.writes,trained=trial.count)


def room_setup():
    pin=subprocess.check_output(["git","-C",str(ATANOR),"rev-parse","HEAD"],text=True).strip()
    if pin!="dc05860ae6837bcdd4de0be34e811bf0848e444f":
        raise ValueError("expected pinned ATANOR")
    sys.path.insert(0,str(ATANOR))
    import torch
    torch.set_num_threads(1)


def rooms():
    room_setup()
    import torch
    from .dark import compound
    from atanor_os.emerge.room3d_gpu import World
    from atanor_os.emerge.room_vary import make_room
    core = LokiMemory.load(ROOT/"checkpoints/v4-room-grid.npz")
    trial = Trial(core,[16])
    old = []
    for seed in range(1,13):
        frames,actions = compound(seed,384,5)
        core.cut()
        core.observe(frames[0])
        for i,a in enumerate(actions):
            x,y = frames[i:i+2]
            base = core.imagine([int(a)])
            trial.train(x,int(a),y,base,0.)
            core.observe(y,int(a))
            if i%24==0:
                old.append((x,int(a),y))
    trial.sleep()
    scores = summarize([errors(trial.predictions(x,a),y) for x,a,y in old])
    results = []
    for seed in range(6401,6413):
        w = World(1,make_room(seed),device="cpu",seed=seed,drain=0)
        rng = np.random.default_rng(seed)
        next_scores,effect_scores = [],[]
        for i in range(90):
            x = w.picture(8,4).numpy().ravel()
            a = int(rng.integers(5))
            predictions = trial.predictions(x,a)
            if i%3==0:
                mobile = trial.predictions(x,0)
                still = trial.predictions(x,4)
                effect = {k:v-still[k] for k,v in mobile.items()}
                # Re-anchor deployed paired head at the same visible cue.
                judge_core = deepcopy(core)
                judge_core.cut()
                judge_core.observe(x)
                effect["v4_paired"] = judge_core.predict_effect([0],[4])
                arms = []
                for act in (1,0):
                    copy = deepcopy(w)
                    copy.step(torch.tensor([act]))
                    arms.append(copy.picture(8,4).numpy().ravel())
                effect_scores.append(errors(effect,arms[0]-arms[1]))
            w.step(torch.tensor([(a+1)%5]))
            next_scores.append(errors(predictions,w.picture(8,4).numpy().ravel()))
        results.append(dict(seed=seed,scores=dict(novel=summarize(next_scores),effect=summarize(effect_scores))))
    return dict(rows=results,revisited=scores,writes=trial.writes,trained=trial.count)


def gates(rows,metric,needed):
    result = {}
    for k in ("v4","fifo_r16"):
        ours = np.array([r["scores"][metric]["r16"] for r in rows])
        other = np.array([r["scores"][metric][k] for r in rows])
        wins = int(sum(ours<other))
        reduction = float(1-ours.mean()/other.mean())
        result[k] = dict(mean=float(ours.mean()),baseline=float(other.mean()),wins=wins,reduction=reduction,
                         passed=wins>=needed and reduction>=.1)
    return result


def timing():
    rng=np.random.default_rng(6188)
    c=LongTermCortex(96)
    core=LokiMemory(96,grid_shape=(4,8,3))
    from loki_memory.cortex import CorticalMemory
    m=CorticalMemory(core,c)
    x=rng.random(96,dtype=np.float32)
    m.observe(x)
    m.observe(1-x,0)
    c.sleep()
    times={k:[] for k in ("core","combined","cortex","core_observe","combined_observe","sleep")}
    single=deepcopy(core)
    for i in range(600):
        a=i%5
        y=rng.random(96,dtype=np.float32)
        for name,call in (("core",lambda:core.imagine([a])),("combined",lambda:m.predict(a)),
                          ("cortex",lambda:c.predict(x,a,x)),("core_observe",lambda:single.observe(y,a)),
                          ("combined_observe",lambda:m.observe(y,a)),("sleep",lambda:c.sleep())):
            before=perf_counter()
            call()
            times[name].append((perf_counter()-before)*1000)
        x=y
    resources={}
    for n in (4,16,64):
        c=LongTermCortex(96,n)
        fifo=Buffer(96,c.state_bytes)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"field.npz"
            c.save(path)
            resources[str(n)]=dict(array_bytes=c.state_bytes,file_bytes=path.stat().st_size,
                                   fifo_capacity=len(fifo.keys),fifo_bytes=sum(v.nbytes for v in vars(fifo).values()
                                   if isinstance(v,np.ndarray))+16)
    return dict(ms={k:np.percentile(v[20:],[50,95]).tolist() for k,v in times.items()},
                resources=resources,core_bytes=core.state_bytes)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("task",choices=["structured","rooms","timing"])
    ap.add_argument("--seed",type=int)
    args=ap.parse_args()
    if args.task=="structured":
        if args.seed:
            write("v5_confirm_seed_"+str(args.seed),structured(args.seed))
        else:
            rows=[structured(s) for s in range(6301,6305)]
            write("v5_confirm_structured",dict(rows=rows,gates={k:gates(rows,k,3) for k in ("early","novel")}))
    elif args.task=="rooms":
        result=rooms()
        result["gates"]={k:gates(result["rows"],k,10) for k in ("novel","effect")}
        write("v5_confirm_rooms",result)
    else:
        write("v5_confirm_timing",timing())
