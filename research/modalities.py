"""Synthetic interventional test. Only the fixture owns the delay line and wiring."""
from collections import deque
from copy import deepcopy
import numpy as np
from loki_memory import LokiMemory
from .reference import mse, setup, write


class Plant:
    def __init__(self, rng, wiring):
        self.angle = rng.uniform(-np.pi,np.pi)
        self.level = rng.uniform(-.4,.4)
        self.heat = rng.uniform(-.4,.4)
        self.queue = deque([0.0]*192)
        self.wiring = wiring

    def see(self):
        state = np.array([np.sin(self.angle),np.cos(self.angle),self.level,self.heat])
        sense = 0.5 + 0.45 * (self.wiring @ state)
        return sense.astype(np.float32)

    def step(self, a):
        turn, move, lever = a
        self.angle += .11 * turn
        self.level = .97*self.level+.03*move*np.cos(self.angle)
        self.queue.append(lever)
        self.heat = .96*self.heat+.04*self.queue.popleft()
        return self.see()


def command(i):
    return np.array([1 if i & (1<<j) else -1 for j in range(3)], np.float32)


def run():
    old, _, _, _, _ = setup()
    rows = []
    for seed in range(1701,1709):
        rng = np.random.default_rng(seed)
        wiring = rng.normal(size=(64,4))
        wiring /= np.abs(wiring).sum(axis=1,keepdims=True)
        wiring[32] = [0,0,0,1]  # judge's thermometer is one anonymous sensory component
        mem, v1 = LokiMemory(64), old(64,tile=1)
        for _ in range(6):
            p = Plant(rng,wiring)
            mem.cut()
            v1.cut()
            mem.observe(p.see())
            v1.observe(p.see())
            for t in range(512):
                if t % 16 == 0:
                    i = int(rng.integers(8))
                x = p.step(command(i))
                mem.observe(x,command(i))
                v1.observe(x,i)
        metrics = {}
        def add(k,a,b):
            metrics.setdefault(k,[]).append(mse(a,b))
        for trial in range(12):
            p = Plant(rng,wiring)
            start = p.see()
            mem.cut()
            v1.cut()
            for g in range(4):
                visible = np.arange(64)//16 == g
                cue = np.where(visible,start,np.nan)
                for k,model in (("v2",mem),("v1",v1)):
                    add(f"group{g}_{k}",model.recall(cue,visible)[~visible],start[~visible])
            # Re-anchor both from the cue via read-only reconstruction for v2,
            # nearest live address for v1. No held-out observations are written.
            visible = np.ones(64,bool)
            addr = v1.locate(start,visible)
            future, preds = [], []
            fixed_low = rng.integers(4,size=16)
            for sign in (0,1):
                plant = deepcopy(p)
                ids = np.repeat(fixed_low,16)
                ids[:64] += 4*sign
                plan = [command(i) for i in ids]
                for act in plan:
                    target = plant.step(act)
                pred = mem.recall(start,visible,plan)
                perm = np.random.default_rng(seed*100+trial).permutation(len(plan))
                wrong = mem.recall(start,visible,[plan[j] for j in perm])
                last = start
                old_pred = v1.imagine(ids,start=addr)
                for k,y in (("v2",pred),("v1",old_pred),("shuffled",wrong),("last",last)):
                    add("blind256_"+k,y,target)
                    add("heat256_"+k,y[32],target[32])
                future.append(target[32])
                preds.append(pred[32])
            add("causal_difference_v2",preds[1]-preds[0],future[1]-future[0])
            add("causal_difference_zero",0.,future[1]-future[0])
        # Independent bits in a never-seen stimulus; no correlations across masks.
        fresh = LokiMemory(64)
        for _ in range(256):
            fresh.observe(rng.integers(2,size=64).astype(np.float32))
        correct = total = 0
        for _ in range(128):
            x = rng.integers(2,size=64)
            mask = np.arange(64)%4==0
            correct += int(np.sum((fresh.recall(np.where(mask,x,np.nan),mask)[~mask]>=.5)==x[~mask]))
            total += int((~mask).sum())
        rows.append(dict(seed=seed,mse={k:float(np.mean(v)) for k,v in metrics.items()},iid_correct=correct,iid_total=total))
    write("modalities",dict(rows=rows,mean_mse={k:float(np.mean([r["mse"][k] for r in rows])) for k in rows[0]["mse"]},
                            iid_accuracy=sum(r["iid_correct"] for r in rows)/sum(r["iid_total"] for r in rows)))


if __name__ == "__main__":
    run()
