from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from loki_memory import LokiMemory
from .modalities import Plant,command
from .reference import setup,write
from .v3_common import prior_version,compare


def fit_inner(streams, seed):
    """Train the upstream Inner GRU and head on 288-step windows including long blackouts."""
    from atanor_os.emerge.inner import Inner
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Inner(64,8,h=32)
    gru = torch.nn.GRU(73,32,batch_first=True)
    for name,p in model.g.named_parameters():
        getattr(gru,name+"_l0").data.copy_(p)
    opt = torch.optim.Adam([*gru.parameters(),*model.o.parameters()],lr=.003)
    for i in range(100):
        batch,acts = [],[]
        for j in range(8):
            frames,actions = streams[rng.integers(len(streams))]
            start = int(rng.integers(len(actions)-288+1))
            batch.append(frames[start:start+289])
            acts.append(actions[start:start+288])
        x = torch.tensor(np.stack(batch))
        a = F.one_hot(torch.tensor(np.stack(acts)),8).float()
        shut = torch.arange(288)[None,:]>=torch.tensor(rng.integers(1,48,size=(8,1)))
        inp = torch.cat((x[:,:-1]*~shut[:,:,None],a,shut[:,:,None]),dim=2)
        pred = model.o(gru(inp)[0])
        loss = (pred-x[:,1:]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for name,p in model.g.named_parameters():
        p.data.copy_(getattr(gru,name+"_l0"))
    return model


@torch.no_grad()
def blind_inner(model, start, plan):
    h = model.zero(1)
    observations = torch.tensor(start[None])
    for i,a in enumerate(plan):
        h,y = model.step(h,observations,torch.tensor([a]),torch.tensor([float(i>0)]))
    return y.numpy().ravel().clip(0,1)


class ExponentialsOnly(LokiMemory):
    def _tick(self,traces,u):
        result = super()._tick(traces,u)
        result[5:] = 0
        return result


def world(seed,v2,*,ablate=False):
    rng = np.random.default_rng(seed)
    wiring = rng.normal(size=(64,4))
    wiring /= np.abs(wiring).sum(axis=1,keepdims=True)
    wiring[32] = [0,0,0,1]
    models = {"v2":v2(64),"v3":LokiMemory(64)}
    if ablate:
        models["exp_only"] = ExponentialsOnly(64)
    streams = []
    for _ in range(6):
        plant = Plant(rng,wiring)
        frames,actions = [plant.see()],[]
        for m in models.values():
            m.cut()
            m.observe(frames[0])
        for t in range(512):
            if t%16==0:
                act = int(rng.integers(8))
            u = command(act)
            frames.append(plant.step(u))
            actions.append(act)
            for m in models.values():
                m.observe(frames[-1],u)
        streams.append((np.stack(frames),np.array(actions)))
    inner = fit_inner(streams,seed)
    rows = []
    for trial in range(12):
        plant = Plant(rng,wiring)
        start = plant.see()
        shared = np.repeat(rng.integers(4,size=16),16)
        for kind in ("pulse","timing"):
            # Timing arms have equal lever exposure, but at different times.
            duration = 64 if kind=="pulse" else int(rng.integers(16,64))
            duty = 1 if kind=="pulse" else int(rng.integers(1,4))
            arms = []
            for j in range(2):
                ids = shared.copy()
                high = (j+trial)%2
                if high:
                    ids[:duration:duty] += 4
                elif kind=="timing":
                    ids[144:144+duration:duty] += 4
                p = deepcopy(plant)
                plan = [command(int(a)) for a in ids]
                for u in plan:
                    final = p.step(u)
                record = {"truth":float(final[32]),"last":float(start[32]),"inner":float(blind_inner(inner,start,ids)[32])}
                for name,m in models.items():
                    m.cut()
                    record[name] = float(m.recall(start,np.ones(64,bool),plan,carry=False)[32])
                arms.append(record)
            diff = {k:arms[1][k]-arms[0][k] for k in arms[0]}
            rows.append(dict(kind=kind,duration=duration,duty=duty,difference=diff,arms=arms))
    out = dict(seed=seed,trials=rows)
    for kind in ("pulse","timing"):
        trials = [v for v in rows if v["kind"]==kind]
        for metric in ("difference","endpoint","sign"):
            scores = {}
            for name in (*models,"last","inner"):
                if metric=="endpoint":
                    data = [(a[name]-a["truth"])**2 for v in trials for a in v["arms"]]
                else:
                    diffs = [v["difference"] for v in trials]
                    data = ([(d[name]-d["truth"])**2 for d in diffs] if metric=="difference"
                            else [d[name]*d["truth"]>0 for d in diffs])
                scores[name] = float(np.mean(data))
            out[kind+"_"+metric] = scores
    return out


def run(seeds=range(3201,3209),file="v3_heat"):
    setup()
    torch.set_num_threads(1)
    rows = [world(s,prior_version()) for s in seeds]
    gates = compare(rows,"pulse_difference",("v2","inner","last"),.5,7)
    sign = float(np.mean([r["pulse_sign"]["v3"] for r in rows]))
    write(file,dict(rows=rows,gates=gates,sign=sign,passed=sign>=.9 and all(g["passed"] for g in gates.values())))


if __name__=="__main__":
    run()
