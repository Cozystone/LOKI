"""Isolate the effect of forward versus use with matched rotations and blind initial states."""
from copy import deepcopy
import numpy as np
import torch
from .dark import compound
from .reference import mse,write
from .v3_common import compare
from .v3_dark import trained_models


@torch.no_grad()
def endpoints(world,models,inner,h,last,commands):
    simulation = deepcopy(world)
    for a in commands:
        simulation.step(torch.tensor([(int(a)+1)%5]))
    result = {"truth":simulation.picture(8,4).numpy().ravel(),"zero":last}
    result.update({name:model.imagine(commands) for name,model in models.items()})
    state = h
    for t,a in enumerate(commands):
        state,y = inner.step(state,torch.tensor(last[None]),torch.tensor([int(a)]),
                             torch.tensor([float(t>0)]))
    result["inner"] = y.numpy().ravel().clip(0,1)
    return result


def room(seed,base,base_inner,ref,controls=4):
    from atanor_os.emerge.room3d_gpu import World
    from atanor_os.emerge.room_vary import make_room
    rng = np.random.default_rng(seed+17000)
    # Recreate the public world at the same final visible state for branching.
    world = World(1,make_room(seed),device="cpu",seed=seed,drain=0)
    frames,actions = compound(seed,248,controls)
    models,inner = deepcopy(base),deepcopy(base_inner)
    ref.train_inner(inner,[(frames[:129],actions[:128])],64,seed)
    for model in models.values():
        model.cut()
        model.observe(frames[0])
    h = inner.zero(1)
    records = []
    for t,a in enumerate(actions):
        with torch.no_grad():
            h,_ = inner.step(h,torch.tensor(frames[t:t+1]),torch.tensor([a]))
        world.step(torch.tensor([(int(a)+1)%5]))
        current = frames[t+1]
        for model in models.values():
            model.observe(current,int(a))
        if t+1 in (128,168,208,248):
            np.testing.assert_array_equal(world.picture(8,4).numpy().ravel(),current)
            count = 4*(1+len(records))
            reference = controls-1
            shared = rng.choice([1,2,reference],size=20)
            indices = rng.choice(20,count,replace=False)
            plans = [shared.copy(),shared.copy()]
            plans[len(records)%2][indices] = 0
            plans[1-len(records)%2][indices] = reference
            p,q = [endpoints(world,models,inner,h,current,plan) for plan in plans]
            effect = p["truth"]-q["truth"]
            scores = {k:mse(p[k]-q[k],effect) for k in p if k!="truth"}
            records.append(dict(n_forward=count,mse=scores,identical=bool(np.all(effect==0))))
    return dict(seed=seed,trials=records,difference={k:float(np.mean([v["mse"][k] for v in records]))
                                                      for k in records[0]["mse"]})


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--idle",action="store_true")
    idle = parser.parse_args().idle
    controls = 5 if idle else 4
    base,inner,ref = trained_models(lattice=True,controls=controls)
    start = 4001 if idle else 3901
    rows = [room(i,base,inner,ref,controls) for i in range(start,start+12)]
    gates = compare(rows,"difference",("zero","inner","v2"),.1,10)
    write("v3_forward_idle" if idle else "v3_forward",dict(rows=rows,gates=gates,passed=all(g["passed"] for g in gates.values())))
