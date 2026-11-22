from copy import deepcopy
import numpy as np
import torch
from loki_memory import LokiMemory
from .reference import mse, setup, write


def compound(seed, length):
    from atanor_os.emerge.room3d_gpu import World
    from atanor_os.emerge.room_vary import make_room
    rng = np.random.default_rng(seed)
    w = World(1,make_room(seed),device="cpu",seed=seed,drain=0)
    frames = [w.picture(8,4).numpy().ravel()]
    actions = []
    action = 0
    for t in range(length):
        if rng.random() < .42:
            action = int(rng.integers(4))
        actions.append(action)
        w.step(torch.tensor([action+1]))  # forward, turn in two directions, use
        frames.append(w.picture(8,4).numpy().ravel())
    return np.stack(frames), np.array(actions)


def score(seed, mode, stream, old, recurrence, base, ref):
    frames, acts = stream
    inner = deepcopy(base)
    ref.train_inner(inner,[(frames[:129],acts[:128])],64,seed)
    states = [LokiMemory(96), old(96)]
    for s in states:
        s.observe(frames[0])
    cycle = recurrence(96) if mode != "compound" else None
    h, incoming = inner.zero(1),torch.tensor(frames[:1])
    previous_dark = False
    errors = {}
    with torch.no_grad():
        for i,a in enumerate(acts):
            dark = i>=128 and (i-128)%60 >=40
            h,pred = inner.step(h,incoming,torch.tensor([a]),torch.tensor([float(previous_dark)]))
            if cycle:
                cyc = cycle.step(a,None if dark else frames[i+1])
            if dark:
                outputs = {"v2":states[0].advance(a),"v1":states[1].advance(a),"inner":pred.numpy().ravel().clip(0,1)}
                if cycle:
                    outputs["v1_cycle"] = cyc
                for k, x in outputs.items():
                    errors.setdefault(k,[]).append(mse(x,frames[i+1]))
            else:
                for s in states:
                    s.observe(frames[i+1],a)
            incoming = torch.zeros_like(incoming) if dark else torch.tensor(frames[i+1:i+2])
            previous_dark = dark
    return dict(seed=seed,mse={k:float(np.mean(v)) for k,v in errors.items()})


def main():
    old, recurrence, _, ref, _ = setup()
    from atanor_os.emerge.inner import Inner
    torch.set_num_threads(1)
    result = {}
    for mode in ("rotation","unobserved_forward","compound"):
        if mode != "unobserved_forward":
            torch.manual_seed(803)
            base = Inner(96,4 if mode=="compound" else 2,h=32)
            train = [compound(s,208) if mode=="compound" else ref.record(s,208,False) for s in range(1,13)]
            ref.train_inner(base,train,500,803)
        rows = []
        for seed in range(101,113):
            stream = compound(seed,368) if mode=="compound" else ref.record(seed,368,mode=="unobserved_forward")
            rows.append(score(seed,mode,stream,old,recurrence,base,ref))
        means = {k:float(np.mean([r["mse"][k] for r in rows])) for k in rows[0]["mse"]}
        wins = {k:sum(r["mse"]["v2"]<r["mse"][k] for r in rows) for k in means if k!="v2"}
        result[mode] = dict(rows=rows, mean_mse=means,wins=wins,
                            passed=all(n==12 and means["v2"]<=.9*means[k] for k,n in wins.items()))
    write("dark",result)


if __name__ == "__main__":
    main()
