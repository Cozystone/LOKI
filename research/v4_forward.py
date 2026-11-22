"""Frozen v3 and factual/paired learning controls for blind translation tests."""
import argparse
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from loki_memory import LokiMemory
from .dark import compound
from .reference import ROOT, setup, write, mse


def frozen_v3():
    root = Path(os.environ.get("LOKI_V3_REFERENCE",ROOT/".git/loki-v3"))
    pin = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if pin!="6d689b5807fd84b488c9743eaa856d5406b9e8cf":
        raise ValueError("wrong v3 reference")
    name = "pinned_loki_v3"
    spec = importlib.util.spec_from_file_location(name,root/"loki_memory/__init__.py",
                                                 submodule_search_locations=[str(root/"loki_memory")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.LokiMemory


def observe(model,frames,actions):
    model.cut()
    model.observe(frames[0])
    for x,a in zip(frames[1:],actions):
        model.observe(x,int(a))


def plans(rng,n,count,flip):
    original = rng.choice([1,2,4],size=(n,20))
    result = np.stack([original.copy(),original.copy()])
    for j in range(n):
        steps = rng.choice(20,int(count[j]),replace=False)
        result[int(flip[j]),j,steps] = 0
        result[1-int(flip[j]),j,steps] = 4
    return result


def simulate(world, actions):
    sim = deepcopy(world)
    frames = [sim.picture(8,4).numpy().reshape(sim.n,96)]
    for act in actions.T:
        sim.step(torch.tensor((act+1)%5))
        frames.append(sim.picture(8,4).numpy().reshape(sim.n,96))
    return np.stack(frames,axis=1)


def pairs(seed,n=24):
    from atanor_os.emerge.room3d_gpu import World
    from atanor_os.emerge.room_vary import make_room
    rng = np.random.default_rng(seed+800)
    w = World(n,make_room(seed),device="cpu",seed=seed,drain=0)
    acts = []
    images = [w.picture(8,4).numpy().reshape(n,96)]
    act = np.zeros(n,int)
    for t in range(80):
        act = np.where(rng.random(n)<.42,rng.integers(5,size=n),act)
        acts.append(act)
        w.step(torch.tensor((act+1)%5))
        images.append(w.picture(8,4).numpy().reshape(n,96))
        if t in (39,79):
            cmds = plans(rng,n,rng.integers(2,19,size=n),rng.integers(2,size=n))
            arms = np.stack([simulate(w,branch) for branch in cmds])
            for j in range(n):
                yield (np.array(images)[:,j],np.array(acts)[:,j],cmds[:,j],arms[:,j])


def fit_inner(model, samples):
    rng = np.random.default_rng(4029)
    opt = torch.optim.Adam(model.parameters(),lr=.001)
    arrays = [torch.tensor(np.stack(v)) for v in zip(*samples)]
    for _ in range(600):
        idx = rng.integers(len(samples),size=24)
        prefix,previous,acts,frames = [v[idx] for v in arrays]
        h = model.zero(len(idx))
        for t in range(previous.shape[1]):
            h,_ = model.step(h,prefix[:,t],previous[:,t])
        h = h.repeat_interleave(2,0)
        acts = acts.flatten(0,1)
        frames = frames.flatten(0,1)
        loss = 0.
        for t in range(20):
            h,y = model.step(h,frames[:,0] if t==0 else torch.zeros_like(frames[:,0]),
                             acts[:,t],torch.full((len(h),),float(t>0)))
            loss += (y-frames[:,t+1]).square().mean()/20
        diff = (y-frames[:,-1]).reshape(-1,2,96)
        loss += (diff[:,0]-diff[:,1]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def train(ref):
    from atanor_os.emerge.inner import Inner
    torch.manual_seed(803)
    streams = [compound(seed,208,5) for seed in range(1,13)]
    inner = Inner(96,5,h=32)
    ref.train_inner(inner,streams,500,803)
    v3 = frozen_v3()(96,grid_shape=(4,8,3))
    v4 = LokiMemory(96,grid_shape=(4,8,3))
    for frames,actions in streams:
        observe(v3,frames,actions)
        observe(v4,frames,actions)
    matched = deepcopy(v3)
    samples = []
    for seed in range(1,13):
        for prefix,previous,cmd,frames in pairs(seed):
            for m in (matched,v4):
                observe(m,prefix,previous)
            v4.learn_effect(*cmd,frames[0,-1],frames[1,-1])
            for arm,act in zip(frames,cmd):
                for m in (matched,v4):
                    observe(m,arm,act)
            samples.append((prefix[-9:-1],previous[-8:],cmd,frames))
    tuned = deepcopy(inner)
    fit_inner(tuned,samples)
    return {"v3":v3,"v3_matched":matched,"v4":v4},{"inner":inner,"inner_matched":tuned}


@torch.no_grad()
def neural_diff(model,h,frame,cmds):
    h = h.repeat(2,1)
    x = torch.tensor(frame[None]).repeat(2,1)
    for i,cmd in enumerate(cmds.T):
        h,y = model.step(h,x,torch.tensor(cmd),torch.full((2,),float(i>0)))
    y = y.numpy().clip(0,1)
    return y[0]-y[1]


def evaluate(seed,base,networks,ref):
    from atanor_os.emerge.room3d_gpu import World
    from atanor_os.emerge.room_vary import make_room
    world = World(1,make_room(seed),device="cpu",seed=seed,drain=0)
    frames,acts = compound(seed,248,5)
    models,nets = deepcopy(base),deepcopy(networks)
    hidden = {k:m.zero(1) for k,m in nets.items()}
    for net in nets.values():
        ref.train_inner(net,[(frames[:129],acts[:128])],64,seed)
    for m in models.values():
        observe(m,frames[:1],[])
    rng = np.random.default_rng(seed+17000)
    rows = []
    for t,a in enumerate(acts):
        with torch.no_grad():
            for k,net in nets.items():
                hidden[k],_ = net.step(hidden[k],torch.tensor(frames[t:t+1]),torch.tensor([a]))
        world.step(torch.tensor([(int(a)+1)%5]))
        current = frames[t+1]
        for m in models.values():
            m.observe(current,int(a))
        if t+1 not in (128,168,208,248):
            continue
        i = len(rows)
        cmd = plans(rng,1,[4*(i+1)],[i%2])[:,0]
        estimate = {k:neural_diff(net,hidden[k],current,cmd) for k,net in nets.items()}
        estimate["zero"] = np.zeros_like(current)
        for k,m in models.items():
            estimate[k] = (m.predict_effect(*cmd) if k=="v4" else m.imagine(cmd[0])-m.imagine(cmd[1]))
        np.testing.assert_array_equal(world.picture(8,4).numpy().ravel(),current)
        # The judge advances independent copies only after forecasts are complete.
        truth = simulate(world,cmd[:1])[0,-1]-simulate(world,cmd[1:])[0,-1]
        rows.append(dict(substitutions=4*(i+1),identical=not np.any(truth).item(),actions=cmd.tolist(),
                         mse={k:mse(p,truth) for k,p in estimate.items()}))
    return dict(seed=seed,pairs=rows,mean={k:float(np.mean([r["mse"][k] for r in rows])) for k in rows[0]["mse"]})


def report(rows):
    avg = {k:float(np.mean([r["mean"][k] for r in rows])) for k in rows[0]["mean"]}
    gates = {}
    for k in avg.keys()-{"v4"}:
        wins = sum(r["mean"]["v4"]<r["mean"][k] for r in rows)
        gain = 1-avg["v4"]/avg[k]
        gates[k] = dict(wins=wins,reduction=gain,passed=wins>=10 and gain>=.1)
    return dict(rows=rows,mean=avg,gates=gates,passed=all(g["passed"] for g in gates.values()))


def store_training(directory,models,nets):
    directory.mkdir(exist_ok=True,parents=True)
    for name,model in models.items():
        model.save(directory/(name+".npz"))
    for name,net in nets.items():
        np.savez(directory/(name+".npz"),**{k:v.detach().numpy() for k,v in net.state_dict().items()})


def restore_training(directory):
    from atanor_os.emerge.inner import Inner
    old = frozen_v3()
    models = {k:cls.load(directory/(k+".npz")) for k,cls in
              (("v4",LokiMemory),("v3",old),("v3_matched",old))}
    nets = {}
    for name in ("inner","inner_matched"):
        net = Inner(96,5,h=32)
        with np.load(directory/(name+".npz"),allow_pickle=False) as data:
            net.load_state_dict({k:torch.tensor(data[k]) for k in data.files})
        nets[name] = net
    return models,nets


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=["train","dev","confirm"])
    args = parser.parse_args()
    _,_,_,ref,_ = setup()
    torch.set_num_threads(1)
    directory = Path(subprocess.check_output(["git","rev-parse","--absolute-git-dir"],text=True).strip())/"v4-training"
    if args.stage=="train":
        store_training(directory,*train(ref))
    else:
        models,nets = restore_training(directory)
        start = 4001 if args.stage=="dev" else 5101
        write("v4_"+args.stage,report([evaluate(s,models,nets,ref) for s in range(start,start+12)]))
