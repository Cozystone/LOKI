"""Paired transfer of the published ret6e retina and Core into held-out 3D mazes."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from .reference import ROOT, write
from .v6_maze import Maze, prepare, best, FORWARD, LEFT, RIGHT, LEVELS


PIN="e4f401594f47de45e7eba7126813c31754eb1cf2"
DIGEST="5fed86beb6f10deab1fe8caec4298e7ddb49983a306b7a7c53c221a948fcc023"


def load_host():
    reference=Path(os.environ.get("ATANOR_V6_REFERENCE",ROOT/".git/atanor-v6"))
    if subprocess.check_output(["git","-C",str(reference),"rev-parse","HEAD"],text=True).strip()!=PIN:
        raise ValueError("expected pinned ATANOR checkout")
    checkpoint=reference/"models/room3d_ret6e.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=DIGEST:
        raise ValueError("ret6e checkpoint mismatch")
    sys.path.insert(0,str(reference))
    from atanor_os.emerge import room3d_live as room
    from atanor_os.emerge.core import Core
    room.set_slots(room.CHROMA,1+room.CHROMA)
    room.set_motors(False,both=True)
    net=Core(room.DX,room.M,room.INSIDE,room.PLACE)
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    missing,extra=net.load_state_dict(state,strict=False)
    if extra or set(missing).intersection(dict(net.named_parameters())):
        raise ValueError("unloaded trained parameters")
    return room,net.eval()


def observe_eye(eye,im):
    # Native 64×40 retina wiring. Repeat pixels of the same 16×16 observation.
    sensor=im[(np.arange(40)*16//40)[:,None],(np.arange(64)*16//64)[None,:]]
    eye.take(sensor)
    return torch.from_numpy(eye.sense())[None]


def native_step(world,p,a):
    q=world.step(p,LEFT if a in (2,5) else RIGHT) if a in (2,3,5,6) else p
    return world.step(q,FORWARD) if a in (1,5,6) else q


def feel_action(eye,p,q):
    turn=(q[2]-p[2]+2)%4-2
    moved=p[:2]!=q[:2]
    eye.w._felt[:]=0.
    eye.w._felt[:2]=[2.*moved,turn*np.pi/2]
    eye.step()


def normalized(score):
    span=np.ptp(score)
    return (score-score.mean())/span if span>1e-7 else np.zeros_like(score)


def episode(room,net,world,poses,model,mode,seed,flipped):
    rng=np.random.default_rng(seed)
    body=SimpleNamespace(store=LEVELS.copy(),_felt=np.zeros(6))
    eye=room.Retina(body)
    eye.POOL="max"
    eye.reflex=True
    x=observe_eye(eye,world.view(poses[0]))
    for p,q in zip(poses,poses[1:]):
        feel_action(eye,p,q)
        x=observe_eye(eye,world.view(q))
    p=poses[-1]
    if flipped:
        for _ in range(2):
            q=world.step(p,RIGHT)
            feel_action(eye,p,q)
            x=observe_eye(eye,world.view(q))
            p=q
    changed=0
    for i in range(2*len(world.guide)):
        acts=[a for a in range(7) if a in (0,4) or a in (2,3) or native_step(world,p,a)[:2]!=p[:2]]
        base=normalized(net.pull(x)[0].numpy()[0,acts])
        if mode=="core":
            scores=base
        else:
            recalled=model.recall(world.view(p))
            target=observe_eye(deepcopy(eye),recalled)
            predicted=net.foresee(x)[0,acts,3:]
            proximity=-((predicted-target[:,3:])**2).mean(1).numpy()
            scores=base+normalized(proximity)
        na=best(scores,rng)
        changed+=int(base[na]<base.max()-1e-8)
        q=native_step(world,p,acts[na])
        feel_action(eye,p,q)
        x=observe_eye(eye,world.view(q))
        p=q
        if p[:2]==poses[0][:2]:
            break
    return dict(success=int(p[:2]==poses[0][:2]),decisions=i+1,overrode_base=changed)


@torch.inference_mode()
def run():
    torch.set_num_threads(1)
    room,net=load_host()
    rows=[]
    for seed in range(7201,7225):
        world=Maze(seed)
        model,shuffled,poses=prepare(world,seed)
        for flipped in (False,True):
            rows.append(dict(seed=seed,flipped=flipped,results={mode:episode(
                room,net,world,poses,shuffled if mode=="shuffled" else model,mode,seed,flipped
            ) for mode in ("core","core_recall","shuffled")}))
    summary={}
    for flipped in (False,True):
        group=[r["results"] for r in rows if r["flipped"]==flipped]
        summary[str(flipped)]={m:{k:int(sum(r[m][k] for r in group)) for k in group[0][m]} for m in group[0]}
    return dict(pin=PIN,model="room3d_ret6e.pt",sha256=DIGEST,trainable_params=sum(p.numel() for p in net.parameters()),
                relief_examples=int(net.rel_n),rows=rows,totals=summary)


if __name__=="__main__":
    result=run()
    write("v6_atanor",result)
    print(result["totals"])
