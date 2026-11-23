"""Paired transfer of the published ret6e retina and Core into held-out 3D mazes."""
import argparse
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


def scan_native(world,pose,actions):
    # A translation is undone using only the trained body's forward and turn motors.
    undo_forward=(2,2,1,2,2)
    undo={0:(),1:undo_forward,2:(3,),3:(2,),4:(),5:(*undo_forward,3),6:(*undo_forward,2)}
    views=[]
    ticks=0
    for a in actions:
        trial=native_step(world,pose,a)
        views.append(world.view(trial))
        for back in undo[a]:
            trial=native_step(world,trial,back)
        if trial!=pose:
            raise ValueError("nonreversible candidate probe")
        ticks+=1+len(undo[a])
    return views,ticks


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
    changed=ticks=0
    for i in range(2*len(world.guide)):
        acts=[a for a in range(7) if a in (0,4) or a in (2,3) or native_step(world,p,a)[:2]!=p[:2]]
        base=normalized(net.pull(x)[0].numpy()[0,acts])
        if mode=="core":
            scores=base
        elif mode in ("scan","scan_shuffled","scan_visual"):
            views,cost=scan_native(world,p,acts)
            ticks+=cost
            rank=normalized(model.return_scores(world.view(p),acts,views))
            scores=rank if mode=="scan_visual" else base+rank
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
    return dict(success=int(p[:2]==poses[0][:2]),decisions=i+1,overrode_base=changed,scan_steps=ticks)


def oracle(world,poses):
    p=poses[-1]
    commands=[2,2]+[1 if a==FORWARD else (3 if a==LEFT else 2) for a in reversed(world.guide)]
    for i,a in enumerate(commands):
        p=native_step(world,p,a)
        if p[:2]==poses[0][:2]:
            return dict(success=int(i+1<=2*len(world.guide)),decisions=i+1)
    return dict(success=0,decisions=len(commands))


@torch.inference_mode()
def run(native=False):
    torch.set_num_threads(1)
    room,net=load_host()
    rows=[]
    start=7301 if native else 7201
    modes=("core","core_recall","scan","scan_shuffled","scan_visual") if native else ("core","core_recall","shuffled")
    for seed in range(start,start+24):
        world=Maze(seed)
        model,shuffled,poses=prepare(world,seed)
        for flipped in ((False,) if native else (False,True)):
            results={mode:episode(room,net,world,poses,shuffled if "shuffled" in mode else model,
                                  mode,seed,flipped) for mode in modes}
            if native:
                results["oracle"]=oracle(world,poses)
            rows.append(dict(seed=seed,flipped=flipped,results=results))
    summary={}
    for flipped in ((False,) if native else (False,True)):
        group=[r["results"] for r in rows if r["flipped"]==flipped]
        summary[str(flipped)]={m:{k:int(sum(r[m][k] for r in group)) for k in group[0][m]} for m in group[0]}
    return dict(pin=PIN,model="room3d_ret6e.pt",sha256=DIGEST,trainable_params=sum(p.numel() for p in net.parameters()),
                relief_examples=int(net.rel_n),rows=rows,totals=summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--native",action="store_true")
    native=parser.parse_args().native
    result=run(native)
    write("v6_atanor_native" if native else "v6_atanor",result)
    print(result["totals"])
