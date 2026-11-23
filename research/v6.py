"""Preregistered raw photo recall and need-dependent T-maze decisions."""
import argparse
import json
import platform
from itertools import product
from time import perf_counter_ns

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from loki_memory import ReverseMemory
from .reference import ROOT, mse, write


def retina(im):
    return np.asarray(im.convert("RGB").resize((16,16), Image.Resampling.BOX), np.float32)/255


def display(x):
    return Image.fromarray(np.rint(255*x).astype(np.uint8), "RGB")


def photos():
    from skimage import data
    pairs = (("astronaut","camera"),("coffee","coins"),("rocket","moon"),("chelsea","grass"))
    frames = []
    memory = ReverseMemory()
    body = np.ones(2, np.float32)
    for past, present in pairs:
        raw = Image.fromarray(getattr(data,past)()).convert("RGB")
        cue = retina(Image.fromarray(getattr(data,present)()))
        previous = retina(raw)
        memory.cut()
        memory.observe(previous,body)
        memory.observe(cue,body,0)
        frames.append((raw,previous,cue))
    memory.cut()
    priors = np.stack([previous for _,previous,_ in frames])
    recalls = []
    rows = []
    size = 176
    cell = 206
    plate = Image.new("RGB",(5*cell+24,4*(size+32)+70),"#f7f5ef")
    drawer = ImageDraw.Draw(plate)
    font = ImageFont.load_default(size=16)
    drawer.text((14,8),"LOKI v6   /   One exposure per transition",font=ImageFont.load_default(size=22),fill="#232d35")
    for col,label in enumerate(("Original photograph","Original at sensor","Current cue","Native recall","Mean of pasts")):
        drawer.text((14+cell*col,43),label,font=font,fill="#28323a")
    for i, (raw,previous,cue) in enumerate(frames):
        recalled = memory.recall(cue)
        recalls.append(recalled)
        mean = priors.mean(axis=0)
        rows.append(dict(past=pairs[i][0],cue=pairs[i][1],recall=mse(recalled,previous),
                         copy_cue=mse(cue,previous),mean=mse(mean,previous)))
        for col,img in enumerate((raw,display(previous),display(cue),display(recalled),display(mean))):
            top = 68+i*(size+32)
            if col==0:
                img=img.copy()
                img.thumbnail((size,size))
            else:
                img=img.resize((size,size),Image.Resampling.NEAREST)
            plate.paste(img,(14+cell*col,top))
        drawer.text((14,top+size+1),pairs[i][0],font=font,fill="#28323a")
        drawer.text((14+2*cell,top+size+1),pairs[i][1],font=font,fill="#28323a")
    out = ROOT/"examples"
    plate.save(out/"v6-photo-recall.png")
    np.savez(out/"v6-photo-native.npz",original=priors,cue=np.stack([c for _,_,c in frames]),recall=np.stack(recalls))
    ambiguous = ReverseMemory()
    for previous in priors[:2]:
        ambiguous.cut()
        ambiguous.observe(previous,body)
        ambiguous.observe(frames[0][2],body,0)
    ambiguous.cut()
    mixed = ambiguous.recall(frames[0][2])
    means = {k:float(np.mean([row[k] for row in rows])) for k in ("recall","copy_cue","mean")}
    return dict(rows=rows,means=means,passed=all(means["recall"]<=.9*means[k] for k in ("mean","copy_cue")),
                ambiguous_mse={k:mse(mixed,v) for k,v in zip(("first","second","midpoint"),(priors[0],priors[1],priors[:2].mean(0)))})


class Fork:
    """Fixed-heading body in a tiled T. Physical coordinates stay in the evaluator."""
    def __init__(self,seed,swap):
        rng = np.random.default_rng(seed)
        self.sign = rng.choice([-1,1])
        self.swap = swap
        self.x = 0
        self.body = np.full(2,.2,np.float32)
        self.floor = {(x,0) for x in range(-3,4)} | {(0,y) for y in range(-3,0)}
        palette = rng.uniform(.16,.9,(15,3))
        self.views = {}
        for x in range(-3,4):
            view = np.empty((16,16,3),np.float32)
            view[:8] = palette[0]*.65
            view[8:] = palette[1]*.7
            for col in range(16):
                dx = (col-7.5)/7
                for depth in np.arange(.05,5.,.05):
                    px,py = x+.5+dx*depth,.5+depth
                    tile = (int(np.floor(px)),int(np.floor(py)))
                    if tile not in self.floor:
                        color = palette[2+(tile[0]+5)%11]
                        height = min(8,max(1,round(3/depth)))
                        view[8-height:8+height,col] = color*(.83+.15*np.cos(px*5))
                        break
            if abs(x)==3:
                channel = int(x>0)^self.swap
                view[9:13,5:11] = ([.05,.58,.86],[.8,.36,.12])[channel]
            self.views[x] = view

    def step(self,action):
        self.x = int(np.clip(self.x+self.sign*(2*action-1),-3,3))
        if abs(self.x)==3:
            self.body[int(self.x>0)^self.swap] = 1.
        return self.views[self.x]

    def fresh(self):
        self.body.fill(.2)
        self.x=0


def excursions(room, order, dark):
    model = ReverseMemory()
    for outward in (order,1-order):
        room.fresh()
        model.cut()
        model.observe(room.views[0],room.body)
        for i,action in enumerate([outward]*3+[1-outward]*3):
            observed = room.step(action)
            model.observe(None if dark and i==3 else observed,room.body,action)
    model.cut()
    return model


def achieved(room,action,need):
    room.fresh()
    room.body[:] = 1.
    room.body[need] = .2
    for _ in range(3):
        room.step(action)
    return float(room.body[need]>.9)


def forks():
    rows=[]
    for seed,dark in product(range(7101,7125),(False,True)):
        by_policy={}
        flips=[]
        for swap,order in product(range(2),repeat=2):
            room=Fork(seed,swap)
            model=excursions(room,order,dark)
            cue=room.views[0]
            assert np.array_equal(cue,Fork(seed,1-swap).views[0])
            pair=[]
            for need in range(2):
                body=np.ones(2,np.float32)
                body[need]=.2
                variants={
                    "full":model.pull(cue,body,(0,1)),
                    "equal_need":model.pull(cue,[.6,.6],(0,1)),
                    "wrong_need":model.pull(cue,body[::-1],(0,1)),
                    "no_path":model.pull(cue,body,(0,1),use_path=False),
                    "no_sight":model.pull(cue,body,(0,1),use_sight=False),
                    "empty":np.zeros(2),
                }
                for name,scores in variants.items():
                    success=.5 if abs(float(scores[0]-scores[1]))<=1e-8 else achieved(room,int(np.argmax(scores)),need)
                    by_policy.setdefault(name,[]).append(success)
                    if name=="full":
                        pair.append(success)
            flips.append(float(pair==[1.,1.]))
        rows.append(dict(seed=seed,dark=dark,score={k:float(np.mean(v)) for k,v in by_policy.items()},
                         both_needs_correct=float(np.mean(flips))))
    summary={}
    for dark in (False,True):
        group=[r for r in rows if r["dark"]==dark]
        scores={k:float(np.mean([r["score"][k] for r in group])) for k in group[0]["score"]}
        full=np.array([r["score"]["full"] for r in group])
        rng=np.random.default_rng(7046)
        ci=np.percentile(rng.choice(full,size=(10000,24)).mean(axis=1),[2.5,97.5]).tolist()
        wins=int(sum(full>.5))
        summary[str(dark)]=dict(score=scores,ci95=ci,rooms_above_chance=wins,
                                both_needs_correct=float(np.mean([r["both_needs_correct"] for r in group])),
                                passed=scores["full"]>=.7 and ci[0]>.5 and wins>=18)
    return dict(rows=rows,summary=summary)


def latency():
    rng=np.random.default_rng(7006)
    memory=ReverseMemory()
    body=np.array([.2,.8],np.float32)
    memory.observe(rng.random((16,16,3),dtype=np.float32),body)
    samples=[]
    for i in range(620):
        x=rng.random((16,16,3),dtype=np.float32)
        start=perf_counter_ns()
        memory.observe(x,body if i%2 else body[::-1],i%2)
        mid=perf_counter_ns()
        memory.pull(x,body,(0,1))
        end=perf_counter_ns()
        memory.recall(x)
        last=perf_counter_ns()
        if i>=20:
            samples.append(np.array([mid-start,end-mid,end-start,last-end])/1e6)
    return dict(state_bytes=memory.state_bytes,core_bytes=memory.core.state_bytes,
                platform=platform.platform(),cpu=platform.processor(),numpy=np.__version__,samples=len(samples),
                ms_p50_p95={k:np.percentile(c,[50,95]).tolist() for k,c in zip(
                    ("observe","pull","observe_plus_pull","recall"),np.array(samples).T)})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("task",choices=("photos","forks","latency"))
    task=parser.parse_args().task
    result={"photos":photos,"forks":forks,"latency":latency}[task]()
    write("v6_"+task,result)
    print(json.dumps(result.get("summary",result.get("means",result)),indent=2))
