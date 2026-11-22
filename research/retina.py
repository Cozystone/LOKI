"""Exploratory native ATANOR retinal-front-end test on eight additional rooms."""
import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from loki_memory import LokiMemory
from loki_memory.atanor import AtanorMemory
from .reference import mse, setup, write


def run(seed, old):
    from atanor_os.emerge.room_body import RoomBody
    from atanor_os.emerge.room_vary import condition
    from atanor_os.emerge.eye_gpu import RetinaGPU
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    w = RoomBody(1,*condition(seed),seed=seed,device="cpu")
    retina = RetinaGPU(1,160,256,device="cpu")
    adapter = AtanorMemory(w.c)
    actions = rng.integers(90,size=144)
    frames = []
    for t in range(145):
        if t:
            w.step(torch.tensor([actions[t-1]]))
        raw = w.raw()
        eye = adaptive_avg_pool2d(retina.look(raw[-1]),(4,4)).numpy().ravel()
        body = np.concatenate([x.numpy().ravel() for x in raw[:-1]])
        features = np.concatenate((eye,0.5+0.5*body/(1+np.abs(body))))
        frames.append(features)
    field, baseline = LokiMemory(len(features)),old(len(features),tile=1)
    vision = np.arange(len(features)) < len(eye)
    totals = {}
    def score(k,x,y):
        totals.setdefault(k,[]).append(mse(x,y))
    for i,x in enumerate(frames):
        a = int(actions[i-1]) if i else None
        if 40<=i<=137:
            for name,mask in (("body_to_retina",~vision),("retina_to_body",vision)):
                cue = np.where(mask,x,np.nan)
                for kind,p in (("v2",field.recall(cue,mask)),("v1",baseline.recall(cue,mask)),
                               ("prior_only",field.imagine()),("v2_no_context",field.recall(cue,mask,carry=False))):
                    score(name+"_"+kind,p[~mask],x[~mask])
            for n in (1,8):
                plan = actions[i-1:i-1+n]
                for kind,p in (("v2",field.imagine(map(adapter.motor,plan))),
                               ("v1",baseline.imagine(plan)),("last",frames[i-1])):
                    for name,mask in (("retina",vision),("body",~vision)):
                        score(f"future{n}_{name}_{kind}",p[mask],frames[i+n-1][mask])
        field.observe(x,adapter.motor(a) if i else None)
        baseline.observe(x,a)
    return dict(seed=seed,visual_dimensions=len(eye),body_dimensions=len(body),
                mse={k:float(np.mean(v)) for k,v in totals.items()})


if __name__=="__main__":
    old,*_ = setup()
    torch.set_num_threads(1)
    rows = [run(s,old) for s in range(2101,2109)]
    write("retina",dict(rows=rows,mean_mse={k:float(np.mean([r["mse"][k] for r in rows])) for k in rows[0]["mse"]}))
