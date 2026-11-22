import numpy as np
from loki_memory import LokiMemory
from .reference import mse,setup,write
from .v3_common import prior_version,compare


def measure(seed,version,scenes,important):
    rng = np.random.default_rng(seed)
    for _ in range(13):
        rng.random(65536,dtype=np.float32)
    waves = rng.normal(size=(8,64)).astype(np.float32)
    waves -= waves.mean(axis=1,keepdims=True)
    waves /= np.sqrt((waves**2).mean(axis=1,keepdims=True))
    models = {"v2":version(65536),"v3":LokiMemory(65536),"off":LokiMemory(65536,sleep=False)}
    mask = np.repeat(np.arange(1024)%4==0,64)
    probes = {}
    out = dict(seed=seed,state_bytes=models["v3"].state_bytes)
    for i in range(2049 if important else 241):
        frame = scenes(rng,1,waves)[0]
        if important:
            frame = .5+(frame-.5)*(1.6 if i in (0,80,160) else .38)
            frame = frame.clip(0,1)
        if i in (0,80,160):
            probes[i] = frame.copy()
        for m in models.values():
            m.observe(frame,1 if i else None)
        if i in (240,2048):
            for t,x in [*probes.items(),("last",frame)]:
                cue = np.where(mask,x,np.nan)
                out[f"t{i}_probe{t}"] = {name:mse(m.recall(cue,mask)[~mask],x[~mask]) for name,m in models.items()}
                out[f"t{i}_probe{t}"]["null"] = mse(x[~mask],.5)
            out[f"t{i}_middle"] = {k:(out[f"t{i}_probe80"][k]+out[f"t{i}_probe160"][k])/2 for k in (*models,"null")}
    return out


def run():
    _,_,scenes,_,_ = setup()
    v2 = prior_version()
    seeds = [522,*range(3401,3413)]
    plain = [measure(s,v2,scenes,False) for s in seeds]
    marked = [measure(s,v2,scenes,True) for s in seeds[1:]]
    first = np.mean([r["t240_probe0"]["v3"] for r in plain[1:]])
    old = np.mean([r["t240_probe0"]["v2"] for r in plain[1:]])
    gates = {f"t{i}":compare(marked,f"t{i}_middle",("off",),.05,10) for i in (240,2048)}
    write("v3_retention",dict(plain=plain,important=marked,first_gate=bool(max(first,plain[0]["t240_probe0"]["v3"])<=.01 and first<=1.1*old),
                              sleep_gates=gates,mid_vs_v2={f"t{i}":compare(marked,f"t{i}_middle",("v2",),0,10) for i in (240,2048)}))


if __name__=="__main__":
    run()
