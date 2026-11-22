"""Oracle lower bound for v3's fixed pooling/readout on the middle-scene judge."""
import numpy as np
from loki_memory import LokiMemory
from .reference import setup,write


def score(seed,scenes):
    rng = np.random.default_rng(seed)
    for _ in range(13):
        rng.random(65536,dtype=np.float32)
    waves = rng.normal(size=(8,64)).astype(np.float32)
    waves -= waves.mean(axis=1,keepdims=True)
    waves /= np.sqrt((waves**2).mean(axis=1,keepdims=True))
    mask = np.repeat(np.arange(1024)%4!=0,64)
    group = LokiMemory(65536).group
    out = {}
    for i in range(161):
        frame = scenes(rng,1,waves)[0]
        if i in (0,80,160):
            x = (.5+(frame-.5)*1.6).clip(0,1)
            best = x.reshape(-1,group).mean(axis=1).repeat(group)
            out[str(i)] = float(np.mean((x[mask]-best[mask])**2))
    return dict(seed=seed,oracle=out)


if __name__=="__main__":
    _,_,scenes,_,_ = setup()
    write("v3_pooling_floor",[score(s,scenes) for s in range(3701,3713)])
