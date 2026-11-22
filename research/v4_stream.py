"""Measure actual distinct input bytes, fixed state and hidden-sample recall."""
import platform
import time
import numpy as np
from loki_memory import LokiMemory
from .reference import mse,write


def main():
    model = LokiMemory(65536)
    rng = np.random.default_rng(7024)
    first = rng.random(model.width,dtype=np.float32)
    mask = np.repeat(np.arange(1024)%4==0,64)
    cue = np.where(mask,first,np.nan)
    initial_bytes = model.state_bytes
    start = time.perf_counter()
    count = 0
    probes = []
    for i in range(65536):
        x = first if i==0 else rng.random(model.width,dtype=np.float32)
        model.observe(x,i%4 if i else None)
        count += x.nbytes
        if i in (0,240,4095,65535):
            probes.append(dict(frames=i+1,bytes_seen=count,state_bytes=model.state_bytes,
                               hidden_mse=mse(model.recall(cue,mask)[~mask],first[~mask])))
    write("v4_stream",dict(input="independent uniform float32",bytes_seen=count,
                           before=initial_bytes,after=model.state_bytes,seconds=time.perf_counter()-start,
                           constant_half_mse=mse(first[~mask],.5),probes=probes,
                           platform=platform.platform(),numpy=np.__version__))


if __name__=="__main__":
    main()
