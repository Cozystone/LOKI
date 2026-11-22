"""Stress independent, single-exposure effects as an interference counterexample."""
import numpy as np
from loki_memory.cortex import LongTermCortex
from .v5 import Buffer
from .reference import mse,write


def main():
    rng=np.random.default_rng(6580)
    c=LongTermCortex(96)
    fifo=Buffer(96,c.state_bytes)
    probes=[]
    result=[]
    for i in range(24000):
        x=rng.uniform(.25,.75,96).astype(np.float32)
        a=int(rng.integers(5))
        y=x+.18*rng.choice([-1,1],96).astype(np.float32)
        if i<64:
            probes.append((x,a,y))
        fifo.add(x,a,y)
        c.imprint(x,a,y,x,integrity_change=.2)
        if (i+1)%256==0:
            c.sleep()
        if i+1 in (64,256,512,1024,4096,24000):
            c.sleep()
            result.append(dict(events=i+1,cortex=float(np.mean([mse(c.predict(x,a,x),y) for x,a,y in probes])),
                               fifo=float(np.mean([mse(fifo.predict(x,a,x),y) for x,a,y in probes])),
                               no_change=float(np.mean([mse(x,y) for x,a,y in probes]))))
    write("v5_interference",dict(rows=result,seed=6580,first_probes=len(probes),fifo_frames=len(fifo.keys)))


if __name__=="__main__":
    main()
