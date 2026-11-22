"""Count persistent state and real CPU streaming cost, excluding the fixture's generator."""
import platform
from time import perf_counter
import tracemalloc
import numpy as np
import psutil
from loki_memory import LokiMemory
from .reference import write


def measure():
    process = psutil.Process()
    rng = np.random.default_rng(4077)
    m = LokiMemory(65536)
    shape = m.matrix.shape
    start_rss = process.memory_info().rss
    start = perf_counter()
    observed = 0
    compute = 0.
    for t in range(4096):
        x = rng.random(m.width,dtype=np.float32)
        tick = perf_counter()
        m.observe(x,t%4)
        compute += perf_counter()-tick
        observed += x.nbytes
    elapsed = perf_counter()-start
    if m.matrix.shape!=shape:
        raise AssertionError("persistent field grew")
    tracemalloc.start()
    m.observe(x,0)
    _,peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,
                bytes_seen=observed,frames=4096,field_bytes=m.state_bytes,wall_seconds=elapsed,
                observe_seconds=compute,mean_observe_ms=compute/4096*1000,
                rss_before=start_rss,rss_after=process.memory_info().rss,transient_write_peak=peak,
                layouts={str(n):LokiMemory(n,grid_shape=g).state_bytes for n,g in
                         [(64,None),(96,(4,8,3)),(1920,(20,32,3)),(1964,(20,32,3)),(65536,None)]})


if __name__=="__main__":
    write("v3_resources",measure())
