"""Separate persistent field, Python working set and transient write allocations."""
from time import perf_counter
import tracemalloc
import numpy as np
import psutil
from loki_memory import LokiMemory
from .reference import write


def main():
    proc = psutil.Process()
    imported = proc.memory_info().rss
    rng = np.random.default_rng(1523)
    m = LokiMemory(65536)
    x = rng.random(m.width,dtype=np.float32)
    for i in range(20):
        m.observe(x,i%3)
    tracemalloc.start()
    m.observe(x,0)
    _,peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mask = np.arange(m.width)%4==0
    times = []
    for _ in range(256):
        t = perf_counter()
        m.recall(x,mask)
        times.append(1000*(perf_counter()-t))
    info = proc.memory_info()
    write("footprint",dict(matrix_bytes=m.state_bytes, imported_rss_bytes=imported, final_rss_bytes=info.rss,
                           os_peak_rss_bytes=getattr(info,"peak_wset",None),traced_write_peak_bytes=peak,
                           recall_ms_p50_p95=np.percentile(times,[50,95]).tolist()))


if __name__=="__main__":
    main()
