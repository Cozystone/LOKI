"""Run on the physical target: python -m research.v6_device [--seconds 900]."""
import argparse
import json
import os
import platform
from time import perf_counter, process_time, sleep


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seconds",type=int,default=900)
    args=ap.parse_args()
    if args.seconds<1:
        ap.error("seconds must be positive")
    for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
        os.environ[name]="1"
    import numpy as np
    from loki_memory import ReverseMemory
    m=ReverseMemory()
    rng=np.random.default_rng(6061)
    body=np.array([.2,.8],np.float32)
    m.observe(rng.random((16,16,3),dtype=np.float32),body)
    ms=[]
    missed=0
    cpu=process_time()
    start=perf_counter()
    for i in range(10*args.seconds):
        frame=rng.random((16,16,3),dtype=np.float32)
        before=perf_counter()
        m.observe(frame,body if i%2 else body[::-1],i%2)
        m.pull(frame,body,(0,1))
        end=perf_counter()
        ms.append((end-before)*1000)
        slack=start+(i+1)*.1-end
        missed+=int(slack<0)
        if slack>0:
            sleep(slack)
    elapsed=perf_counter()-start
    p95=float(np.percentile(ms,95))
    print(json.dumps(dict(seconds=elapsed,ticks=len(ms),state_bytes=m.state_bytes,ms_p95=p95,
                          missed_ticks=missed,cpu_core_percent=100*(process_time()-cpu)/elapsed,
                          machine=platform.machine(),cpu=platform.processor(),platform=platform.system(),
                          meets_15min_timing=(args.seconds>=900 and p95<=10 and missed==0)),indent=2))


if __name__=="__main__":
    main()
