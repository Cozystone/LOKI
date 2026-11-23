"""Exact-output and write-amortization measurements for portable v7."""
from pathlib import Path
import tempfile
from time import perf_counter_ns
import numpy as np
from loki_memory import PortableMemory
from .reference import write


def main():
    rng=np.random.default_rng(7750)
    x=rng.random((257,16,16,3),dtype=np.float32)
    y=rng.random((257,8),dtype=np.float32)
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        for size in (1,16):
            m=PortableMemory.create(Path(tmp)/str(size),"host")
            m.observe(x[0],y[0])
            times=[]
            begin=perf_counter_ns()
            for i in range(1,257,size):
                start=perf_counter_ns()
                with m.wake() as w:
                    for j in range(i,i+size):
                        w.observe(x[j],y[j])
                times.append((perf_counter_ns()-start)/1e6)
            rows.append(dict(window=size,windows=len(times),write_ms_per_tick=(perf_counter_ns()-begin)/1e6/256,
                             commit_inclusive_window_p95_ms=float(np.percentile(times,95)),
                             logical_payload=len(times)*m._field.data.nbytes))
            if size==1:
                expected=m._field.data.copy()
            else:
                np.testing.assert_array_equal(expected,m._field.data)
        samples=[[],[]]; max_diff=0.
        with m.wake() as w:
            for i in range(2020):
                cue=x[i%257]
                pair={}
                for k in (i%2,1-i%2):
                    before=perf_counter_ns()
                    pair[k]=w.recall_signals(cue) if k else w.recall(cue)[1:]
                    elapsed=(perf_counter_ns()-before)/1e6
                    if i>=20:
                        samples[k].append(elapsed)
                max_diff=max(max_diff,float(np.max(np.abs(pair[0][0]-pair[1][0]))),abs(pair[0][1]-pair[1][1]))
        assert max_diff<2e-5
        old=m._field.data.copy()
        try:
            with m.wake() as w:
                w.observe(x[0],y[0])
                raise ArithmeticError()
        except ArithmeticError:
            pass
        np.testing.assert_array_equal(old,m._field.data)
    result=dict(writes=rows,read_p50_p95_ms=np.percentile(samples,[50,95],axis=1).T.tolist(),
                max_signal_difference=max_diff,weights_bit_identical=True,rollback_bit_identical=True)
    write("v7_efficiency",result)
    print(result)
if __name__=="__main__":
    main()
