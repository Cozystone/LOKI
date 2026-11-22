import argparse
from time import perf_counter
import platform
import numpy as np
from loki_memory import LokiMemory
from .reference import write


def run(target):
    rng = np.random.default_rng(3101)
    frame = np.empty((1024, 64), np.float32)
    patterns = rng.uniform(-.15,.15,(8,64)).astype(np.float32)
    mem = LokiMemory(frame.size)
    steps = (target + frame.nbytes-1)//frame.nbytes
    times, blocks, retention = [], [], {}
    visible = np.repeat(np.arange(1024)%4==0,64)
    start = perf_counter()
    for i in range(steps):
        np.add(patterns[rng.integers(8,size=1024)], rng.uniform(.3,.7,(1024,1)).astype(np.float32), out=frame)
        t = perf_counter()
        mem.observe(frame.ravel(), i%3)
        if i%64 == 0:
            times.append(1000*(perf_counter()-t))
        if i == 0:
            first = frame.ravel().copy()  # judge only
        if i in (0,240,10000,steps-1):
            prediction = mem.recall(np.where(visible,first,np.nan),visible)
            retention[str(i)] = float(np.square(prediction[~visible]-first[~visible]).mean())
        if (i+1)%max(1,steps//8) == 0:
            blocks.append([i+1, mem.state_bytes, perf_counter()-start])
    elapsed = perf_counter()-start
    size = steps*frame.nbytes
    assert np.isfinite(mem.matrix).all() and mem.state_bytes <= 262144
    write("stream", dict(steps=steps,input_bytes=size,state_bytes=mem.state_bytes, matrix_shape=mem.matrix.shape,
                         elapsed_seconds=elapsed, MB_s=size/elapsed/1e6, seconds_per_TB_estimate=1e12*elapsed/size,
                         sampled_write_ms_p50_p95=np.percentile(times,[50,95]).tolist(),blocks=blocks,
                         first_scene_hidden_mse=retention,cpu=platform.processor(),numpy=np.__version__))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bytes", type=int, default=16*2**30)
    run(p.parse_args().bytes)
