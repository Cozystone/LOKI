"""Wall time per completed action for 16-action commits on the same synthetic stream."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from loki_memory import BodyMemory, ContextMemory
from .reference import ROOT


def measure(root):
    rng=np.random.default_rng(9550)
    frames=rng.random((17,16,16,3),dtype=np.float32)
    signals=rng.random((17,8),dtype=np.float32)
    commands=rng.random((16,8),dtype=np.float32)
    # Loop the stream across batch boundaries.
    frames[-1],signals[-1]=frames[0],signals[0]
    raw=ContextMemory.create(root/"raw","A")
    raw.observe(frames[0],signals[0])
    live=BodyMemory.create(root/"live","A")
    times=[[],[]]
    for n in range(48):
        for i in ((0,1) if n%2 else (1,0)):
            m=(raw,live)[i]
            start=perf_counter_ns()
            with m.wake() as w:
                for j,command in enumerate(commands):
                    if i:
                        t=w.begin_action(frames[j],signals[j],command)
                        assert w.finish_action(t,frames[j+1],signals[j+1])
                    else:
                        history=np.stack((signals[j],command,signals[j+1]))
                        w.observe_transition(frames[j+1],history,signals[j+1])
            if n>=8:
                times[i].append((perf_counter_ns()-start)/16000)
    assert np.array_equal(raw._field.data,live._field.data)
    return dict(raw_us=np.percentile(times[0],[50,95]).tolist(),
                live_us=np.percentile(times[1],[50,95]).tolist(),
                field_bytes=live._field.data.nbytes,live_state_bytes=live.state_bytes,
                batches_us_per_tick=times)


if __name__=="__main__":
    with TemporaryDirectory() as d:
        result=measure(Path(d))
    (ROOT/"measurements/v8_handoff_latency.json").write_text(json.dumps(result,indent=2)+"\n")
    print({k:v for k,v in result.items() if k!="batches_us_per_tick"})
