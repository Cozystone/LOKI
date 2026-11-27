"""Synchronized complete cue/write/link latency. Population is part of the workload."""

import json
from time import perf_counter_ns
from pathlib import Path
import numpy as np
import torch
from loki_memory.trace_store import TraceStore


def bench(device, batch):
    s = TraceStore(3, bodies=batch, device=device)
    # Fully populated deterministic fixture for the measured hit path.
    rng = np.random.default_rng(101)
    x = torch.tensor(rng.random((1024, 32)), device=device, dtype=torch.float32)
    ctx = torch.tensor(rng.random((1024, 16)), device=device, dtype=torch.float32)
    pos = torch.tensor(rng.random((1024, 2)) * 8, device=device, dtype=torch.float32)
    s.features[:, :32] = x
    s.features[:, 32:48] = ctx
    s.features[:, 48:50] = pos
    s.visits.fill_(3)
    s.frame.zero_()
    s.located.fill_(True)
    live = torch.ones(batch, dtype=torch.bool, device=device)
    rise = x[:batch, :3] * 0
    drop = rise[:, 0]
    prev = torch.arange(batch, device=device)
    result = {}
    for mode in ("hit", "allocate", "pressure"):
        if mode == "allocate":
            s.visits.zero_()
        elif mode == "pressure":
            s.visits.fill_(3)
        times = []
        for i in range(140):
            idx = (torch.arange(batch, device=device) + batch * i) % 1024
            a, p, c = x[idx], pos[idx], ctx[idx]
            if mode == "pressure":
                a = 1 - a
                c = 1 - c
            if mode == "allocate" and i % max(1, 1024 // batch) == 0:
                s.visits.zero_()
                s.nbr.fill_(-1)
                s.count.zero_()
                prev = torch.full_like(prev, -1)
            if device == "cuda":
                torch.cuda.synchronize()
            start = perf_counter_ns()
            s.cue(a, p, c)
            found = s.write(
                a,
                p,
                c,
                rise,
                drop,
                140 * list(("hit", "allocate", "pressure")).index(mode) + i,
                live,
            )
            s.link(prev, found, live)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= 20:
                times.append((perf_counter_ns() - start) / 1e6)
            prev = found
        result[mode] = np.percentile(times, [50, 95]).tolist()
    return result


if __name__ == "__main__":
    torch.set_num_threads(1)
    rows = {"cpu1": bench("cpu", 1)}
    if torch.cuda.is_available():
        rows["gpu64"] = bench("cuda", 64)
        rows["gpu"] = torch.cuda.get_device_name()
    print(rows)
    Path("measurements/v13_latency.json").write_text(json.dumps(rows, indent=2))
