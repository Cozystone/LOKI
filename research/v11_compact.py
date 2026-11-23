"""Evaluate the bounded-workspace planning kernel on both final cohorts."""

import json
from time import perf_counter_ns
import numpy as np
from .reference import ROOT
from .v11_world import measure
from loki_memory import BeliefModel


if __name__ == "__main__":
    result = {}
    for name, start, changed in (("sealed", 12701, False), ("holdout", 14001, True)):
        result[name] = [
            measure(s, 16, holdout=changed, compact=True)
            for s in range(start, start + 24)
        ]
    rng = np.random.default_rng(92)
    m = BeliefModel(16)
    inputs = rng.random((96, 24), dtype="f4")
    for _ in range(1000):
        i, j = rng.integers(96, size=2)
        m.fit(inputs[i], rng.integers(4), inputs[j], adapt=False)
    b = m.infer(inputs[0])
    times = []
    for i in range(101):
        start = perf_counter_ns()
        m.compact_step(b, inputs[i % 96])
        if i:
            times.append((perf_counter_ns() - start) / 1e6)
    result["latency_ms_p50_p95"] = np.percentile(times, [50, 95]).tolist()
    result["array_bytes"] = sum(
        a.nbytes for a in vars(m).values() if isinstance(a, np.ndarray)
    )
    (ROOT / "measurements/v11_compact.json").write_text(json.dumps(result, indent=1))
