"""Full-width synthetic regression and warmed per-body timing (includes journey queries)."""

import json
from pathlib import Path
from time import perf_counter_ns
import numpy as np
from loki_memory.journey import LokiV12
from loki_memory.journey_121 import LokiV121
from .v12_life import Body


def measure(seed):
    rng = np.random.default_rng(seed)
    w = Body(seed)
    organs = [
        cls(str(seed), w.widths, w.motors, place_dim=30) for cls in (LokiV12, LokiV121)
    ]
    maes = np.zeros((3, 140))
    timings = [[], []]
    for t in range(2000):
        a = int(rng.integers(90))
        u = w.commands[a]
        prev = w.raw.copy()
        value = w.step(a)
        place = np.resize(w.place(), 30)
        preds = []
        for k, m in enumerate(organs):
            start = perf_counter_ns()
            preds.append(m.forecast(u))
            m.observe(value, u, place=place)
            m.journeys(1 - value[:3])
            if t >= 1000:
                timings[k].append((perf_counter_ns() - start) / 1e6)
        if t >= 1000:
            maes += np.abs(np.stack([*preds, prev]) - value)
    return dict(
        seed=seed,
        mae=maes.mean(axis=1).tolist(),
        ticks=1000,
        ms=[np.percentile(x, [50, 95]).tolist() for x in timings],
        state_bytes=[m.matrix.nbytes for m in organs],
    )


if __name__ == "__main__":
    rows = [measure(i) for i in range(25201, 25205)]
    for r in rows:
        r["mae"] = (np.array(r["mae"]) / r["ticks"]).tolist()
    Path("measurements/v12_1_width.json").write_text(json.dumps(rows, indent=1))
