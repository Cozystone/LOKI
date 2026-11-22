"""100,000 high-dimensional observations, streamed through the public API."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from loki_memory import LokiMemory


def run():
    width, n = 65536, 100000
    rng = np.random.default_rng(1123)
    # The fixture owns these recurring physical signals. The organ sees one at a time.
    templates = rng.uniform(-0.18, 0.18, (16, 64)).astype(np.float32)
    signals = (templates[rng.integers(16, size=(12, 1024))]
               + rng.uniform(0.3, 0.7, (12, 1024, 1))).astype(np.float32).reshape(12, width)
    memory = LokiMemory(width)
    initial = memory.state_bytes
    state = 0
    ms = []
    interval = perf_counter()
    for t in range(n):
        action = int(rng.integers(3))
        state = (state + (1, -1, 5)[action]) % len(signals)
        memory.observe(signals[state], action)
        if (t + 1) % 10000 == 0:
            now = perf_counter()
            ms.append((now - interval) / 10)
            assert memory.state_bytes == initial
            interval = now
    visible = np.arange(1024) % 4 == 0
    hidden = ~np.repeat(visible, 64)
    recalled = memory.recall(signals[0], visible)
    return dict(steps=n, width=width, distinct_source_scenes=12, streamed_float32_bytes=n*width*4,
                state_bytes=initial, used=memory.used, capacity=memory.capacity,
                ms_per_write_by_10000_steps=ms, first_scene_hidden_mse=float(((recalled[hidden]-signals[0, hidden])**2).mean()))


if __name__ == "__main__":
    result = run()
    Path("endurance_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result)
