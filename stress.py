"""Streaming footprint, round trip and timing for the recurrence experiment."""
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tracemalloc

import numpy as np
from loki_memory.recurrence import RecurrenceMemory


def run():
    rng = np.random.default_rng(61)
    signals = rng.uniform(0, 1, (19, 96)).astype(np.float32)
    model = RecurrenceMemory(96)
    state_before = model.state_bytes
    z = visible = 0
    start = perf_counter()
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "recurrence.npz"
        for t in range(100000):
            action = int(rng.integers(2))
            z = (z + (1 if action == 0 else -1)) % len(signals)
            reading = signals[z] if t % 60 < 40 else None
            visible += int(reading is not None)
            prediction = model.step(action, reading)
            if t == 49999:
                model.save(path)
                restored = RecurrenceMemory.load(path)
            if 50000 <= t < 50128:
                np.testing.assert_array_equal(prediction, restored.step(action, reading))
        elapsed = perf_counter() - start
        size = path.stat().st_size
    tracemalloc.start()
    for i in range(100):
        model.step(i % 2, signals[i % len(signals)])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert state_before == model.state_bytes <= 65536
    return dict(steps=100000, visible_sensor_float32_bytes=visible * 96 * 4, numeric_state_bytes=model.state_bytes,
                checkpoint_bytes=size, resume_exact_steps=128, seconds=elapsed, mean_ms=elapsed / 100,
                python_traced_peak_above_resident_bytes=peak, host=platform.processor())


if __name__ == "__main__":
    result = run()
    print(result)
    Path("stress_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
