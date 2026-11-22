"""Process-level working set for each organ in a fresh interpreter (judge only)."""
import json
import subprocess
import sys
from pathlib import Path


def measure(kind):
    import numpy as np
    import psutil
    from loki_memory import LokiMemory
    from loki_memory.recurrence import RecurrenceMemory

    proc = psutil.Process()
    baseline = proc.memory_info().rss
    width = 65536 if kind == "general" else 96
    model = LokiMemory(width) if kind == "general" else RecurrenceMemory(width)
    rng = np.random.default_rng(120)
    sampled = proc.memory_info().rss
    for step in range(160):
        x = rng.random(width, dtype=np.float32)
        if kind == "general":
            model.observe(x, step % 3)
            model.imagine([1, 2])
        else:
            model.step(step % 2, x)
        sampled = max(sampled, proc.memory_info().rss)
    info = proc.memory_info()
    return dict(imported_process_rss=baseline, final_rss=info.rss, sampled_high_rss=sampled,
                os_peak_working_set=getattr(info, "peak_wset", None), state_bytes=model.state_bytes)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(json.dumps(measure(sys.argv[1])))
    else:
        results = {kind: json.loads(subprocess.check_output([sys.executable, "-B", __file__, kind], text=True))
                   for kind in ("general", "recurrence")}
        Path("footprint_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(results)
