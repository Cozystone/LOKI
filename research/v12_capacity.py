"""Exercise the production observation kernel continuously, taking fixed-size checkpoints."""

from pathlib import Path
import tempfile, json
import numpy as np
from numba import njit
from loki_memory.journey import LokiV12, _arrive


@njit
def stream(arrays, start, end):
    y = np.empty(2, np.float32)
    key = np.empty(1, np.float32)
    u = np.zeros(2, np.float32)
    mask = np.ones(2, np.bool_)
    for tick in range(start, end):
        i = tick % 8
        key[0] = i / 7
        y[0] = (tick % 32) / 31
        y[1] = key[0]
        u[0] = tick % 2
        u[1] = 1 - u[0]
        _arrive(arrays, y, mask, u, key, True, tick > 0)


if __name__ == "__main__":
    m = LokiV12("capacity", [1, 1], [2], place_dim=1)
    rows = []
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m"
        start = 0
        for end in (10000, 10000000):
            stream(m._arrays(), start, end)
            m.save(p)
            rows.append([end, p.stat().st_size, m.matrix.nbytes])
            start = end
    Path("measurements/v12_capacity.json").write_text(json.dumps(rows))
