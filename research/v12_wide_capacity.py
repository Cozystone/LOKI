"""10M chronological arrivals on the full 140/14 contract using the production update."""

import tempfile
import json
from time import perf_counter
from pathlib import Path
import numpy as np
from numba import njit
from loki_memory.journey import LokiV12, _arrive
from .v12_life import Body


@njit
def run(arrays, commands, movement, samples, places, start, end, pos):
    visible = np.ones(140, np.bool_)
    for t in range(start, end):
        a = (t + 7 * (t // 13)) % len(commands)
        pos = (pos + movement[a]) % len(places)
        _arrive(arrays, samples[pos, a], visible, commands[a], places[pos], True, t > 0)
    return pos


if __name__ == "__main__":
    w = Body(884)
    m = LokiV12("wide-capacity", w.widths, w.motors, place_dim=2)
    samples = np.empty((9, len(w.commands), 140), np.float32)
    for i in range(9):
        w.pos = i
        for a, u in enumerate(w.commands):
            samples[i, a] = w.obs(u)
            samples[i, a, 0] = i / 8
    places = np.stack(
        (np.cos(np.arange(9) * 2 * np.pi / 9), np.sin(np.arange(9) * 2 * np.pi / 9)), 1
    ).astype("f4")
    pos = 0
    start = 0
    rows = []
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m"
        for end in (10000, 10000000):
            t = perf_counter()
            pos = run(
                m._arrays(),
                w.commands,
                w.move[w.physical_option],
                samples,
                places,
                start,
                end,
                pos,
            )
            m.save(p)
            rows.append(
                dict(
                    events=end,
                    file_bytes=p.stat().st_size,
                    state_bytes=m.matrix.nbytes,
                    elapsed_s=perf_counter() - t,
                )
            )
            start = end
    Path("measurements/v12_wide_capacity.json").write_text(json.dumps(rows, indent=2))
