"""High-throughput sufficient-statistic update for exact, unique emission anchors."""

import numpy as np
from numba import njit


@njit
def accumulate(observations, commands, results, emissions, n, counts):
    for k in range(len(commands)):
        indices = np.empty(2, np.int64)
        for side in range(2):
            x = observations[k] if side == 0 else results[k]
            best = 1.0e30
            idx = 0
            for j in range(n):
                d = 0.0
                for h in range(24):
                    d += (x[h] - emissions[j, h]) ** 2
                if d < best:
                    idx = j
                    best = d
                if d == 0:
                    break
            if best > 24e-9 and n < len(emissions):
                idx = n
                n += 1
                emissions[idx] = x
            indices[side] = idx
        counts[commands[k], indices[0], indices[1]] += 1
    return n
