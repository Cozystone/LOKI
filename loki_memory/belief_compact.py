"""Optional fixed-workspace decision kernel for revision 16 (Numba, no dense T copy)."""

import numpy as np
from numba import njit


@njit
def _row(count, motor, i, reverse, n, row):
    total = 0.0
    for j in range(n):
        row[j] = count[motor, i, j]
        total += row[j]
    if total == 0 and reverse[motor] >= 0:
        for j in range(n):
            row[j] = int(j != i and count[reverse[motor], j, i] > 0)
            total += row[j]
    if total == 0:
        row[i] = 1.0
    else:
        for j in range(n):
            row[j] /= total


@njit
def _reverse(count, n):
    inverse = np.full(4, -1, np.int32)
    for a in range(4):
        best = 0.75
        for other in range(4):
            hits = 0
            candidates = 0
            for i in range(n):
                present = False
                for j in range(n):
                    if i != j and count[a, i, j] > 0:
                        present = True
                        break
                if present:
                    for j in range(n):
                        if i != j and count[other, j, i] > 0:
                            candidates += 1
                            hits += int(count[a, i, j] > 0)
            confidence = hits / max(4, candidates)
            if confidence > best:
                best = confidence
                inverse[a] = other
    return inverse


@njit
def decide(count, emission, attempts, belief, terminal):
    """Return action and predicted belief. Updates only episode-local visit weights."""
    n = len(belief)
    inverse = _reverse(count, n)
    # Sparse rows up to eight successors. Dense rows are streamed if they overflow.
    indices = np.empty((4, n, 8), np.uint8)
    weights = np.empty((4, n, 8), np.float32)
    lengths = np.zeros((4, n), np.uint8)
    risk = np.empty((4, n), np.float32)
    row = np.empty(n, np.float32)
    for a in range(4):
        for i in range(n):
            _row(count, a, i, inverse, n, row)
            entropy = 0.0
            k = 0
            for j in range(n):
                if row[j] > 0:
                    entropy -= row[j] * np.log(row[j])
                    if k < 8:
                        indices[a, i, k] = j
                        weights[a, i, k] = row[j]
                    k += 1
            lengths[a, i] = k
            risk[a, i] = 0.96 * np.exp(-entropy) / (1 + 0.12 * attempts[a, i])
    value = terminal.astype(np.float32)
    q = np.zeros((4, n), np.float32)
    for _ in range(32):
        for a in range(4):
            for i in range(n):
                v = 0.0
                if lengths[a, i] <= 8:
                    for k in range(lengths[a, i]):
                        v += weights[a, i, k] * value[indices[a, i, k]]
                else:
                    _row(count, a, i, inverse, n, row)
                    for j in range(n):
                        v += row[j] * value[j]
                q[a, i] = risk[a, i] * v
        for i in range(n):
            value[i] = max(terminal[i], q[0, i], q[1, i], q[2, i], q[3, i])
    utilities = np.zeros(4, np.float64)
    for a in range(4):
        for i in range(n):
            utilities[a] += q[a, i] * belief[i]
    if belief.max() < 0.85:
        groups = np.arange(n)
        for i in range(n):
            for j in range(i):
                equal = True
                for k in range(24):
                    if np.rint(10000 * emission[i, k]) != np.rint(
                        10000 * emission[j, k]
                    ):
                        equal = False
                        break
                if equal:
                    groups[i] = groups[j]
                    break
        grouped = np.empty(n, np.float64)
        mix = np.empty(n, np.float64)
        for a in range(4):
            mix.fill(0.0)
            conditional = 0.0
            for i in range(n):
                if not belief[i]:
                    continue
                grouped.fill(0.0)
                _row(count, a, i, inverse, n, row)
                for j in range(n):
                    grouped[groups[j]] += row[j]
                for j in range(n):
                    mix[j] += belief[i] * grouped[j]
                    if grouped[j] > 0:
                        conditional -= belief[i] * grouped[j] * np.log(grouped[j])
            entropy = 0.0
            for prob in mix:
                if prob > 0:
                    entropy -= prob * np.log(prob)
            utilities[a] += 0.08 * (entropy - conditional)
    scale = 0.16 * max(0.05, utilities.max())
    for a in range(4):
        visits = 0.0
        for i in range(n):
            visits += attempts[a, i] * belief[i]
        utilities[a] -= scale * np.log1p(visits)
    a = int(np.argmax(utilities))
    prior = np.zeros(n, np.float64)
    for i in range(n):
        attempts[a, i] += belief[i]
        if belief[i] > 0:
            _row(count, a, i, inverse, n, row)
            for j in range(n):
                prior[j] += belief[i] * row[j]
    return a, prior
