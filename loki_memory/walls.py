"""Bounded research instruments. Inputs are observations; no method chooses an action."""

import hashlib
import numpy as np


def unit(x):
    a = np.asarray(x, dtype=np.float32)
    if not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError("expected finite normalized observations")
    return a


def _snapshot(tag, *arrays):
    """Self-describing little-endian numeric state including short-lived traces."""
    import struct

    data = bytearray(tag)
    for a in arrays:
        a = np.asarray(a)
        dtype = a.dtype.newbyteorder("<")
        data.extend(dtype.str.encode("ascii") + b"\0")
        data.extend(struct.pack("<I", a.ndim))
        data.extend(np.asarray(a.shape, dtype="<u8").tobytes())
        data.extend(a.astype(dtype, copy=False).tobytes())
    return bytes(data)


class ContextWitness:
    """Defer changes to long-lived contexts until the end of a bounded encounter."""

    def __init__(self, width=12, channels=2, banks=8, edges=96, window=64, rare=False):
        self.rare = rare
        self.low = np.ones((2, banks), "f4") if rare else np.empty((0, banks), "f4")
        self.width = width
        size = 2 * width + 2 + channels
        self.edges = np.zeros((banks, edges, size), "f4")
        self.count = np.zeros((banks, edges), "i4")
        self.pending = np.zeros((window, size), "f4")
        self.total = np.zeros(banks, "f4")
        self.prev = np.zeros(width, "f4")
        self.pos = np.zeros(2, "f4")
        self.cut()

    def cut(self):
        self.t = 0
        self.ready = False
        self.total.fill(0)
        self.low.fill(1)

    def _distance(self, bank, row):
        d = (bank - row) ** 2
        w = self.width
        return (
            d[..., : 2 * w].mean(-1) / 0.015
            + d[..., 2 * w : 2 * w + 2].mean(-1) / 0.04
            + d[..., 2 * w + 2 :].mean(-1) / 0.12
        )

    def evidence(self):
        active = np.flatnonzero(self.count.any(1))
        if not len(active) or self.t < 4:
            return -1, 0.0
        scores = self.total[active] / self.t
        if self.rare:
            scores *= self.low[1, active]
        best = int(scores.argmax())
        alt = max(0.50, np.max(np.delete(scores, best), initial=0.0))
        support = float(np.clip((scores[best] - alt) / 0.40, 0, 1))
        return (int(active[best]) if support >= 0.8 else -1), support

    def observe(self, view, pos, relief):
        view, pos, relief = map(unit, (view, pos, relief))
        if (
            view.shape != self.prev.shape
            or pos.shape != (2,)
            or relief.shape != (self.pending.shape[1] - 2 * self.width - 2,)
        ):
            raise ValueError("incompatible observation binding")
        if self.ready:
            row = np.r_[self.prev, view, (pos - self.pos + 1) / 2, relief]
            self.pending[self.t % len(self.pending)] = row
            distance = self._distance(self.edges, row)
            score = np.exp(-np.where(self.count > 0, distance, np.inf).min(1))
            if self.rare:
                self.low[1] = np.minimum(self.low[1], np.maximum(self.low[0], score))
                self.low[0] = np.minimum(self.low[0], score)
            self.total += score
            self.t += 1
        self.prev[:] = view
        self.pos[:] = pos
        self.ready = True
        return self.evidence()

    def consolidate(self):
        chosen, _ = self.evidence()
        if self.t < 4:
            return -1
        if chosen < 0:
            free = np.flatnonzero(~self.count.any(1))
            if not len(free):
                return -1
            chosen = int(free[0])
        table = self.edges[chosen]
        counts = self.count[chosen]
        for t in range(max(0, self.t - len(self.pending)), self.t):
            row = self.pending[t % len(self.pending)]
            dist = np.where(counts, self._distance(table, row), np.inf)
            j = int(dist.argmin())
            if dist[j] > 0.3:
                free = np.flatnonzero(counts == 0)
                if not len(free):
                    continue
                j = int(free[0])
            counts[j] = min(65535, int(counts[j]) + 1)
            table[j] += (row - table[j]) / min(64, counts[j])
        self.cut()
        return chosen

    def recall_edge(self, context, previous, displacement):
        if not 0 <= context < len(self.count) or not self.count[context].any():
            return None
        previous, displacement = unit(previous), unit(displacement)
        table = self.edges[context]
        w = self.width
        cost = ((table[:, :w] - previous) ** 2).mean(1) / 0.015 + (
            (table[:, 2 * w : 2 * w + 2] - displacement) ** 2
        ).mean(1) / 0.04
        j = np.where(self.count[context], cost, np.inf).argmin()
        return table[j, w : 2 * w].copy(), table[j, 2 * w + 2 :].copy()

    def state(self):
        return _snapshot(
            b"LOKIW102" if self.rare else b"LOKIW101",
            np.asarray([self.t, self.ready], dtype="<i8"),
            self.edges,
            self.count,
            self.pending,
            self.total,
            self.prev,
            self.pos,
            *([self.low] if self.rare else []),
        )

    def sha(self):
        return hashlib.sha256(self.state()).hexdigest()


class DelayedEvidence:
    """Centered eligibility under logged randomized following probabilities."""

    def __init__(self):
        self.decay = np.exp(-1 / np.array([16.0, 64.0, 192.0]))
        self.trace = np.zeros(3)
        self.weight = np.zeros(3)
        self.samples = 0

    def state(self):
        return _snapshot(
            b"LOKIW201", self.decay, self.trace, self.weight, np.int64(self.samples)
        )

    def observe_levels(self, follow, propensity, before, after):
        before, after = unit([before, after])
        return self.observe(follow, propensity, float(after - before))

    def cut(self):
        self.trace.fill(0)

    def observe(self, follow, propensity, integrity_change):
        # follow=None means no choice opportunity at this beat.
        if not -1 <= integrity_change <= 1:
            raise ValueError("integrity change outside range")
        self.trace *= self.decay
        if follow is not None:
            if not 0 < propensity < 1 or follow not in (0, 1):
                raise ValueError("unsupported contrast")
            self.trace += (follow - propensity) / (propensity * (1 - propensity))
            self.samples += 1
        self.weight += integrity_change * self.trace
        return self.weight.mean() / max(1, self.samples)


class PopulationPrior:
    """Equal weight per experienced individual, capped prior effective sample size."""

    def __init__(self):
        self.stats = np.zeros(4)

    def state(self):
        return _snapshot(b"LOKIW301", self.stats)

    def add_body(self, successes, trials):
        if not 0 <= successes <= trials or trials <= 0:
            raise ValueError("invalid counts")
        m = successes / trials
        self.stats += [1, m, m * m, m * (1 - m) / max(1, trials - 1)]

    def estimate(self, successes, trials):
        n, total, squares, noise = self.stats
        if not 0 <= successes <= trials:
            raise ValueError("invalid counts")
        if n < 2:
            return (1 + successes) / (2 + trials)
        mean = total / n
        var = max(1e-6, (squares - total * mean) / (n - 1) - noise / n)
        strength = np.clip(mean * (1 - mean) / var - 1, 1, 32)
        return (mean * strength + successes) / (strength + trials)


class LagSignature:
    """Streaming centered lag correlations; no correspondence or shared clock."""

    def __init__(self, channels, lags=(1, 2, 4, 8)):
        self.lags = np.array(lags)
        self.history = np.zeros((max(lags), channels))
        self.moments = np.zeros((len(lags), 5, channels))
        self.n = 0

    def observe(self, raw):
        x = unit(raw)
        if x.shape != self.history.shape[1:]:
            raise ValueError("wrong binding width")
        for i, lag in enumerate(self.lags):
            if self.n >= lag:
                y = self.history[(self.n - lag) % len(self.history)]
                self.moments[i] += [x, y, x * x, y * y, x * y]
        self.history[self.n % len(self.history)] = x
        self.n += 1

    def state(self):
        return _snapshot(
            b"LOKIW401", self.history, self.moments, self.lags, np.int64(self.n)
        )

    def signature(self):
        if self.n <= max(self.lags):
            raise ValueError("insufficient observations")
        m = self.moments / (self.n - self.lags)[:, None, None]
        x, y, xx, yy, xy = m.transpose(1, 0, 2)
        return (
            (xy - x * y) / np.sqrt(np.maximum(1e-12, (xx - x * x) * (yy - y * y)))
        ).T

    def match(self, other):
        from scipy.optimize import linear_sum_assignment

        d = ((self.signature()[:, None] - other.signature()[None]) ** 2).mean(-1)
        return linear_sum_assignment(d)[1]
