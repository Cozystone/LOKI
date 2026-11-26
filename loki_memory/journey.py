"""Bounded online body and resource-journey memory. Inputs carry no world labels."""

import heapq
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import NamedTuple
import struct
import tempfile

import numpy as np
from numba import njit


@dataclass(frozen=True)
class Recollection:
    steps: np.ndarray
    bedrock_change: np.ndarray
    confidence: float
    source: str = "recalled"


class Journey(NamedTuple):
    journey_id: int
    bedrock_change: np.ndarray
    expected_beats: float
    confidence: float

    @property
    def source(self):
        return "recalled"


class Belief(NamedTuple):
    entropy: float
    confidence: float
    surprise: float

    @property
    def source(self):
        return "recalled"


@njit(cache=True)
def _features(x, u, d):
    f = np.zeros(d, np.float32)
    f[0] = 1
    m = len(u)
    for i in range(m):
        f[1 + i] = u[i]
    k = (d - 1 - m) // 2
    for j in range(len(x)):
        f[1 + m + j % k] += (x[j] - 0.5) / (1 + (len(x) - 1) // k)
    for i in range(k):
        for j in range(m):
            f[1 + m + k + (i + 3 * j) % k] += f[1 + m + i] * u[j]
    return f


@njit(cache=True)
def _body(x, u, weights):
    f = _features(x, u, weights.shape[0])
    pred = x.copy()
    for k in range(len(f)):
        for j in range(len(x)):
            pred[j] += weights[k, j] * f[k]
    return np.minimum(1.0, np.maximum(0.0, pred)).astype(np.float32), f


@njit(cache=True)
def _update(x, u, y, seen, previous, weights, precision, quality):
    pred, f = _body(x, u, weights)
    v = precision @ f
    gain = v / (1 + f @ v)
    # One shared feature precision; unobservable outputs never supply targets.
    for k in range(len(f)):
        for l in range(len(f)):
            precision[k, l] -= gain[k] * v[l]
        for j in range(len(x)):
            if seen[j] and previous[j]:
                weights[k, j] += gain[k] * (y[j] - pred[j])
    for j in range(len(x)):
        if seen[j] and previous[j]:
            quality[j] += 0.015 * (
                (1.0 if abs(pred[j] - y[j]) <= 0.05 else 0.0) - quality[j]
            )


@njit(cache=True)
def _arrive(arrays, y, mask, u, key, located, live):
    (
        ctrl,
        x,
        seen,
        quality,
        mean,
        var,
        w,
        prec,
        centers,
        visits,
        dest,
        count,
        motors,
        changes,
        hits,
        base,
        relief,
        last,
        since,
        order,
        posterior,
        duration,
    ) = arrays
    src = int(ctrl[1])
    if live:
        _update(x, u, y, mask, seen, w, prec, quality)
    novelty = 0.0
    observed = 0
    for k in range(len(x)):
        if mask[k]:
            novelty += (y[k] - mean[k]) ** 2 / (var[k] + 0.0025)
            observed += 1
    if observed and ctrl[7] > 10:
        ctrl[8] = max(0.0, novelty / observed - 3.0)
    ctrl[7] = min(ctrl[7] + 1, 100000)
    if located:
        n = int(ctrl[0])
        j = 0
        dist = 1.0e30
        for i in range(n):
            d = np.mean((centers[i] - key) ** 2)
            if d < dist:
                dist = d
                j = i
        if n < len(visits) and (not n or dist > 0.0025):
            j = n
            ctrl[0] += 1
            centers[j] = key
            if n == 0:
                dist = 1.0
        elif live:
            best = 0.0
            for k in range(dest.shape[1]):
                i = int(dest[src, k])
                if i < 0 or np.mean((centers[i] - key) ** 2) > dist + 0.0001:
                    continue
                c = count[src, k]
                score = c * np.exp(5 * np.sum((motors[src, k] / c - 1) * u))
                if score > best:
                    best = score
                    j = i
        parent = int(ctrl[4])
        slot = int(ctrl[5])
        if (
            live
            and parent >= 0
            and parent != src
            and slot >= 0
            and src != j
            and dest[parent, slot] == src
        ):
            conflict = False
            compatible = False
            for k in range(dest.shape[1]):
                c = count[src, k]
                if c > 1 and np.mean((motors[src, k] / c - u) ** 2) < 1e-6:
                    if dest[src, k] == j:
                        compatible = True
                    else:
                        conflict = True
            if conflict and not compatible:
                n = int(ctrl[0])
                clone = n
                for i in range(n):
                    if (
                        i != src
                        and np.mean((centers[i] - centers[src]) ** 2) < 1e-8
                        and np.any(dest[i] == j)
                    ):
                        clone = i
                        break
                if clone < len(visits):
                    if clone == n:
                        ctrl[0] += 1
                        centers[n] = centers[src]
                        visits[n] = 1
                    dest[parent, slot] = clone
                    src = clone
        visits[j] += 1
        centers[j] += (key - centers[j]) / min(100.0, visits[j])
        if live:
            slot = np.argmin(count[src])
            for k in range(dest.shape[1]):
                if dest[src, k] == j:
                    slot = k
                    break
            if dest[src, slot] != j:
                dest[src, slot] = j
                count[src, slot] = 0
                motors[src, slot] = 0
                changes[src, slot] = 0
            count[src, slot] += 1
            motors[src, slot] += u
            ctrl[4] = src
            ctrl[5] = slot
            for h in range(len(last)):
                since[h] += 1
                if mask[h] and seen[h]:
                    delta = y[h] - x[h]
                    changes[src, slot, h] += (delta - changes[src, slot, h]) / count[
                        src, slot
                    ]
                    if delta > 0:
                        hits[j, h] += 1
                        relief[j, h] += (delta - relief[j, h]) / hits[j, h]
                        base[j, h] += (y[h] - base[j, h]) / hits[j, h]
                        old = int(last[h])
                        if old != j:
                            if old >= 0:
                                order[h, old, j] += 1
                                duration[h, old, j] += (
                                    since[h] - duration[h, old, j]
                                ) / order[h, old, j]
                            last[h] = j
                            since[h] = 0
        else:
            ctrl[4] = -1
            ctrl[5] = -1
        n = int(ctrl[0])
        norm = 0.0
        prior = np.zeros(n, np.float32)
        if src >= 0:
            for i in range(n):
                if posterior[i] <= 0:
                    continue
                scores = np.zeros(dest.shape[1], np.float32)
                for k in range(len(scores)):
                    c = count[i, k]
                    if c > 0:
                        scores[k] = c * np.exp(5 * np.sum((motors[i, k] / c - 1) * u))
                total = scores.sum()
                if total > 0:
                    for k in range(len(scores)):
                        if scores[k] > 0:
                            prior[int(dest[i, k])] += posterior[i] * scores[k] / total
                else:
                    prior[i] += posterior[i]
        for i in range(n):
            cost = np.mean((centers[i] - key) ** 2)
            posterior[i] = np.exp(-max(0.0, cost - dist) / 0.0025) * (
                0.995 * prior[i] + 0.005 / n
            )
            norm += posterior[i]
        if norm > 0:
            posterior[:n] /= norm
        else:
            posterior[:n] = 0
            posterior[j] = 1
        ctrl[1] = j
        ctrl[3] = dist
    else:
        ctrl[1] = -1
        ctrl[4] = -1
        ctrl[5] = -1
    ctrl[2] = 1
    for i in range(len(x)):
        if mask[i]:
            residual = y[i] - mean[i]
            rate = 1 / min(50.0, ctrl[7])
            mean[i] += rate * residual
            var[i] += rate * ((1 - rate) * residual**2 - var[i])
        x[i] = y[i]
        seen[i] = mask[i]


class LokiV12:
    """widths=(bedrock width, other vector/audio width, pooled image width, ...)."""

    def __init__(
        self, contract_signature, widths, motors, budget=262144, place_dim=None
    ):
        self.widths = tuple(int(i) for i in widths)
        self.motors = tuple(int(i) for i in motors)
        self.signature = (
            contract_signature
            if isinstance(contract_signature, str)
            else json.dumps(contract_signature, sort_keys=True, separators=(",", ":"))
        )
        self.budget = int(budget)
        self.place_dim = int(place_dim or 0)
        if (
            not self.widths
            or self.widths[0] < 1
            or min(self.widths) < 0
            or not self.motors
            or min(self.motors) < 1
            or self.place_dim < 0
        ):
            raise ValueError("invalid anonymous body dimensions")
        self.width = sum(self.widths)
        self.bed = self.widths[0]
        n = min(self.width, max(192, self.bed + 1))
        m = min(sum(self.motors), 24)
        d = 1 + m + 2 * min(16, n)
        p = min(self.place_dim or n, 16)
        e = 6
        b = self.bed
        # Metadata is included in both snapshot and budget.
        self.config = dict(
            contract_signature=self.signature,
            widths=self.widths,
            motors=self.motors,
            budget=budget,
            place_dim=place_dim,
        )
        self.header = json.dumps(
            self.config, separators=(",", ":"), ensure_ascii=False
        ).encode()
        base = d * n + d * d + 5 * n + 9 + 2 * b
        per = p + 2 + b * 3 + e * (2 + m + b)
        for s in range(64, 3, -1):
            if 12 + len(self.header) + 4 * (base + s * per + 2 * b * s * s) <= budget:
                break
        else:
            raise ValueError("contract cannot fit the specified budget")
        self.matrix = np.zeros(base + s * per + 2 * b * s * s, np.float32)
        offset = 0

        def view(*shape):
            nonlocal offset
            size = int(np.prod(shape))
            out = self.matrix[offset : offset + size].reshape(shape)
            offset += size
            return out

        self.w = view(d, n)
        self.prec = view(d, d)
        np.fill_diagonal(self.prec, 4)
        self.x = view(n)
        self.seen = view(n)
        self.quality = view(n)
        self.mean = view(n)
        self.var = view(n)
        self.ctrl = view(9)
        self.centers = view(s, p)
        self.visits = view(s)
        self.posterior = view(s)
        self.relief = view(s, b)
        self.base_at = view(s, b)
        self.hits = view(s, b)
        self.dest = view(s, e)
        self.dest.fill(-1)
        self.count = view(s, e)
        self.motor_sum = view(s, e, m)
        self.change = view(s, e, b)
        self.order = view(b, s, s)
        self.duration = view(b, s, s)
        self.last = view(b)
        self.last.fill(-1)
        self.since = view(b)
        assert offset == self.matrix.size
        self.cut()

    @staticmethod
    def _fold(x, n):
        if len(x) == n:
            return x
        idx = np.arange(len(x)) % n
        return (
            np.bincount(idx, weights=x, minlength=n)
            / np.maximum(1, np.bincount(idx, minlength=n))
        ).astype("f4")

    def _motor(self, u):
        u = np.asarray(u, dtype="f4")
        if u.shape != (sum(self.motors),) or not np.all((u == 0) | (u == 1)):
            raise ValueError("invalid motor vector")
        start = 0
        for size in self.motors:
            if u[start : start + size].sum() != 1:
                raise ValueError("each branch requires one active option")
            start += size
        return self._fold(u, self.motor_sum.shape[-1])

    def _sense(self, values, visible):
        x = np.asarray(values, dtype="f4")
        mask = (
            np.ones(self.width, bool) if visible is None else np.asarray(visible, bool)
        )
        if (
            x.shape != (self.width,)
            or mask.shape != x.shape
            or not np.all(np.isfinite(x[mask]))
            or np.any((x[mask] < 0) | (x[mask] > 1))
        ):
            raise ValueError("invalid observation or visibility")
        n = len(self.x)
        b = self.bed
        if n == self.width:
            return np.where(mask, x, self.x), mask
        hist = np.arange(self.width - b) % (n - b)
        count = np.bincount(hist, weights=mask[b:], minlength=n - b)
        pooled = np.bincount(
            hist, weights=np.where(mask[b:], x[b:], 0), minlength=n - b
        )
        return np.r_[
            np.where(mask[:b], x[:b], self.x[:b]),
            np.where(count > 0, pooled / np.maximum(1, count), self.x[b:]),
        ].astype("f4"), np.r_[mask[:b], count > 0]

    def _out(self, x):
        if len(x) == self.width:
            return x.copy()
        return np.r_[x[: self.bed], np.resize(x[self.bed :], self.width - self.bed)]

    def observe(self, values, motor_onehot=None, visible=None, place=None):
        y, mask = self._sense(values, visible)
        u = None if motor_onehot is None else self._motor(motor_onehot)
        src = int(self.ctrl[1])
        live = src >= 0 and self.ctrl[2] == 1 and u is not None
        if place is not None:
            place = np.asarray(place, dtype="f4")
            if place.shape != (self.place_dim,) or not np.isfinite(place).all():
                raise ValueError("invalid place observation")
        key_input = place if self.place_dim else y
        located = key_input is not None and (self.place_dim or mask.any())
        key = (
            self._fold(key_input, self.centers.shape[1])
            if located
            else np.zeros(self.centers.shape[1], dtype="f4")
        )
        if u is None:
            u = np.zeros(self.motor_sum.shape[-1], dtype="f4")
        _arrive(self._arrays(), y, mask, u, key, bool(located), bool(live))

    def _arrays(self):
        return (
            self.ctrl,
            self.x,
            self.seen,
            self.quality,
            self.mean,
            self.var,
            self.w,
            self.prec,
            self.centers,
            self.visits,
            self.dest,
            self.count,
            self.motor_sum,
            self.change,
            self.hits,
            self.base_at,
            self.relief,
            self.last,
            self.since,
            self.order,
            self.posterior,
            self.duration,
        )

    def forecast(self, motor_onehot):
        return self._out(_body(self.x, self._motor(motor_onehot), self.w)[0])

    def belief(self):
        mass = self.posterior[: int(self.ctrl[0])]
        if not len(mass) or mass.sum() == 0:
            return Belief(0.0, 0.0, float(self.ctrl[3]))
        return Belief(
            float(-mass @ np.log(mass + 1e-30)),
            float(mass.max() * np.exp(-self.ctrl[3] / 0.0025 - self.ctrl[8])),
            float(self.ctrl[3]),
        )

    def confidence(self):
        # Pre-outcome support weighted by the online hit rate from observed predictions.
        return self._out(
            self.quality * np.exp(-self.ctrl[3] / 0.01 - self.ctrl[8]) * self.seen
        )

    def foresee(self, motor_sequence):
        x = self.x.copy()
        n = int(self.ctrl[0])
        state = int(self.ctrl[1])
        mass = self.posterior[:n].copy()
        rows = []
        certainty = float(self.confidence().mean())
        for motor in motor_sequence:
            u = self._motor(motor)
            y, _ = _body(x, u, self.w)
            if n and mass.any():
                counts = self.count[:n]
                motor_means = self.motor_sum[:n] / np.maximum(1, counts[:, :, None])
                scores = counts * np.exp(5 * ((motor_means - 1) @ u))
                norm = scores.sum(axis=1)
                prob = scores / np.maximum(norm[:, None], 1e-30) * mass[:, None]
                lost = mass * (norm == 0)
                mass = (
                    np.bincount(
                        np.maximum(self.dest[:n].astype(int), 0).ravel(),
                        weights=prob.ravel(),
                        minlength=n,
                    )
                    + lost
                )
                delta = np.sum(prob[:, :, None] * self.change[:n], axis=(0, 1))
                y[: self.bed] = np.clip(
                    x[: self.bed]
                    + delta
                    + lost.sum() * (y[: self.bed] - x[: self.bed]),
                    0,
                    1,
                )
                certainty *= float(np.clip(1 - 0.02 * (1 - np.max(mass)), 0, 1))
            rows.append(self._out(y))
            x = y
        return Recollection(
            np.asarray(rows, dtype="f4").reshape(-1, self.width),
            x[: self.bed] - self.x[: self.bed],
            certainty,
        )

    def imagine(self, actions=()):
        actions = tuple(actions)
        if len(actions) == 1:
            return self.forecast(actions[0])
        rows = self.foresee(actions).steps
        return rows[-1] if len(rows) else self._out(self.x)

    def advance(self, motor):
        # An imagined tick breaks the chain of measured outcomes.
        y, _ = _body(self.x, self._motor(motor), self.w)
        n = int(self.ctrl[0])
        u = self._motor(motor)
        if n:
            counts = self.count[:n]
            scores = counts * np.exp(
                5 * ((self.motor_sum[:n] / np.maximum(1, counts[:, :, None]) - 1) @ u)
            )
            norm = scores.sum(axis=1)
            b = self.posterior[:n].copy()
            mass = scores / np.maximum(norm[:, None], 1e-30) * b[:, None]
            self.posterior[:n] = np.bincount(
                np.maximum(self.dest[:n].astype(int), 0).ravel(),
                weights=mass.ravel(),
                minlength=n,
            ) + b * (norm == 0)
        self.x[:] = y
        self.ctrl[1] = -1
        self.ctrl[2] = 0
        self.ctrl[4:6] = -1
        self.seen.fill(0)
        return self._out(y)

    def next_resource(self, channel):
        prev = int(self.last[channel])
        n = int(self.ctrl[0])
        counts = self.order[channel, prev, :n].copy() if prev >= 0 else np.zeros(n)
        return counts / max(1, counts.sum())

    def resource_belief(self, channel):
        """Anonymous experienced cues and mass for the next distinct relief location."""
        if not 0 <= channel < self.bed:
            raise ValueError("invalid bedrock channel")
        p = self.next_resource(channel)
        return {"cues": self.centers[: len(p)].copy(), "mass": p, "source": "recalled"}

    def resource_at(self, channel, ahead=0):
        """One-change occupancy estimate using the observed inter-arrival interval."""
        estimate = self.resource_belief(channel)
        last = int(self.last[channel])
        if last >= 0 and estimate["mass"].any():
            interval = max(
                1,
                float(
                    self.duration[channel, last, : len(estimate["mass"])]
                    @ estimate["mass"]
                ),
            )
            phase = (self.since[channel] + ahead - interval) / max(1.0, 0.1 * interval)
            switched = 1 / (1 + np.exp(-np.clip(phase, -40, 40)))
            estimate["mass"] *= switched
            estimate["mass"][last] += 1 - switched
        return estimate

    def goal_example(self, journey_id):
        i = int(journey_id)
        if not 0 <= i < int(self.ctrl[0]) or not self.hits[i].any():
            raise ValueError("no experienced replenishment here")
        return {
            "cue": self.centers[i].copy(),
            "bedrock": self.base_at[i].copy(),
            "observed_channels": self.hits[i] > 0,
            "source": "recalled",
        }

    def predict_effect(self, left, right):
        return self.imagine(left) - self.imagine(right)

    def journeys(self, lack_vector):
        lack = np.asarray(lack_vector, float)
        if (
            lack.shape != (self.bed,)
            or not np.isfinite(lack).all()
            or np.any((lack < 0) | (lack > 1))
        ):
            raise ValueError("invalid lack")
        src = int(self.ctrl[1])
        if src < 0:
            return []
        queue = [(0.0, src, 1.0)]
        seen = set()
        result = []
        cumulative = {src: np.zeros(self.bed, np.float32)}
        costs = {src: 0.0}
        while queue:
            beats, i, conf = heapq.heappop(queue)
            if i in seen:
                continue
            seen.add(i)
            if np.any(self.hits[i] > 0):
                delta = cumulative[i].copy()
                # Arrival examples give a measured conditional endpoint for channels replenished here.
                observed = self.hits[i] > 0
                delta[observed] = self.base_at[i, observed] - (1 - lack[observed])
                result.append(
                    Journey(i, delta, max(1.0, beats), conf * self.belief().confidence)
                )
            for k in np.flatnonzero(self.count[i] > 0):
                j = int(self.dest[i, k])
                if j in seen:
                    continue
                c = self.count[i, k]
                motor = self.motor_sum[i, k] / c
                scores = self.count[i] * np.exp(
                    5
                    * (
                        (self.motor_sum[i] / np.maximum(1, self.count[i, :, None]) - 1)
                        @ motor
                    )
                )
                prob = float(scores[k] / max(1e-30, scores.sum()))
                wait = 1 / max(1e-6, prob)
                if beats + wait < costs.get(j, float("inf")):
                    costs[j] = beats + wait
                    cumulative[j] = cumulative[i] + self.change[i, k]
                    heapq.heappush(queue, (costs[j], j, conf * c / (1 + c)))
        return sorted(
            result,
            key=lambda r: float(np.mean(lack**2 - (lack - r.bedrock_change) ** 2)),
            reverse=True,
        )

    def cut(self):
        self.posterior.fill(0)
        self.ctrl[1:6] = [-1, 0, 0, -1, -1]
        self.x.fill(0)
        self.seen.fill(0)
        self.last.fill(-1)
        self.since.fill(0)

    def save(self, path, private=True):
        path = Path(path)
        matrix = self.matrix.copy()

        def clear(arr, value=0):
            start = (arr.ctypes.data - self.matrix.ctypes.data) // 4
            matrix[start : start + arr.size] = value

        if private:
            for arr in (
                self.x,
                self.mean,
                self.var,
                self.seen,
                self.since,
                self.posterior,
            ):
                clear(arr)
            clear(self.last, -1)
            matrix[(self.ctrl.ctypes.data - self.matrix.ctypes.data) // 4 + 1] = -1
            if not self.place_dim:
                # Sensory anchors can contain a single camera/audio impression.
                for arr in (
                    self.centers,
                    self.visits,
                    self.count,
                    self.change,
                    self.motor_sum,
                    self.order,
                    self.duration,
                    self.hits,
                    self.base_at,
                    self.relief,
                    self.ctrl,
                ):
                    clear(arr)
                clear(self.dest, -1)
                idx = (self.ctrl.ctypes.data - self.matrix.ctypes.data) // 4
                matrix[idx + 1] = -1
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temp = f.name
        try:
            with open(temp, "wb") as f:
                f.write(
                    b"LOKI12A\0" + struct.pack("<I", len(self.header)) + self.header
                )
                f.write(matrix.astype("<f4", copy=False).tobytes())
            os.replace(temp, path)
        finally:
            Path(temp).unlink(missing_ok=True)

    @classmethod
    def load(cls, path, binding):
        # Full constructor descriptor permits a clean start on new hardware/versions.
        fresh = cls(**binding) if isinstance(binding, dict) else None
        with open(path, "rb") as f:
            if f.read(8) != b"LOKI12A\0":
                if fresh is not None:
                    return fresh
                raise ValueError("unknown version; supply current contract")
            length = struct.unpack("<I", f.read(4))[0]
            if length > 262144:
                raise ValueError("invalid header")
            cfg = json.loads(f.read(length))
            if fresh is not None and json.loads(fresh.header) != cfg:
                return fresh
            if fresh is None and str(binding) != cfg["contract_signature"]:
                return cls(**dict(cfg, contract_signature=str(binding)))
            m = fresh if fresh is not None else cls(**cfg)
            data = f.read(m.matrix.nbytes + 1)
            if len(data) != m.matrix.nbytes:
                raise ValueError("invalid checkpoint length")
            m.matrix[:] = np.frombuffer(data, dtype="<f4")
            return m
